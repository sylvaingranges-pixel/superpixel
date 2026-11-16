"""Chaîne complète : lecture -> contrôles -> bruit -> fusion par tuiles -> rendu."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from . import baseline as baseline_mod
from .cfa import rgb_from_rggb, to_rggb
from .dng_io import FrameInfo, list_dng_files, load_normalized, read_burst_info
from .gpu import GpuStatus, detect_gpu
from .noise import NoiseModel, resolve_noise_model
from .params import FusionParams
from .render import RenderParams, apply_orientation, camera_to_srgb_matrix, render
from .tiling import (GB, MIN_TILE, MemoryEstimate, choose_tile_size, estimate_memory,
                     fuse_tiled, output_shape, plan_tiles)
from .validation import Issue, check_brightness, has_errors, validate_burst

ProgressFn = Callable[[float, str], None]


class BurstError(ValueError):
    """Rafale inutilisable (les détails sont dans ``issues``)."""

    def __init__(self, issues: Sequence[Issue]):
        self.issues = list(issues)
        super().__init__("\n".join(str(i) for i in self.issues if i.level == "erreur"))


class GpuUnavailable(RuntimeError):
    pass


class Engine(Protocol):
    name: str

    def prepare(self, ref_rggb: np.ndarray, status: Callable[[str], None] | None = None) -> None: ...

    def fuse_tile(self, ref: np.ndarray, comp: np.ndarray,
                  on_frame: Callable[[], None] | None = None) -> np.ndarray: ...


def _noop(_f: float, _m: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Analyse (rapide : métadonnées seulement)
# --------------------------------------------------------------------------- #
def resolve_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        paths = list_dng_files(source)
    elif isinstance(source, (str, Path)):
        paths = [Path(source)]
    else:
        paths = sorted(Path(p) for p in source)
    paths = [p for p in paths if p.suffix.lower() == ".dng"]
    if not paths:
        raise FileNotFoundError("Aucun fichier .dng trouvé.")
    return paths


@dataclass
class Analysis:
    infos: list[FrameInfo]
    issues: list[Issue]
    memory: dict[float, MemoryEstimate]

    @property
    def ok(self) -> bool:
        return not has_errors(self.issues)

    def table(self) -> list[dict]:
        return [i.summary() for i in self.infos]

    def text(self) -> str:
        lines = [f"{len(self.infos)} images — {self.infos[0].make} {self.infos[0].model} — "
                 f"{self.infos[0].mode_label}"]
        lines += [str(i) for i in self.issues]
        lines.append("Mémoire estimée :")
        lines += [f"  ×{s:g} : {m.describe()}" for s, m in self.memory.items()]
        return "\n".join(lines)


def analyze(source, scales: Sequence[float] = (1.0, 1.5, 2.0)) -> Analysis:
    infos = read_burst_info(resolve_paths(source))
    issues = validate_burst(infos)
    ref = infos[0]
    memory = {s: estimate_memory(ref.height, ref.width, len(infos), s) for s in scales}
    return Analysis(infos, issues, memory)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
@dataclass
class Burst:
    infos: list[FrameInfo]            # infos[0] = image de référence
    stack_native: np.ndarray          # (N, H, W) float32 normalisé, disposition CFA d'origine
    issues: list[Issue]
    reference_name: str
    sharpness: list[float] = field(default_factory=list)

    @property
    def cfa(self) -> np.ndarray:
        return self.infos[0].cfa

    @property
    def stack(self) -> np.ndarray:
        """Pile en disposition RGGB (vue, sans copie)."""
        return to_rggb(self.stack_native, self.cfa)


def _sharpness(frame: np.ndarray) -> float:
    """Netteté : variance du laplacien sur la luminance binnée 2x2 (insensible au CFA)."""
    import cv2

    h, w = frame.shape
    lum = frame[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    return float(cv2.Laplacian(lum.astype(np.float32), cv2.CV_32F).var())


def load_stack(source, reference: str = "sharpest", max_frames: int | None = None,
               progress: ProgressFn = _noop) -> Burst:
    paths = resolve_paths(source)
    infos = read_burst_info(paths)
    issues = validate_burst(infos)
    if has_errors(issues):
        raise BurstError(issues)
    if max_frames:
        infos = infos[:max_frames]
    ref = infos[0]
    n, h, w = len(infos), ref.height, ref.width
    stack = np.empty((n, h, w), dtype=np.float32)
    for k, info in enumerate(infos):
        progress(0.12 * k / n, f"Lecture {info.path.name} ({k + 1}/{n})")
        load_normalized(info, out=stack[k])
    issues += check_brightness([float(f[::8, ::8].mean()) for f in stack])

    sharp = [_sharpness(f) for f in stack]
    r = int(np.argmax(sharp)) if reference == "sharpest" else 0
    if r != 0:  # échange en place (pas de copie de la pile entière)
        tmp = stack[0].copy()
        stack[0] = stack[r]
        stack[r] = tmp
        infos[0], infos[r] = infos[r], infos[0]
        sharp[0], sharp[r] = sharp[r], sharp[0]
    if reference == "sharpest" and min(sharp) < 0.6 * max(sharp):
        issues.append(Issue("info", "Netteté très variable dans la rafale (flou de bougé ?) : "
                                    f"référence = {infos[0].path.name}, la plus nette."))
    return Burst(infos, stack, issues, infos[0].path.name, sharp)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
@dataclass
class FusionResult:
    fused: np.ndarray            # sRGB uint16 (H·s, W·s, 3), orienté
    single: np.ndarray           # image seule rawpy, même taille et même rendu
    noise: NoiseModel
    issues: list[Issue]
    engine: str
    tiles: int
    tile_size: int | None
    snr: float | None
    seconds: float
    reference_name: str
    n_frames: int
    scale: float
    config_dump: str = ""

    def report(self) -> str:
        h, w = self.fused.shape[:2]
        lines = [
            f"Moteur : {self.engine}",
            f"{self.n_frames} images, référence {self.reference_name}, échelle ×{self.scale:g} "
            f"→ {w}x{h} ({w * h / 1e6:.0f} MP)",
            self.noise.describe(),
            f"SNR estimé : {self.snr:.1f} dB" if self.snr is not None else "",
            f"Tuiles : {self.tiles}" + (f" de {self.tile_size} px" if self.tile_size else " (image entière)"),
            f"Durée : {self.seconds:.0f} s",
        ]
        lines += [str(i) for i in self.issues]
        return "\n".join(line for line in lines if line)


def render_params_for(info: FrameInfo, params: FusionParams) -> RenderParams:
    return RenderParams(white_balance=info.white_balance,
                        camera_to_srgb=camera_to_srgb_matrix(info.xyz_to_cam),
                        sharpen_amount=params.sharpen_amount, sharpen_radius=params.sharpen_radius,
                        exposure_ev=params.exposure_ev)


def render_single(info: FrameInfo, out_shape: tuple[int, int], rp: RenderParams) -> np.ndarray:
    """Image seule (rawpy), agrandie à ``out_shape`` (avant orientation), puis rendue."""
    lin = baseline_mod.demosaic_linear(info)
    lin = baseline_mod.upscale(lin, out_shape)
    return apply_orientation(render(lin, rp), info.orientation)


def _host_memory_warning(est: MemoryEstimate) -> list[Issue]:
    try:
        import psutil

        avail = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        return []
    if est.host_bytes > avail:
        return [Issue("avertissement",
                      f"RAM disponible {avail / GB:.1f} Go < besoin estimé {est.host_bytes / GB:.1f} Go : "
                      "risque d'échec. Réduisez l'échelle ou le nombre d'images.")]
    return []


def fuse_burst(burst: Burst, params: FusionParams, progress: ProgressFn = _noop,
               engine: Engine | None = None, gpu: GpuStatus | None = None) -> FusionResult:
    """Fusionne une rafale chargée. Sans ``engine``, utilise l'algorithme amont sur GPU."""
    params.validate()
    t0 = time.perf_counter()
    issues = list(burst.issues)
    ref = burst.infos[0]
    n, h, w = burst.stack_native.shape
    if n < 2:
        raise BurstError([Issue("erreur", "Il faut au moins 2 images.")])

    free_vram = None
    if engine is None:
        gpu = gpu or detect_gpu()
        if not gpu.ready:
            raise GpuUnavailable(gpu.message)
        free_vram = gpu.free_bytes
    stack = burst.stack

    progress(0.13, "Modèle de bruit…")
    noise = resolve_noise_model(ref.noise_profile, ref.iso, stack, params.noise_alpha,
                                params.noise_beta, prefer_estimate=params.estimate_noise)
    if not noise.reliable:
        issues.append(Issue("avertissement", f"Modèle de bruit peu fiable ({noise.source}) : "
                                             "la robustesse peut être mal calibrée."))
    if engine is None:
        from .engine import UpstreamEngine

        engine = UpstreamEngine(params, noise)
    elif hasattr(engine, "noise"):
        engine.noise = noise  # type: ignore[attr-defined]

    est = estimate_memory(h, w, n, params.scale)
    issues += _host_memory_warning(est)

    progress(0.16, "Préparation de l'algorithme…")
    engine.prepare(stack[0], status=lambda m: progress(0.18, m))

    tile_size = choose_tile_size(h, w, params.scale, free_vram, params.tile_size)
    oh, ow = output_shape(h, w, params.scale)
    while True:
        tiles = plan_tiles(h, w, tile_size)
        total = len(tiles) * n
        done = [0]

        def on_frame():
            done[0] += 1
            progress(0.25 + 0.6 * done[0] / total,
                     f"Fusion : tuile {done[0] // n + (1 if done[0] % n else 0)}/{len(tiles)}, "
                     f"image {(done[0] - 1) % n + 1}/{n}")

        def fuse_fn(ref_t, comp_t):
            return engine.fuse_tile(ref_t, comp_t, on_frame=on_frame)

        progress(0.25, "Fusion (la 1re exécution compile les kernels CUDA : ~1 min)…")
        try:
            linear = fuse_tiled(stack, params.scale, tiles, fuse_fn)
            break
        except Exception as error:  # noqa: BLE001
            from .engine import GpuOutOfMemory

            if not isinstance(error, GpuOutOfMemory):
                raise
            current = tile_size or max(h, w)
            if current <= MIN_TILE:
                raise
            tile_size = max(MIN_TILE, (current * 2 // 3) // 64 * 64)
            issues.append(Issue("info", f"Mémoire GPU insuffisante : nouvel essai avec des tuiles "
                                        f"de {tile_size} px."))

    nan_count = int(np.count_nonzero(~np.isfinite(linear)))
    if nan_count:
        warnings.warn(f"{nan_count} valeurs non finies remplacées par 0")
        np.nan_to_num(linear, copy=False)

    progress(0.88, "Rendu couleur…")
    rp = render_params_for(ref, params)
    fused = apply_orientation(render(rgb_from_rggb(linear, burst.cfa), rp), ref.orientation)
    del linear
    progress(0.94, "Image seule de référence (rawpy)…")
    single = render_single(ref, (oh, ow), rp)
    progress(1.0, "Terminé")
    return FusionResult(
        fused=fused, single=single, noise=noise, issues=issues,
        engine=getattr(engine, "name", type(engine).__name__), tiles=len(tiles),
        tile_size=tile_size, snr=getattr(engine, "snr", None),
        seconds=time.perf_counter() - t0, reference_name=burst.reference_name, n_frames=n,
        scale=params.scale,
        config_dump=engine.describe_config() if hasattr(engine, "describe_config") else "",
    )


def run(source, params: FusionParams, progress: ProgressFn = _noop,
        engine: Engine | None = None) -> FusionResult:
    gpu = None
    if engine is None:
        gpu = detect_gpu()
        if not gpu.ready:  # échouer tôt, avant de charger des Go de données
            raise GpuUnavailable(gpu.message)
    burst = load_stack(source, params.reference, params.max_frames, progress)
    return fuse_burst(burst, params, progress, engine=engine, gpu=gpu)
