"""Adaptateur vers l'implémentation amont (handheld_super_resolution, licence MIT).

On n'y réécrit rien : on construit la ``Config`` amont, on reproduit la courte
préparation de ``super_resolution.process`` (profil de bruit, table de correction,
SNR, réglages dépendant du SNR) puis on appelle ``super_resolution.main`` tuile par
tuile. ``process`` n'est pas utilisé directement car il exige un tag NoiseProfile,
ne trouve pas les fichiers ``.DNG`` en majuscules et traite toujours l'image entière.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Callable

import numpy as np

from .noise import NoiseModel, get_noise_lut
from .params import FusionParams


class GpuOutOfMemory(RuntimeError):
    pass


def _is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "out of memory" in text or "out_of_memory" in text or "outofmemory" in text


def free_gpu_memory() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        from numba import cuda

        cuda.current_context().deallocations.clear()
    except Exception:  # noqa: BLE001
        pass


def build_config(params: FusionParams, noise: NoiseModel):
    """``Config`` amont à partir de nos paramètres (post-traitement amont désactivé)."""
    from handheld_super_resolution.config import SNR_BASED, Config

    cfg = Config()
    cfg.scale = float(params.scale)
    cfg.mode = "bayer"
    cfg.verbose = 0
    cfg.debug = False
    cfg.noise_model.alpha = tuple(noise.alpha)
    cfg.noise_model.beta = tuple(noise.beta)

    rob = cfg.robustness
    rob.enabled = bool(params.robustness)
    rob.noise_correction = bool(params.robustness and params.noise_correction)
    rob.save_mask = False
    rob.t, rob.s1, rob.s2, rob.Mt = params.t, params.s1, params.s2, params.mt

    cfg.merging.kernel_type = params.kernel_type
    k = cfg.merging.kernel
    k.k_detail = SNR_BASED if params.k_detail is None else float(params.k_detail)
    k.k_denoise = SNR_BASED if params.k_denoise is None else float(params.k_denoise)
    k.D_th = SNR_BASED if params.d_th is None else float(params.d_th)
    k.D_tr = SNR_BASED if params.d_tr is None else float(params.d_tr)
    k.k_stretch, k.k_shrink = float(params.k_stretch), float(params.k_shrink)

    # Rendu fait par drone_burst_sr (identique pour la fusion et l'image seule).
    cfg.postprocessing.enabled = False
    return cfg


class UpstreamEngine:
    """Fusion d'une tuile RGGB normalisée par ``super_resolution.main``."""

    name = "Handheld Multi-Frame Super-Resolution (IPOL, GPU)"

    def __init__(self, params: FusionParams, noise: NoiseModel):
        self.params = params
        self.noise = noise
        self.config = None
        self.snr: float | None = None

    def prepare(self, ref_rggb: np.ndarray, status: Callable[[str], None] | None = None) -> None:
        from handheld_super_resolution.params import runtime_config, update_snr_config
        from handheld_super_resolution.utils_image import estimate_image_snr

        cfg = runtime_config(build_config(self.params, self.noise))
        if cfg.robustness.enabled and cfg.robustness.noise_correction:
            if status:
                status("Table de correction du bruit (Monte Carlo, une fois par profil)…")
            lut = get_noise_lut(self.noise, trials=self.params.lut_trials)
            sigma, dsq = lut.sigma_noise_sq, lut.d_noise_sq
        else:
            sigma = dsq = np.zeros(2, dtype=np.float32)

        # SNR estimé une seule fois sur l'image entière puis imposé à toutes les tuiles,
        # pour que la taille des blocs d'alignement et les noyaux soient cohérents.
        step = max(1, int(np.sqrt(ref_rggb.size / 4e6)))
        sub = _subsample_cfa(ref_rggb, step)
        self.snr = float(estimate_image_snr(sub, self.noise.alpha, self.noise.beta))
        update_snr_config(cfg, self.snr)
        cfg.noise_model.sigma_sq_curve = np.asarray(sigma, dtype=np.float32).tolist()
        cfg.noise_model.d_sq_curve = np.asarray(dsq, dtype=np.float32).tolist()
        self.config = cfg

    def fuse_tile(self, ref: np.ndarray, comp: np.ndarray,
                  on_frame: Callable[[], None] | None = None) -> np.ndarray:
        from handheld_super_resolution import super_resolution as hsr
        from handheld_super_resolution.params import sanitize_config
        from handheld_super_resolution.utils import DEFAULT_NUMPY_FLOAT_TYPE

        if self.config is None:
            raise RuntimeError("prepare() doit être appelé avant fuse_tile()")
        cfg = copy.deepcopy(self.config)
        sanitize_config(cfg, ref.shape)
        try:
            with _count_merges(hsr, on_frame):
                out, _ = hsr.main(ref.astype(DEFAULT_NUMPY_FLOAT_TYPE, copy=False),
                                  comp.astype(DEFAULT_NUMPY_FLOAT_TYPE, copy=False), cfg)
        except Exception as error:  # noqa: BLE001
            if _is_oom(error):
                free_gpu_memory()
                raise GpuOutOfMemory(str(error)) from error
            raise
        return out

    def describe_config(self) -> str:
        return self.config.dump() if self.config is not None else ""


def _subsample_cfa(x: np.ndarray, step: int) -> np.ndarray:
    """Garde un bloc 2x2 sur ``step`` dans chaque direction (phase CFA conservée)."""
    if step <= 1:
        return np.ascontiguousarray(x)
    g = 2 * step
    h, w = (x.shape[0] // g) * g, (x.shape[1] // g) * g
    blocks = x[:h, :w].reshape(h // g, g, w // g, g)[:, :2, :, :2]
    return np.ascontiguousarray(blocks).reshape(h // step, w // step)


@contextmanager
def _count_merges(module, callback: Callable[[], None] | None):
    """Compte les appels à ``merge`` (un par image) pour la barre de progression.

    ``super_resolution.main`` enveloppe la fonction globale ``merge`` à chaque appel :
    la remplacer temporairement suffit, sans modifier le code amont.
    """
    if callback is None:
        yield
        return
    original = module.merge

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        callback()
        return result

    module.merge = counted
    try:
        yield
    finally:
        module.merge = original
