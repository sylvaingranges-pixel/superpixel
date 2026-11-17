"""Auto-test de bout en bout sur une rafale synthétique (décalages sous-pixel connus).

``python -m drone_burst_sr.selftest`` (GPU requis) génère une rafale DNG, la fusionne à ×2
et la compare à une seule image dématricée par rawpy puis agrandie :

- **détail** (texture fine non périodique) : corrélation avec la vérité
  entre 0,45 et 0,85 cycle/pixel, au-delà de la limite de Nyquist d'une image seule. Cette
  mesure ignore le flou et ne compte que le détail juste (pas le moiré ni le bruit) ;
- **aplats** : PSNR dans les zones uniformes, c'est-à-dire le débruitage ;
- **basses fréquences** : PSNR sur la mire (à titre indicatif : une fusion est plus douce
  qu'un agrandissement Lanczos, l'accentuation du rendu compense).

Trois fusions sont comparées : réglages par défaut, robustesse désactivée, et une
**référence** CPU qui connaît les décalages exacts (gain atteignable avec un alignement
parfait). Une planche de comparaison PNG est écrite à côté.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import RenderParams, linear_to_srgb, render
from .synthetic import (GRATING_U, SR_BAND, TEXTURE_V, SyntheticBurst, make_burst,
                        scene_regions, shift_and_add_reference, write_burst)

PSNR_REGIONS = ("aplats", "basses_freq")


@dataclass
class Score:
    psnr: dict          # PSNR (dB) : "global", "aplats" (bruit), "basses_freq"
    corr: dict          # corrélation avec la vérité dans la bande SR : "texture"


def _gray(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _register(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Recale ``img`` sur ``ref`` (translation sous-pixel) : neutralise les conventions de centre."""
    win = cv2.createHanningWindow(ref.shape[1::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(_gray(ref), _gray(img), win)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img.astype(np.float32), m, img.shape[1::-1], flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def _psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = (a - b) ** 2
    mse = float(err[mask].mean()) if mask is not None else float(err.mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def band_correlation(img: np.ndarray, gt: np.ndarray, box: tuple[int, int, int, int],
                     scale: float, band: tuple[float, float] = SR_BAND) -> float:
    """Corrélation entre ``img`` et la vérité, limitée à une bande de fréquences.

    ``band`` est en cycles par pixel capteur. Insensible au flou (un filtre de phase nulle ne
    change pas la corrélation) : elle mesure si le détail restitué est le *bon* détail, et non
    du moiré ou du bruit. 1 = parfait, ~0 = aucun détail juste dans cette bande.
    """
    y0, y1, x0, x1 = box
    a, g = _gray(img[y0:y1, x0:x1]), _gray(gt[y0:y1, x0:x1])
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.fft2((a - a.mean()) * win)
    G = np.fft.fft2((g - g.mean()) * win)
    fy = np.fft.fftfreq(a.shape[0])[:, None] * scale
    fx = np.fft.fftfreq(a.shape[1])[None, :] * scale
    f = np.hypot(fy, fx)
    m = (f >= band[0]) & (f <= band[1])
    den = np.sqrt(np.sum(np.abs(A[m]) ** 2) * np.sum(np.abs(G[m]) ** 2))
    return float(np.real(np.sum(A[m] * np.conj(G[m]))) / max(den, 1e-12))


def _boxes(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    """Rectangle de mesure du détail (pixels de sortie) : la texture fine.

    Le réseau de lignes n'est pas mesuré : sur un motif périodique, la corrélation de bande
    récompense aussi des répliques parasites (un moteur sans fusion y obtient un bon score).
    Il reste visible sur la planche de comparaison.
    """
    pad = 16
    u0, u1 = GRATING_U
    return {"texture": (int(TEXTURE_V[0] * h) + pad, int(TEXTURE_V[1] * h) - pad,
                        int(u0 * w) + pad, int(u1 * w) - pad)}


def score(img: np.ndarray, gt: np.ndarray, regions: dict[str, np.ndarray], scale: float,
          border: int = 16) -> Score:
    img = _register(img, gt)
    keep = np.zeros(gt.shape[:2], bool)
    keep[border:-border, border:-border] = True
    psnr = {"global": _psnr(img, gt, keep)}
    for name in PSNR_REGIONS:
        m = regions[name] & keep
        psnr[name] = _psnr(img, gt, m) if m.any() else float("nan")
    corr = {k: band_correlation(img, gt, box, scale) for k, box in _boxes(*gt.shape[:2]).items()}
    return Score(psnr, corr)


def evaluate(fused16: np.ndarray, single16: np.ndarray, burst: SyntheticBurst, scale: float) -> dict:
    gt = linear_to_srgb(burst.ground_truth(scale)).astype(np.float32)
    lr_h, lr_w = burst.raws[0].shape
    regions = scene_regions(gt.shape[0], gt.shape[1], lr_h, lr_w)
    f = score(fused16.astype(np.float32) / 65535, gt, regions, scale)
    s = score(single16.astype(np.float32) / 65535, gt, regions, scale)
    gains = {k: f.psnr[k] - s.psnr[k] for k in f.psnr}
    corr_gain = {k: f.corr[k] - s.corr[k] for k in f.corr}
    return {
        "fusion": f, "image_seule": s, "gains_db": gains, "gains_corr": corr_gain,
        "gain_psnr_db": gains["aplats"],
        # Critères : du détail juste au-delà de Nyquist sur une texture réaliste, et moins de bruit.
        "plus_net": corr_gain["texture"] > 0.10 and gains["aplats"] > 0.5,
    }


def reference_fusion(burst: SyntheticBurst, scale: float) -> np.ndarray:
    """Fusion de référence connaissant les décalages exacts, rendue en sRGB 16 bits."""
    return render(shift_and_add_reference(burst, scale), RenderParams(sharpen_amount=0.0))


def contact_sheet(path: Path, gt: np.ndarray, single16: np.ndarray, fusions: dict[str, np.ndarray],
                  burst: SyntheticBurst, scale: float, size: int = 192) -> Path:
    """Planche PNG : vérité / image seule / fusions, sur la mire et les deux réseaux."""
    h, w = gt.shape[:2]
    lr_h, lr_w = burst.raws[0].shape
    from .synthetic import GRATING_H_U, GRATING_U, GRATING_V, STAR_CENTER, STAR_SPOKES

    # Centres : anneau SR de la mire, et zone ~0,6 cycle/px de chaque réseau.
    star_r = STAR_SPOKES / 2 / (2 * np.pi * 0.6)  # rayon (px capteur) où la mire vaut 0,6 cycle/px
    centers = [
        (STAR_CENTER[0], STAR_CENTER[1] + star_r / lr_w),
        ((GRATING_V[0] + GRATING_V[1]) / 2, GRATING_U[0] + 0.62 * (GRATING_U[1] - GRATING_U[0])),
        (GRATING_H_U[0] + 0.62 * (GRATING_H_U[1] - GRATING_H_U[0]), (GRATING_U[0] + GRATING_U[1]) / 2),
    ]
    cols = {"vérité": gt, "image seule": single16.astype(np.float32) / 65535}
    cols.update({k: v.astype(np.float32) / 65535 for k, v in fusions.items()})
    rows = []
    for cv_, cu_ in centers:
        y0 = int(np.clip(cv_ * h - size / 2, 0, h - size))
        x0 = int(np.clip(cu_ * w - size / 2, 0, w - size))
        tiles = [cv2.resize(img[y0:y0 + size, x0:x0 + size], (2 * size, 2 * size),
                            interpolation=cv2.INTER_NEAREST) for img in cols.values()]
        rows.append(np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=1) for t in tiles], 1))
    sheet = (np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)
    header = np.full((36, sheet.shape[1], 3), 255, np.uint8)
    for k, name in enumerate(cols):
        label = name.encode("ascii", "replace").decode().replace("?", "e")
        cv2.putText(header, label, (k * (2 * size + 8) + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.concatenate([header, sheet], 0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return path


def run_selftest(scale: float = 2.0, n_frames: int = 10, height: int = 512, width: int = 768,
                 alpha: float = 1e-3, beta: float = 1e-5, workdir: str | Path | None = None,
                 engine_factory=None, seed: int = 0, out_dir: str | Path | None = ".",
                 variants: bool = True) -> dict:
    """Fusionne une rafale synthétique et compare à l'image seule. Bruit par défaut ≈ ISO 800."""
    from .params import FusionParams
    from .pipeline import fuse_burst, load_stack

    burst = make_burst(height, width, n_frames=n_frames, alpha=alpha, beta=beta, seed=seed)
    tmp = tempfile.TemporaryDirectory() if workdir is None else None
    folder = Path(workdir or tmp.name)  # type: ignore[union-attr]
    write_burst(burst, folder)
    runs = {"fusion": FusionParams(scale=scale, reference="first", sharpen_amount=0.0,
                                   lut_trials=2_000_000)}
    if variants:
        runs["sans robustesse"] = FusionParams(scale=scale, reference="first", sharpen_amount=0.0,
                                               robustness=False)
    results, metrics = {}, {}
    for name, params in runs.items():
        loaded = load_stack(folder, params.reference)
        engine = engine_factory(params) if engine_factory else None
        results[name] = fuse_burst(loaded, params, engine=engine)
        metrics[name] = evaluate(results[name].fused, results[name].single, burst, scale)
    ref16 = reference_fusion(burst, scale)
    metrics["référence (décalages exacts)"] = evaluate(ref16, results["fusion"].single, burst, scale)
    main = metrics["fusion"]
    main["rapport"] = results["fusion"].report()
    main["variantes"] = metrics
    if out_dir is not None:
        gt = linear_to_srgb(burst.ground_truth(scale)).astype(np.float32)
        fusions = {k: r.fused for k, r in results.items()}
        fusions["reference"] = ref16
        main["planche"] = contact_sheet(Path(out_dir) / "selftest_comparaison.png", gt,
                                        results["fusion"].single, fusions, burst, scale)
    if tmp is not None:
        tmp.cleanup()
    return main


def format_report(m: dict) -> str:
    cols = ("global", *PSNR_REGIONS)
    head = (f"{'':30s}" + "".join(f"{'PSNR ' + k:>18s}" for k in cols)
            + f"{'détail texture':>16s}")
    def row(name, sc):
        return (f"{name:30s}" + "".join(f"{sc.psnr[k]:15.2f} dB" for k in cols)
                + f"{sc.corr['texture']:16.2f}")
    lines = [m["rapport"], "", head, row("image seule", m["image_seule"])]
    lines += [row(name, mm["fusion"]) for name, mm in m["variantes"].items()]
    lines += [
        "",
        "« détail texture » = corrélation avec la vérité entre 0,45 et 0,85 cycle/pixel, sur une",
        "texture fine non périodique (au-delà de ce qu'une image seule peut résoudre). Insensible",
        "au flou : 0 = moiré ou bruit, 1 = parfait. Réseau de lignes et mire : voir la planche.",
        "La « référence » connaît les décalages exacts : c'est le gain atteignable avec un",
        "alignement parfait.",
    ]
    if "planche" in m:
        lines.append(f"Planche de comparaison : {m['planche']}")
    lines.append("✅ La fusion restitue du détail juste au-delà d'une image seule et réduit le bruit."
                 if m["plus_net"] else
                 "❌ La fusion ne bat pas l'image seule sur la texture fine ou sur le bruit : envoyez "
                 "ce rapport et la planche PNG.")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m drone_burst_sr.selftest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--out-dir", default=".", help="dossier de la planche PNG")
    p.add_argument("--no-variants", action="store_true", help="ne pas refaire la fusion sans robustesse")
    args = p.parse_args(argv)
    m = run_selftest(args.scale, args.frames, out_dir=args.out_dir, variants=not args.no_variants)
    print(format_report(m))
    return 0 if m["plus_net"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
