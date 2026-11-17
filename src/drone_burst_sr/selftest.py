"""Auto-test de bout en bout sur une rafale synthétique (décalages sous-pixel connus).

``python -m drone_burst_sr.selftest`` (GPU requis) génère une rafale DNG, la fusionne à ×2
et la compare, zone par zone, à une seule image dématricée par rawpy puis agrandie :

- **super_resolution** : réseaux et mire entre 0,45 et 0,85 cycle/pixel, au-delà de la
  limite de Nyquist d'une image seule. Seule la fusion peut y restituer le détail ;
- **aplats** : zones uniformes, où l'on mesure le débruitage ;
- **basses_freq** : détails qu'une image seule résout déjà (contrôle : pas de dégradation).

La fusion est faite deux fois : réglages par défaut, puis robustesse désactivée. Si la
seconde est nettement meilleure, la robustesse rejette trop d'images. Une planche de
comparaison PNG est écrite à côté.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import linear_to_srgb
from .synthetic import SyntheticBurst, make_burst, scene_regions, write_burst

REGIONS = ("super_resolution", "aplats", "basses_freq")


@dataclass
class Score:
    psnr: dict          # PSNR (dB) par zone, + "global"
    hf_ratio: float     # part du détail fin de la vérité terrain retrouvée (zone SR)


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


def score(img: np.ndarray, gt: np.ndarray, regions: dict[str, np.ndarray], border: int = 16) -> Score:
    """PSNR par zone et part du détail fin retrouvé, après recalage, en sRGB [0, 1]."""
    img = _register(img, gt)
    keep = np.zeros(gt.shape[:2], bool)
    keep[border:-border, border:-border] = True
    psnr = {"global": _psnr(img, gt, keep)}
    for name in REGIONS:
        m = regions[name] & keep
        psnr[name] = _psnr(img, gt, m) if m.any() else float("nan")
    lap_a = cv2.Laplacian(_gray(img), cv2.CV_32F)[regions["super_resolution"] & keep]
    lap_b = cv2.Laplacian(_gray(gt), cv2.CV_32F)[regions["super_resolution"] & keep]
    lap_b = lap_b - lap_b.mean()
    # Projection sur le détail réel : le bruit et le moiré (non corrélés à la vérité) ne comptent pas.
    hf = float(np.dot(lap_a - lap_a.mean(), lap_b) / max(np.dot(lap_b, lap_b), 1e-12))
    return Score(psnr, hf)


def evaluate(fused16: np.ndarray, single16: np.ndarray, burst: SyntheticBurst, scale: float) -> dict:
    gt = linear_to_srgb(burst.ground_truth(scale)).astype(np.float32)
    lr_h, lr_w = burst.raws[0].shape
    regions = scene_regions(gt.shape[0], gt.shape[1], lr_h, lr_w)
    f = score(fused16.astype(np.float32) / 65535, gt, regions)
    s = score(single16.astype(np.float32) / 65535, gt, regions)
    gains = {k: f.psnr[k] - s.psnr[k] for k in f.psnr}
    return {
        "fusion": f, "image_seule": s, "gains_db": gains,
        "gain_psnr_db": gains["super_resolution"],
        # Critères : détail fin nettement mieux restitué, bruit réduit, pas de dégradation ailleurs.
        "plus_net": (gains["super_resolution"] > 0.5 and f.hf_ratio > s.hf_ratio
                     and gains["aplats"] > 0.5 and gains["basses_freq"] > -0.5),
    }


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
    main = metrics["fusion"]
    main["rapport"] = results["fusion"].report()
    main["variantes"] = metrics
    if out_dir is not None:
        gt = linear_to_srgb(burst.ground_truth(scale)).astype(np.float32)
        main["planche"] = contact_sheet(Path(out_dir) / "selftest_comparaison.png", gt,
                                        results["fusion"].single,
                                        {k: r.fused for k, r in results.items()}, burst, scale)
    if tmp is not None:
        tmp.cleanup()
    return main


def format_report(m: dict) -> str:
    lines = [m["rapport"], "", f"{'':18s}" + "".join(f"{k:>18s}" for k in ("global", *REGIONS)) + "   détail fin"]
    single = m["image_seule"]
    lines.append(f"{'image seule':18s}" + "".join(f"{single.psnr[k]:15.2f} dB" for k in ("global", *REGIONS))
                 + f"   {single.hf_ratio:6.2f}")
    for name, mm in m["variantes"].items():
        f = mm["fusion"]
        lines.append(f"{name:18s}" + "".join(f"{f.psnr[k]:15.2f} dB" for k in ("global", *REGIONS))
                     + f"   {f.hf_ratio:6.2f}")
    lines.append("")
    g = m["gains_db"]
    lines.append("Gains de la fusion : " + ", ".join(f"{k} {g[k]:+.2f} dB" for k in ("global", *REGIONS)))
    if "sans robustesse" in m["variantes"]:
        g2 = m["variantes"]["sans robustesse"]["gains_db"]
        lines.append("Sans robustesse   : " + ", ".join(f"{k} {g2[k]:+.2f} dB" for k in ("global", *REGIONS)))
    if "planche" in m:
        lines.append(f"Planche de comparaison : {m['planche']}")
    lines.append("✅ La fusion restitue des détails au-delà d'une image seule et réduit le bruit."
                 if m["plus_net"] else
                 "❌ La fusion ne bat pas l'image seule sur tous les critères : envoyez ce rapport "
                 "et la planche PNG.")
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
