"""Auto-test de bout en bout sur une rafale synthétique (décalages sous-pixel connus).

``python -m drone_burst_sr.selftest`` (GPU requis) : génère une rafale DNG, la fusionne
à ×2 et vérifie que le résultat est plus fidèle et plus net qu'une seule image
dématricée par rawpy puis agrandie.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import linear_to_srgb
from .synthetic import SyntheticBurst, make_burst, write_burst


@dataclass
class Score:
    psnr: float
    hf_ratio: float   # part des hautes fréquences de la vérité terrain retrouvée (1 = parfait)


def _register(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Recale ``img`` sur ``ref`` (translation sous-pixel) : neutralise les conventions de centre."""
    a = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img.astype(np.float32), m, img.shape[1::-1], flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def score(img: np.ndarray, gt: np.ndarray, border: int = 16) -> Score:
    """PSNR et part du détail fin retrouvé, après recalage, en sRGB [0, 1]."""
    img = _register(img, gt)
    sl = (slice(border, -border), slice(border, -border))
    a, b = img[sl], gt[sl]
    mse = float(np.mean((a - b) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    la, lb = lap(a).ravel(), lap(b).ravel()
    lb = lb - lb.mean()
    # Projection sur le détail réel : le bruit (non corrélé à la vérité) ne compte pas.
    hf = float(np.dot(la - la.mean(), lb) / max(np.dot(lb, lb), 1e-12))
    return Score(psnr, hf)


def evaluate(fused16: np.ndarray, single16: np.ndarray, burst: SyntheticBurst, scale: float) -> dict:
    gt = linear_to_srgb(burst.ground_truth(scale)).astype(np.float32)
    f = score(fused16.astype(np.float32) / 65535, gt)
    s = score(single16.astype(np.float32) / 65535, gt)
    return {"fusion": f, "image_seule": s,
            "gain_psnr_db": f.psnr - s.psnr,
            "plus_net": f.hf_ratio > s.hf_ratio and f.psnr > s.psnr}


def run_selftest(scale: float = 2.0, n_frames: int = 10, height: int = 512, width: int = 768,
                 workdir: str | Path | None = None, engine=None, seed: int = 0) -> dict:
    from .params import FusionParams
    from .pipeline import run

    burst = make_burst(height, width, n_frames=n_frames, seed=seed)
    tmp = tempfile.TemporaryDirectory() if workdir is None else None
    folder = Path(workdir or tmp.name)  # type: ignore[union-attr]
    write_burst(burst, folder)
    # Rendu neutre (pas d'accentuation) pour comparer à la vérité terrain.
    params = FusionParams(scale=scale, reference="first", sharpen_amount=0.0, lut_trials=2_000_000)
    result = run(folder, params, engine=engine)
    metrics = evaluate(result.fused, result.single, burst, scale)
    metrics["rapport"] = result.report()
    if tmp is not None:
        tmp.cleanup()
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m drone_burst_sr.selftest", description=__doc__)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--frames", type=int, default=10)
    args = p.parse_args(argv)
    m = run_selftest(args.scale, args.frames)
    print(m["rapport"])
    for k in ("image_seule", "fusion"):
        s = m[k]
        print(f"{k:12s} PSNR {s.psnr:6.2f} dB   détail fin retrouvé {s.hf_ratio:5.2f}")
    print(f"Gain PSNR : {m['gain_psnr_db']:+.2f} dB")
    print("✅ La fusion est plus fidèle et plus nette que l'image seule." if m["plus_net"]
          else "❌ La fusion n'est pas meilleure que l'image seule : à examiner.")
    return 0 if m["plus_net"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
