"""Rafales synthétiques : scène connue, décalages sous-pixel connus, bruit Poisson-Gauss.

Sert aux tests et à l'auto-test GPU (``python -m drone_burst_sr.selftest``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])

_CFA_CODES = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}


def make_scene(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Scène linéaire RVB (H, W, 3) riche en détails : mire, textures, aplats."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    yn, xn = y / height, x / width
    img = np.zeros((height, width, 3))

    # Fond : aplats en damier de niveaux variés (utile pour l'estimation du bruit).
    levels = np.linspace(0.03, 0.8, 12)
    cell = max(height, width) // 8
    idx = ((y // cell) * 5 + (x // cell)).astype(int) % len(levels)
    base = levels[idx]
    tint = np.stack([np.ones_like(base), 0.9 + 0.1 * np.sin(idx), 0.85 + 0.15 * np.cos(idx)], -1)
    img[:] = base[..., None] * tint

    # Mire de Siemens (hautes fréquences radiales) au centre-gauche.
    cy, cx, r = 0.5 * height, 0.3 * width, 0.22 * min(height, width)
    ang = np.arctan2(y - cy, x - cx)
    rad = np.hypot(y - cy, x - cx)
    star = 0.45 + 0.35 * np.sign(np.sin(36 * ang))
    m = rad < r
    img[m] = star[m, None] * np.array([1.0, 0.95, 0.9])

    # Réseau de fines lignes à fréquence croissante (droite).
    m2 = (xn > 0.6) & (yn > 0.15) & (yn < 0.45)
    freq = 30 + 220 * (xn - 0.6) / 0.4
    lines = 0.4 + 0.3 * np.sin(2 * np.pi * freq * xn)
    img[m2] = lines[m2, None] * np.array([0.9, 1.0, 1.1])

    # Texture aléatoire filtrée (bas droite).
    m3 = (xn > 0.6) & (yn > 0.6) & (yn < 0.9)
    tex = ndimage.gaussian_filter(rng.standard_normal((height, width)), 1.2)
    tex = 0.35 + 0.15 * tex / tex.std()
    img[m3] = tex[m3, None] * np.array([1.1, 1.0, 0.8])
    return np.clip(img, 0.0, 0.95)


def sample_frame(scene: np.ndarray, factor: int, shift: tuple[float, float]) -> np.ndarray:
    """Image RVB basse résolution (intégration sur la surface du pixel), décalée de (dy, dx) pixels."""
    h, w = scene.shape[0] // factor, scene.shape[1] // factor
    blurred = ndimage.uniform_filter(scene, size=(factor, factor, 1), mode="nearest")
    dy, dx = shift
    yy = (np.arange(h)[:, None] + 0.5 + dy) * factor - 0.5
    xx = (np.arange(w)[None, :] + 0.5 + dx) * factor - 0.5
    yy, xx = np.broadcast_arrays(yy, xx)
    out = np.empty((h, w, 3))
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(blurred[..., c], [yy, xx], order=1, mode="nearest")
    return out


def mosaic(rgb: np.ndarray, cfa: str = "RGGB") -> np.ndarray:
    codes = np.array(_CFA_CODES[cfa]).reshape(2, 2)
    out = np.empty(rgb.shape[:2])
    for i in range(2):
        for j in range(2):
            out[i::2, j::2] = rgb[i::2, j::2, codes[i, j]]
    return out


@dataclass
class SyntheticBurst:
    raws: list[np.ndarray]        # uint16 CFA
    shifts: list[tuple[float, float]]
    scene: np.ndarray             # vérité terrain haute résolution (linéaire)
    factor: int
    black: int
    white: int
    alpha: float
    beta: float
    cfa: str

    def ground_truth(self, scale: float) -> np.ndarray:
        """Image idéale (linéaire RVB) à l'échelle demandée, alignée sur l'image 0."""
        h, w = self.raws[0].shape
        out_h, out_w = round(h * scale), round(w * scale)
        f = self.factor / scale
        blurred = ndimage.uniform_filter(self.scene, size=(max(f, 1), max(f, 1), 1), mode="nearest")
        yy = (np.arange(out_h)[:, None] + 0.5) * f - 0.5
        xx = (np.arange(out_w)[None, :] + 0.5) * f - 0.5
        yy, xx = np.broadcast_arrays(yy, xx)
        return np.stack([ndimage.map_coordinates(blurred[..., c], [yy, xx], order=1, mode="nearest")
                         for c in range(3)], -1)


def make_burst(height: int = 256, width: int = 384, n_frames: int = 10, factor: int = 4,
               alpha: float = 2e-4, beta: float = 4e-6, cfa: str = "RGGB",
               shifts: Sequence[tuple[float, float]] | None = None,
               black: int = 256, white: int = 4095, seed: int = 0) -> SyntheticBurst:
    rng = np.random.default_rng(seed)
    scene = make_scene(height * factor, width * factor, seed=seed)
    if shifts is None:
        shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-2.0, 2.0, 2)) for _ in range(n_frames - 1)]
    raws = []
    for s in shifts:
        x = mosaic(sample_frame(scene, factor, s), cfa)
        noisy = x + np.sqrt(np.maximum(alpha * x + beta, 0)) * rng.standard_normal(x.shape)
        raw = np.clip(np.round(noisy * (white - black) + black), 0, white).astype(np.uint16)
        raws.append(raw)
    return SyntheticBurst(raws, list(shifts), scene, factor, black, white, alpha, beta, cfa)


def _srational(values):
    out = []
    for v in values:
        out += [int(round(v * 1_000_000)), 1_000_000]
    return out


def write_dng(path: str | Path, raw: np.ndarray, *, black: int = 256, white: int = 4095,
              cfa: str = "RGGB", as_shot_neutral=(1.0, 1.0, 1.0), iso: int = 100,
              exposure_time: tuple[int, int] = (1, 500), f_number: tuple[int, int] = (17, 10),
              noise_profile: Sequence[float] | None = None, xyz_to_cam: np.ndarray | None = None,
              make: str = "DJI", model: str = "Synthetic") -> Path:
    """Écrit un DNG Bayer minimal lisible par LibRaw/rawpy et exifread."""
    import tifffile

    path = Path(path)
    if xyz_to_cam is None:
        xyz_to_cam = np.linalg.inv(SRGB_TO_XYZ)  # caméra RVB == sRGB linéaire
    tags = [
        (271, "s", 0, make, True),
        (272, "s", 0, model, True),
        (274, "H", 1, 1, True),
        (33421, "H", 2, (2, 2), True),
        (33422, "B", 4, _CFA_CODES[cfa], True),
        (33434, "2I", 1, exposure_time, True),
        (33437, "2I", 1, f_number, True),
        (34855, "H", 1, int(iso), True),
        (50706, "B", 4, (1, 4, 0, 0), True),
        (50707, "B", 4, (1, 1, 0, 0), True),
        (50708, "s", 0, f"{make} {model}", True),
        (50713, "H", 2, (2, 2), True),
        (50714, "H", 4, (black,) * 4, True),
        (50717, "H", 1, int(white), True),
        (50721, "2i", 9, _srational(np.asarray(xyz_to_cam).ravel()), True),
        (50728, "2I", 3, _srational(as_shot_neutral), True),
        (50778, "H", 1, 21, True),
    ]
    if noise_profile is not None:
        tags.append((51041, "d", len(noise_profile), tuple(float(v) for v in noise_profile), True))
    tifffile.imwrite(path, np.ascontiguousarray(raw, dtype=np.uint16), photometric=32803,
                     compression=None, metadata=None, extratags=tags, software="drone_burst_sr",
                     subfiletype=0)
    return path


def write_burst(burst: SyntheticBurst, folder: str | Path, with_noise_profile: bool = True,
                iso: int = 100, ext: str = ".DNG") -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    profile = (burst.alpha, burst.beta) if with_noise_profile else None
    return [write_dng(folder / f"SYN_{k:04d}{ext}", raw, black=burst.black, white=burst.white,
                      cfa=burst.cfa, iso=iso, noise_profile=profile)
            for k, raw in enumerate(burst.raws)]
