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


# Géométrie de la scène, en coordonnées normalisées (u = x/largeur, v = y/hauteur) pour
# pouvoir définir les mêmes zones de mesure sur n'importe quelle grille (basse ou haute
# résolution). Les fréquences sont exprimées en cycles par pixel de l'image du capteur.
STAR_CENTER = (0.5, 0.25)          # (v, u)
STAR_RADIUS = 0.21                 # × min(hauteur, largeur)
STAR_SPOKES = 96
GRATING_V = (0.10, 0.42)           # réseau de lignes verticales (fréquence croissante selon u)
TEXTURE_V = (0.58, 0.90)           # texture fine non périodique (comme toits, gravier, feuillage)
GRATING_H_U = TEXTURE_V            # compatibilité
GRATING_U = (0.55, 0.95)
GRATING_F = (0.20, 0.85)           # cycles / pixel capteur ; > 0.5 = au-delà de Nyquist d'une image
SR_BAND = (0.45, 0.85)             # bande où seule la fusion peut restituer le détail


def _chirp_phase(t: np.ndarray, t0: float, t1: float, n_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Phase d'un chirp linéaire de GRATING_F[0] à GRATING_F[1] cycles/px sur [t0, t1]."""
    f0, f1 = GRATING_F
    d = np.clip(t - t0, 0, t1 - t0)
    phase = 2 * np.pi * n_px * (f0 * d + (f1 - f0) * d * d / (2 * (t1 - t0)))
    freq = f0 + (f1 - f0) * d / (t1 - t0)
    return phase, freq


def _layout(height: int, width: int, lr_h: int, lr_w: int):
    """Grilles et masques des éléments de la scène sur une grille height x width."""
    v = (np.arange(height)[:, None] + 0.5) / height
    u = (np.arange(width)[None, :] + 0.5) / width
    v, u = np.broadcast_arrays(v, u)
    # Mire : distances en pixels capteur.
    dy = (v - STAR_CENTER[0]) * lr_h
    dx = (u - STAR_CENTER[1]) * lr_w
    rad = np.hypot(dy, dx)
    star = rad < STAR_RADIUS * min(lr_h, lr_w)
    gv = (u >= GRATING_U[0]) & (u <= GRATING_U[1]) & (v >= GRATING_V[0]) & (v <= GRATING_V[1])
    tex = (u >= GRATING_U[0]) & (u <= GRATING_U[1]) & (v >= TEXTURE_V[0]) & (v <= TEXTURE_V[1])
    return u, v, dy, dx, rad, star, gv, tex


def _fine_texture(height: int, width: int, factor: int, seed: int) -> np.ndarray:
    """Texture aléatoire à large bande (spectre ∝ f^-0.5) jusqu'à 0,85 cycle/pixel capteur."""
    rng = np.random.default_rng(seed + 1234)
    noise = rng.standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None] * factor   # cycles / pixel capteur
    fx = np.fft.rfftfreq(width)[None, :] * factor
    f = np.hypot(fy, fx)
    amp = np.where(f > 0.02, np.maximum(f, 0.02) ** -0.5, 0.0)
    amp *= np.exp(-np.maximum(f - GRATING_F[1], 0) ** 2 / (2 * 0.03 ** 2))  # coupure douce à 0,85
    tex = np.fft.irfft2(np.fft.rfft2(noise) * amp, s=(height, width))
    return tex / tex.std()


def make_scene(height: int, width: int, seed: int = 0, factor: int = 4) -> np.ndarray:
    """Scène linéaire RVB (H, W, 3) haute résolution (``factor`` × la taille capteur).

    Contient des aplats (bruit), une mire de Siemens, un réseau de lignes à fréquence
    croissante et une texture fine non périodique. Réseau, mire et texture vont au-delà de
    la fréquence de Nyquist d'une image seule : c'est là que la super-résolution joue.
    """
    lr_h, lr_w = height // factor, width // factor
    u, v, dy, dx, rad, star, gv, tex = _layout(height, width, lr_h, lr_w)
    img = np.zeros((height, width, 3))

    # Fond : aplats en damier de niveaux variés (bruit, estimation du bruit).
    levels = np.linspace(0.03, 0.8, 12)
    idx = (np.floor(v * 6) * 5 + np.floor(u * 8)).astype(int) % len(levels)
    base = levels[idx]
    tint = np.stack([np.ones_like(base), 0.9 + 0.1 * np.sin(idx), 0.85 + 0.15 * np.cos(idx)], -1)
    img[:] = base[..., None] * tint

    ang = np.arctan2(dy, dx)
    img[star] = (0.45 + 0.35 * np.sign(np.sin(STAR_SPOKES / 2 * ang)))[star, None] * np.array([1.0, 0.95, 0.9])

    phase_u, _ = _chirp_phase(u, *GRATING_U, lr_w)
    img[gv] = (0.45 + 0.3 * np.sin(phase_u))[gv, None] * np.array([0.95, 1.0, 1.05])
    texture = 0.42 + 0.12 * _fine_texture(height, width, factor, seed)
    img[tex] = texture[tex, None] * np.array([1.05, 1.0, 0.9])
    return np.clip(img, 0.0, 0.95)


def scene_regions(height: int, width: int, lr_h: int, lr_w: int) -> dict[str, np.ndarray]:
    """Zones de mesure sur une grille height x width (ex. la sortie ×2).

    - ``super_resolution`` : texture fine non périodique (cas des vraies photos) ;
    - ``motifs_periodiques`` : réseau et anneau de la mire dans la bande SR_BAND ; cas
      difficile pour l'alignement local (risque de se caler sur la mauvaise période) ;
    - ``aplats`` : intérieur des aplats (mesure du bruit) ;
    - ``basses_freq`` : mire sous la fréquence de Nyquist d'une image seule.
    """
    u, v, dy, dx, rad, star, gv, tex = _layout(height, width, lr_h, lr_w)
    _, fu = _chirp_phase(u, *GRATING_U, lr_w)
    with np.errstate(divide="ignore"):
        f_star = STAR_SPOKES / 2 / (2 * np.pi * np.maximum(rad, 1e-6))  # cycles / px
    lo, hi = SR_BAND
    periodic = (gv & (fu >= lo) & (fu <= hi)) | (star & (f_star >= lo) & (f_star <= hi))
    low = star & (f_star < 0.35)
    m = 4 / min(lr_h, lr_w)  # marge de 4 px capteur à l'intérieur de la texture
    sr = (tex & (u > GRATING_U[0] + m) & (u < GRATING_U[1] - m)
          & (v > TEXTURE_V[0] + m) & (v < TEXTURE_V[1] - m))
    # Aplats : loin des bords des cases (3 px capteur), hors mire, réseau et texture.
    mu, mv = 3 / lr_w, 3 / lr_h
    cu, cv = (u * 8) % 1, (v * 6) % 1
    inner = (cu > mu * 8) & (cu < 1 - mu * 8) & (cv > mv * 6) & (cv < 1 - mv * 6)
    margin = 6 / min(lr_h, lr_w)
    near_star = rad < (STAR_RADIUS + margin) * min(lr_h, lr_w)
    near_g = ((u > GRATING_U[0] - margin) & (u < GRATING_U[1] + margin)
              & (((v > GRATING_V[0] - margin) & (v < GRATING_V[1] + margin))
                 | ((v > TEXTURE_V[0] - margin) & (v < TEXTURE_V[1] + margin))))
    flats = inner & ~near_star & ~near_g
    return {"super_resolution": sr, "motifs_periodiques": periodic, "aplats": flats, "basses_freq": low}


def shift_and_add_reference(burst: "SyntheticBurst", scale: float) -> np.ndarray:
    """Fusion de référence connaissant les décalages exacts (RVB linéaire, CPU).

    Simple accumulation pondérée (noyau gaussien) des échantillons Bayer de toutes les images
    sur la grille de sortie. Sert uniquement de point de comparaison dans l'auto-test : elle
    montre ce qu'on peut gagner quand l'alignement est parfait.
    """
    h, w = burst.raws[0].shape
    oh, ow = round(h * scale), round(w * scale)
    codes = np.array(_CFA_CODES[burst.cfa]).reshape(2, 2)
    num = np.zeros((3, oh * ow))
    den = np.zeros((3, oh * ow))
    for raw, (dy, dx) in zip(burst.raws, burst.shifts):
        x = (raw.astype(np.float64) - burst.black) / (burst.white - burst.black)
        for i0 in range(2):
            for j0 in range(2):
                c = int(codes[i0, j0])
                sigma = (0.5 if c == 1 else 0.7) * scale  # en pixels de sortie
                ii, jj = np.mgrid[i0:h:2, j0:w:2]
                py = (ii + 0.5 + dy) * scale - 0.5
                px = (jj + 0.5 + dx) * scale - 0.5
                val = x[i0::2, j0::2]
                by, bx = np.floor(py).astype(int), np.floor(px).astype(int)
                r = int(np.ceil(2 * sigma))
                for oy in range(-r + 1, r + 1):
                    for ox in range(-r + 1, r + 1):
                        ty, tx = by + oy, bx + ox
                        ok = (ty >= 0) & (ty < oh) & (tx >= 0) & (tx < ow)
                        wgt = np.exp(-((ty - py) ** 2 + (tx - px) ** 2) / (2 * sigma ** 2))[ok]
                        flat = (ty * ow + tx)[ok]
                        num[c] += np.bincount(flat, wgt * val[ok], minlength=oh * ow)
                        den[c] += np.bincount(flat, wgt, minlength=oh * ow)
    out = num / np.maximum(den, 1e-12)
    return out.reshape(3, oh, ow).transpose(1, 2, 0).astype(np.float32)


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
    scene = make_scene(height * factor, width * factor, seed=seed, factor=factor)
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
