"""Rendu couleur commun à la fusion et à l'image seule de référence.

Même ordre que le post-traitement amont (``raw2rgb.postprocess``) : RVB caméra
linéaire -> balance des blancs -> matrice caméra -> sRGB linéaire -> écrêtage ->
accentuation (masque flou) -> courbe sRGB. On l'applique à l'identique aux deux
images pour que la comparaison avant/après ne mesure que l'effet de la fusion.
Le traitement se fait par bandes pour limiter la mémoire sur les très grandes images.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])


@dataclass(frozen=True)
class RenderParams:
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    camera_to_srgb: np.ndarray | None = None
    sharpen_amount: float = 1.5   # valeur par défaut de l'amont
    sharpen_radius: float = 3.0
    exposure_ev: float = 0.0


def camera_to_srgb_matrix(xyz_to_cam: np.ndarray | None) -> np.ndarray:
    """Matrice RVB caméra -> sRGB linéaire déduite de la ColorMatrix DNG (comme l'amont)."""
    if xyz_to_cam is None or np.linalg.norm(xyz_to_cam) == 0:
        return np.eye(3, dtype=np.float32)
    rgb2cam = np.asarray(xyz_to_cam, dtype=np.float64) @ SRGB_TO_XYZ
    rgb2cam /= rgb2cam.sum(axis=1, keepdims=True)
    return np.linalg.inv(rgb2cam).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _render_band(band: np.ndarray, p: RenderParams, matrix: np.ndarray, gain: np.ndarray) -> np.ndarray:
    rgb = band.astype(np.float32, copy=True)
    rgb *= gain
    rgb = np.clip(rgb.reshape(-1, 3) @ matrix.T, 0.0, 1.0).reshape(band.shape).astype(np.float32)
    if p.sharpen_amount > 0 and p.sharpen_radius > 0:
        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=p.sharpen_radius, sigmaY=p.sharpen_radius,
                                borderType=cv2.BORDER_REFLECT)
        rgb += p.sharpen_amount * (rgb - blur)
    return linear_to_srgb(rgb)


def render(linear: np.ndarray, params: RenderParams, band_rows: int = 1024) -> np.ndarray:
    """RVB caméra linéaire (float dans [0, 1], ou uint16 sur 0-65535) -> sRGB uint16."""
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError(f"Image RVB attendue, reçu {linear.shape}")
    scale = 1.0 / 65535.0 if linear.dtype == np.uint16 else 1.0
    matrix = params.camera_to_srgb if params.camera_to_srgb is not None else np.eye(3, dtype=np.float32)
    gain = (np.asarray(params.white_balance, dtype=np.float32) * np.float32(2.0 ** params.exposure_ev)
            * np.float32(scale))
    h = linear.shape[0]
    margin = int(np.ceil(4 * params.sharpen_radius)) + 2 if params.sharpen_amount > 0 else 0
    out = np.empty(linear.shape, dtype=np.uint16)
    for y0 in range(0, h, band_rows):
        y1 = min(h, y0 + band_rows)
        a0, a1 = max(0, y0 - margin), min(h, y1 + margin)
        band = _render_band(linear[a0:a1], params, matrix, gain)
        out[y0:y1] = np.round(band[y0 - a0: y0 - a0 + (y1 - y0)] * 65535.0).astype(np.uint16)
    return out


def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Applique le tag EXIF Orientation (1 à 8)."""
    ops = {
        1: lambda a: a,
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.swapaxes(a[::-1, ::-1], 0, 1),
        8: lambda a: np.rot90(a, 1),
    }
    return np.ascontiguousarray(ops.get(int(orientation or 1), ops[1])(img))
