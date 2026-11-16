"""Exports TIFF 16 bits / JPEG et petites aides d'affichage (aperçu, recadrage 100 %)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import __version__


def to_uint8(img16: np.ndarray) -> np.ndarray:
    return ((img16.astype(np.uint32) + 128) // 257).astype(np.uint8)


def _ascii(text: str) -> str:
    """Les chaînes TIFF doivent être en ASCII 7 bits : on retire accents et symboles."""
    import unicodedata

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").strip()


def save_tiff16(path: str | Path, img16: np.ndarray, description: str = "") -> Path:
    """TIFF RVB 16 bits, non compressé (compatibilité maximale), sRGB."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img16.dtype != np.uint16 or img16.ndim != 3:
        raise ValueError("image RVB uint16 attendue")
    tifffile.imwrite(
        path, img16, photometric="rgb", compression=None,
        bigtiff=img16.nbytes > 3.9 * 1024 ** 3,
        description=_ascii(description) or "drone-burst-sr (sRGB, 16 bits)",
        software=f"drone-burst-sr {__version__}", metadata=None,
    )
    return path


def save_jpeg(path: str | Path, img16: np.ndarray, quality: int = 92,
              max_side: int | None = None) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = preview(img16, max_side) if max_side else to_uint8(img16)
    Image.fromarray(img).save(path, quality=quality, subsampling=0, optimize=True)
    return path


def preview(img16: np.ndarray, max_side: int = 2048) -> np.ndarray:
    """Aperçu uint8 réduit (INTER_AREA) pour l'affichage web."""
    h, w = img16.shape[:2]
    f = min(1.0, max_side / max(h, w))
    img = img16 if f >= 1 else cv2.resize(img16, (max(1, round(w * f)), max(1, round(h * f))),
                                          interpolation=cv2.INTER_AREA)
    return to_uint8(img)


def crop(img: np.ndarray, cx: float, cy: float, size: int = 512) -> np.ndarray:
    """Recadrage size x size centré sur (cx, cy) en coordonnées relatives [0, 1], à 100 %."""
    h, w = img.shape[:2]
    s = min(size, h, w)
    x0 = int(round(cx * w - s / 2))
    y0 = int(round(cy * h - s / 2))
    x0 = min(max(x0, 0), w - s)
    y0 = min(max(y0, 0), h - s)
    return img[y0:y0 + s, x0:x0 + s]
