"""Image « avant » : une seule image de la rafale, dématricée normalement par rawpy.

rawpy (LibRaw, AHD) produit du RVB caméra linéaire 16 bits, sans balance des blancs
ni luminosité automatique. On l'agrandit en Lanczos à la taille de la fusion, puis on
applique exactement le même rendu couleur que la fusion (``render.render``).
"""

from __future__ import annotations

import cv2
import numpy as np

from .dng_io import FrameInfo


def demosaic_linear(info: FrameInfo) -> np.ndarray:
    """RVB caméra linéaire uint16 (0 = noir, 65535 = blanc), mêmes dimensions que la fusion à ×1."""
    import rawpy

    with rawpy.imread(str(info.path)) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            no_auto_bright=True,
            adjust_maximum_thr=0.0,  # pas de recalage du niveau de blanc sur le max mesuré
            output_bps=16,
            user_flip=0,
        )
    return np.ascontiguousarray(rgb[: info.height, : info.width])


def upscale(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Agrandissement Lanczos (uint16 conservé)."""
    h, w = shape
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
