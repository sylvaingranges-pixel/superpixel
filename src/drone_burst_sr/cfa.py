"""Manipulations du motif CFA.

L'algorithme amont travaille en disposition RGGB. Comme lui, on y ramène
n'importe quel motif Bayer par simple retournement de l'image (opération qui est
sa propre inverse). Les dimensions doivent être paires pour que le retournement
conserve la phase du motif.
"""

from __future__ import annotations

import numpy as np

# Position (ligne, colonne) dans le motif 2x2 en disposition RGGB.
RGGB_POSITIONS = {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}
# Ordre des plans attendu par l'amont pour alpha/beta : R, G1, B, G2.
UPSTREAM_PLANE_ORDER = ("R", "G1", "B", "G2")


def flip_axes_for(cfa: np.ndarray) -> tuple[int, ...]:
    """Axes (relatifs : -2 = lignes, -1 = colonnes) à retourner pour obtenir du RGGB."""
    cfa = np.asarray(cfa)
    pos = tuple(int(v) for v in np.argwhere(cfa == 0)[0])
    if cfa[1 - pos[0], 1 - pos[1]] != 2:
        raise ValueError(f"Motif CFA non Bayer : {cfa.tolist()}")
    return {(0, 0): (), (0, 1): (-1,), (1, 0): (-2,), (1, 1): (-2, -1)}[pos]


def to_rggb(x: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    """Retourne une image (..., H, W) pour qu'elle soit en RGGB (vue, pas de copie)."""
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(f"Dimensions impaires non supportées : {x.shape}")
    axes = flip_axes_for(cfa)
    return np.flip(x, axis=axes) if axes else x


def rgb_from_rggb(rgb: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    """Annule le retournement sur une image RVB (H, W, 3) issue de l'espace RGGB."""
    axes = tuple(a - 1 for a in flip_axes_for(cfa))  # décale : le dernier axe est la couleur
    return np.flip(rgb, axis=axes) if axes else rgb


def planes(x_rggb: np.ndarray) -> dict[str, np.ndarray]:
    """Les quatre sous-plans (vues) d'une image ou pile en disposition RGGB."""
    return {name: x_rggb[..., i::2, j::2] for name, (i, j) in RGGB_POSITIONS.items()}
