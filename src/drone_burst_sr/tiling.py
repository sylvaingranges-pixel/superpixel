"""Estimation mémoire (RAM / VRAM) et traitement par tuiles.

Les tuiles sont découpées à l'identique dans toutes les images de la rafale, avec
des coordonnées paires (phase du CFA conservée) et un recouvrement. Chaque tuile est
fusionnée indépendamment par l'algorithme amont, puis les tuiles sont recombinées
avec un fondu linéaire dans les zones de recouvrement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

GB = 1024 ** 3

# L'alignement amont utilise une pyramide de facteurs (1, 2, 4, 4) = 32 et des tuiles
# de 16 à 64 px : une tuile d'au moins 1024 px de côté passe toujours ``sanitize_config``.
MIN_TILE = 1024
DEFAULT_OVERLAP = 192       # pixels d'entrée (CFA) ; > déplacement max attendu + noyaux
EDGE_GUARD = 48             # pixels d'entrée ignorés au bord intérieur d'une tuile

# Modèle d'occupation GPU de ``super_resolution.main`` (octets par pixel CFA d'entrée),
# établi en relisant les allocations amont : image de référence et courante, niveaux de
# gris, pyramide, FFT par tuiles, statistiques locales, robustesse, noyaux (~48 o/px) ;
# accumulateurs num/den en float32 RVB (24 o par pixel de sortie).
GPU_BYTES_PER_INPUT_PX = 48
GPU_BYTES_PER_OUTPUT_PX = 24
GPU_CONTEXT_BYTES = int(0.9 * GB)  # contexte CUDA + PyTorch + kernels compilés
SAFETY = 1.25


@dataclass(frozen=True)
class MemoryEstimate:
    host_bytes: int
    device_bytes_full: int     # VRAM si on traite l'image entière d'un coup
    output_shape: tuple[int, int]

    def describe(self) -> str:
        h, w = self.output_shape
        return (f"Sortie {w}x{h} ({w * h / 1e6:.0f} MP). RAM estimée ≈ {self.host_bytes / GB:.1f} Go, "
                f"VRAM sans tuiles ≈ {self.device_bytes_full / GB:.1f} Go.")


def output_shape(h: int, w: int, scale: float) -> tuple[int, int]:
    return round(scale * h), round(scale * w)


def device_bytes(tile_h: int, tile_w: int, scale: float) -> int:
    px_in = tile_h * tile_w
    px_out = px_in * scale * scale
    return int(SAFETY * (GPU_BYTES_PER_INPUT_PX * px_in + GPU_BYTES_PER_OUTPUT_PX * px_out)
               + GPU_CONTEXT_BYTES)


def estimate_memory(h: int, w: int, n_frames: int, scale: float) -> MemoryEstimate:
    """RAM : pile float32 + accumulateurs RVB/poids float32 + sorties uint16 (fusion, référence)."""
    oh, ow = output_shape(h, w, scale)
    px_out = oh * ow
    host = (n_frames * h * w * 4          # pile normalisée
            + px_out * (3 * 4 + 4)        # accumulateur linéaire + poids
            + 2 * px_out * 3 * 2          # rendus 16 bits (fusion, image seule)
            + h * w * 3 * 4 * 2           # dématriçage rawpy de l'image de référence
            + int(1.5 * GB))              # Python, Gradio, copies temporaires
    return MemoryEstimate(int(host), device_bytes(h, w, scale), (oh, ow))


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    starts = sorted({s - (s % 2) for s in starts})
    return starts


def plan_tiles(h: int, w: int, tile_size: int | None, overlap: int = DEFAULT_OVERLAP,
               min_tile: int = MIN_TILE) -> list[Tile]:
    """Grille de tuiles paires couvrant l'image. ``tile_size=None`` : une seule tuile."""
    if tile_size is None or (tile_size >= h and tile_size >= w):
        return [Tile(0, h, 0, w)]
    t = max(int(tile_size), min_tile, 2 * overlap + 2)
    t -= t % 2
    th, tw = min(t, h), min(t, w)
    return [Tile(y, y + th, x, x + tw)
            for y in _starts(h, th, overlap) for x in _starts(w, tw, overlap)]


def choose_tile_size(h: int, w: int, scale: float, free_vram_bytes: int | None,
                     forced: int | None = None) -> int | None:
    """Plus grande tuile carrée tenant dans la VRAM libre (None = image entière)."""
    if forced is not None:
        return forced if forced > 0 else None
    if free_vram_bytes is None or device_bytes(h, w, scale) <= free_vram_bytes:
        return None
    budget = free_vram_bytes - GPU_CONTEXT_BYTES
    per_px = SAFETY * (GPU_BYTES_PER_INPUT_PX + GPU_BYTES_PER_OUTPUT_PX * scale * scale)
    side = int(math.sqrt(budget / per_px)) if budget > 0 else 0
    side = max(MIN_TILE, side - side % 64)
    return side if side < max(h, w) else None


def _ramp(n: int, lead: bool, trail: bool, overlap: int, guard: int) -> np.ndarray:
    """Poids 1D : 0 sur la garde, rampe linéaire sur le recouvrement, 1 ailleurs."""
    w = np.ones(n, dtype=np.float32)
    ramp_len = max(overlap - 2 * guard, 1)
    d = np.arange(n, dtype=np.float32)
    if lead:
        w = np.minimum(w, np.clip((d - guard) / ramp_len, 0, 1))
    if trail:
        w = np.minimum(w, np.clip((n - 1 - d - guard) / ramp_len, 0, 1))
    return w


def tile_weights(tile: Tile, h: int, w: int, scale: float, overlap: int = DEFAULT_OVERLAP,
                 guard: int = EDGE_GUARD) -> np.ndarray:
    th, tw = output_shape(*tile.shape, scale)
    wy = _ramp(th, tile.y0 > 0, tile.y1 < h, round(overlap * scale), round(guard * scale))
    wx = _ramp(tw, tile.x0 > 0, tile.x1 < w, round(overlap * scale), round(guard * scale))
    return wy[:, None] * wx[None, :]


class TileAccumulator:
    """Recombine des tuiles RVB linéaires (fondu pondéré) dans une image pleine taille."""

    def __init__(self, h: int, w: int, scale: float, tiles: list[Tile], overlap: int = DEFAULT_OVERLAP):
        self.h, self.w, self.scale, self.overlap = h, w, scale, overlap
        self.single = len(tiles) == 1
        oh, ow = output_shape(h, w, scale)
        self.image = np.zeros((oh, ow, 3), dtype=np.float32)
        self.weight = None if self.single else np.zeros((oh, ow), dtype=np.float32)

    def add(self, tile: Tile, rgb: np.ndarray) -> None:
        oy, ox = round(tile.y0 * self.scale), round(tile.x0 * self.scale)
        th, tw = rgb.shape[:2]
        if self.single:
            self.image[:] = rgb
            return
        wt = tile_weights(tile, self.h, self.w, self.scale, self.overlap)
        if wt.shape != (th, tw):
            raise ValueError(f"Taille de tuile inattendue {rgb.shape} (attendu {wt.shape})")
        self.image[oy:oy + th, ox:ox + tw] += rgb * wt[..., None]
        self.weight[oy:oy + th, ox:ox + tw] += wt  # type: ignore[index]

    def result(self) -> np.ndarray:
        if not self.single:
            np.divide(self.image, np.maximum(self.weight, 1e-12)[..., None], out=self.image)  # type: ignore[arg-type]
            self.weight = None
        return self.image


def iterate_tiles(stack: np.ndarray, tiles: list[Tile]) -> Iterator[tuple[Tile, np.ndarray, np.ndarray]]:
    """(tuile, référence contiguë, autres images contiguës) pour chaque tuile."""
    for t in tiles:
        ref = np.ascontiguousarray(stack[0, t.y0:t.y1, t.x0:t.x1])
        comp = np.ascontiguousarray(stack[1:, t.y0:t.y1, t.x0:t.x1])
        yield t, ref, comp


def fuse_tiled(stack: np.ndarray, scale: float, tiles: list[Tile],
               fuse_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
               on_tile: Callable[[int, int], None] | None = None,
               overlap: int = DEFAULT_OVERLAP) -> np.ndarray:
    """Applique ``fuse_fn(ref, comp) -> (h·s, w·s, 3)`` tuile par tuile et recombine."""
    _, h, w = stack.shape
    acc = TileAccumulator(h, w, scale, tiles, overlap)
    for k, (tile, ref, comp) in enumerate(iterate_tiles(stack, tiles)):
        if on_tile:
            on_tile(k, len(tiles))
        out = fuse_fn(ref, comp)
        expected = output_shape(*tile.shape, scale)
        if out.shape[:2] != expected:
            raise ValueError(f"La fusion a renvoyé {out.shape}, attendu {expected}")
        acc.add(tile, np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0))
    return acc.result()
