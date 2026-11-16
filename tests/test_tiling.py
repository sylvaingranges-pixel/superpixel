import numpy as np
import pytest

from drone_burst_sr.tiling import (GB, choose_tile_size, estimate_memory, fuse_tiled, output_shape,
                                   plan_tiles)


def _nearest_engine(scale):
    """Moteur factice local et invariant par translation : out[i] = ref[floor(i / s)]."""
    def fuse(ref, comp):
        h, w = output_shape(*ref.shape, scale)
        yi = np.minimum((np.arange(h) / scale).astype(int), ref.shape[0] - 1)
        xi = np.minimum((np.arange(w) / scale).astype(int), ref.shape[1] - 1)
        out = ref[yi][:, xi]
        return np.repeat(out[..., None], 3, axis=-1)
    return fuse


def test_tiles_cover_image_with_even_coordinates():
    tiles = plan_tiles(3000, 4000, 1100, overlap=128)
    assert len(tiles) > 1
    cover = np.zeros((3000, 4000), bool)
    for t in tiles:
        assert t.y0 % 2 == 0 and t.x0 % 2 == 0 and t.y1 % 2 == 0 and t.x1 % 2 == 0
        cover[t.y0:t.y1, t.x0:t.x1] = True
    assert cover.all()


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_tiled_fusion_equals_full_image(scale):
    rng = np.random.default_rng(0)
    stack = rng.random((3, 600, 800), dtype=np.float32)
    full = fuse_tiled(stack, scale, plan_tiles(600, 800, None), _nearest_engine(scale))
    tiles = plan_tiles(600, 800, 300, overlap=100, min_tile=64)
    assert len(tiles) >= 6
    tiled = fuse_tiled(stack, scale, tiles, _nearest_engine(scale), overlap=100)
    assert tiled.shape == (*output_shape(600, 800, scale), 3)
    assert np.allclose(tiled, full, atol=1e-5)


def test_memory_estimates_scale_with_output():
    m12 = estimate_memory(3024, 4032, 10, 2.0)
    m48 = estimate_memory(6048, 8064, 10, 2.0)
    assert m48.device_bytes_full > 3 * m12.device_bytes_full
    assert 4 * GB < m48.device_bytes_full < 12 * GB  # ~8 Go : tuiles nécessaires sur un T4 chargé


def test_choose_tile_size():
    assert choose_tile_size(3024, 4032, 2.0, 15 * GB) is None          # 12 MP x2 tient sur un T4
    side = choose_tile_size(6048, 8064, 2.0, 4 * GB)
    assert side is not None and side >= 1024
    assert choose_tile_size(6048, 8064, 2.0, None, forced=0) is None
    assert choose_tile_size(6048, 8064, 2.0, None, forced=2048) == 2048
