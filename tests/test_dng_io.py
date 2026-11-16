import numpy as np
import pytest

from drone_burst_sr.baseline import demosaic_linear
from drone_burst_sr.cfa import rgb_from_rggb, to_rggb
from drone_burst_sr.dng_io import list_dng_files, load_normalized, read_frame_info
from drone_burst_sr.synthetic import make_burst, write_burst, write_dng


def test_list_is_case_insensitive(tmp_path):
    raw = np.full((64, 64), 1000, np.uint16)
    write_dng(tmp_path / "a.DNG", raw)
    write_dng(tmp_path / "b.dng", raw)
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in list_dng_files(tmp_path)] == ["a.DNG", "b.dng"]


@pytest.mark.parametrize("cfa", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_metadata_and_normalisation(tmp_path, cfa):
    raw = np.random.default_rng(0).integers(256, 4095, (64, 96)).astype(np.uint16)
    path = write_dng(tmp_path / "x.DNG", raw, black=256, white=4095, cfa=cfa,
                     as_shot_neutral=(0.5, 1.0, 0.8), iso=400, exposure_time=(1, 1000),
                     noise_profile=(1e-4, 2e-6))
    info = read_frame_info(path)
    assert info.cfa_string == cfa
    assert (info.width, info.height) == (96, 64)
    assert info.white_level == 4095
    assert np.allclose(info.black_2x2, 256)
    assert info.white_balance == pytest.approx((2.0, 1.0, 1.25), rel=1e-3)
    assert info.iso == 400
    assert info.exposure_time == pytest.approx(1e-3)
    assert info.noise_profile == ((1e-4,), (2e-6,))
    assert info.xyz_to_cam is not None and info.xyz_to_cam.shape == (3, 3)
    x = load_normalized(info)
    assert x.dtype == np.float32
    assert np.allclose(x, (raw.astype(np.float64) - 256) / (4095 - 256), atol=1e-6)


def test_odd_dimensions_are_cropped_to_even(tmp_path):
    path = write_dng(tmp_path / "odd.dng", np.full((65, 97), 500, np.uint16))
    info = read_frame_info(path)
    assert (info.width, info.height) == (96, 64)
    assert load_normalized(info).shape == (64, 96)


CFA_CODES = {"RGGB": [[0, 1], [1, 2]], "BGGR": [[2, 1], [1, 0]],
             "GRBG": [[1, 0], [2, 1]], "GBRG": [[1, 2], [0, 1]]}


@pytest.mark.parametrize("cfa", list(CFA_CODES))
def test_to_rggb_puts_red_top_left_and_is_invertible(cfa):
    from drone_burst_sr.synthetic import mosaic

    rgb = np.stack([np.full((8, 10), v) for v in (1.0, 2.0, 3.0)], -1)  # R=1, G=2, B=3
    code = np.array(CFA_CODES[cfa])
    flipped = to_rggb(mosaic(rgb, cfa), code)
    assert flipped[0, 0] == 1 and flipped[0, 1] == 2 and flipped[1, 0] == 2 and flipped[1, 1] == 3
    marker = np.random.default_rng(0).random((8, 10, 3))
    assert np.array_equal(rgb_from_rggb(to_rggb(marker[..., 0], code)[..., None].repeat(3, -1), code)[..., 0],
                          marker[..., 0])


def test_rawpy_baseline_matches_normalisation(tmp_path):
    burst = make_burst(64, 96, n_frames=1, alpha=1e-12, beta=1e-14)
    path = write_burst(burst, tmp_path)[0]
    info = read_frame_info(path)
    lin = demosaic_linear(info).astype(np.float64) / 65535
    x = load_normalized(info)
    ratio = np.median(lin[0::2, 0::2, 0] / np.maximum(x[0::2, 0::2], 1e-3))
    assert ratio == pytest.approx(1.0, abs=0.01)  # même échelle noir/blanc que la fusion
