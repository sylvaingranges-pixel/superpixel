import numpy as np

from drone_burst_sr.dng_io import read_burst_info
from drone_burst_sr.synthetic import write_dng
from drone_burst_sr.validation import check_brightness, has_errors, validate_burst


def _write(tmp_path, name, shape=(64, 96), **kw):
    return write_dng(tmp_path / name, np.full(shape, 1000, np.uint16), **kw)


def _messages(issues, level):
    return " ".join(i.message for i in issues if i.level == level)


def test_consistent_burst_has_no_warning_about_exposure(tmp_path):
    paths = [_write(tmp_path, f"{k}.DNG", noise_profile=(1e-4, 1e-6)) for k in range(8)]
    issues = validate_burst(read_burst_info(paths))
    assert not has_errors(issues)
    assert "ISO" not in _messages(issues, "avertissement")
    assert "Temps de pose" not in _messages(issues, "avertissement")


def test_detects_iso_exposure_and_wb_changes(tmp_path):
    paths = [_write(tmp_path, "0.DNG"),
             _write(tmp_path, "1.DNG", iso=200, exposure_time=(1, 250), as_shot_neutral=(0.6, 1, 0.7))]
    issues = validate_burst(read_burst_info(paths))
    warn = _messages(issues, "avertissement")
    assert "ISO" in warn and "Temps de pose" in warn and "Balance des blancs" in warn
    assert not has_errors(issues)


def test_detects_size_mismatch_as_error(tmp_path):
    paths = [_write(tmp_path, "0.DNG"), _write(tmp_path, "1.DNG", shape=(64, 128))]
    issues = validate_burst(read_burst_info(paths))
    assert has_errors(issues)
    assert "Dimensions" in _messages(issues, "erreur")


def test_single_frame_is_an_error(tmp_path):
    assert has_errors(validate_burst(read_burst_info([_write(tmp_path, "0.DNG")])))


def test_quad_bayer_notice(tmp_path, monkeypatch):
    import drone_burst_sr.dng_io as dng_io

    monkeypatch.setattr(dng_io, "QUAD_BAYER_MIN_MEGAPIXELS", 0.001)
    paths = [_write(tmp_path, f"{k}.DNG") for k in range(3)]
    issues = validate_burst(read_burst_info(paths))
    assert "48 MP" in _messages(issues, "avertissement")


def test_brightness_check():
    assert check_brightness([0.30, 0.31, 0.29]) == []
    issues = check_brightness([0.30, 0.30, 0.40], ["a.DNG", "b.DNG", "c.DNG"])
    assert issues and "c.DNG" in issues[0].message and "b.DNG" not in issues[0].message
