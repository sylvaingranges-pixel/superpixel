"""Chaîne complète sans GPU (moteur factice) + test GPU réel (ignoré sans GPU)."""

import numpy as np
import pytest
import tifffile

from drone_burst_sr import cli
from drone_burst_sr.params import FusionParams
from drone_burst_sr.pipeline import BurstError, analyze, run
from drone_burst_sr.render import RenderParams, apply_orientation, linear_to_srgb, render
from drone_burst_sr.selftest import evaluate, run_selftest


def test_pipeline_with_fake_engine(burst_dir, bilinear_engine):
    params = FusionParams(scale=1.5, sharpen_amount=0.0)
    steps = []
    result = run(burst_dir, params, progress=lambda f, m: steps.append(f), engine=bilinear_engine(1.5))
    assert result.fused.shape == (192, 288, 3) and result.fused.dtype == np.uint16
    assert result.single.shape == result.fused.shape
    assert result.noise.source == "NoiseProfile DNG"
    assert steps[-1] == 1.0 and steps == sorted(steps)
    assert "×1.5" in result.report()


def test_reference_is_sharpest(tmp_path, bilinear_engine):
    from drone_burst_sr.synthetic import make_burst, write_burst
    from scipy import ndimage

    burst = make_burst(128, 192, n_frames=4, seed=2)
    burst.raws[0] = ndimage.uniform_filter(burst.raws[0].astype(float), 5).astype(np.uint16)  # floue
    write_burst(burst, tmp_path)
    result = run(tmp_path, FusionParams(scale=1.0), engine=bilinear_engine(1.0))
    assert result.reference_name != "SYN_0000.DNG"


def test_fake_burst_errors(tmp_path):
    from drone_burst_sr.synthetic import write_dng

    write_dng(tmp_path / "a.DNG", np.full((64, 64), 900, np.uint16))
    write_dng(tmp_path / "b.DNG", np.full((64, 96), 900, np.uint16))
    assert not analyze(tmp_path).ok
    with pytest.raises(BurstError):
        run(tmp_path, FusionParams(), engine=object())


def test_render_identity_and_orientation():
    lin = np.full((4, 6, 3), 0.5, np.float32)
    out = render(lin, RenderParams(sharpen_amount=0.0))
    assert out.dtype == np.uint16
    assert np.allclose(out / 65535, linear_to_srgb(np.float32(0.5)), atol=1e-4)
    img = np.arange(6).reshape(2, 3)
    assert apply_orientation(img, 6).shape == (3, 2)
    assert apply_orientation(img, 3)[0, 0] == 5


def test_cli_check_and_no_gpu(burst_dir, tmp_path, capsys, monkeypatch):
    assert cli.main(["--input", str(burst_dir), "--check"]) == 0
    out = capsys.readouterr().out
    assert "6 images" in out and "Mémoire estimée" in out
    from drone_burst_sr import gpu

    monkeypatch.setattr(gpu, "detect_gpu", lambda: gpu.GpuStatus(ready=False, message="pas de GPU"))
    assert cli.main(["--input", str(burst_dir), "--out", str(tmp_path / "r.tif")]) == 2
    assert not (tmp_path / "r.tif").exists()


def test_cli_writes_tiff_and_jpeg(burst_dir, tmp_path, monkeypatch, bilinear_engine):
    from drone_burst_sr import gpu, pipeline

    monkeypatch.setattr("drone_burst_sr.gpu.detect_gpu",
                        lambda: gpu.GpuStatus(ready=True, name="fake", message="ok"))
    original = pipeline.run
    monkeypatch.setattr(pipeline, "run",
                        lambda src, params, progress=None: original(src, params, progress,
                                                                    engine=bilinear_engine(params.scale)))
    out = tmp_path / "res.tif"
    assert cli.main(["--input", str(burst_dir), "--scale", "2", "--out", str(out), "-q"]) == 0
    img = tifffile.imread(out)
    assert img.dtype == np.uint16 and img.shape == (256, 384, 3)
    assert out.with_suffix(".jpg").exists()


def test_selftest_metrics_reward_true_super_resolution():
    import cv2

    from drone_burst_sr.synthetic import make_burst

    burst = make_burst(256, 384, n_frames=1, alpha=1e-3, beta=1e-5)
    gt = (linear_to_srgb(burst.ground_truth(2.0)) * 65535).astype(np.uint16)
    single = (linear_to_srgb(burst.ground_truth(1.0)) * 65535).astype(np.uint16)
    single = cv2.resize(single, gt.shape[1::-1], interpolation=cv2.INTER_LANCZOS4)
    m = evaluate(gt, single, burst, 2.0)          # « fusion » parfaite vs image seule
    assert m["plus_net"] and m["gains_corr"]["texture"] > 0.3
    blurred = cv2.GaussianBlur(gt, (0, 0), 0.8)   # un flou modéré ne doit pas être puni
    assert evaluate(blurred, single, burst, 2.0)["gains_corr"]["texture"] > 0.2
    assert not evaluate(single, single, burst, 2.0)["plus_net"]  # aucune fusion : échec


def test_known_shift_reference_beats_single_frame(tmp_path):
    """Valide la mesure : avec les vrais décalages, la fusion de référence gagne en détail."""
    import dataclasses

    from drone_burst_sr.dng_io import read_frame_info
    from drone_burst_sr.pipeline import render_single
    from drone_burst_sr.selftest import reference_fusion
    from drone_burst_sr.synthetic import make_burst, write_burst

    burst = make_burst(256, 384, n_frames=10, alpha=1e-3, beta=1e-5)
    single = render_single(read_frame_info(write_burst(burst, tmp_path)[0]), (512, 768),
                           RenderParams(sharpen_amount=0.0))
    good = evaluate(reference_fusion(burst, 2.0), single, burst, 2.0)
    assert good["plus_net"], good
    wrong = dataclasses.replace(burst, shifts=[(-y, -x) for y, x in burst.shifts])
    assert not evaluate(reference_fusion(wrong, 2.0), single, burst, 2.0)["plus_net"]


def test_selftest_runs_end_to_end_with_fake_engine(tmp_path, bilinear_engine):
    from drone_burst_sr.selftest import format_report

    m = run_selftest(height=256, width=384, n_frames=4, out_dir=tmp_path,
                     engine_factory=lambda p: bilinear_engine(p.scale))
    assert not m["plus_net"]  # le moteur factice ne fusionne rien
    assert (tmp_path / "selftest_comparaison.png").exists()
    assert "détail texture" in format_report(m)


@pytest.mark.gpu
def test_gpu_fusion_is_sharper_than_single_frame():
    """Rafale synthétique à décalages sous-pixel connus : la fusion ×2 doit battre l'image seule."""
    from drone_burst_sr.selftest import format_report

    m = run_selftest(scale=2.0, n_frames=10, out_dir=None)
    assert m["plus_net"], format_report(m)
