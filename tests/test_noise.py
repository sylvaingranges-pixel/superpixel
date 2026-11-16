import numpy as np
import pytest

from drone_burst_sr.noise import (estimate_from_stack, expand_profile, generic_profile,
                                  resolve_noise_model)
from drone_burst_sr.synthetic import make_burst


def _stack(burst):
    return np.stack([(r.astype(np.float32) - burst.black) / (burst.white - burst.black)
                     for r in burst.raws])


@pytest.mark.parametrize("alpha,beta", [(2e-4, 4e-6), (1e-3, 2e-5)])
def test_estimation_recovers_known_noise_curve(alpha, beta):
    burst = make_burst(512, 768, n_frames=8, alpha=alpha, beta=beta, seed=1)
    model = estimate_from_stack(_stack(burst))
    assert model.source == "estimé"
    a = np.array(model.alpha)
    assert np.all(np.abs(a / alpha - 1) < 0.2), a / alpha
    # Ce qui compte pour l'algorithme : la variance prédite sur la plage utile.
    for level in (0.1, 0.4, 0.8):
        predicted = a * level + np.array(model.beta)
        true = alpha * level + beta
        assert np.all(np.abs(predicted / true - 1) < 0.2), (level, predicted / true)


def test_expand_profile():
    a, b = expand_profile([1, 2, 3], [4, 5, 6])
    assert a == (1, 2, 3, 2) and b == (4, 5, 6, 5)
    assert expand_profile([1], [2]) == ((1,) * 4, (2,) * 4)
    with pytest.raises(ValueError):
        expand_profile([1, 2], [3, 4])


def test_resolution_order():
    tag = ((1e-4,), (1e-6,))
    assert resolve_noise_model(tag, 100, user_alpha=[2e-4], user_beta=[3e-6]).source == "utilisateur"
    assert resolve_noise_model(tag, 100).source == "NoiseProfile DNG"
    assert resolve_noise_model(None, 400).source == "générique"
    assert not generic_profile(400).reliable


@pytest.mark.upstream
def test_noise_lut_is_generated_and_cached(tmp_path):
    from drone_burst_sr.noise import NoiseModel, get_noise_lut

    model = NoiseModel((2e-4,) * 4, (4e-6,) * 4, "test")
    lut = get_noise_lut(model, trials=200_000, cache_dir=tmp_path)
    assert lut.sigma_noise_sq.shape == lut.d_noise_sq.shape
    assert len(list((tmp_path / "luts").glob("*.npz"))) == 1
    again = get_noise_lut(model, trials=200_000, cache_dir=tmp_path)
    assert np.array_equal(again.sigma_noise_sq, lut.sigma_noise_sq)
