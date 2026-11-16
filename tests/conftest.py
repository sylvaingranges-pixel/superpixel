import importlib.util

import cv2
import numpy as np
import pytest

from drone_burst_sr.synthetic import make_burst, write_burst
from drone_burst_sr.tiling import output_shape


def _gpu_ready() -> bool:
    try:
        from drone_burst_sr.gpu import detect_gpu

        return detect_gpu().ready
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    gpu = None
    upstream = importlib.util.find_spec("handheld_super_resolution") is not None
    for item in items:
        if "gpu" in item.keywords:
            gpu = _gpu_ready() if gpu is None else gpu
            if not gpu:
                item.add_marker(pytest.mark.skip(reason="GPU NVIDIA + extra [gpu] requis"))
        if "upstream" in item.keywords and not upstream:
            item.add_marker(pytest.mark.skip(reason="paquet handheld-super-resolution absent"))


class BilinearEngine:
    """Moteur factice CPU : dématriçage bilinéaire de la référence + agrandissement.

    Ne fait aucune fusion ; sert à tester la plomberie (tuiles, rendu, export) sans GPU.
    """

    name = "factice (tests)"

    def __init__(self, scale):
        self.scale = scale
        self.snr = None
        self.calls = 0

    def prepare(self, ref, status=None):
        pass

    def fuse_tile(self, ref, comp, on_frame=None):
        self.calls += 1
        raw = np.clip(ref * 65535, 0, 65535).astype(np.uint16)
        rgb = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 65535  # RGGB
        h, w = output_shape(*ref.shape, self.scale)
        if on_frame:
            for _ in range(comp.shape[0] + 1):
                on_frame()
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)


@pytest.fixture
def bilinear_engine():
    return BilinearEngine


@pytest.fixture(scope="session")
def small_burst():
    return make_burst(128, 192, n_frames=6, seed=3)


@pytest.fixture
def burst_dir(tmp_path, small_burst):
    write_burst(small_burst, tmp_path / "rafale")
    return tmp_path / "rafale"
