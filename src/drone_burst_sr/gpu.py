"""Détection du GPU NVIDIA et de la pile CUDA nécessaire à l'algorithme amont."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

COLAB_HINT = (
    "Sans GPU NVIDIA, utilisez le notebook Google Colab fourni "
    "(notebooks/drone_burst_sr_colab.ipynb) : il lance cette même interface sur un GPU "
    "gratuit. L'analyse des DNG et l'estimation du bruit restent disponibles en local."
)


@dataclass
class GpuStatus:
    ready: bool
    name: str | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    driver: str | None = None
    upstream_installed: bool = False
    message: str = ""
    in_colab: bool = False

    def describe(self) -> str:
        if self.ready:
            mem = f"{self.free_bytes / 1024**3:.1f} Go libres / {self.total_bytes / 1024**3:.1f} Go" \
                if self.total_bytes else "mémoire inconnue"
            return f"✅ GPU : {self.name} ({mem}, pilote {self.driver or '?'})"
        return f"❌ {self.message}"


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, total, free, driver = [p.strip() for p in out[0].split(",")[:4]]
    return {"name": name, "total": int(float(total)) * 1024**2,
            "free": int(float(free)) * 1024**2, "driver": driver}


def detect_gpu() -> GpuStatus:
    """Vérifie : GPU NVIDIA présent, paquet amont installé, PyTorch CUDA et Numba CUDA."""
    smi = _nvidia_smi()
    upstream = importlib.util.find_spec("handheld_super_resolution") is not None
    status = GpuStatus(ready=False, upstream_installed=upstream, in_colab=in_colab())
    if smi:
        status.name, status.total_bytes = smi["name"], smi["total"]
        status.free_bytes, status.driver = smi["free"], smi["driver"]

    if smi is None and not upstream:
        status.message = ("Aucun GPU NVIDIA détecté. L'algorithme Handheld Multi-Frame "
                          "Super-Resolution (Numba CUDA + PyTorch) ne fonctionne pas sur CPU : "
                          "le code amont crée ses données directement sur « cuda » et aucun repli "
                          "CPU n'existe. " + COLAB_HINT)
        return status
    if not upstream:
        status.message = ("GPU détecté mais l'algorithme n'est pas installé. "
                          "Installez l'extra GPU : pip install \".[gpu]\"")
        return status

    try:
        import torch
    except ImportError:
        status.message = "PyTorch absent : réinstallez avec pip install \".[gpu]\"."
        return status
    if not torch.cuda.is_available():
        status.message = ("PyTorch ne voit pas de GPU CUDA (PyTorch installé en version CPU, ou "
                          "pas de GPU NVIDIA). " + COLAB_HINT)
        return status
    try:
        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError("numba.cuda.is_available() = False")
        cuda.get_current_device()
    except Exception as error:  # noqa: BLE001 - message utilisateur
        status.message = (f"Numba CUDA indisponible ({error}). Si le pilote NVIDIA est antérieur "
                          "à CUDA 13, installez la variante CUDA 12 : "
                          "pip install \"numba-cuda[cu12]\" (voir README).")
        return status

    free, total = torch.cuda.mem_get_info()
    status.ready = True
    status.free_bytes, status.total_bytes = int(free), int(total)
    status.name = status.name or torch.cuda.get_device_name(0)
    status.message = "GPU prêt."
    return status
