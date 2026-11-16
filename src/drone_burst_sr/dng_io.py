"""Lecture des DNG et de leurs métadonnées (rawpy + exifread, exiftool facultatif).

Tout ce qui concerne le capteur est lu ici : motif CFA, niveaux de noir et de
blanc, balance des blancs, matrice couleur, ISO, temps de pose, profil de bruit.
Les données sont normalisées exactement comme dans l'implémentation amont
(``(x - noir) / (blanc - noir)`` par canal, sans écrêtage).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Le Mini 4 Pro produit 4032x3024 (12 MP, vrai Bayer) ou 8064x6048
# (48 MP, Quad Bayer remosaïqué en Bayer par l'appareil). La règle ne vaut que pour
# les drones DJI : un hybride de 33 MP (Sony α7 IV) est un vrai Bayer.
QUAD_BAYER_MIN_MEGAPIXELS = 30.0
QUAD_BAYER_MAKES = ("DJI",)

# Formats RAW Bayer lus par rawpy/LibRaw et dont les tags EXIF sont lisibles par exifread
# (conteneurs TIFF). Le X-Trans de Fujifilm (.RAF) n'est pas Bayer : refusé à la lecture.
RAW_EXTENSIONS = (".dng", ".arw", ".nef", ".cr2", ".orf", ".rw2", ".pef")

# Illuminants DNG ayant une température de couleur définie (même logique que
# l'amont : on prend la matrice dont l'illuminant est le plus proche de D65).
_ILLUMINANT_CCT = {
    1: 5500, 3: 2850, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 17: 2856,
    18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200,
}

_COLOR_NAMES = {0: "R", 1: "G", 2: "B"}


class UnsupportedDNGError(ValueError):
    """Le fichier n'est pas un DNG Bayer exploitable."""


@dataclass
class FrameInfo:
    """Métadonnées d'une image de la rafale."""

    path: Path
    width: int
    height: int
    cfa: np.ndarray                      # 2x2, 0=R 1=G 2=B (couleurs vues dans l'image)
    black_levels: tuple[float, float, float, float]  # dans l'ordre rawpy (color_desc RGBG)
    black_2x2: np.ndarray                # niveau de noir par position du motif 2x2
    white_level: float
    white_balance: tuple[float, float, float]        # multiplicateurs R, G, B (G = 1)
    xyz_to_cam: np.ndarray | None        # 3x3, matrice ColorMatrix DNG choisie
    iso: float | None
    exposure_time: float | None          # secondes
    f_number: float | None
    exposure_bias: float | None
    make: str
    model: str
    orientation: int
    noise_profile: tuple[tuple[float, ...], tuple[float, ...]] | None  # (alpha, beta)
    baseline_exposure: float | None
    datetime: str | None
    extra: dict = field(default_factory=dict)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def is_quad_bayer_mode(self) -> bool:
        is_dji = any(m.lower() in self.make.lower() for m in QUAD_BAYER_MAKES)
        return is_dji and self.megapixels >= QUAD_BAYER_MIN_MEGAPIXELS

    @property
    def mode_label(self) -> str:
        mp = f"{self.megapixels:.0f}" if self.megapixels >= 10 else f"{self.megapixels:.1f}"
        if self.is_quad_bayer_mode:
            return f"{mp} MP (Quad Bayer remosaïqué)"
        return f"{mp} MP (Bayer)"

    @property
    def cfa_string(self) -> str:
        return "".join(_COLOR_NAMES[int(c)] for c in self.cfa.ravel())

    def summary(self) -> dict:
        """Résumé lisible (pour l'interface et la CLI)."""
        return {
            "fichier": self.path.name,
            "taille": f"{self.width}x{self.height}",
            "mode": self.mode_label,
            "CFA": self.cfa_string,
            "noir": "/".join(f"{b:g}" for b in self.black_levels),
            "blanc": f"{self.white_level:g}",
            "ISO": _fmt(self.iso, "{:g}"),
            "pose": _fmt_exposure(self.exposure_time),
            "f/": _fmt(self.f_number, "{:.1f}"),
            "BdB (R,G,B)": ", ".join(f"{v:.3f}" for v in self.white_balance),
            "NoiseProfile": "oui" if self.noise_profile else "non",
        }


def _fmt(value, pattern: str) -> str:
    return "?" if value is None else pattern.format(value)


def _fmt_exposure(t: float | None) -> str:
    if t is None:
        return "?"
    if t < 1:
        return f"1/{round(1 / t)} s"
    return f"{t:g} s"


# --------------------------------------------------------------------------- #
# Fichiers
# --------------------------------------------------------------------------- #
def is_raw_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def list_raw_files(folder: str | Path) -> list[Path]:
    """Liste les RAW (DNG, ARW…) d'un dossier, sans tenir compte de la casse (DJI écrit ``.DNG``).

    Si le dossier mélange plusieurs formats, seul le plus fréquent est gardé (ex. des DNG
    convertis à côté des ARW d'origine) pour ne pas fusionner deux fois la même photo.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and is_raw_file(p))
    exts = [p.suffix.lower() for p in files]
    if len(set(exts)) > 1:
        main = max(set(exts), key=exts.count)
        files = [p for p in files if p.suffix.lower() == main]
    return files


list_dng_files = list_raw_files  # compatibilité


# --------------------------------------------------------------------------- #
# Tags EXIF
# --------------------------------------------------------------------------- #
def _read_tags(path: Path) -> dict:
    import exifread

    with open(path, "rb") as fh:
        return exifread.process_file(fh, details=False)


def _tag(tags: dict, *names: str):
    for name in names:
        for prefix in ("EXIF ", "Image "):
            if prefix + name in tags:
                return tags[prefix + name]
        if name in tags:
            return tags[name]
    return None


def _to_float(value) -> float:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "num") and hasattr(value, "den"):
        return float(value.num) / float(value.den) if value.den else float("nan")
    return float(value)


def _tag_float(tags: dict, *names: str) -> float | None:
    tag = _tag(tags, *names)
    if tag is None:
        return None
    try:
        return _to_float(tag.values[0] if hasattr(tag, "values") else tag)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def _tag_values(tags: dict, *names: str) -> list[float] | None:
    tag = _tag(tags, *names)
    if tag is None:
        return None
    try:
        return [_to_float(v) for v in tag.values]
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _tag_str(tags: dict, *names: str) -> str | None:
    tag = _tag(tags, *names)
    return None if tag is None else str(tag).strip()


def _noise_profile_from_tags(tags: dict):
    """Tag DNG NoiseProfile (0xC761) : paires (S, O) avec var = S·x + O, x dans [0, 1]."""
    values = _tag_values(tags, "Tag 0xC761", "NoiseProfile")
    if not values or len(values) % 2:
        return None
    alpha = tuple(values[0::2])
    beta = tuple(values[1::2])
    if any(not np.isfinite(v) or v < 0 for v in alpha + beta):
        return None
    return alpha, beta


def _color_matrix_from_tags(tags: dict) -> np.ndarray | None:
    candidates = []
    for matrix_name, illum_name in (("Tag 0xC621", "Tag 0xC65A"), ("Tag 0xC622", "Tag 0xC65B")):
        values = _tag_values(tags, matrix_name)
        if not values or len(values) != 9:
            continue
        illum = _tag_float(tags, illum_name)
        cct = _ILLUMINANT_CCT.get(int(illum)) if illum is not None else None
        distance = abs(cct - 6504) if cct is not None else float("inf")
        candidates.append((distance, np.asarray(values, dtype=np.float64).reshape(3, 3)))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def _cfa_from_raw(raw) -> np.ndarray:
    """Motif CFA 2x2 (0=R, 1=G, 2=B) vu depuis le coin haut-gauche de l'image visible."""
    if raw.raw_type.name != "Flat" or raw.num_colors != 3:
        raise UnsupportedDNGError(
            "DNG non Bayer (LinearRaw ou multi-plan) : la super-résolution a besoin "
            "des données CFA brutes."
        )
    colors = np.asarray(raw.raw_colors_visible)
    if colors.shape[0] < 4 or colors.shape[1] < 4:
        raise UnsupportedDNGError("Image trop petite.")
    block = colors[:2, :2]
    if not np.array_equal(np.tile(block, (2, 2)), colors[:4, :4]):
        raise UnsupportedDNGError(
            "Motif CFA non périodique 2x2 (Quad Bayer non remosaïqué ?) : non supporté. "
            "Le mode 48 MP du Mini 4 Pro est normalement remosaïqué en Bayer par l'appareil."
        )
    desc = raw.color_desc.decode("ascii", "replace")
    mapping = {"R": 0, "G": 1, "B": 2}
    try:
        cfa = np.vectorize(lambda idx: mapping[desc[int(idx)]])(block)
    except (KeyError, IndexError) as error:
        raise UnsupportedDNGError(f"Description des couleurs inattendue : {desc!r}") from error
    cfa = cfa.astype(np.int64)
    if sorted(cfa.ravel().tolist()) != [0, 1, 1, 2]:
        raise UnsupportedDNGError(f"Motif CFA non Bayer : {cfa.tolist()}")
    return cfa


def read_frame_info(path: str | Path) -> FrameInfo:
    """Lit les métadonnées d'un DNG (sans charger durablement les pixels)."""
    import rawpy

    path = Path(path)
    tags = _read_tags(path)
    with rawpy.imread(str(path)) as raw:
        cfa = _cfa_from_raw(raw)
        h, w = raw.raw_image_visible.shape
        black = tuple(float(b) for b in raw.black_level_per_channel)
        colors = np.asarray(raw.raw_colors_visible[:2, :2])
        black_2x2 = np.array([[black[int(c)] for c in row] for row in colors], dtype=np.float32)
        white = float(raw.white_level)
        cam_wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)
        rawpy_matrix = np.asarray(raw.rgb_xyz_matrix[:3], dtype=np.float64)

    # Balance des blancs (multiplicateurs). rawpy les déduit d'AsShotNeutral.
    if cam_wb.size < 3 or cam_wb[1] <= 0 or not np.all(np.isfinite(cam_wb[:3])):
        neutral = _tag_values(tags, "Tag 0xC628", "AsShotNeutral")
        if neutral and len(neutral) == 3 and min(neutral) > 0:
            cam_wb = np.array([1 / v for v in neutral])
        else:
            warnings.warn(f"{path.name} : balance des blancs absente, neutre utilisée.")
            cam_wb = np.ones(3)
    wb = tuple(float(v / cam_wb[1]) for v in cam_wb[:3])

    xyz_to_cam = _color_matrix_from_tags(tags)
    if xyz_to_cam is None and np.linalg.norm(rawpy_matrix) > 0:
        xyz_to_cam = rawpy_matrix

    exposure = _tag_float(tags, "ExposureTime")
    if exposure is None:
        shutter = _tag_float(tags, "ShutterSpeedValue")
        exposure = 2.0 ** (-shutter) if shutter is not None else None

    orientation = _tag_float(tags, "Orientation")
    return FrameInfo(
        path=path,
        width=w - (w % 2),
        height=h - (h % 2),
        cfa=cfa,
        black_levels=black,  # type: ignore[arg-type]
        black_2x2=black_2x2,
        white_level=white,
        white_balance=wb,  # type: ignore[arg-type]
        xyz_to_cam=xyz_to_cam,
        iso=_tag_float(tags, "ISOSpeedRatings", "PhotographicSensitivity", "RecommendedExposureIndex"),
        exposure_time=exposure,
        f_number=_tag_float(tags, "FNumber"),
        exposure_bias=_tag_float(tags, "ExposureBiasValue"),
        make=_tag_str(tags, "Make") or "",
        model=_tag_str(tags, "Model") or "",
        orientation=int(orientation) if orientation else 1,
        noise_profile=_noise_profile_from_tags(tags),
        baseline_exposure=_tag_float(tags, "Tag 0xC62A", "BaselineExposure"),
        datetime=_tag_str(tags, "DateTimeOriginal", "DateTime"),
    )


def read_burst_info(paths: Iterable[str | Path]) -> list[FrameInfo]:
    return [read_frame_info(p) for p in paths]


def load_normalized(info: FrameInfo, out: np.ndarray | None = None) -> np.ndarray:
    """Charge les pixels CFA, normalisés (noir -> 0, blanc -> 1) par canal, en float32.

    Dimensions ramenées à des valeurs paires pour conserver la phase du CFA.
    ``out`` permet d'écrire directement dans une pile préallouée (économie de RAM).
    """
    import rawpy

    with rawpy.imread(str(info.path)) as raw:
        data = raw.raw_image_visible[: info.height, : info.width]
        if out is None:
            out = np.empty((info.height, info.width), dtype=np.float32)
        np.copyto(out, data, casting="unsafe")
    for i in range(2):
        for j in range(2):
            black = info.black_2x2[i, j]
            out[i::2, j::2] -= black
            out[i::2, j::2] *= 1.0 / (info.white_level - black)
    return out


def load_raw_uint16(info: FrameInfo) -> np.ndarray:
    """Pixels CFA bruts (non normalisés), recadrés en dimensions paires."""
    import rawpy

    with rawpy.imread(str(info.path)) as raw:
        return raw.raw_image_visible[: info.height, : info.width].copy()


# --------------------------------------------------------------------------- #
# exiftool (facultatif, pour un contrôle croisé)
# --------------------------------------------------------------------------- #
EXIFTOOL_TAGS = (
    "Make", "Model", "ImageWidth", "ImageHeight", "CFAPattern2", "CFARepeatPatternDim",
    "BlackLevel", "WhiteLevel", "AsShotNeutral", "ColorMatrix1", "ColorMatrix2",
    "CalibrationIlluminant1", "CalibrationIlluminant2", "NoiseProfile", "ISO",
    "ExposureTime", "FNumber", "BaselineExposure",
)


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def exiftool_dump(paths: Sequence[str | Path], tags: Sequence[str] = EXIFTOOL_TAGS) -> list[dict]:
    """Lit quelques tags avec exiftool (s'il est installé). Renvoie [] sinon."""
    exe = shutil.which("exiftool")
    if exe is None or not paths:
        return []
    cmd = [exe, "-j", "-n", *[f"-{t}" for t in tags], *[str(p) for p in paths]]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []


def info_as_dict(info: FrameInfo) -> dict:
    d = asdict(info)
    d["path"] = str(info.path)
    d["cfa"] = info.cfa_string
    d["black_2x2"] = info.black_2x2.tolist()
    d["xyz_to_cam"] = None if info.xyz_to_cam is None else info.xyz_to_cam.tolist()
    d["mode"] = info.mode_label
    return d
