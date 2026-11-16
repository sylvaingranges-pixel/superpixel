"""Contrôles de cohérence d'une rafale avant fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .dng_io import FrameInfo

Level = Literal["erreur", "avertissement", "info"]

MIN_FRAMES = 2
RECOMMENDED_MIN_FRAMES = 6
RECOMMENDED_MAX_FRAMES = 20


@dataclass(frozen=True)
class Issue:
    level: Level
    message: str

    def __str__(self) -> str:
        icon = {"erreur": "❌", "avertissement": "⚠️", "info": "ℹ️"}[self.level]
        return f"{icon} {self.message}"


QUAD_BAYER_NOTE = (
    "Mode 48 MP détecté : le capteur Quad Bayer est « remosaïqué » en Bayer par le "
    "drone. Une partie des détails est déjà interpolée et le bruit est corrélé "
    "spatialement, ce qui réduit fortement le gain de la super-résolution. Pour le "
    "meilleur résultat, photographiez en mode 12 MP (vrai Bayer) et utilisez une "
    "échelle ×2 ; en 48 MP, préférez ×1 (fusion = débruitage + dématriçage)."
)


HANDHELD_NOTE = (
    "Appareil photo : la super-résolution a besoin de petits décalages entre les images. "
    "Photographiez la rafale à main levée (le tremblement naturel suffit, stabilisation "
    "activée ou non) ; sur trépied, les images sont identiques et seul le débruitage opère."
)


def _rel_spread(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or np.max(np.abs(v)) == 0:
        return 0.0
    return float((v.max() - v.min()) / np.max(np.abs(v)))


def validate_burst(infos: Sequence[FrameInfo]) -> list[Issue]:
    """Vérifie dimensions, CFA, exposition, ISO, ouverture et balance des blancs."""
    issues: list[Issue] = []
    n = len(infos)
    if n < MIN_FRAMES:
        issues.append(Issue("erreur", f"Il faut au moins {MIN_FRAMES} images (reçu : {n})."))
        return issues
    if n < RECOMMENDED_MIN_FRAMES:
        issues.append(Issue("avertissement",
                            f"Seulement {n} images : le gain sera limité (8 à 15 conseillées)."))
    if n > RECOMMENDED_MAX_FRAMES:
        issues.append(Issue("info", f"{n} images : au-delà de ~15-20 le gain devient marginal "
                                    "et le temps/mémoire augmentent."))

    ref = infos[0]
    sizes = {(i.width, i.height) for i in infos}
    if len(sizes) > 1:
        issues.append(Issue("erreur", "Dimensions différentes dans la rafale : "
                                      + ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
                                      + " (mélange 12 MP / 48 MP ?)."))
    cfas = {i.cfa_string for i in infos}
    if len(cfas) > 1:
        issues.append(Issue("erreur", f"Motifs CFA différents : {sorted(cfas)}."))

    models = {(i.make, i.model) for i in infos}
    if len(models) > 1:
        issues.append(Issue("avertissement", f"Appareils différents : {sorted(models)}."))

    def check(name: str, values, tol: float, fmt=str, required: bool = True):
        known = [v for v in values if v is not None]
        if required and len(known) < len(values):
            issues.append(Issue("avertissement", f"{name} absent de certains fichiers."))
        if len(known) >= 2 and _rel_spread(known) > tol:
            uniq = sorted({round(v, 6) for v in known})
            issues.append(Issue("avertissement",
                                f"{name} différent(e) selon les images : "
                                + ", ".join(fmt(v) for v in uniq)
                                + ". Passez en exposition manuelle (M) pour toute la rafale."))

    check("ISO", [i.iso for i in infos], 0.01, lambda v: f"{v:g}")
    check("Temps de pose", [i.exposure_time for i in infos], 0.01,
          lambda v: f"1/{round(1 / v)} s" if v and v < 1 else f"{v:g} s")
    check("Ouverture", [i.f_number for i in infos], 0.01, lambda v: f"f/{v:.1f}")
    check("Correction d'exposition", [i.exposure_bias for i in infos], 0.0, lambda v: f"{v:+.1f} EV",
          required=False)

    wbs = np.array([i.white_balance for i in infos])
    if np.any(np.ptp(wbs, axis=0) / np.maximum(wbs.mean(axis=0), 1e-9) > 0.02):
        issues.append(Issue("avertissement",
                            "Balance des blancs différente selon les images (BdB auto ?). "
                            "La couleur de l'image de référence sera utilisée ; fixez la BdB."))

    if any(i.white_level != ref.white_level for i in infos) or any(
            not np.allclose(i.black_2x2, ref.black_2x2) for i in infos):
        issues.append(Issue("avertissement", "Niveaux de noir/blanc différents selon les images."))

    if ref.is_quad_bayer_mode:
        issues.append(Issue("avertissement", QUAD_BAYER_NOTE))
    elif "dji" in ref.make.lower():
        issues.append(Issue("info", f"Mode {ref.mode_label} : c'est le mode qui donne le meilleur gain."))
    else:
        issues.append(Issue("info", HANDHELD_NOTE))

    if ref.noise_profile is None:
        issues.append(Issue("info", "Pas de tag NoiseProfile : la courbe de bruit sera estimée "
                                    "à partir de la rafale."))
    if ref.xyz_to_cam is None:
        issues.append(Issue("avertissement", "Pas de matrice couleur DNG : couleurs non calibrées."))
    return issues


def check_brightness(means: Sequence[float], names: Sequence[str] | None = None,
                     tol: float = 0.08) -> list[Issue]:
    """Compare la luminosité moyenne mesurée des images à la référence (auto-exposition ?)."""
    m = np.asarray(means, dtype=np.float64)
    ref = m[0] if m[0] > 0 else np.median(m)
    if ref <= 0:
        return []
    dev = np.abs(m / ref - 1)
    labels = list(names) if names is not None else [str(k + 1) for k in range(len(m))]
    bad = [labels[int(k)] for k in np.nonzero(dev > tol)[0]]
    if bad:
        return [Issue("avertissement",
                      f"Luminosité moyenne différente de plus de {tol:.0%} pour : {', '.join(bad)} "
                      f"(écart max {dev.max():.0%}). Exposition auto, scintillement ou changement "
                      "de lumière ? Ces images pèseront peu dans la fusion.")]
    return []


def has_errors(issues: Sequence[Issue]) -> bool:
    return any(i.level == "erreur" for i in issues)


def format_issues(issues: Sequence[Issue]) -> str:
    return "\n".join(str(i) for i in issues)
