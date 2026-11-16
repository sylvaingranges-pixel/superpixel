"""Paramètres utilisateur (sans dépendance au GPU)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

ALLOWED_SCALES = (1.0, 1.5, 2.0)


@dataclass
class FusionParams:
    # Échelle de sortie (1 = dématriçage + débruitage multi-images).
    scale: float = 2.0
    # Image de référence : la plus nette de la rafale, ou la première.
    reference: Literal["sharpest", "first"] = "sharpest"

    # Robustesse (rejet des zones mal alignées / en mouvement) — valeurs amont.
    # R = clamp(S · exp(-d²/σ²) - t, 0, 1), S valant s1 ou s2 selon le flux local.
    robustness: bool = True
    noise_correction: bool = True   # σ² et d² corrigés par le modèle de bruit (table Monte Carlo)
    t: float = 0.12                 # retranché à R : plus grand = rejet plus strict
    s1: float = 2.0                 # gain là où le flux optique varie localement (mouvement)
    s2: float = 12.0                # gain dans les zones au flux cohérent (statiques)
    mt: float = 0.8                 # variation locale du flux (px) au-delà de laquelle on applique s1

    # Noyaux de fusion. None = « auto » (valeur choisie par l'amont selon le SNR).
    kernel_type: Literal["steerable", "iso"] = "steerable"
    k_detail: float | None = None   # taille de base des noyaux (petit = plus net, plus bruité)
    k_denoise: float | None = None  # agrandissement des noyaux dans les zones lisses (débruitage)
    d_th: float | None = None       # seuil de « zone lisse » sur la force du gradient
    d_tr: float | None = None       # largeur de transition zone lisse / zone détaillée
    k_stretch: float = 4.0          # étirement le long des contours
    k_shrink: float = 2.0           # rétrécissement perpendiculaire aux contours

    # Rendu (appliqué à l'identique à la fusion et à l'image seule).
    sharpen_amount: float = 1.5
    sharpen_radius: float = 3.0
    exposure_ev: float = 0.0

    # Bruit : None = NoiseProfile DNG, sinon estimation sur la rafale.
    noise_alpha: Sequence[float] | None = None
    noise_beta: Sequence[float] | None = None
    estimate_noise: bool = False    # ignorer le tag NoiseProfile et estimer sur la rafale
    lut_trials: int = 10_000_000    # tirages Monte Carlo de la table de bruit amont

    # Mémoire : None = automatique selon la VRAM libre ; 0 = pas de tuiles ; >0 = côté en px.
    tile_size: int | None = None
    max_frames: int | None = None

    def validate(self) -> "FusionParams":
        if self.scale < 1 or self.scale > 3:
            raise ValueError("L'échelle doit être comprise entre 1 et 3 (1, 1.5 ou 2 conseillés).")
        if self.scale * 2 != round(self.scale * 2):
            raise ValueError("L'échelle doit être un multiple de 0,5 (tuiles alignées sur le CFA).")
        for name in ("t", "s1", "s2", "mt", "k_stretch", "k_shrink", "sharpen_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être > 0")
        if self.sharpen_amount < 0:
            raise ValueError("sharpen_amount doit être >= 0")
        if (self.noise_alpha is None) != (self.noise_beta is None):
            raise ValueError("noise_alpha et noise_beta doivent être fournis ensemble")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
