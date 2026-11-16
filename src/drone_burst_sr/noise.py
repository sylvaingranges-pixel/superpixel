"""Modèle de bruit capteur : var(x) = alpha·x + beta (x normalisé dans [0, 1]).

Ordre des sources :

1. valeurs fournies par l'utilisateur ;
2. tag DNG ``NoiseProfile`` (0xC761) s'il est présent ;
3. estimation à partir de la rafale (méthode documentée dans
   ``docs/estimation_bruit.md``) ;
4. en dernier recours, profil générique mis à l'échelle de l'ISO.

Le module s'exécute aussi seul :
``python -m drone_burst_sr.noise --input dossier/ [--json bruit.json] [--plot bruit.png]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .cfa import UPSTREAM_PLANE_ORDER, planes

Profile4 = tuple[float, float, float, float]


@dataclass
class NoiseModel:
    alpha: Profile4          # ordre amont : R, G1, B, G2
    beta: Profile4
    source: str              # "utilisateur" | "NoiseProfile DNG" | "estimé" | "générique"
    reliable: bool = True
    details: dict = field(default_factory=dict)

    def describe(self) -> str:
        a = ", ".join(f"{v:.3g}" for v in self.alpha)
        b = ", ".join(f"{v:.3g}" for v in self.beta)
        flag = "" if self.reliable else " (peu fiable)"
        return f"Bruit [{self.source}{flag}] alpha(R,G1,B,G2)=({a}) beta=({b})"


def expand_profile(alpha: Sequence[float], beta: Sequence[float]) -> tuple[Profile4, Profile4]:
    """Étend un profil à 1, 3 (R, G, B) ou 4 valeurs vers l'ordre R, G1, B, G2."""
    alpha = tuple(float(v) for v in alpha)
    beta = tuple(float(v) for v in beta)
    if len(alpha) != len(beta):
        raise ValueError("alpha et beta doivent avoir le même nombre de valeurs")
    if len(alpha) == 1:
        return alpha * 4, beta * 4  # type: ignore[return-value]
    if len(alpha) == 3:
        return (alpha[0], alpha[1], alpha[2], alpha[1]), (beta[0], beta[1], beta[2], beta[1])
    if len(alpha) == 4:
        return alpha, beta  # type: ignore[return-value]
    raise ValueError(f"profil de bruit à 1, 3 ou 4 valeurs attendu, reçu {len(alpha)}")


def generic_profile(iso: float | None) -> NoiseModel:
    """Profil du Pixel 4 (README amont), mis à l'échelle de l'ISO. Dernier recours."""
    g = max(float(iso or 100), 50.0) / 100.0
    a, b = expand_profile([1.80710882e-4 * g], [3.1937599182128e-6 * g * g])
    return NoiseModel(a, b, "générique", reliable=False,
                      details={"note": "profil Pixel 4 mis à l'échelle ISO, à remplacer par une estimation"})


# --------------------------------------------------------------------------- #
# Estimation à partir de la rafale
# --------------------------------------------------------------------------- #
def _block_view(x: np.ndarray, b: int) -> np.ndarray:
    """(…, H, W) -> (…, nby, nbx, b, b) (vue)."""
    *lead, h, w = x.shape
    nby, nbx = h // b, w // b
    x = x[..., : nby * b, : nbx * b]
    x = x.reshape(*lead, nby, b, nbx, b)
    return np.moveaxis(x, -3, -2)


def _plane_block_stats(plane_stack: np.ndarray, block: int, max_shift: float):
    """Statistiques par bloc d'un plan CFA : intensité, variance temporelle, critère de platitude."""
    n = plane_stack.shape[0]
    stack = plane_stack.astype(np.float32, copy=False)
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=1)
    vmin = stack.min(axis=0)
    vmax = stack.max(axis=0)

    mb = _block_view(mean, block).reshape(*_block_view(mean, block).shape[:2], -1)
    vb = _block_view(var, block).reshape(mb.shape)
    lo = _block_view(vmin, block).reshape(mb.shape)
    hi = _block_view(vmax, block).reshape(mb.shape)

    # Régression plane (gradient lent) dans chaque bloc, coordonnées centrées orthogonales.
    c = np.arange(block, dtype=np.float32) - (block - 1) / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    sxx = float(np.sum(xx * xx))
    intensity = mb.mean(axis=-1)
    centered = mb - intensity[..., None]
    gx = (centered @ xx) / sxx
    gy = (centered @ yy) / sxx
    resid_var = (np.sum(centered ** 2, axis=-1) - (gx ** 2 + gy ** 2) * sxx) / (block * block - 3)
    # Médiane (et non moyenne) des variances pixel à pixel : robuste aux quelques pixels
    # de bord contaminés par un contour voisin qui « entre » dans le bloc quand le drone
    # bouge. La médiane d'une variance empirique à n-1 degrés de liberté est biaisée vers
    # le bas : on corrige par médiane(chi2_k)/k.
    from scipy.stats import chi2

    k = n - 1
    # Seul l'intérieur du bloc sert à mesurer la variance ; la marge sert au test de
    # platitude (un contour dans la marge fait rejeter le bloc).
    margin = min(int(np.ceil(max_shift / 2)) + 1, block // 4)  # décalage en pixels du plan CFA
    inner = _block_view(var, block)[..., margin:block - margin, margin:block - margin]
    inner = inner.reshape(*inner.shape[:2], -1)
    temporal_var = np.median(inner, axis=-1) / (chi2.median(k) / k)

    floor = float(np.min(vmin))
    clipped_low = np.mean(lo <= floor + 1e-6, axis=-1) > 0.01
    clipped_high = hi.max(axis=-1) > 0.97
    # Zone plate : la variance spatiale résiduelle de la moyenne temporelle doit être
    # celle attendue du seul bruit (var/N), et le gradient ne doit pas « fuir » dans la
    # variance temporelle à cause des petits déplacements du drone.
    expected = temporal_var / n
    flat = resid_var < 1.5 * expected
    slope_leak = (gx ** 2 + gy ** 2) * max_shift ** 2 < 0.1 * temporal_var
    ok = flat & slope_leak & ~clipped_low & ~clipped_high & (temporal_var > 0)
    return intensity[ok].ravel(), temporal_var[ok].ravel(), int(ok.size)


def _fit_affine(i: np.ndarray, v: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Moindres carrés relatifs robustes de v = a·i + b, avec a, b >= 0."""
    keep = np.ones(i.size, dtype=bool)
    a = b = 0.0
    for _ in range(3):
        w = 1.0 / np.maximum(v[keep], 1e-30) ** 2
        A = np.stack([i[keep], np.ones(keep.sum())], axis=1)
        sw = np.sqrt(w)
        a, b = np.linalg.lstsq(A * sw[:, None], v[keep] * sw, rcond=None)[0]
        if b < 0:
            b = 0.0
            a = float(np.sum(w * i[keep] * v[keep]) / max(np.sum(w * i[keep] ** 2), 1e-30))
        if a < 0:
            a = 0.0
            b = float(np.sum(w * v[keep]) / np.sum(w))
        rel = (v - (a * i + b)) / np.maximum(v, 1e-30)
        mad = np.median(np.abs(rel[keep] - np.median(rel[keep]))) * 1.4826
        new_keep = np.abs(rel) <= max(3 * mad, 0.05)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(a), float(b), keep


def estimate_from_stack(stack_rggb: np.ndarray, block: int = 16, n_bins: int = 24,
                        max_frames: int = 10, max_shift: float = 3.0) -> NoiseModel:
    """Estime alpha/beta par plan CFA à partir d'une pile (N, H, W) normalisée en RGGB.

    Principe : dans les zones plates de la scène, un léger déplacement entre images ne
    change pas la valeur des pixels ; la variance temporelle pixel à pixel y mesure donc
    directement le bruit. On la regroupe par niveau d'intensité et on ajuste
    var = alpha·I + beta.
    """
    if stack_rggb.ndim != 3 or stack_rggb.shape[0] < 2:
        raise ValueError("Il faut une pile d'au moins 2 images (N, H, W)")
    stack = stack_rggb[:max_frames]
    alpha, beta, info = {}, {}, {}
    reliable = True
    for name, plane in planes(stack).items():
        inten, tvar, n_blocks = _plane_block_stats(plane, block, max_shift)
        entry = {"blocs_total": n_blocks, "blocs_plats": int(inten.size)}
        if inten.size < 30:
            reliable = False
            entry["erreur"] = "pas assez de zones plates"
            info[name] = entry
            alpha[name], beta[name] = np.nan, np.nan
            continue
        bins = int(max(3, min(n_bins, inten.size // 10)))
        edges = np.unique(np.quantile(inten, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(edges, inten, side="right") - 1, 0, len(edges) - 2)
        bi, bv, bc = [], [], []
        for k in range(len(edges) - 1):
            sel = idx == k
            if sel.sum() >= 5:
                bi.append(float(np.median(inten[sel])))
                bv.append(float(np.median(tvar[sel])))
                bc.append(int(sel.sum()))
        bi_a, bv_a = np.array(bi), np.array(bv)
        if len(bi) < 3:
            reliable = False
            entry["erreur"] = "trop peu de niveaux d'intensité"
            info[name] = entry
            alpha[name], beta[name] = np.nan, np.nan
            continue
        a, b, keep = _fit_affine(bi_a, bv_a)
        span = float(bi_a.max() - bi_a.min())
        if span < 0.05 or len(bi) < 5:
            reliable = False
            entry["avertissement"] = f"plage d'intensité étroite ({span:.3f}) : alpha/beta mal séparés"
        entry.update({"intensite": bi, "variance": bv, "effectif": bc,
                      "retenus": keep.tolist(), "alpha": a, "beta": b})
        info[name] = entry
        alpha[name], beta[name] = a, b

    # Plans en échec : on reprend la moyenne des plans réussis.
    good = [k for k in alpha if np.isfinite(alpha[k])]
    if not good:
        raise RuntimeError("Estimation du bruit impossible : aucune zone plate exploitable.")
    for k in alpha:
        if not np.isfinite(alpha[k]):
            alpha[k] = float(np.mean([alpha[g] for g in good]))
            beta[k] = float(np.mean([beta[g] for g in good]))
    a4 = tuple(float(alpha[k]) for k in UPSTREAM_PLANE_ORDER)
    b4 = tuple(float(beta[k]) for k in UPSTREAM_PLANE_ORDER)
    return NoiseModel(a4, b4, "estimé", reliable=reliable,  # type: ignore[arg-type]
                      details={"plans": info, "images": int(stack.shape[0]), "bloc": block})


def resolve_noise_model(noise_profile, iso, stack_rggb: np.ndarray | None = None,
                        user_alpha: Sequence[float] | None = None,
                        user_beta: Sequence[float] | None = None,
                        prefer_estimate: bool = False) -> NoiseModel:
    """Choisit la source du modèle de bruit (voir la docstring du module)."""
    if (user_alpha is None) != (user_beta is None):
        raise ValueError("alpha et beta doivent être fournis ensemble")
    if user_alpha is not None:
        a, b = expand_profile(user_alpha, user_beta)  # type: ignore[arg-type]
        return NoiseModel(a, b, "utilisateur")
    if noise_profile is not None and not prefer_estimate:
        a, b = expand_profile(*noise_profile)
        return NoiseModel(a, b, "NoiseProfile DNG")
    if stack_rggb is not None:
        try:
            return estimate_from_stack(stack_rggb)
        except RuntimeError as error:
            warnings.warn(f"{error} Profil générique utilisé.")
    return generic_profile(iso)


# --------------------------------------------------------------------------- #
# Table de correction (Monte Carlo amont), mise en cache
# --------------------------------------------------------------------------- #
def lut_cache_dir() -> Path:
    import os

    base = os.environ.get("DRONE_BURST_SR_CACHE")
    if base:
        return Path(base)
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "drone_burst_sr"


def get_noise_lut(model: NoiseModel, trials: int = 10_000_000, cache_dir: Path | None = None):
    """Table σ²/d² de l'amont pour ce profil (générée sur CPU puis mise en cache).

    Nécessite l'extra ``[gpu]`` (paquet ``handheld_super_resolution``).
    """
    from handheld_super_resolution.monte_carlo import generate_noise_lut
    from handheld_super_resolution.noise_lut import load_noise_lut, save_noise_lut

    cache_dir = Path(cache_dir) if cache_dir else lut_cache_dir()
    key = hashlib.sha1(json.dumps([list(model.alpha), list(model.beta), trials]).encode()).hexdigest()[:16]
    path = cache_dir / "luts" / f"lut_{key}.npz"
    if path.is_file():
        try:
            return load_noise_lut(path, expected_alpha=model.alpha, expected_beta=model.beta)
        except ValueError:
            path.unlink(missing_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lut = generate_noise_lut(model.alpha, model.beta, trials=trials, show_progress=False)
    save_noise_lut(path, lut, trials=trials, seed=0, source_dng="drone_burst_sr")
    return load_noise_lut(path, expected_alpha=model.alpha, expected_beta=model.beta)


# --------------------------------------------------------------------------- #
# Ligne de commande
# --------------------------------------------------------------------------- #
def save_plot(model: NoiseModel, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"R": "tab:red", "G1": "tab:green", "G2": "tab:olive", "B": "tab:blue"}
    for name, entry in model.details.get("plans", {}).items():
        if "intensite" not in entry:
            continue
        i = np.array(entry["intensite"])
        v = np.array(entry["variance"])
        ax.plot(i, v, "o", color=colors[name], ms=4, label=f"{name} mesuré")
        xs = np.linspace(0, i.max() * 1.05, 50)
        ax.plot(xs, entry["alpha"] * xs + entry["beta"], "-", color=colors[name], lw=1)
    ax.set_xlabel("intensité normalisée")
    ax.set_ylabel("variance temporelle")
    ax.set_title("Courbe de bruit estimée : var = alpha·I + beta")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    from .pipeline import load_stack

    parser = argparse.ArgumentParser(
        prog="python -m drone_burst_sr.noise",
        description="Estime la courbe de bruit (alpha, beta) d'une rafale DNG.")
    parser.add_argument("--input", required=True, type=Path, help="dossier contenant la rafale DNG")
    parser.add_argument("--json", type=Path, help="écrit le résultat en JSON")
    parser.add_argument("--plot", type=Path, help="graphique PNG (nécessite matplotlib)")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args(argv)

    burst = load_stack(args.input, max_frames=args.max_frames)
    ref = burst.infos[0]
    est = estimate_from_stack(burst.stack)
    print(est.describe())
    if not est.reliable:
        print("⚠️  Estimation peu fiable (peu de zones plates ou plage d'intensité étroite).")
    if ref.noise_profile is not None:
        tag = NoiseModel(*expand_profile(*ref.noise_profile), "NoiseProfile DNG")
        print(tag.describe())
        ratio = np.array(est.alpha) / np.maximum(np.array(tag.alpha), 1e-30)
        print("Rapport alpha estimé / tag :", ", ".join(f"{r:.2f}" for r in ratio))
    else:
        print("Pas de tag NoiseProfile dans les DNG.")
    if args.json:
        args.json.write_text(json.dumps({"alpha": est.alpha, "beta": est.beta, "source": est.source,
                                         "fiable": est.reliable, "iso": ref.iso,
                                         "details": est.details}, indent=2, ensure_ascii=False))
        print(f"Écrit : {args.json}")
    if args.plot:
        print(f"Graphique : {args.plot}" if save_plot(est, args.plot)
              else "matplotlib absent : pas de graphique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
