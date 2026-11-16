"""Ligne de commande.

Exemples ::

    python -m drone_burst_sr --input rafale/ --scale 2 --out result.tif
    python -m drone_burst_sr --input rafale/ --check          # contrôle seul, sans GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .params import FusionParams


def _float_or_auto(value: str):
    return None if value.lower() == "auto" else float(value)


def build_parser() -> argparse.ArgumentParser:
    d = FusionParams()
    p = argparse.ArgumentParser(
        prog="python -m drone_burst_sr",
        description="Super-résolution multi-images d'une rafale DNG (DJI Mini 4 Pro).")
    p.add_argument("--input", "-i", required=True, type=Path, help="dossier de la rafale (.DNG)")
    p.add_argument("--out", "-o", type=Path, default=Path("result.tif"), help="TIFF 16 bits de sortie")
    p.add_argument("--scale", type=float, default=d.scale, choices=[1.0, 1.5, 2.0])
    p.add_argument("--jpeg", type=Path, help="aperçu JPEG (défaut : même nom que --out en .jpg)")
    p.add_argument("--no-jpeg", action="store_true", help="ne pas écrire d'aperçu JPEG")
    p.add_argument("--single-out", type=Path,
                   help="écrit aussi l'image seule (rawpy, même taille et rendu) en TIFF 16 bits")
    p.add_argument("--check", action="store_true",
                   help="analyse et contrôles seulement (ne nécessite pas de GPU)")
    p.add_argument("--reference", choices=["sharpest", "first"], default=d.reference)
    p.add_argument("--max-frames", type=int)

    g = p.add_argument_group("robustesse")
    g.add_argument("--no-robustness", action="store_true")
    g.add_argument("--no-noise-correction", action="store_true")
    g.add_argument("--t", type=float, default=d.t)
    g.add_argument("--s1", type=float, default=d.s1)
    g.add_argument("--s2", type=float, default=d.s2)
    g.add_argument("--mt", type=float, default=d.mt)

    g = p.add_argument_group("noyaux (« auto » = choisi selon le SNR)")
    g.add_argument("--kernel", choices=["steerable", "iso"], default=d.kernel_type)
    g.add_argument("--k-detail", type=_float_or_auto, default=None)
    g.add_argument("--k-denoise", type=_float_or_auto, default=None)
    g.add_argument("--d-th", type=_float_or_auto, default=None)
    g.add_argument("--d-tr", type=_float_or_auto, default=None)
    g.add_argument("--k-stretch", type=float, default=d.k_stretch)
    g.add_argument("--k-shrink", type=float, default=d.k_shrink)

    g = p.add_argument_group("rendu")
    g.add_argument("--sharpen", type=float, default=d.sharpen_amount, help="0 = sans accentuation")
    g.add_argument("--sharpen-radius", type=float, default=d.sharpen_radius)
    g.add_argument("--ev", type=float, default=d.exposure_ev, help="correction d'exposition (IL)")

    g = p.add_argument_group("bruit")
    g.add_argument("--alpha", type=float, nargs="+", help="1, 3 ou 4 valeurs (R, G1, B, G2)")
    g.add_argument("--beta", type=float, nargs="+")
    g.add_argument("--estimate-noise", action="store_true", help="ignorer le tag NoiseProfile")
    g.add_argument("--lut-trials", type=int, default=d.lut_trials)

    g = p.add_argument_group("mémoire")
    g.add_argument("--tile-size", type=int, default=None,
                   help="côté des tuiles en px (0 = image entière ; défaut : automatique)")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def params_from_args(a: argparse.Namespace) -> FusionParams:
    return FusionParams(
        scale=a.scale, reference=a.reference, robustness=not a.no_robustness,
        noise_correction=not a.no_noise_correction, t=a.t, s1=a.s1, s2=a.s2, mt=a.mt,
        kernel_type=a.kernel, k_detail=a.k_detail, k_denoise=a.k_denoise, d_th=a.d_th,
        d_tr=a.d_tr, k_stretch=a.k_stretch, k_shrink=a.k_shrink, sharpen_amount=a.sharpen,
        sharpen_radius=a.sharpen_radius, exposure_ev=a.ev, noise_alpha=a.alpha, noise_beta=a.beta,
        estimate_noise=a.estimate_noise, lut_trials=a.lut_trials, tile_size=a.tile_size,
        max_frames=a.max_frames,
    ).validate()


def main(argv=None) -> int:
    from .export import save_jpeg, save_tiff16
    from .gpu import detect_gpu
    from .pipeline import BurstError, GpuUnavailable, analyze, run

    args = build_parser().parse_args(argv)
    try:
        params = params_from_args(args)
    except ValueError as error:
        print(f"Paramètre invalide : {error}", file=sys.stderr)
        return 1

    try:
        analysis = analyze(args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    print(analysis.text())
    gpu = detect_gpu()
    print(gpu.describe())
    if args.check:
        return 0 if analysis.ok else 1
    if not analysis.ok:
        return 1
    if not gpu.ready:
        return 2

    last = [-1]

    def progress(frac: float, msg: str) -> None:
        if args.quiet:
            return
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r[{pct:3d} %] {msg[:90]:<90}", end="", flush=True)

    try:
        result = run(args.input, params, progress)
    except BurstError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    except GpuUnavailable as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
    print()
    print(result.report())

    save_tiff16(args.out, result.fused, description=result.report())
    print(f"TIFF 16 bits : {args.out}")
    if not args.no_jpeg:
        jpeg = args.jpeg or args.out.with_suffix(".jpg")
        save_jpeg(jpeg, result.fused)
        print(f"Aperçu JPEG  : {jpeg}")
    if args.single_out:
        save_tiff16(args.single_out, result.single, description="image seule (rawpy)")
        print(f"Image seule  : {args.single_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
