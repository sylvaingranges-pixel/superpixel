"""Estime la courbe de bruit (alpha, beta) d'une rafale DNG.

Usage : python scripts/estimate_noise.py --input dossier/ [--json bruit.json] [--plot bruit.png]
(équivalent à ``python -m drone_burst_sr.noise``). Méthode : docs/estimation_bruit.md
"""

from drone_burst_sr.noise import main

if __name__ == "__main__":
    raise SystemExit(main())
