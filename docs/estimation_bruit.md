# Estimation de la courbe de bruit (alpha, beta)

L'algorithme utilise un modèle de bruit capteur de type Poisson-Gauss, par plan CFA :

```
var(x) = alpha · x + beta        (x = valeur normalisée : noir -> 0, blanc -> 1)
```

- **alpha** : bruit photonique (proportionnel au signal, augmente avec l'ISO) ;
- **beta** : bruit de lecture (indépendant du signal, ∝ ISO²).

Il sert à trois endroits de l'implémentation amont :

1. la transformée d'Anscombe généralisée, avant le calcul des gradients des noyaux ;
2. le SNR global, qui choisit la taille des blocs d'alignement et les valeurs « auto » des noyaux ;
3. la table de correction Monte Carlo de la robustesse (σ² et d² attendus du seul bruit).

Le dépôt amont fournit les coefficients du Pixel 4. Ils ne conviennent pas à un capteur de drone.

## Ordre de priorité dans drone-burst-sr

1. **Valeurs fournies** : options `--alpha` et `--beta` (1, 3 ou 4 valeurs : R, G1, B, G2).
2. **Tag DNG `NoiseProfile` (0xC761)**, s'il est présent. Le format DNG le définit exactement
   ainsi : paires (S, O) avec `var = S·x + O`, dans l'ordre des plans de couleur.
   Pour savoir si vos fichiers l'ont :
   `exiftool -NoiseProfile DJI_0001.DNG`
3. **Estimation sur la rafale** : méthode ci-dessous. Elle sert aussi si l'on coche
   « Ignorer le NoiseProfile » ou si l'on passe `--estimate-noise`.
4. **Dernier recours** : le profil du Pixel 4 mis à l'échelle de l'ISO, avec un avertissement.

## Méthode d'estimation à partir de la rafale

Script : `python -m drone_burst_sr.noise --input rafale/ --json bruit.json --plot bruit.png`
(ou `python scripts/estimate_noise.py …`). Ce script fonctionne **sans GPU**.

Idée : une rafale en vol stationnaire contient N mesures de la même scène. Dans une **zone plate**,
le petit déplacement du drone ne change pas la valeur vraie des pixels. La **variance temporelle**
d'un pixel à travers les N images y mesure donc directement le bruit, sans être faussée par
la texture, contrairement aux méthodes sur une seule image.

Étapes, pour chacun des 4 plans CFA (R, G1, G2, B), sur 10 images au plus :

1. Pour chaque pixel, on calcule la moyenne temporelle `m`, la variance temporelle `v`
   (ddof = 1), ainsi que le minimum et le maximum.
2. On découpe le plan en blocs de 16×16 pixels. On y ajuste un plan `m ≈ a + bx·x + by·y`
   pour absorber les dégradés lents (ciel, eau calme…).
3. **Test de platitude.** Un bloc est retenu si :
   - la variance spatiale résiduelle de `m` après ajustement reste inférieure à 1,5 fois
     celle qu'on attend du seul bruit, c'est-à-dire `var/N` ;
   - le dégradé ajusté, multiplié par un déplacement de 3 px, n'ajoute pas plus de 10 % à la
     variance (sinon le mouvement du drone contaminerait la mesure) ;
   - il n'y a pas d'écrêtage : rien au-dessus de 0,97, et pas de pixels collés au plancher.
4. **Mesure robuste.** Dans chaque bloc retenu, on prend la **médiane** des variances
   pixel à pixel, mesurée seulement à l'intérieur du bloc (la marge de 2 px ne sert qu'au test
   de platitude). On corrige ensuite le biais de la médiane d'une variance empirique :
   on divise par `médiane(χ²ₖ)/k`, avec k = N−1. La médiane neutralise les quelques pixels
   contaminés par un contour voisin qui « entre » dans le bloc quand le drone bouge.
5. On regroupe les blocs en au plus 24 classes d'intensité de même effectif, et on prend
   la médiane de chaque classe.
6. **Ajustement** de `v = alpha·I + beta` par moindres carrés relatifs (poids 1/v²), avec
   2 passes de rejet des classes aberrantes (écart relatif > 3 MAD) et la contrainte
   alpha, beta ≥ 0.

L'estimation est signalée **peu fiable** s'il y a moins de 30 blocs plats, moins de 5 classes,
ou une plage d'intensité inférieure à 0,05. Dans ce cas, alpha et beta sont mal séparés.

## Précision mesurée (tests sur des rafales synthétiques)

`tests/test_noise.py` génère des rafales à bruit Poisson-Gauss connu, avec des décalages
sous-pixel aléatoires de ±2 px :

- **alpha** est retrouvé à environ 5–10 % près ;
- **beta**, beaucoup plus petit, est moins précis (il dépend des zones sombres disponibles).
  Mais la **variance prédite** reste juste à mieux que 20 % sur toute la plage utile
  (0,1 à 0,8), et c'est cette variance qu'utilise l'algorithme.

## Conseils

- Une scène avec des **aplats à plusieurs niveaux de luminosité** (ciel, route, toits, ombres)
  donne la meilleure estimation. Une scène très texturée (forêt, herbe) en donne peu.
- Le bruit dépend de l'ISO : estimez-le **une fois par ISO** sur une bonne rafale, notez
  alpha et beta (`--json`), puis réutilisez-les avec `--alpha/--beta` sur les rafales difficiles.
- Si le DNG contient un `NoiseProfile`, comparez-le à l'estimation : le script affiche le rapport
  entre les deux. Un écart de plus de 30 % justifie d'utiliser `--estimate-noise`.
- L'eau agitée, les feuillages au vent ou les nuages qui bougent gonflent la variance temporelle.
  La médiane et le test de platitude en éliminent l'essentiel, mais pas tout.
