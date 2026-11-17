# drone-burst-sr

**Super-résolution et débruitage multi-images pour rafales RAW**, pensé pour le drone
DJI Mini 4 Pro (DNG) et utilisable aussi avec un hybride comme le **Sony α7 IV** (ARW, voir § 4.1).

On prend une rafale d'une dizaine de photos RAW en vol stationnaire. Le léger dérapage du drone
décale chaque image d'une fraction de pixel. En fusionnant ces images, on obtient une image plus
définie et moins bruitée que chacune d'elles.

L'algorithme de fusion est **Handheld Multi-Frame Super-Resolution** (Wronski et al., Google,
SIGGRAPH 2019, celui du Pixel 3), dans l'implémentation publiée sur IPOL en 2023 par
**Jamy Lafenetre, Gabriele Facciolo et Thomas Eboli**
([dépôt](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution), licence MIT).
Ce projet **ne le réécrit pas**. Il l'utilise comme dépendance et l'entoure de ce qui manque
pour un usage drone :

- lecture et contrôle des RAW (DNG, ARW Sony, et autres formats Bayer lus par LibRaw) ;
- modèle de bruit adapté au capteur ;
- traitement par tuiles pour la mémoire ;
- rendu couleur identique avant/après ;
- interface web et ligne de commande.

---

## Sommaire

1. [Ce qu'il faut savoir d'abord](#1-ce-quil-faut-savoir-dabord)
2. [Installation](#2-installation)
3. [Utilisation](#3-utilisation)
4. [Conseils de prise de vue au drone](#4-conseils-de-prise-de-vue-au-drone)
   (et [avec un Sony α7 IV](#41-avec-un-sony-α7-iv-ou-un-autre-hybride))
5. [Limites réalistes](#5-limites-réalistes)
6. [Détails techniques](#6-détails-techniques)
7. [Tests](#7-tests)
8. [Licence et citation](#8-licence-et-citation)

---

## 1. Ce qu'il faut savoir d'abord

- **Il faut un GPU NVIDIA.** L'implémentation amont repose sur Numba CUDA et PyTorch, et crée
  ses données directement sur le GPU (`device="cuda"`). **Il n'existe pas de repli CPU
  raisonnable** : le simulateur CUDA de Numba ne gère pas la partie PyTorch, et il serait de
  toute façon des milliers de fois trop lent.
  - **Sans GPU (par exemple un PC Windows portable)**, utilisez **Google Colab** : GPU T4
    gratuit, notebook fourni, et la même interface s'ouvre dans votre navigateur
    ([§ 2.1](#21-sans-gpu--google-colab-recommandé)).
  - En local sans GPU, l'outil sert quand même à **analyser** une rafale (métadonnées,
    cohérence, mémoire nécessaire) et à **estimer la courbe de bruit**.
- **Photographiez en mode 12 MP.** Le mode 48 MP du Mini 4 Pro est un capteur *Quad Bayer*
  « remosaïqué » en Bayer par le drone : une partie des détails est déjà interpolée et le bruit
  est corrélé. Le gain de la fusion y est bien plus faible. L'interface le rappelle.

## 2. Installation

Python **3.11** (3.12 et 3.13 fonctionnent aussi). Toutes les dépendances sont déclarées dans
`pyproject.toml`.

| Installation | Commande | Contenu |
|---|---|---|
| Légère | `pip install .` | analyse, contrôles, estimation du bruit, interface (sans fusion) |
| Complète | `pip install ".[gpu]"` | + algorithme amont (PyTorch, Numba CUDA) : **GPU NVIDIA requis** |
| Tests | `pip install ".[dev]"` | + pytest |

### 2.1 Sans GPU : Google Colab (recommandé)

1. Ouvrez `notebooks/drone_burst_sr_colab.ipynb` dans Colab (*Fichier → Importer le notebook →
   GitHub*, ou glissez le fichier).
2. *Exécution → Modifier le type d'exécution → **GPU T4***.
3. Si le dépôt est **privé**, deux possibilités :
   - **sans jeton** : *Code → Download ZIP* sur GitHub, déposez le ZIP dans Google Drive et
     indiquez son chemin dans la variable `LOCAL_SOURCE` du notebook ;
   - **avec jeton** : un jeton GitHub en lecture seule dans les *Secrets* Colab, sous le nom
     `GITHUB_TOKEN`.

   La connexion GitHub de Colab ne sert qu'à ouvrir et enregistrer des notebooks. Elle ne
   permet pas à `pip` de lire un dépôt privé.
4. Exécutez les cellules dans l'ordre :
   - installation ;
   - vérification CUDA (bascule automatique en CUDA 12 si le pilote de Colab est ancien) ;
   - auto-test ;
   - interface, ouverte par un lien `https://….gradio.live` dans votre navigateur.
5. Pour les grosses rafales, déposez les DNG sur **Google Drive** et indiquez le dossier dans
   l'interface, ou utilisez la cellule « ligne de commande ». C'est plus rapide que l'envoi
   par le navigateur.

Ce que permet la version gratuite de Colab : GPU T4 (16 Go de VRAM), environ 12 Go de RAM,
sessions limitées dans le temps. C'est suffisant pour 12 MP ×2. Le 48 MP ×2 passe aussi
(environ 9,5 Go de RAM estimés), mais c'est juste.

### 2.2 Windows

```powershell
# Python 3.11 depuis python.org (cocher « Add to PATH »)
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install .                 # PC sans GPU : analyse + estimation du bruit
```

Avec un **GPU NVIDIA** sous Windows :

- L'auteur de l'implémentation amont recommande **WSL2** (Ubuntu), à cause de problèmes connus de
  Numba sous Windows. Suivez alors les instructions Linux dans WSL2.
- En natif, installez d'abord **PyTorch avec CUDA**. Sur Windows, pip installe sinon la version
  CPU. Prenez la commande indiquée sur pytorch.org (*Get Started*), par exemple :
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install ".[gpu]"
  ```
- La variante CUDA 13 de Numba exige un pilote NVIDIA récent (≥ 580). Avec un pilote plus
  ancien : `pip install "numba-cuda[cu12]" "cuda-bindings>=12.9,<13" "cuda-toolkit[cccl,cudart,nvcc,nvrtc]==12.*"`.

### 2.3 Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ".[gpu]"          # ou « pip install . » sans GPU
python -m drone_burst_sr.selftest   # vérifie toute la chaîne GPU (≈ 2 min)
```

Les bibliothèques CUDA sont fournies par les paquets pip ; seul un **pilote NVIDIA** est
nécessaire. `exiftool` est facultatif (`sudo apt install libimage-exiftool-perl`) : il sert
seulement à vérifier les métadonnées.

## 3. Utilisation

### 3.1 Interface web

```bash
drone-burst-sr-ui             # ou : python -m drone_burst_sr.app   (--share pour un lien public)
```

1. **Glissez-déposez** les DNG de la rafale, ou indiquez un dossier côté serveur (Drive dans Colab).
2. **Analyser la rafale** affiche :
   - les métadonnées de chaque image : CFA, niveaux de noir et de blanc, ISO, temps de pose,
     balance des blancs, présence d'un NoiseProfile ;
   - les **avertissements** : ISO, exposition ou balance des blancs différents, mélange de
     tailles, mode 48 MP… ;
   - la **mémoire nécessaire** pour chaque échelle.
3. Choisissez l'**échelle** : ×1, ×1,5 ou ×2. ×2 est proposé par défaut en 12 MP, ×1 en 48 MP.
   Les **paramètres avancés** sont repliés ; les valeurs par défaut sont celles de l'amont.
4. **Lancer la fusion** affiche une barre de progression (lecture, bruit, puis tuile et image en
   cours). La 1re exécution compile les kernels CUDA, ce qui prend environ une minute de plus.
5. Résultat :
   - un **comparateur avant/après** à curseur. L'« avant » est l'image de référence seule,
     dématricée normalement par rawpy, agrandie en Lanczos, avec exactement le même rendu
     couleur ;
   - un **zoom 100 %** : cliquez sur la vignette à l'endroit à examiner ;
   - les **téléchargements** : TIFF 16 bits (sRGB), aperçu JPEG, et l'image seule en TIFF
     pour vos propres comparaisons.

### 3.2 Ligne de commande

```bash
python -m drone_burst_sr --input dossier/ --scale 2 --out result.tif
```

Le TIFF 16 bits et `result.jpg` sont écrits. Options utiles :

| Option | Effet |
|---|---|
| `--check` | analyse et contrôles seulement (sans GPU) |
| `--scale {1,1.5,2}` | facteur d'échelle |
| `--single-out seule.tif` | écrit aussi l'image seule rawpy (même taille, même rendu) |
| `--reference {sharpest,first}` | image de référence : la plus nette (défaut) ou la première |
| `--t --s1 --s2 --mt` | robustesse (voir § 6.4) ; `--no-robustness` pour la désactiver |
| `--kernel {steerable,iso}`, `--k-detail`, `--k-denoise`, `--d-th`, `--d-tr` | noyaux (`auto` = selon le SNR) |
| `--k-stretch --k-shrink` | forme des noyaux le long et en travers des contours |
| `--sharpen 1.5 --sharpen-radius 3 --ev 0` | rendu (accentuation, exposition) |
| `--alpha … --beta …`, `--estimate-noise` | modèle de bruit (voir § 6.2) |
| `--tile-size N` | côté des tuiles en px (0 = image entière ; défaut : selon la VRAM libre) |

Codes de sortie : 0 = succès, 1 = rafale ou paramètres invalides, 2 = pas de GPU utilisable.

### 3.3 Autres commandes

```bash
python -m drone_burst_sr.noise --input dossier/ --json bruit.json --plot bruit.png   # sans GPU
python -m drone_burst_sr.selftest                                                    # GPU
exiftool -a -G1 -CFAPattern2 -BlackLevel -WhiteLevel -AsShotNeutral -ColorMatrix1 -NoiseProfile DJI_0001.DNG
```

## 4. Conseils de prise de vue au drone

- **Mode photo 12 MP, format RAW (DNG)**, en rafale (AEB désactivé) ou en déclenchements
  rapprochés. 8 à 15 images suffisent ; au-delà le gain est marginal.
- **Exposition manuelle (M)** : ISO, vitesse et **balance des blancs fixes**. L'outil signale
  toute variation, et une image plus claire ou plus sombre est en grande partie rejetée.
- **ISO le plus bas possible**, et une vitesse assez rapide pour éviter le flou de bougé :
  1/500 s ou plus vite si le vent secoue le drone. Une image floue n'apporte rien à la
  super-résolution. L'outil choisit par défaut **la plus nette** comme référence.
- **Vol stationnaire**, idéalement par vent faible. Le petit dérapage est **souhaitable** : c'est
  lui qui apporte les décalages sous-pixel. Inutile de bouger volontairement : quelques pixels
  suffisent, les grands déplacements (> 30–50 px) gênent l'alignement.
- **Scène statique** : voitures, piétons, eau agitée, arbres au vent et ombres de nuages
  mobiles sont détectés et rejetés par la robustesse. Dans ces zones on revient au niveau
  d'une seule image.
- **Parallaxe** : le drone ne fait pas que tourner, il **se déplace**. Les sujets proches
  (bâtiment près du drone, arbres en premier plan) bougent alors par rapport au fond et
  l'alignement local peut échouer. Volez assez **haut et loin** du sujet, et méfiez-vous des
  scènes à fort relief proche.
- Évitez le **zoom numérique** (déjà interpolé) et gardez la même focale.

### 4.1 Avec un Sony α7 IV (ou un autre hybride)

Les fichiers **ARW** sont acceptés directement, sans conversion en DNG. Les autres RAW Bayer lus
par LibRaw (NEF, CR2, ORF, RW2, PEF) aussi. Le X-Trans de Fujifilm (RAF) n'est pas un capteur
Bayer : il est refusé.

Le principe est celui du Pixel de Google : **à main levée**, le tremblement naturel entre les
images d'une rafale fournit les décalages sous-pixel.

- **Rafale à main levée**, mode continu rapide, 8 à 15 images. La stabilisation (SteadyShot) peut
  rester active : il reste toujours assez de micro-décalages. **Sur trépied**, les images sont
  identiques au pixel près : la fusion débruite mais n'ajoute pas de détails.
- **Exposition manuelle**, ISO fixe, **balance des blancs fixe**. Vitesse d'au moins
  1/(2 × focale) pour éviter le flou de bougé. L'outil prend la plus nette comme référence.
- **Format RAW** : *non compressé* ou *compressé sans perte*, en **taille L**. Évitez le RAW
  *compressé* (avec perte), dont les artefacts près des contours contrastés gênent l'alignement.
  Les tailles M/S, si le boîtier les propose, ne sont pas des données Bayer brutes : refusées.
- **Couleurs** : pas de matrice DNG dans un ARW. On utilise la matrice de l'α7 IV intégrée à
  LibRaw.
- **Bruit** : un ARW n'a pas de tag NoiseProfile. La courbe est **estimée sur la rafale**
  automatiquement (§ 6.2). Vous pouvez aussi l'estimer une fois par ISO
  (`python -m drone_burst_sr.noise`) et la réutiliser avec `--alpha/--beta`.
- **Gain attendu** : l'α7 IV a un **filtre passe-bas optique**, qui atténue déjà les fréquences
  que la super-résolution cherche à reconstruire. Le gain en résolution sera plus modeste qu'avec
  un capteur sans filtre (α7R, la plupart des drones). Le débruitage (≈ √N) reste entier, et
  c'est souvent l'intérêt principal en basse lumière.
- **Mémoire** (33 MP, 10 images, estimations) : ×1 ≈ 4,3 Go de RAM et 3,6 Go de VRAM ;
  ×2 (131 MP) ≈ 6,9 Go de RAM et 6,4 Go de VRAM. Ça passe sur le T4 de Colab.

## 5. Limites réalistes

- **Résolution** : le gain réel en détails est d'environ **×1,5 à ×2 en linéaire** dans le
  meilleur cas : scène statique, mode 12 MP, bonne mise au point, 10 images ou plus. Une sortie
  « ×2 » n'est pas toujours deux fois plus détaillée ; l'optique du Mini 4 Pro limite aussi.
- **Bruit** : la réduction est d'environ **√N** dans les zones lisses et bien alignées, soit à peu
  près ×3 pour 10 images (environ 1,5 IL). Elle est moindre dans les zones rejetées (mouvement,
  parallaxe) et en ×2, où chaque pixel de sortie reçoit moins d'échantillons.
- **Mode 48 MP** : gain faible (Quad Bayer remosaïqué). Utilisez-le en ×1 comme débruitage.
- **Couleur** : rendu simple (balance des blancs « tel que pris », matrice DNG, sRGB, sans courbe
  de tons). Le TIFF 16 bits est fait pour être retravaillé (Lightroom, darktable…).
  L'orientation EXIF est appliquée.
- **Mémoire** : voir § 6.3. Les grandes sorties pèsent lourd : TIFF 48 MP ×2 ≈ 1,2 Go.
- **Validation sur GPU** : la chaîne complète a été testée sans GPU, avec un moteur factice. Le
  test GPU de bout en bout (`selftest`) est fourni mais **n'a pas pu être exécuté par l'auteur
  de ce dépôt**, faute de GPU : lancez-le en premier dans Colab.

## 6. Détails techniques

### 6.1 Intégration de l'algorithme : pip depuis git plutôt qu'un submodule

- **Licence vérifiée : MIT**. Utilisation et redistribution libres, à condition de garder la
  mention de copyright (voir `THIRD_PARTY_NOTICES.md`).
- L'amont est un **vrai paquet Python** (pyproject, hatchling) : pip installe aussi ses
  dépendances (PyTorch, numba-cuda), ce qu'un submodule ne fait pas.
- La version est **figée sur un commit** (`27c4655`, 30/09/2026) dans l'extra `[gpu]` : on
  installe toujours le même code. On ne monte de version que volontairement, tests à l'appui.
  L'amont prévient lui-même que son code évolue beaucoup.
- Pas de `git clone --recursive` ni de `sys.path` à gérer, ce qui est fragile sous Windows. Et la
  partie GPU reste **optionnelle**, ce qui permet l'installation légère.
- Un submodule ne serait utile que pour **modifier** l'amont, ce dont on n'a pas besoin.

`engine.py` est un court adaptateur. Il construit la `Config` amont, reproduit
la courte préparation de `super_resolution.process` (profil de bruit, table de correction, SNR),
puis appelle `super_resolution.main` tuile par tuile. On n'appelle pas `process()` directement
pour trois raisons :

- il **exige** un tag NoiseProfile ;
- il ne trouve pas les fichiers `.DNG` en majuscules écrits par DJI ;
- il traite toujours l'image entière.

La progression image par image s'obtient en enveloppant temporairement la fonction `merge`
amont, sans modifier son code.

### 6.2 DNG du Mini 4 Pro et courbe de bruit

Lecture avec **rawpy** (LibRaw) et **exifread**, plus **exiftool** en contrôle croisé s'il est
installé :

- **motif CFA**, déduit des couleurs réelles du coin haut-gauche de l'image visible (rejet des
  DNG non Bayer ou non périodiques 2×2). Tous les motifs Bayer sont ramenés en RGGB par
  retournement, comme dans l'amont ;
- **niveaux de noir** par canal et **niveau de blanc**, puis normalisation
  `(x − noir)/(blanc − noir)` ;
- **balance des blancs** (AsShotNeutral) et **matrice couleur** (ColorMatrix1/2 : celle dont
  l'illuminant est le plus proche de D65, comme l'amont) ;
- **ISO, temps de pose, ouverture, correction d'exposition** pour les contrôles de cohérence ;
- **mode 12 / 48 MP**, d'après la taille de l'image.

Le modèle de bruit `var = alpha·x + beta` est pris, dans l'ordre :

1. dans les valeurs fournies ;
2. dans le tag **`NoiseProfile`** ;
3. dans une **estimation sur la rafale** (variance temporelle dans les zones plates), documentée
   dans [`docs/estimation_bruit.md`](docs/estimation_bruit.md) ;
4. en dernier recours, dans le profil générique.

La table de correction Monte Carlo de l'amont est générée pour ce profil (≈ 10 s sur 4 cœurs)
et mise en cache dans `~/.cache/drone_burst_sr` ou `%LOCALAPPDATA%\drone_burst_sr`.

> Je n'avais pas de DNG réel du Mini 4 Pro pour développer : la lecture a été validée sur des DNG
> synthétiques (4 motifs CFA). Si vos fichiers posent problème, la commande `exiftool` du § 3.3
> donne les informations utiles.

### 6.3 Mémoire et tuiles

Estimations calculées par `tiling.estimate_memory` pour 10 images, à confirmer par la mesure :

| Mode | Échelle | Sortie | RAM | VRAM sans tuiles |
|---|---|---|---|---|
| 12 MP | ×1 | 12 MP | ≈ 2,5 Go | ≈ 1,9 Go |
| 12 MP | ×2 | 49 MP | ≈ 3,5 Go | ≈ 2,9 Go |
| 48 MP | ×1 | 49 MP | ≈ 5,7 Go | ≈ 5,0 Go |
| 48 MP | ×1,5 | 110 MP | ≈ 7,3 Go | ≈ 6,7 Go |
| 48 MP | ×2 | 195 MP | ≈ 9,5 Go | ≈ 9,1 Go |

La VRAM est dominée par les accumulateurs de sortie (2 × RVB float32 à la taille de sortie).

**Découpage en tuiles.** Si la VRAM libre ne suffit pas, l'image est découpée automatiquement en
tuiles carrées :

- même découpe dans toutes les images, coordonnées **paires** pour garder la phase du CFA ;
- côté d'au moins 1024 px (contrainte de la pyramide d'alignement) ;
- recouvrement de 192 px.

Chaque tuile est fusionnée indépendamment, puis les tuiles sont recombinées avec un fondu
linéaire, en ignorant les 48 px de bord de chaque tuile. Le SNR est estimé une seule fois sur
l'image entière et imposé à toutes les tuiles, pour que les réglages restent cohérents.

En cas de manque de mémoire GPU, le traitement recommence automatiquement avec des tuiles plus
petites. Un test vérifie que la recombinaison par tuiles donne exactement le même résultat que
le traitement de l'image entière.

### 6.4 Paramètres

Les valeurs par défaut sont celles de l'amont.

| Paramètre | Défaut | Rôle |
|---|---|---|
| robustesse `t` | 0,12 | retranché à la robustesse R : plus grand = rejet plus strict (moins de fantômes, moins de débruitage) |
| `s1` | 2 | gain de R là où le flux optique varie localement (mouvement) |
| `s2` | 12 | gain de R dans les zones statiques |
| `Mt` | 0,8 px | variation locale du flux au-delà de laquelle on applique `s1` |
| noyau | steerable | noyaux orientés selon les contours (`iso` = isotropes) |
| `k_detail`, `k_denoise`, `D_th`, `D_tr` | auto | choisis par l'amont selon le SNR mesuré |
| `k_stretch` / `k_shrink` | 4 / 2 | étirement le long / rétrécissement en travers des contours |
| accentuation | 1,5 (rayon 3) | masque flou, appliqué à l'identique aux deux images |

Avec `R = clamp(S·exp(−d²/σ²) − t, 0, 1)`, où S vaut `s1` ou `s2`.

En cas de **fantômes** (sujets en mouvement dédoublés) : augmentez `t` ou baissez `s1`.
Si le résultat est **trop bruité** dans des zones pourtant statiques : baissez `t`.

### 6.5 Structure du dépôt

```
src/drone_burst_sr/
  dng_io.py      lecture DNG et métadonnées
  validation.py  contrôles de cohérence de la rafale
  noise.py       modèle de bruit, estimation, table amont (+ CLI)
  tiling.py      estimation mémoire, tuiles, fondu
  engine.py      adaptateur vers handheld_super_resolution (GPU)
  pipeline.py    chaîne complète
  render.py      rendu couleur commun
  baseline.py    image seule rawpy
  export.py      TIFF 16 bits / JPEG / aperçus
  gpu.py         détection GPU
  app.py         interface Gradio
  cli.py         ligne de commande
  synthetic.py   rafales synthétiques (tests, auto-test)
  selftest.py    auto-test GPU de bout en bout
notebooks/drone_burst_sr_colab.ipynb
docs/estimation_bruit.md
scripts/estimate_noise.py
tests/
```

## 7. Tests

```bash
pip install ".[dev]"
pytest -q            # ~15 s, sans GPU ; le test GPU est ignoré automatiquement
pytest -m gpu        # sur une machine avec GPU (ou dans Colab)
```

Ce qui est couvert :

- lecture de DNG synthétiques (4 motifs CFA, noir/blanc, balance des blancs, ISO, NoiseProfile,
  dimensions impaires) ;
- échelle de l'image rawpy identique à celle de la fusion ;
- contrôles de cohérence de la rafale ;
- estimation du bruit sur des rafales à bruit connu ;
- tuiles : équivalence avec le traitement de l'image entière ;
- ligne de commande, rendu et orientation ;
- chaîne complète avec un moteur factice CPU.

Le **test GPU** (`python -m drone_burst_sr.selftest`) génère une rafale synthétique à décalages
sous-pixel connus et la compare, **zone par zone**, à une seule image dématricée par rawpy puis
agrandie. Il vérifie trois choses :

- les lignes plus fines que la limite de Nyquist d'une image seule sont mieux restituées
  (PSNR et part du détail fin retrouvée) ;
- le bruit des aplats baisse ;
- les détails déjà résolus ne se dégradent pas.

Il refait aussi la fusion sans robustesse, pour repérer un rejet excessif, et écrit une planche
de comparaison `selftest_comparaison.png`. Le rapport affiche le **taux de fusion** : la part des
autres images réellement utilisée. La CI GitHub Actions lance les
tests CPU sous Ubuntu et Windows.

## 8. Licence et citation

Code de ce dépôt : MIT (voir `LICENSE`). L'algorithme vient du paquet `handheld-super-resolution`,
MIT, © 2023 Jamy Lafenetre (voir `THIRD_PARTY_NOTICES.md`). Si ce travail vous sert, citez :

- J. Lafenetre, G. Facciolo, T. Eboli, *Implementing Handheld Burst Super-Resolution*,
  Image Processing On Line, 2023.
- B. Wronski et al., *Handheld Multi-Frame Super-Resolution*, ACM TOG (SIGGRAPH), 2019.
