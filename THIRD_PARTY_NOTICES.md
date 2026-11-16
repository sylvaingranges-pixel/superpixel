# Composants tiers

## Handheld Multi-Frame Super-Resolution (implémentation IPOL)

- Dépôt : https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution
- Version utilisée : commit `27c4655ae4752898e0ae6e9737697c4d89c95226` (30/09/2026)
- Mode d'intégration : dépendance pip optionnelle (`pip install ".[gpu]"`). Aucun fichier de ce
  dépôt n'est copié ou modifié ici.
- Licence : MIT

```
MIT License

Copyright (c) 2023 Jamy Lafenetre (jamy.lafenetre@ens-paris-saclay.fr)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Citation

```bibtex
@article{lafenetre23handheld,
  title   = {Implementing Handheld Burst Super-Resolution},
  author  = {Lafenetre, Jamy and Facciolo, Gabriele and Eboli, Thomas},
  journal = {Image Processing On Line},
  year    = {2023},
}

@article{wronski2019handheld,
  title   = {Handheld Multi-Frame Super-Resolution},
  author  = {Wronski, Bartlomiej and Garcia-Dorado, Ignacio and Ernst, Manfred and
             Kelly, Damien and Krainin, Michael and Liang, Chia-Kai and Levoy, Marc and
             Milanfar, Peyman},
  journal = {ACM Transactions on Graphics (SIGGRAPH)},
  volume  = {38},
  number  = {4},
  year    = {2019},
}
```

## Autres dépendances

rawpy (LibRaw), exifread, tifffile, OpenCV, Pillow, NumPy, SciPy, psutil et Gradio sont installés
depuis PyPI selon leurs propres licences (MIT, BSD, Apache 2.0, LGPL pour LibRaw via rawpy).
