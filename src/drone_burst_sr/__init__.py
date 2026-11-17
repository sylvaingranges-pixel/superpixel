"""drone-burst-sr : super-résolution multi-images pour rafales DNG de drone.

L'algorithme de fusion est celui de Wronski et al. (SIGGRAPH 2019) dans
l'implémentation IPOL de Lafenetre, Facciolo et Eboli (paquet
``handheld-super-resolution``, licence MIT). Ce paquet n'en est qu'un
adaptateur : lecture/contrôle des DNG, modèle de bruit, tuiles, rendu,
interface web et ligne de commande.
"""

__version__ = "0.2.0"
