"""Interface web locale (Gradio).

Lancement : ``drone-burst-sr-ui`` ou ``python -m drone_burst_sr.app [--share] [--port 7860]``.
"""

from __future__ import annotations

import argparse
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import numpy as np

from .export import crop, preview, save_jpeg, save_tiff16, to_uint8
from .gpu import detect_gpu, in_colab
from .params import FusionParams
from .pipeline import BurstError, GpuUnavailable, analyze, fuse_burst, load_stack, resolve_paths
from .validation import QUAD_BAYER_NOTE

PREVIEW_SIDE = 2048
_RESULTS: "OrderedDict[str, object]" = OrderedDict()  # résultats pleine résolution (côté serveur)
_MAX_RESULTS = 2

SCALES = {"1x": 1.0, "1.5x": 1.5, "2x": 2.0}

INTRO = """
# drone-burst-sr — super-résolution de rafales DNG

Fusionne une rafale de ~10 photos RAW prises en vol stationnaire (DJI Mini 4 Pro) avec
l'algorithme *Handheld Multi-Frame Super-Resolution* (Wronski et al. 2019), implémentation
IPOL de Lafenetre, Facciolo & Eboli.

> **Conseil** : le mode **12 MP** (vrai Bayer) donne le meilleur gain. En 48 MP, le capteur
> Quad Bayer est remosaïqué par le drone : détails en partie interpolés, gain faible.
"""


def _store(result) -> str:
    key = uuid.uuid4().hex
    _RESULTS[key] = result
    while len(_RESULTS) > _MAX_RESULTS:
        _RESULTS.popitem(last=False)
    return key


def _paths(files, folder: str) -> list[Path]:
    if folder and folder.strip():
        return resolve_paths(Path(folder.strip()).expanduser())
    if not files:
        raise gr.Error("Déposez les fichiers DNG de la rafale (ou indiquez un dossier).")
    return resolve_paths([getattr(f, "name", f) for f in files])


def on_analyze(files, folder):
    try:
        analysis = analyze(_paths(files, folder))
    except (FileNotFoundError, ValueError) as error:
        raise gr.Error(str(error)) from error
    table = analysis.table()
    headers = list(table[0].keys())
    rows = [[r[h] for h in headers] for r in table]
    text = analysis.text().replace("\n", "  \n")
    scale = "1x" if analysis.infos[0].is_quad_bayer_mode else "2x"
    return text, gr.Dataframe(value=rows, headers=headers), gr.Radio(value=scale)


def _make_on_run(engine_factory):
    def handler(*args, progress=gr.Progress()):
        return on_run(*args, progress=progress, engine_factory=engine_factory)
    return handler


def on_run(files, folder, scale, reference, robustness, noise_correction, t, s1, s2, mt,
           kernel_type, auto_kernels, k_detail, k_denoise, d_th, d_tr, k_stretch, k_shrink,
           sharpen, radius, ev, estimate_noise, tile_size, progress=gr.Progress(),
           engine_factory=None):
    params = FusionParams(
        scale=SCALES[scale], reference=reference, robustness=robustness,
        noise_correction=noise_correction, t=t, s1=s1, s2=s2, mt=mt, kernel_type=kernel_type,
        k_detail=None if auto_kernels else k_detail, k_denoise=None if auto_kernels else k_denoise,
        d_th=None if auto_kernels else d_th, d_tr=None if auto_kernels else d_tr,
        k_stretch=k_stretch, k_shrink=k_shrink, sharpen_amount=sharpen, sharpen_radius=radius,
        exposure_ev=ev, estimate_noise=estimate_noise,
        tile_size=None if not tile_size else int(tile_size),
    )
    try:
        params.validate()
        engine, gpu = None, None
        if engine_factory is not None:  # tests : moteur injecté, pas de GPU
            engine = engine_factory(params)
        else:
            gpu = detect_gpu()
            if not gpu.ready:
                raise GpuUnavailable(gpu.message)
        report = lambda f, m: progress(f, desc=m)  # noqa: E731
        burst = load_stack(_paths(files, folder), params.reference, params.max_frames, report)
        result = fuse_burst(burst, params, report, engine=engine, gpu=gpu)
    except (GpuUnavailable, BurstError, ValueError, FileNotFoundError) as error:
        raise gr.Error(str(error), duration=None) from error

    progress(1.0, desc="Écriture des fichiers…")
    out_dir = Path(tempfile.mkdtemp(prefix="drone_burst_sr_"))
    tag = f"x{params.scale:g}"
    tif = save_tiff16(out_dir / f"fusion_{tag}.tif", result.fused, description=result.report())
    jpg = save_jpeg(out_dir / f"fusion_{tag}.jpg", result.fused)
    single = save_tiff16(out_dir / f"image_seule_{tag}.tif", result.single,
                         description="image seule (rawpy)")
    key = _store(result)
    before, after = preview(result.single, PREVIEW_SIDE), preview(result.fused, PREVIEW_SIDE)
    zoom = _zoom(result, 0.5, 0.5, 512)
    report_md = result.report().replace("\n", "  \n")
    return (report_md, (before, after), after, zoom, [str(tif), str(jpg), str(single)], key)


def _zoom(result, cx: float, cy: float, size: int):
    return to_uint8(crop(result.single, cx, cy, size)), to_uint8(crop(result.fused, cx, cy, size))


def on_select(key, size, evt: gr.SelectData):
    result = _RESULTS.get(key)
    if result is None:
        raise gr.Error("Lancez d'abord un traitement.")
    h, w = result.fused.shape[:2]
    f = min(1.0, PREVIEW_SIDE / max(h, w))
    x, y = evt.index[0], evt.index[1]
    return _zoom(result, (x + 0.5) / (w * f), (y + 0.5) / (h * f), int(size))


def build_app(engine_factory=None) -> gr.Blocks:
    """``engine_factory(params)`` ne sert qu'aux tests (moteur factice sans GPU)."""
    d = FusionParams()
    gpu = detect_gpu()
    with gr.Blocks(title="drone-burst-sr") as demo:
        gr.Markdown(INTRO)
        gr.Markdown(f"**{gpu.describe()}**")
        result_key = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Rafale DNG (glisser-déposer plusieurs fichiers)",
                                file_count="multiple", file_types=[".dng", ".DNG"], type="filepath")
                folder = gr.Textbox(label="… ou dossier sur le serveur (ex. Google Drive monté dans Colab)",
                                    placeholder="/content/drive/MyDrive/rafale")
                analyze_btn = gr.Button("1. Analyser la rafale")
                scale = gr.Radio(list(SCALES), value="2x", label="Facteur d'échelle")
                with gr.Accordion("Paramètres avancés", open=False):
                    gr.Markdown("**Robustesse** — rejet des zones mal alignées (sujets mobiles, parallaxe).")
                    robustness = gr.Checkbox(value=d.robustness, label="Robustesse activée")
                    noise_corr = gr.Checkbox(value=d.noise_correction, label="Correction par le modèle de bruit")
                    t = gr.Slider(0.0, 0.5, value=d.t, step=0.01, label="t (plus grand = rejet plus strict)")
                    s1 = gr.Slider(0.5, 10, value=d.s1, step=0.5, label="s1 (gain en zone de mouvement)")
                    s2 = gr.Slider(1, 30, value=d.s2, step=0.5, label="s2 (gain en zone statique)")
                    mt = gr.Slider(0.1, 3, value=d.mt, step=0.05, label="Mt (seuil de mouvement, px)")
                    gr.Markdown("**Noyaux de fusion**")
                    kernel_type = gr.Radio(["steerable", "iso"], value=d.kernel_type,
                                           label="Type (steerable = orienté selon les contours)")
                    auto_k = gr.Checkbox(value=True, label="Noyaux automatiques (selon le SNR mesuré)")
                    k_detail = gr.Slider(0.1, 0.8, value=0.29, step=0.01, interactive=False,
                                         label="k_detail (petit = plus net, plus bruité)")
                    k_denoise = gr.Slider(1, 8, value=4.0, step=0.1, interactive=False,
                                          label="k_denoise (zones lisses : plus grand = plus lissé)")
                    d_th = gr.Slider(0.3, 1.5, value=0.76, step=0.01, interactive=False, label="D_th")
                    d_tr = gr.Slider(0.5, 3, value=1.12, step=0.01, interactive=False, label="D_tr")
                    k_stretch = gr.Slider(1, 8, value=d.k_stretch, step=0.1, label="k_stretch (le long des contours)")
                    k_shrink = gr.Slider(1, 4, value=d.k_shrink, step=0.1, label="k_shrink (en travers des contours)")
                    gr.Markdown("**Rendu** (appliqué à l'identique aux deux images)")
                    sharpen = gr.Slider(0, 3, value=d.sharpen_amount, step=0.1, label="Accentuation (0 = aucune)")
                    radius = gr.Slider(0.5, 6, value=d.sharpen_radius, step=0.5, label="Rayon d'accentuation")
                    ev = gr.Slider(-2, 2, value=0, step=0.1, label="Exposition (IL)")
                    gr.Markdown("**Divers**")
                    reference = gr.Radio([("La plus nette", "sharpest"), ("La première", "first")],
                                         value=d.reference, label="Image de référence")
                    estimate_noise = gr.Checkbox(value=False, label="Ignorer le NoiseProfile DNG et estimer le bruit sur la rafale")
                    tile_size = gr.Number(value=0, precision=0, label="Taille des tuiles (0 = automatique selon la VRAM)")
                run_btn = gr.Button("2. Lancer la fusion", variant="primary")

            with gr.Column(scale=2):
                analysis_md = gr.Markdown("Déposez une rafale puis cliquez sur « Analyser ».")
                meta = gr.Dataframe(label="Métadonnées", interactive=False, wrap=True)
                report_md = gr.Markdown()
                slider = gr.ImageSlider(label="Avant (image seule, rawpy) ↔ Après (fusion)",
                                        type="numpy", max_height=800)
                with gr.Row():
                    nav = gr.Image(label="Cliquez sur la zone à examiner à 100 %", type="numpy",
                                   interactive=False, height=320)
                    zoom_size = gr.Dropdown([256, 384, 512, 768], value=512, label="Taille du zoom (px)")
                zoom = gr.ImageSlider(label="Zoom 100 % : avant ↔ après", type="numpy", max_height=800)
                downloads = gr.File(label="Téléchargements (TIFF 16 bits, aperçu JPEG, image seule)",
                                    file_count="multiple", interactive=False)
                gr.Markdown("<small>" + QUAD_BAYER_NOTE + "</small>")

        auto_k.change(lambda a: [gr.Slider(interactive=not a)] * 4, auto_k, [k_detail, k_denoise, d_th, d_tr])
        analyze_btn.click(on_analyze, [files, folder], [analysis_md, meta, scale])
        run_btn.click(
            _make_on_run(engine_factory),
            [files, folder, scale, reference, robustness, noise_corr, t, s1, s2, mt, kernel_type,
             auto_k, k_detail, k_denoise, d_th, d_tr, k_stretch, k_shrink, sharpen, radius, ev,
             estimate_noise, tile_size],
            [report_md, slider, nav, zoom, downloads, result_key],
        )
        nav.select(on_select, [result_key, zoom_size], zoom)
    return demo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="drone-burst-sr-ui", description="Interface web drone-burst-sr")
    p.add_argument("--share", action="store_true", help="lien public temporaire (utile dans Colab)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)
    demo = build_app()
    demo.queue(default_concurrency_limit=1)  # un seul traitement GPU à la fois
    demo.launch(server_name=args.host, server_port=args.port, share=args.share or in_colab(),
                theme=gr.themes.Soft())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
