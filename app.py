"""Demo UI (phase 6): face image -> verdict -> Grad-CAM++ evidence -> evidence cards -> grounded explanation.

Two pages:
  Live analysis     upload an image; the progress strip shows each step (Detect, Heatmap, Retrieve, Explain,
                    Check) as it finishes. Needs the detector checkpoint and a GROQ_API_KEY.
  Gallery · offline test images analysed in advance by every model (outputs/demo/gallery/, built with
                    `python -m src.demo.pipeline` or notebook 08); works without internet or an API key and
                    reveals each image's real label.

    python app.py [--checkpoint path/to/best.pt] [--share]     # then open http://localhost:7860
"""
import argparse
import os

import gradio as gr

from src.config import CHECKPOINT
from src.demo import render
from src.demo.pipeline import GALLERY_DIR, MODELS, DemoPipeline, gallery_result, load_gallery


LIVE_PROMPT = "Upload a face image or a video frame and press Analyse."
GALLERY_PROMPT = "Pick a test image on the left."


def panel_values(res, waiting_for_llm: bool = False, prompt: str = ""):
    if res is None:
        return (render.verdict(None, prompt), None, None, None, render.regions(None), render.summary(None),
                render.explanation(None), render.cards(None))
    return (render.verdict(res), res.face, res.overlay, res.regions, render.regions(res), render.summary(res),
            render.explanation(res, waiting_for_llm), render.cards(res))


def panels(prompt: str):
    """The result components, in the order of panel_values(). prompt: what the empty verdict panel says."""
    with gr.Row(equal_height=False):
        verdict = gr.HTML(render.verdict(None, prompt))
        heat = gr.HTML(render.regions(None))
    with gr.Row():
        kw = dict(interactive=False, buttons=["fullscreen"])
        face = gr.Image(label="Face", **kw)
        overlay = gr.Image(label="Grad-CAM++ heatmap", **kw)
        regions = gr.Image(label="Region boxes · coloured by heat", **kw)
    summary = gr.HTML(render.summary(None))
    expl = gr.HTML(render.explanation(None))
    cards = gr.HTML(render.cards(None))
    return [verdict, face, overlay, regions, heat, summary, expl, cards]


def build(pipe: DemoPipeline, gallery: list):
    def run_live(image, model):
        if image is None:
            raise gr.Error("Upload a face image or a video frame first.")
        if not pipe.ready:
            raise gr.Error(f"No detector checkpoint at {pipe.checkpoint}: copy EfficientNet-B0's best.pt there "
                           f"or start with --checkpoint. The Gallery page works without it.")
        yield (render.progress(-1, running=True), *panel_values(None, prompt="Analysing…"))
        for done, res, err in pipe.run(image, model):
            running = not err and done < len(render.STEPS) - 1
            yield (render.progress(done, err, res.timings, running), *panel_values(res, waiting_for_llm=running and done == 2))

    def show_cached(index, model):
        if index is None:
            return panel_values(None, prompt=GALLERY_PROMPT)
        return panel_values(gallery_result(gallery[index], model))

    def select_cached(evt: gr.SelectData, model):
        return (evt.index, *show_cached(evt.index, model))

    stored = [m for m in MODELS if any(m in e["answers"] for e in gallery)] or list(MODELS)
    with gr.Blocks(title="Explainable Deepfake Detector") as demo:
        gr.HTML(render.header())
        with gr.Tabs():
            with gr.Tab("Live analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        image = gr.Image(type="pil", label="Face image or video frame", sources=["upload", "clipboard"],
                                         height=300)
                        model = gr.Dropdown([(label, m) for m, label in MODELS.items()], value=next(iter(MODELS)),
                                            label="Explanation model (Groq)")
                        go = gr.Button("Analyse", variant="primary", size="lg")
                        steps = gr.HTML(render.progress())
                    with gr.Column(scale=8):
                        live = panels(LIVE_PROMPT)
                go.click(run_live, [image, model], [steps, *live])
            with gr.Tab("Gallery · offline"):
                if not gallery:
                    gr.HTML(f"<div class='dd-banner'>No gallery yet: build it with notebook 08 or "
                            f"<code>python -m src.demo.pipeline</code> (saved to {GALLERY_DIR}).</div>")
                else:
                    gr.HTML("<div class='dd-banner'>Test images analysed in advance by every model: no internet or "
                            "API key needed. Pick one; its real label is shown with the result.</div>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        thumbs = [(str(GALLERY_DIR / f"{e['key']}_face.png"), f"{e['truth']['dataset']} · {i + 1}")
                                  for i, e in enumerate(gallery)]
                        pick = gr.Gallery(thumbs, label="Test images", columns=3, height=420, allow_preview=False,
                                          object_fit="cover")
                        gmodel = gr.Dropdown([(MODELS[m], m) for m in stored], value=stored[0],
                                             label="Explanation model (stored answers)")
                        chosen = gr.State(None)
                    with gr.Column(scale=8):
                        cached = panels(GALLERY_PROMPT)
                pick.select(select_cached, [gmodel], [chosen, *cached])
                gmodel.change(show_cached, [chosen, gmodel], cached)
        gr.HTML("<div class='dd-muted' style='text-align:center;margin-top:8px'>The explanation describes where the "
                "detector found evidence and what the literature says about such artefacts; it is not proof of "
                "manipulation. The LLM never sees the image.</div>")
    return demo


THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT), help="EfficientNet-B0 best.pt")
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed on Kaggle/Colab)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    build(DemoPipeline(args.checkpoint), load_gallery()).launch(
        server_name="127.0.0.1", server_port=args.port, share=args.share, theme=THEME, css=render.CSS,
        allowed_paths=[str(GALLERY_DIR)])
