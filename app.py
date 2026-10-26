"""Local demo UI: upload a face -> verdict -> Grad-CAM++ heatmap -> forensic explanation.

    python app.py      # then open http://localhost:7860
"""
import gradio as gr
from PIL import Image

from src.config import CHECKPOINT
from src.eval.metrics import faithfulness_iou, hallucination_flags
from src.explain.corpus import load_corpus
from src.explain.llm import ForensicExplainer
from src.faces import FaceCropper
from src.xai.gradcam import Explainer, draw_bbox

cam = Explainer() if CHECKPOINT.exists() else None
cropper = FaceCropper()
llm = ForensicExplainer() if ForensicExplainer.available() else None
CITATIONS = {d.id: d.citation for d in load_corpus()}


def render(out, exp) -> str:
    if out is None:
        return "_No LLM configured. Set `LLM_PROVIDER` and its API key (see src/explain/providers.py) and restart._"
    if out.error:
        return f"**Error:** {out.error}"
    flags = hallucination_flags(out, exp)
    refs = "\n".join(f"- `[{c}]` {CITATIONS.get(c, 'unknown source')}" for c in out.citations)
    return (f"**Region:** {out.claimed_region}  \n**Artifact:** `{out.artifact_type}`\n\n{out.explanation}\n\n"
            + (f"**Sources**\n{refs}\n\n" if refs else "")
            + f"<sub>Faithfulness IoU {faithfulness_iou(out.claimed_region, exp.heatmap, regions=exp.regions):.2f} · "
              f"hallucination flags: {', '.join(k for k, v in flags.items() if v and k != 'any_hallucination') or 'none'}</sub>")


def analyse(image: Image.Image, compare: bool, crop: bool = True):
    if image is None:
        return None, "", {}, "", ""
    if cam is None:
        raise gr.Error(f"No trained model found at {CHECKPOINT}. Run `python -m src.classifier.train` first.")
    exp = cam(cropper.crop_pil(image) if crop else image)
    verdict = f"## {'🔴 FAKE' if exp.label == 'fake' else '🟢 REAL'}  \nP(fake) = {exp.prob_fake:.3f}"
    regions = {r: round(exp.region_scores[r], 3) for r, _ in exp.top_regions}  # shares, in ranking order
    grounded = render(llm.explain(exp, grounded=True) if llm else None, exp)
    ungrounded = render(llm.explain(exp, grounded=False), exp) if llm and compare else ""
    return Image.fromarray(draw_bbox(exp.overlay, exp.bbox)), verdict, regions, grounded, ungrounded


with gr.Blocks(title="Explainable Deepfake Detector") as demo:
    gr.Markdown("# Explainable Deepfake Detector\nEfficientNet-B0 · Grad-CAM++ · RAG-grounded forensic explanation"
                + (f" · LLM: `{llm.name}`" if llm else ""))
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Face image")
            compare = gr.Checkbox(label="Also show ungrounded explanation (comparison)", value=False)
            crop = gr.Checkbox(label="Detect and crop the face first (use for full video frames)", value=True)
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            heat = gr.Image(label="Grad-CAM++ overlay (box = hottest region)")
            regions = gr.Label(label="Heatmap mass by facial region", num_top_classes=5)
    with gr.Row():
        grounded = gr.Markdown(label="Grounded explanation")
        ungrounded = gr.Markdown(label="Ungrounded explanation")
    btn.click(analyse, [inp, compare, crop], [heat, verdict, regions, grounded, ungrounded])

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed on Kaggle/Colab)")
    args = ap.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=args.share)
