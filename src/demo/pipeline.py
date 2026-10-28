"""Demo pipeline (phase 6): one face image -> verdict, heatmap, evidence summary, evidence cards and a grounded
explanation, run step by step so the UI can show progress.

The same code precomputes the gallery: a few test images analysed in advance by every model and stored under
outputs/demo/gallery/, so the gallery page works without internet or an API key.

    python -m src.demo.pipeline --data ffpp=data/ffpp celebdf=data/celebdf    # build the gallery
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import CHECKPOINT, IMG_SIZE, OUTPUT_DIR
from src.eval.verify import grounding_flags
from src.explain.corpus import cards_by_id
from src.explain.llm import ForensicExplainer, LLMExplanation
from src.xai.gradcam import BACKGROUND, confidence_level

MODELS = {"openai/gpt-oss-120b": "GPT-OSS-120B", "openai/gpt-oss-20b": "GPT-OSS-20B",
          "qwen/qwen3.8-27b": "Qwen3.8-27B (slower)"}
MAX_TOKENS = {"qwen/qwen3.8-27b": 1000}   # Groq caps Qwen at 1K output tokens per minute
STEPS = ("Detect", "Heatmap", "Retrieve", "Explain", "Check")
SUMMARY_REGIONS = None                     # the LLM reads every region (phases 3-5 list only the top 3)
GALLERY_DIR = OUTPUT_DIR / "demo" / "gallery"
DISPLAY = 448                              # side of the images shown in the UI (the detector sees 224)
TOP_COLORS = ((229, 72, 77), (245, 165, 36), (59, 130, 246))   # top-3 regions: red, amber, blue
METHOD_NAMES = {"Deepfakes": "Deepfakes · face swap", "FaceSwap": "FaceSwap · face swap",
                "FaceShifter": "FaceShifter · face swap", "Face2Face": "Face2Face · re-enactment",
                "NeuralTextures": "NeuralTextures · re-enactment", "CelebDF": "Celeb-DF · face swap",
                "original": "original video"}


@dataclass
class DemoResult:
    face: np.ndarray = None          # what the detector saw, DISPLAY x DISPLAY RGB
    overlay: np.ndarray = None       # Grad-CAM++ overlay with the hottest-region box
    regions: np.ndarray = None       # face with the facial-region boxes, top 3 numbered
    prob_fake: float = None
    label: str = ""
    threshold: float = None
    confidence: str = ""
    face_found: bool = None
    top_regions: list = field(default_factory=list)   # [(region, concentration, share)] for the top 3
    region_source: str = ""
    summary: str = ""
    cards: list = field(default_factory=list)          # ids of the retrieved cards, in retrieval order
    model: str = ""
    answer: LLMExplanation = None
    check: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)        # step -> seconds
    truth: dict = None                                  # gallery images: dataset, file, manipulation, label
    cached: bool = False


def _upscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)


def _box(img, box, color, width):
    x0, y0, x1, y1 = (int(round(v * DISPLAY)) for v in box)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, width, cv2.LINE_AA)
    return x0, y0


def draw_top_regions(face: np.ndarray, regions: dict, top: list) -> np.ndarray:
    """Every facial-region box in thin grey, the top 3 thick and numbered in their colours."""
    out = face.copy()
    names = [r for r, _, _ in top]
    for name, box in regions.items():
        if name not in names:
            _box(out, box, (200, 200, 200), 1)
    for i, name in reversed(list(enumerate(names))):
        if name == BACKGROUND or name not in regions:
            continue
        x0, y0 = _box(out, regions[name], TOP_COLORS[i], 3)
        cv2.rectangle(out, (x0, y0), (x0 + 22, y0 + 22), TOP_COLORS[i], -1)
        cv2.putText(out, str(i + 1), (x0 + 5, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def check_answer(out: LLMExplanation, exp) -> dict:
    """Deterministic checks of a grounded answer against the cards it was given (no NLI model)."""
    if not out.schema_ok:
        return {}
    given, cited = set(out.retrieved), list(dict.fromkeys(out.citations))
    flags = grounding_flags(out, exp)
    return {"hits": [c for c in cited if c in given], "invented": [c for c in cited if c not in given],
            "region_in_top3": out.claimed_region in [r for r, _ in exp.top_regions[:3]],
            "artifact_backed": None if "artifact_not_in_cited_card" not in flags else not flags["artifact_not_in_cited_card"]}


class DemoPipeline:
    def __init__(self, checkpoint=CHECKPOINT):
        self.checkpoint = Path(checkpoint)
        self._explainer, self._retriever, self._llms = None, None, {}

    @property
    def ready(self) -> bool:
        return self.checkpoint.exists()

    @property
    def explainer(self):
        if self._explainer is None:
            from src.xai.gradcam import Explainer

            self._explainer = Explainer(self.checkpoint)
        return self._explainer

    @property
    def retriever(self):
        if self._retriever is None:
            from src.explain.retriever import Retriever

            self._retriever = Retriever()
        return self._retriever

    def llm(self, model: str, min_interval: float = 0) -> ForensicExplainer:
        """A Groq client for `model` (built once; the provider reads its settings when the client is built)."""
        if model not in self._llms:
            os.environ.update(LLM_PROVIDER="groq", LLM_MODEL=model, LLM_REASONING_EFFORT="low",
                              LLM_MIN_INTERVAL=str(min_interval))
            if model in MAX_TOKENS:
                os.environ["LLM_MAX_TOKENS"] = str(MAX_TOKENS[model])
            else:
                os.environ.pop("LLM_MAX_TOKENS", None)
            if not ForensicExplainer.available():
                raise RuntimeError("No GROQ_API_KEY set: live explanations need a Groq key "
                                   "(the gallery works without one)")
            self._llms[model] = ForensicExplainer(retriever=self.retriever, summary_regions=SUMMARY_REGIONS)
        return self._llms[model]

    def run(self, image: Image.Image, model: str, crop: bool = True):
        """Yields (step index just finished, result so far, error message or "") after every step; stops after
        the first error."""
        res, exp = DemoResult(model=model), None
        steps = iter(STEPS)
        try:
            step, t0 = next(steps), time.time()
            image = image.convert("RGB")
            face = self.explainer.faces.crop(np.asarray(image)) if crop and self.explainer.faces else None
            res.face_found = face is not None if crop else None
            image = Image.fromarray(face) if face is not None else image
            x, prob = self.explainer.predict(image)
            res.prob_fake, res.threshold = prob, self.explainer.model.threshold
            res.label = "fake" if prob >= res.threshold else "real"
            res.confidence = confidence_level(prob, res.threshold)
            res.face = _upscale(np.asarray(image.resize((IMG_SIZE, IMG_SIZE))))
            res.timings[step] = time.time() - t0
            yield 0, res, ""

            step, t0 = next(steps), time.time()
            exp = self.explainer.explain(image, x, prob)
            res.top_regions = [(r, exp.region_concentration.get(r, 0.0), exp.region_scores.get(r, 0.0))
                               for r, _ in exp.top_regions[:3]]
            res.region_source, res.summary = exp.region_source, exp.describe(SUMMARY_REGIONS)
            res.overlay = _upscale(exp.overlay)
            x0, y0, x1, y1 = exp.bbox
            cv2.rectangle(res.overlay, (int(x0 * DISPLAY), int(y0 * DISPLAY)),
                          (int(x1 * DISPLAY) - 1, int(y1 * DISPLAY) - 1), (255, 255, 255), 2, cv2.LINE_AA)
            res.regions = draw_top_regions(res.face, exp.regions, res.top_regions)
            res.timings[step] = time.time() - t0
            yield 1, res, ""

            step, t0 = next(steps), time.time()
            res.cards = [d.id for d in self.retriever.retrieve(exp)]
            res.timings[step] = time.time() - t0
            yield 2, res, ""

            step, t0 = next(steps), time.time()
            res.answer = self.llm(model).explain(exp, grounded=True)
            res.timings[step] = time.time() - t0
            if res.answer.error and not res.answer.schema_ok:
                raise RuntimeError(f"the model's reply could not be used ({res.answer.error[:160]})")
            yield 3, res, ""

            step, t0 = next(steps), time.time()
            res.check = check_answer(res.answer, exp)
            res.timings[step] = time.time() - t0
            yield 4, res, ""
        except Exception as e:
            yield STEPS.index(step) - 1, res, f"{step} failed: {e}"


# --- gallery ---------------------------------------------------------------------------------------------------

def save_gallery(entries: list, out_dir: Path = GALLERY_DIR):
    """entries: [(truth dict, {model: DemoResult})], all results of one entry from the same image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (truth, by_model) in enumerate(entries):
        first = next(iter(by_model.values()))
        key = f"{i:02d}"
        for name in ("face", "overlay", "regions"):
            Image.fromarray(getattr(first, name)).save(out_dir / f"{key}_{name}.png")
        meta = {k: v for k, v in asdict(first).items()
                if k not in ("face", "overlay", "regions", "model", "answer", "check", "timings", "truth", "cached")}
        meta["top_regions"] = [list(t) for t in first.top_regions]
        index.append({"key": key, "truth": truth, **meta,
                      "answers": {m: {"answer": r.answer.to_dict(), "check": r.check, "timings": r.timings}
                                  for m, r in by_model.items() if r.answer is not None and r.answer.schema_ok}})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return out_dir / "index.json"


def load_gallery(out_dir: Path = GALLERY_DIR) -> list:
    path = Path(out_dir) / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def gallery_result(entry: dict, model: str, out_dir: Path = GALLERY_DIR) -> DemoResult:
    """The stored result of one gallery image for `model` (answer None if that model has none stored)."""
    out_dir = Path(out_dir)
    stored = entry["answers"].get(model)
    res = DemoResult(**{k: entry[k] for k in ("prob_fake", "label", "threshold", "confidence", "face_found",
                                               "region_source", "summary", "cards")},
                     top_regions=[tuple(t) for t in entry["top_regions"]], truth=entry["truth"], model=model,
                     cached=True)
    for name in ("face", "overlay", "regions"):
        setattr(res, name, np.asarray(Image.open(out_dir / f"{entry['key']}_{name}.png").convert("RGB")))
    if stored:
        res.answer = LLMExplanation(**stored["answer"])
        res.check, res.timings = stored["check"], stored["timings"]
    return res


def pick_gallery(datasets: dict, explainer, pool: int = 8, mistakes: int = 2, seed: int = 0) -> list:
    """One correctly classified test image per manipulation method (and real) per dataset, plus up to
    `mistakes` images the detector gets wrong (a false alarm and a missed fake if both exist)."""
    from src.xai.evaluate_cam import sample_images

    picks, wrong = [], []
    for ds, root in datasets.items():
        cand = sample_images(Path(root), pool * 10, seed)
        for method, group in cand.groupby("method", sort=True):
            chosen = None
            for r in group.itertuples():
                _, prob = explainer.predict(Image.open(r.path).convert("RGB"))
                item = {"dataset": ds, "file": r.file, "path": r.path, "manipulation": method, "label": r.label}
                if (prob >= explainer.model.threshold) == (r.label == "fake"):
                    chosen = chosen or item
                else:
                    wrong.append(item)
                if chosen and len(wrong) >= 2 * mistakes:
                    break
            if chosen:
                picks.append(chosen)
    false_alarm = [w for w in wrong if w["label"] == "real"]
    missed = [w for w in wrong if w["label"] == "fake"]
    extra = (false_alarm[:1] + missed[:1] + false_alarm[1:] + missed[1:])[:mistakes]
    order = np.random.default_rng(seed).permutation(len(picks) + len(extra))   # mistakes not at the end
    allp = picks + extra
    return [allp[i] for i in order]


def build_gallery(pipeline: DemoPipeline, picks: list, models=tuple(MODELS), out_dir: Path = GALLERY_DIR,
                  min_interval: dict = None) -> Path:
    """Analyse every picked image with every model and store the results. min_interval: seconds between
    requests per model (free-tier pacing)."""
    entries = []
    for i, p in enumerate(picks):
        by_model = {}
        for model in models:
            pipeline.llm(model, (min_interval or {}).get(model, 0))
            res, err = None, ""
            for _, res, err in pipeline.run(Image.open(p["path"]), model, crop=False):
                pass
            if err:
                print(f"  {p['file']} · {model}: {err}")
            by_model[model] = res
        entries.append(({k: p[k] for k in ("dataset", "file", "manipulation", "label")}, by_model))
        print(f"{i + 1}/{len(picks)} {p['dataset']} {p['file']}: detector {res.label} ({res.prob_fake:.3f}), "
              f"truth {p['label']}")
    return save_gallery(entries, out_dir)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Precompute the demo gallery (needs the detector and GROQ_API_KEY)")
    ap.add_argument("--data", nargs="+", required=True, help="dataset=face-crop folder, e.g. ffpp=data/ffpp")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--mistakes", type=int, default=2)
    args = ap.parse_args()
    pipeline = DemoPipeline(args.checkpoint)
    datasets = dict(d.split("=", 1) for d in args.data)
    picks = pick_gallery(datasets, pipeline.explainer, mistakes=args.mistakes)
    print(f"saved {build_gallery(pipeline, picks, min_interval={'qwen/qwen3.8-27b': 60})}")


if __name__ == "__main__":
    main()
