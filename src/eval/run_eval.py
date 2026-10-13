"""Phase 5: compare ungrounded vs. RAG-grounded explanations on N test images.

    python -m src.eval.run_eval --n 50
Writes outputs/eval/results.csv and outputs/eval/summary.json.
"""
import argparse
import csv
import json
import random

from PIL import Image
from tqdm import tqdm

from src.classifier.dataset import split_dir
from src.config import OUTPUT_DIR
from src.eval.metrics import faithfulness_iou, hallucination_flags, pointing_hit
from src.explain.llm import ForensicExplainer
from src.xai.gradcam import Explainer


def sample_images(n: int, split: str, fakes_only: bool):
    classes = ["fake"] if fakes_only else ["fake", "real"]
    files = [(p, c) for c in classes for p in sorted((split_dir(split) / c).glob("*.*"))]
    return random.Random(0).sample(files, min(n, len(files)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--split", default="test")
    ap.add_argument("--include-real", action="store_true", help="also sample real images")
    args = ap.parse_args()

    if not ForensicExplainer.available():
        raise SystemExit("No LLM configured: set LLM_PROVIDER and its API key (see src/explain/providers.py).")
    cam, llm = Explainer(), ForensicExplainer()
    rows = []
    for path, true_label in tqdm(sample_images(args.n, args.split, not args.include_real)):
        exp = cam(Image.open(path))
        for grounded in (False, True):
            out = llm.explain(exp, grounded=grounded)
            row = {"image": path.name, "true_label": true_label, "pred_label": exp.label,
                   "prob_fake": round(exp.prob_fake, 4), "top_region": exp.top_regions[0][0], **out.to_dict()}
            if out.error:
                row["error"] = out.error
            else:
                row.update(iou=round(faithfulness_iou(out.claimed_region, exp.heatmap), 4),
                           pointing_hit=pointing_hit(out.claimed_region, exp.heatmap),
                           **hallucination_flags(out, exp))
            rows.append(row)

    out_dir = OUTPUT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows)

    summary = {"llm": llm.name}
    metrics = ["iou", "pointing_hit", "region_unsupported", "artifact_unsupported", "verdict_mismatch",
               "citation_invalid", "any_hallucination"]
    for cond in ("ungrounded", "grounded"):
        ok = [r for r in rows if r["condition"] == cond and not r.get("error")]
        summary[cond] = {"n": len(ok), "errors": sum(1 for r in rows if r["condition"] == cond and r.get("error"))}
        for m in metrics:
            summary[cond][m] = round(sum(float(r[m]) for r in ok) / len(ok), 4) if ok else None
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nLLM: {llm.name}")
    print(f"{'metric':<22}{'ungrounded':>12}{'grounded':>12}")
    for m in metrics:
        u, g = summary["ungrounded"][m], summary["grounded"][m]
        print(f"{m:<22}{u if u is not None else '-':>12}{g if g is not None else '-':>12}")
    print(f"\nsaved {out_dir / 'results.csv'} and summary.json")


if __name__ == "__main__":
    main()
