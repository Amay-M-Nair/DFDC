"""Phase 5 experiment: the held-out test sample, the metrics, no-LLM baselines and scoring of every answer.

The pilot (notebooks 05 and 06) was used to tune prompts, limits and the verifier, so phase 5 reports results on
a fresh sample of test videos that the pilot and worked examples did not use, with the settings frozen.

Baselines answer from fixed rules, scored with the same checks as the LLMs (they have no text, so the NLI,
citation and language checks do not apply):
  top region + first card   the heatmap's hottest region, and the artefact of the retriever's best artefact card
                            for a FAKE verdict ("none" for REAL): the evidence pipeline without the LLM
  always mouth              "mouth / teeth" and teeth_mouth_artifacts for FAKE (the most common pilot answer)
  random plausible          the expected score of a random top-3 region with a random artefact plausible there
                            (exact average over all such combinations)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain.corpus import artifact_taxonomy
from src.explain.llm import LLMExplanation

# name: (group, higher is better, meaning)
METRICS = {
    "valid": ("output", True, "JSON with labels from the allowed lists"),
    "region_in_top3": ("heatmap", True, "the named region is one of the heatmap's top-3 regions"),
    "pointing_hit": ("heatmap", True, "the named region contains the heatmap's hottest pixel"),
    "region_iou": ("heatmap", True, "overlap of the named region with the top 20% of heatmap pixels"),
    "verdict_consistent": ("artefact", True, "an artefact for FAKE, none for REAL"),
    "artifact_plausible": ("artefact", True, "the artefact occurs in one of the top-3 regions (corpus taxonomy)"),
    "method_consistent": ("artefact", True, "the artefact is documented for the image's true manipulation family"),
    "gan_claim": ("artefact", False, "claims a GAN fingerprint (no test image is fully GAN-generated)"),
    "supported_share": ("attribution", True, "sentences entailed by the evidence summary + cited cards (NLI)"),
    "contradicted_share": ("attribution", False, "sentences contradicted by the evidence summary + cited cards"),
    "citation_valid": ("attribution", True, "cites only ids that were provided"),
    "artifact_in_cited_card": ("attribution", True, "a cited artefact card supports the artefact (grounded)"),
    "region_in_cited_card": ("attribution", True, "that card also lists the named region (grounded)"),
    "hedges_when_uncertain": ("language", True, "uses hedging words on a moderate or borderline verdict"),
    "overclaims": ("language", False, "uses certainty words (clearly, definitely, proves...)"),
    "visual_claim": ("language", False, "describes the image as if seen (the LLM never sees it)"),
}
HIGHER_IS_BETTER = {m: v[1] for m, v in METRICS.items()}
CONTINUOUS = ("region_iou", "supported_share", "contradicted_share")   # shares and overlaps: Wilcoxon, not McNemar
# Pre-registered primary comparison: grounded vs. ungrounded, per model, Holm-corrected over all these tests.
PRIMARY = ["supported_share", "contradicted_share", "method_consistent", "gan_claim"]

BASELINES = ("top region + first card", "always mouth", "random plausible")
BASELINE_LLM = "baseline"

# Approximate keyword checks (English); they flag wording, not meaning.
HEDGE = re.compile(r"\b(may|might|could|possibly|possible|perhaps|likely|unlikely|suggests?|appears?|seems?|"
                   r"uncertain|uncertainty|borderline|not (?:certain|sure|conclusive)|caution|tentative)\b", re.I)
CERTAIN = re.compile(r"\b(clearly|definitely|certainly|undoubtedly|unquestionably|obviously|without (?:a )?doubt|"
                     r"proves?|proven|confirms?|conclusively)\b", re.I)
VISUAL = re.compile(r"\b(i (?:can )?see|i notice|we can see|can be seen|(?:is|are) (?:clearly )?visible|"
                    r"looks? (?:natural|unnatural|blurr\w*|smooth|real|fake|distorted)|"
                    r"natural skin texture|consistent tooth alignment)\b", re.I)


def video_of(file: str) -> str:
    return "__".join(Path(file).name.split("__")[:2])


def test_sample(root: Path, n: int, exclude_videos=(), seed: int = 1) -> pd.DataFrame:
    """One frame per test video, balanced across manipulation methods (and real), from videos not in
    `exclude_videos` (the pilot and the worked examples)."""
    files = sorted(Path(root, "test").glob("*/*.jpg"))
    df = pd.DataFrame({"path": [str(p) for p in files], "label": [p.parent.name for p in files],
                       "file": [p.name for p in files]})
    df["method"] = df.file.str.split("__").str[0]
    df["video"] = df.file.map(video_of)
    df = df[~df.video.isin(set(exclude_videos))]
    one = df.groupby("video").sample(1, random_state=seed)
    groups = one.groupby("method")
    per = max(1, n // groups.ngroups)
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in groups]).reset_index(drop=True)


def language_flags(text: str, confidence: str) -> dict:
    flags = {"overclaims": bool(CERTAIN.search(text)), "visual_claim": bool(VISUAL.search(text))}
    if confidence != "high":
        flags["hedges_when_uncertain"] = bool(HEDGE.search(text))
    return flags


def _answer(name, region, artifact):
    return LLMExplanation(name, claimed_region=region, artifact_type=artifact, explanation="", schema_ok=True)


def baseline_answers(exp, retriever) -> dict:
    """{baseline name: list of answers}; a baseline's score is the mean over its list."""
    top3 = [r for r, _ in exp.top_regions[:3]]
    fake = exp.label == "fake"
    first = "none"
    if fake:
        first = next((d.artifact_type for d in retriever.retrieve(exp) if d.kind == "artifact"), "none")
    taxonomy = artifact_taxonomy()
    if fake:
        combos = [(r, a) for r in top3 for a in sorted(taxonomy) if a != "none" and r in taxonomy[a]]
        combos = combos or [(r, a) for r in top3 for a in sorted(taxonomy) if a != "none"]
    else:
        combos = [(r, "none") for r in top3]
    return {"top region + first card": [_answer("top region + first card", top3[0], first)],
            "always mouth": [_answer("always mouth", "mouth / teeth", "teeth_mouth_artifacts" if fake else "none")],
            "random plausible": [_answer("random plausible", r, a) for r, a in combos]}


def _base_row(it, cond, llm):
    exp = it["exp"]
    return {"batch": it.get("batch"), "image": f"{it['dataset']}/{it['file']}", "dataset": it["dataset"],
            "file": it["file"], "manipulation": it["manipulation"], "condition": cond, "llm": llm,
            "verdict": exp.label, "verdict_correct": exp.label == it["label"], "confidence": exp.confidence}


def score(items, results, verifier, retriever=None):
    """Score every answer (and the baselines, when a retriever is given).

    items: dicts with dataset, file, manipulation, label (true), exp (Explanation) and optionally batch.
    results: {(condition, model_id): [LLMExplanation, ...]} in the order of `items`.
    Returns (scores, sentences): one row per (llm, condition, image), and one row per explanation sentence with
    its NLI label. Failed requests (no reply at all, e.g. a rate limit) are kept with request_failed=True and
    every metric left empty, since they say nothing about the model."""
    rows, sentence_rows = [], []
    for (cond, model_id), outs in results.items():
        llm = model_id.split("/")[-1]
        support = verifier.support_batch(outs, [it["exp"] for it in items])
        for it, o, s in zip(items, outs, support):
            row = _base_row(it, cond, llm)
            failed = bool(o.error) and not o.raw
            row.update(artifact=o.artifact_type if o.schema_ok else None, revised=o.revised,
                       request_failed=failed, latency_s=o.latency_s)
            if not failed:
                row.update(verifier.checks(o, it["exp"], it["manipulation"], s))
                if o.schema_ok:
                    row.update(language_flags(o.explanation, it["exp"].confidence))
            rows.append(row)
            sentence_rows += [{**{k: row[k] for k in ("batch", "image", "condition", "llm", "verdict")},
                               "sentence": sent, "nli_label": label, "premise": verifier.premise(o, it["exp"])}
                              for sent, label in zip(s["sentences"], s["labels"])]
    if retriever is not None:
        no_text = ("valid", "citation_valid", "n_sentences", "supported_share", "contradicted_share")
        for it in items:
            for name, answers in baseline_answers(it["exp"], retriever).items():
                checks = pd.DataFrame([verifier.checks(a, it["exp"], it["manipulation"], {}) for a in answers])
                checks = checks.drop(columns=[c for c in no_text if c in checks]).astype(float).mean()
                row = _base_row(it, name, BASELINE_LLM)
                row.update(artifact=answers[0].artifact_type if len(answers) == 1 else None, request_failed=False,
                           **checks.to_dict())
                rows.append(row)
    scores = pd.DataFrame(rows)
    metrics = [m for m in METRICS if m in scores]
    scores[metrics] = scores[metrics].astype(float)
    return scores, pd.DataFrame(sentence_rows)
