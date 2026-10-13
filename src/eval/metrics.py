"""Faithfulness and hallucination metrics for heatmap-grounded explanations."""
import numpy as np

from src.explain.corpus import artifact_taxonomy, load_corpus
from src.xai.gradcam import BACKGROUND, FACE_REGIONS, heatmap_mask


def region_mask(region: str, shape) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if region in FACE_REGIONS:
        x0, y0, x1, y1 = FACE_REGIONS[region]
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    elif region == BACKGROUND:
        mask[:] = True
        for name in FACE_REGIONS:
            mask &= ~region_mask(name, shape)
    return mask


def faithfulness_iou(claimed_region: str, heatmap: np.ndarray, quantile: float = 0.8) -> float:
    """IoU between the region the explanation talks about and the top-(1-q) Grad-CAM++ pixels."""
    a, b = region_mask(claimed_region, heatmap.shape), heatmap_mask(heatmap, quantile)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def pointing_hit(claimed_region: str, heatmap: np.ndarray) -> bool:
    """Does the claimed region contain the heatmap peak? (pointing-game metric)"""
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(region_mask(claimed_region, heatmap.shape)[peak])


def hallucination_flags(llm, exp, k: int = 3) -> dict:
    """llm: LLMExplanation, exp: Explanation. Each flag True == hallucinated claim."""
    taxonomy = artifact_taxonomy()
    top_regions = {r for r, _ in exp.top_regions[:k]}
    valid_ids = {d.id for d in load_corpus()}

    unknown_artifact = llm.artifact_type not in taxonomy
    verdict_mismatch = (exp.label == "real") != (llm.artifact_type == "none")
    # Artifact is unsupported if none of the heatmap's top regions is a place that artefact appears
    artifact_unsupported = unknown_artifact or not (taxonomy[llm.artifact_type] & top_regions)
    region_unsupported = llm.claimed_region not in top_regions
    citation_invalid = any(c not in valid_ids or (llm.retrieved and c not in llm.retrieved) for c in llm.citations)

    flags = {
        "region_unsupported": region_unsupported,
        "artifact_unsupported": artifact_unsupported,
        "verdict_mismatch": verdict_mismatch,
        "citation_invalid": citation_invalid,
    }
    flags["any_hallucination"] = any(flags.values())
    return flags
