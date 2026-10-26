"""Evidence checks for one explanation: the phase 4 verifier, also used to score phase 3 answers.

grounding_flags()   citations and the artifact claim checked against the cards the answer cites
method_consistent() is the claimed artifact documented for the image's true manipulation family?
                    (evaluation only: the LLM never sees the manipulation name)
NLIChecker          is each explanation sentence entailed by the evidence summary plus the cards it cites?
EvidenceVerifier    all of the above plus the heatmap checks in metrics.py; problems() lists failures in plain
                    English for the revision request of the "verified" condition

The NLI check measures attribution (is the sentence supported by what the model was given and cites), not
truth: a correct statement from the model's own knowledge counts as unsupported. Ungrounded answers have no
cards, so their premise is the evidence summary alone.
"""
import re

import numpy as np

from src.eval.metrics import faithfulness_iou, hallucination_flags, pointing_hit
from src.explain.corpus import METHOD_FAMILY, cards_by_id, load_corpus

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS = ("entailment", "neutral", "contradiction")


def sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text.strip()) if s.strip()]


def cited_cards(out) -> list:
    """Cards the answer cites that were actually given to it (ids that were not provided are ignored)."""
    cards = cards_by_id()
    return [cards[c] for c in dict.fromkeys(out.citations) if c in cards and c in out.retrieved]


def grounding_flags(out, exp) -> dict:
    """True = problem. Only for answers that were given cards (grounded, verified); {} otherwise."""
    if not out.retrieved:
        return {}
    flags = {"uncited": not out.citations}
    if exp.label == "fake":
        same_type = [c for c in cited_cards(out) if c.kind == "artifact" and c.artifact_type == out.artifact_type]
        flags["artifact_not_in_cited_card"] = not same_type
        flags["region_not_in_cited_card"] = not any(out.claimed_region in c.regions for c in same_type)
    return flags


def method_consistent(out, manipulation: str):
    """Is the claimed artifact described by an artifact card for the true manipulation's family?
    None when it does not apply (real images, unknown method, artifact "none" or empty)."""
    family = METHOD_FAMILY.get(manipulation)
    if family is None or out.artifact_type in ("", "none"):
        return None
    return any(d.kind == "artifact" and d.artifact_type == out.artifact_type and d.applies_to(family)
               for d in load_corpus())


class NLIChecker:
    """Sentence-level entailment with a cross-encoder NLI model (premise, hypothesis) -> 3 classes."""

    def __init__(self, model: str = NLI_MODEL, device: str = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name, self.batch_size = torch, model, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [labels[name] for name in NLI_LABELS]

    def probs(self, premises: list, hypotheses: list) -> np.ndarray:
        """(n, 3) probabilities of entailment, neutral, contradiction. Long premises are truncated."""
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out.append(self.model(**enc).logits.softmax(-1)[:, self.order].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))


class EvidenceVerifier:
    def __init__(self, nli: NLIChecker = None, revise_on: tuple = ("contradiction",)):
        """nli: NLIChecker, or None to skip the sentence check.
        revise_on: NLI labels that count as a problem in the revision request. The default asks for a revision
        only when a sentence contradicts the evidence; adding "neutral" also flags sentences that are merely not
        entailed, which in practice includes most interpretive sentences and so triggers far more revisions."""
        self.nli, self.revise_on = nli, tuple(revise_on)

    @staticmethod
    def premise(out, exp) -> str:
        return " ".join([exp.describe()] + [c.text for c in cited_cards(out)])

    def support_batch(self, outs, exps) -> list:
        """Per answer: its sentences, their NLI labels and the shares supported/contradicted (one batched call)."""
        pairs, spans = [], []
        for out, exp in zip(outs, exps):
            sents = sentences(out.explanation) if out.schema_ok else []
            spans.append((len(pairs), sents))
            pairs += [(self.premise(out, exp), s) for s in sents]
        p = self.nli.probs([a for a, _ in pairs], [b for _, b in pairs]) if pairs else np.zeros((0, 3))
        results = []
        for start, sents in spans:
            labels = [NLI_LABELS[k] for k in p[start:start + len(sents)].argmax(1)]
            n = len(sents)
            results.append({"sentences": sents, "labels": labels, "n_sentences": n,
                            "supported_share": labels.count("entailment") / n if n else np.nan,
                            "contradicted_share": labels.count("contradiction") / n if n else np.nan})
        return results

    def problems(self, out, exp) -> list:
        """Plain-English problems with one answer (empty list = passes every check)."""
        if not out.schema_ok:
            return ["The answer must be a JSON object with claimed_region from the region list, artifact_type "
                    "from the artefact list and a non-empty explanation" + (f" ({out.error[:120]})." if out.error else ".")]
        top3 = [r for r, _ in exp.top_regions[:3]]
        f, g, found = hallucination_flags(out, exp), grounding_flags(out, exp), []
        if f["region_unsupported"]:
            found.append(f'claimed_region "{out.claimed_region}" is not one of the three regions where the evidence '
                         f"is concentrated ({', '.join(top3)}).")
        if f["verdict_mismatch"]:
            found.append(f'artifact_type "{out.artifact_type}" contradicts the {exp.label.upper()} verdict'
                         + (' (use "none" for a REAL verdict).' if exp.label == "real"
                            else " (a FAKE verdict needs an artefact type)."))
        if f["citation_invalid"]:
            found.append(f"citations include ids that were not provided: "
                         f"{', '.join(c for c in out.citations if c not in out.retrieved)}.")
        if g.get("uncited"):
            found.append("citations is empty; cite the ids of the cards you used.")
        if g.get("artifact_not_in_cited_card"):
            found.append(f'artifact_type "{out.artifact_type}" is not supported by any artefact card you cited.')
        elif g.get("region_not_in_cited_card"):
            found.append(f'none of the cited "{out.artifact_type}" cards lists the region "{out.claimed_region}".')
        if self.nli is not None and self.revise_on:
            s = self.support_batch([out], [exp])[0]
            found += [f'This sentence {"contradicts" if label == "contradiction" else "is not supported by"} the '
                      f'evidence summary or the cited cards: "{sent}"'
                      for sent, label in zip(s["sentences"], s["labels"]) if label in self.revise_on]
        return found

    def checks(self, out, exp, manipulation: str = None, support: dict = None) -> dict:
        """Evaluation row for one answer (True = good). Checks that do not apply are left out (NaN in a table).
        support: this answer's entry from support_batch(), to batch the NLI model over many answers."""
        row = {"valid": out.schema_ok}
        if not out.schema_ok:
            return row
        f = hallucination_flags(out, exp)
        row.update(region_in_top3=not f["region_unsupported"], artifact_plausible=not f["artifact_unsupported"],
                   verdict_consistent=not f["verdict_mismatch"], citation_valid=not f["citation_invalid"])
        g = grounding_flags(out, exp)
        if g:
            row["cites_cards"] = not g["uncited"]
        if "artifact_not_in_cited_card" in g:
            row.update(artifact_in_cited_card=not g["artifact_not_in_cited_card"],
                       region_in_cited_card=not g["region_not_in_cited_card"])
        if getattr(exp, "heatmap", None) is not None:
            row.update(pointing_hit=pointing_hit(out.claimed_region, exp.heatmap, exp.regions),
                       region_iou=faithfulness_iou(out.claimed_region, exp.heatmap, regions=exp.regions))
        consistent = method_consistent(out, manipulation)
        if consistent is not None:
            row["method_consistent"] = consistent
        if exp.label == "fake":
            row["gan_claim"] = out.artifact_type == "spectral_gan_fingerprint"
        if self.nli is not None:
            s = support or self.support_batch([out], [exp])[0]
            row.update(n_sentences=s["n_sentences"], supported_share=s["supported_share"],
                       contradicted_share=s["contradicted_share"])
        return row
