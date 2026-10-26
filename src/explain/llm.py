"""Natural-language forensic explanations: ungrounded (phase 3) vs. RAG-grounded (phase 4).

The LLM never sees the image. It receives only the evidence summary built by Explanation.describe()
(verdict, P(fake), threshold, confidence, top regions) and must answer in a fixed JSON schema with labels
from fixed lists, so every claim can be checked automatically against the heatmap in phase 5.

Both conditions share the same system prompt, evidence summary, label sets and output schema; the only
difference is whether retrieved forensic passages are included (and must be cited).
The LLM backend (hf, Anthropic, OpenRouter, Ollama or any OpenAI-compatible server) is set in providers.py.
"""
import json
import re
import time
from dataclasses import asdict, dataclass, field

from src.explain import providers
from src.explain.corpus import artifact_taxonomy
from src.explain.retriever import Retriever
from src.xai.gradcam import BACKGROUND, FACE_REGIONS

REGIONS = list(FACE_REGIONS) + [BACKGROUND]

SYSTEM = """You are a digital-forensics assistant explaining the decision of a deepfake image classifier \
to a non-expert. You cannot see the image; you only receive the classifier's verdict, its confidence and a \
summary of where its Grad-CAM++ evidence is concentrated. Base everything on that evidence. Be concise, \
cautious and specific. Never claim certainty about the cause of an artefact, and hedge more when the \
confidence is borderline.

If the verdict is FAKE: name the facial region the evidence points to and the most plausible type of \
manipulation artefact there.
If the verdict is REAL: set artifact_type to "none" and explain which regions the classifier relied on to \
judge the face authentic.

Respond with ONLY a JSON object with these keys:
  "claimed_region": one of {regions}
  "artifact_type": one of {artifacts}
  "explanation": 2-4 plain-English sentences for a non-expert
  "citations": list of source ids you relied on (empty list if no sources were provided)"""

GROUNDED_SUFFIX = """

Forensic reference passages are provided below. Base the artifact_type and explanation ONLY on artefact \
categories that these passages support for the highlighted region, and cite the ids of the passages you use."""


@dataclass
class LLMExplanation:
    condition: str
    claimed_region: str = ""
    artifact_type: str = ""
    explanation: str = ""
    citations: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    error: str = ""
    schema_ok: bool = False      # valid JSON with labels from the allowed lists
    llm: str = ""                # provider:model that produced it
    latency_s: float = 0.0
    raw: str = ""                # the model's raw reply, kept for auditing

    def to_dict(self):
        return asdict(self)


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models (e.g. Qwen3, DeepSeek)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(match.group(0))


class ForensicExplainer:
    def __init__(self):
        self.client = providers.ChatClient()
        self.name = providers.describe()
        self.retriever = Retriever()
        self.taxonomy = sorted(artifact_taxonomy())
        self.system = SYSTEM.format(regions=json.dumps(REGIONS), artifacts=json.dumps(self.taxonomy))

    @staticmethod
    def available() -> bool:
        return providers.available()

    def build_prompt(self, exp, grounded: bool):
        """(system prompt, user message, retrieved docs) for one Explanation."""
        system, user = self.system, f"Evidence summary:\n{exp.describe()}"
        docs = []
        if grounded:
            regions = [r for r, _ in exp.top_regions[:3]]
            docs = self.retriever.search(exp.describe(), regions=regions, verdict=exp.label, k=3)
            system += GROUNDED_SUFFIX
            passages = "\n\n".join(f"[{d.id}] ({d.citation})\n{d.text}" for d in docs)
            user += f"\n\nForensic reference passages:\n{passages}"
        return system, user, docs

    def _parse(self, raw: str, condition: str, docs, latency: float) -> LLMExplanation:
        out = LLMExplanation(condition, retrieved=[d.id for d in docs], llm=self.name, latency_s=latency, raw=raw)
        try:
            data = _parse_json(raw)
            out.claimed_region = str(data.get("claimed_region", "")).strip()
            out.artifact_type = str(data.get("artifact_type", "")).strip()
            out.explanation = str(data.get("explanation", "")).strip()
            cites = data.get("citations", [])
            out.citations = [str(c) for c in cites] if isinstance(cites, list) else [str(cites)]
            out.schema_ok = (out.claimed_region in REGIONS and out.artifact_type in self.taxonomy
                             and bool(out.explanation))
        except Exception as e:  # keep going; parse failures are counted, not hidden
            out.error = f"{type(e).__name__}: {e}"
        return out

    def explain_batch(self, exps, grounded: bool):
        """Explanations for several images (batched on the GPU for hf, one API call each otherwise).
        A failed API call is recorded as an error for that image and the rest carry on."""
        condition = "grounded" if grounded else "ungrounded"
        prompts = [self.build_prompt(e, grounded) for e in exps]
        if self.client.provider == "hf":
            start = time.time()
            raws = self.client.chat_batch([(s, u) for s, u, _ in prompts])
            per = (time.time() - start) / max(1, len(exps))
            return [self._parse(r, condition, d, per) for r, (_, _, d) in zip(raws, prompts)]
        results = []
        for s, u, d in prompts:
            try:
                raw = self.client.chat(s, u)
            except Exception as e:
                results.append(LLMExplanation(condition, retrieved=[x.id for x in d], llm=self.name,
                                              error=f"{type(e).__name__}: {e}"))
                continue
            results.append(self._parse(raw, condition, d, self.client.last_latency))  # waits for rate limits excluded
        return results

    def explain(self, exp, grounded: bool) -> LLMExplanation:
        """exp: src.xai.gradcam.Explanation"""
        return self.explain_batch([exp], grounded)[0]
