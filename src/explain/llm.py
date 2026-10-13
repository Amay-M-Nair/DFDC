"""Natural-language forensic explanations: ungrounded vs. RAG-grounded.

Both conditions share the same system prompt, heatmap description, label set and output schema;
the only difference is whether retrieved forensic passages are included (and must be cited).
The LLM backend (Anthropic, OpenRouter, Ollama or any OpenAI-compatible server) is set in providers.py.
"""
import json
import re
from dataclasses import asdict, dataclass, field

from src.explain import providers
from src.explain.corpus import artifact_taxonomy
from src.explain.retriever import Retriever
from src.xai.gradcam import BACKGROUND, FACE_REGIONS

REGIONS = list(FACE_REGIONS) + [BACKGROUND]

SYSTEM = """You are a digital-forensics assistant explaining the decision of a deepfake image classifier \
to a non-expert. You cannot see the image; you only receive the classifier verdict and a summary of its \
Grad-CAM++ heatmap. Be concise, cautious and specific. Never claim certainty about the cause of an artefact.

Respond with ONLY a JSON object with these keys:
  "claimed_region": one of {regions}
  "artifact_type": one of {artifacts}  (use "none" if the verdict is REAL)
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

    def to_dict(self):
        return asdict(self)


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models (e.g. local Qwen/DeepSeek)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(match.group(0))


class ForensicExplainer:
    def __init__(self):
        self.client = providers.ChatClient()
        self.name = providers.describe()
        self.retriever = Retriever()
        self.system = SYSTEM.format(regions=json.dumps(REGIONS), artifacts=json.dumps(sorted(artifact_taxonomy())))

    @staticmethod
    def available() -> bool:
        return providers.available()

    def explain(self, exp, grounded: bool) -> LLMExplanation:
        """exp: src.xai.gradcam.Explanation"""
        condition = "grounded" if grounded else "ungrounded"
        system, user = self.system, f"Heatmap summary:\n{exp.describe()}"
        docs = []
        if grounded:
            regions = [r for r, _ in exp.top_regions[:3]]
            docs = self.retriever.search(exp.describe(), regions=regions, verdict=exp.label, k=3)
            system += GROUNDED_SUFFIX
            passages = "\n\n".join(f"[{d.id}] ({d.citation})\n{d.text}" for d in docs)
            user += f"\n\nForensic reference passages:\n{passages}"
        try:
            data = _parse_json(self.client.chat(system, user))
            return LLMExplanation(condition, str(data.get("claimed_region", "")), str(data.get("artifact_type", "")),
                                  str(data.get("explanation", "")), list(data.get("citations", [])),
                                  [d.id for d in docs])
        except Exception as e:  # keep the pipeline running; failures are reported in eval
            return LLMExplanation(condition, retrieved=[d.id for d in docs], error=f"{type(e).__name__}: {e}")
