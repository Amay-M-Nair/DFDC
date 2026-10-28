"""Natural-language forensic explanations: ungrounded (phase 3) vs. RAG-grounded (phase 4).

The LLM never sees the image. It receives only the evidence summary built by Explanation.describe()
(verdict, P(fake), threshold, confidence, top regions) and must answer in a fixed JSON schema with labels
from fixed lists, so every claim can be checked automatically against the heatmap in phase 5.

Both conditions share the same system prompt, evidence summary, label sets and output schema; the only
difference is that the grounded prompt adds retrieved evidence cards (corpus/) and the instruction to use and
cite them. The "verified" condition checks each grounded answer (src/eval/verify.py) and asks once for a
revision when a check fails.
The LLM backend (Groq, Anthropic, OpenRouter, Ollama or any OpenAI-compatible server) is set in providers.py.
"""
import json
import re
from dataclasses import asdict, dataclass, field, replace

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

Forensic reference cards follow the evidence summary. Each card has an id in square brackets, says what it is \
about and lists the facial regions it applies to. Use them as your only source of forensic knowledge:
- If the verdict is FAKE, choose artifact_type only from the artefact cards, and only a type whose card lists \
your claimed_region. Scope cards say when an artefact does not apply and limitation cards say what the evidence \
cannot show; follow both.
- Support every statement with the evidence summary or a card, and cite the ids of all cards you used."""

REVISION = """

Your previous answer was:
{previous}

Automatic checks found these problems:
{problems}

Rewrite the answer to fix them, using only the evidence summary and the cards above. Respond with ONLY the \
JSON object."""

CARD_ABOUT = {"artifact": "artefact: {}", "scope": "scope of: {}", "limitation": "limitation of the evidence",
              "authentic": "authentic cues"}


def format_card(d) -> str:
    """One evidence card as the LLM sees it."""
    regions = "any" if set(REGIONS) <= set(d.regions) else ", ".join(d.regions)
    return f"[{d.id}] {CARD_ABOUT[d.kind].format(d.artifact_type)} | regions: {regions}\n{d.text} ({d.citation})"


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
    revised: bool = False        # verified condition: a revision was requested
    problems: list = field(default_factory=list)  # verified condition: what the checks found in the grounded answer

    def to_dict(self):
        return asdict(self)


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models (e.g. Qwen3, DeepSeek)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(match.group(0))


def normalise_citation(cid) -> str:
    """Card id as cited, without the brackets or quotes some models copy from the prompt ("[warp-affine]")."""
    return str(cid).strip().strip("[]'\"` ").strip()


class ForensicExplainer:
    def __init__(self, retriever: Retriever = None):
        """retriever: a shared Retriever; one is built on first grounded use if not given."""
        self.client = providers.ChatClient()
        self.name = providers.describe()
        self._retriever = retriever
        self.taxonomy = sorted(artifact_taxonomy())
        self.system = SYSTEM.format(regions=json.dumps(REGIONS), artifacts=json.dumps(self.taxonomy))

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    @staticmethod
    def available() -> bool:
        return providers.available()

    def build_prompt(self, exp, grounded: bool):
        """(system prompt, user message, retrieved cards) for one Explanation."""
        system, user = self.system, f"Evidence summary:\n{exp.describe()}"
        docs = []
        if grounded:
            docs = self.retriever.retrieve(exp)
            system += GROUNDED_SUFFIX
            user += "\n\nForensic reference cards:\n" + "\n\n".join(format_card(d) for d in docs)
        return system, user, docs

    def _parse(self, raw: str, condition: str, docs, latency: float) -> LLMExplanation:
        out = LLMExplanation(condition, retrieved=[d.id for d in docs], llm=self.name, latency_s=latency, raw=raw)
        try:
            data = _parse_json(raw)
            out.claimed_region = str(data.get("claimed_region", "")).strip()
            out.artifact_type = str(data.get("artifact_type", "")).strip()
            out.explanation = str(data.get("explanation", "")).strip()
            cites = data.get("citations", [])
            cites = cites if isinstance(cites, list) else [cites]
            out.citations = [c for c in map(normalise_citation, cites) if c]
            out.schema_ok = (out.claimed_region in REGIONS and out.artifact_type in self.taxonomy
                             and bool(out.explanation))
        except Exception as e:  # keep going; parse failures are counted, not hidden
            out.error = f"{type(e).__name__}: {e}"
        return out

    def explain_batch(self, exps, grounded: bool):
        """Explanations for several images, one API call each.
        A failed API call is recorded as an error for that image and the rest carry on."""
        condition = "grounded" if grounded else "ungrounded"
        prompts = [self.build_prompt(e, grounded) for e in exps]
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

    def revise_batch(self, exps, outs, verifier):
        """The "verified" condition: check each grounded answer and, when a check fails, ask once for a revision
        that lists the problems. Answers that pass are kept unchanged (revised=False).
        verifier: src.eval.verify.EvidenceVerifier"""
        results = []
        for exp, first in zip(exps, outs):
            if first.error and not first.raw:  # the request itself failed: nothing to revise
                results.append(replace(first, condition="verified", problems=["request failed"]))
                continue
            problems = verifier.problems(first, exp)
            if not problems:
                results.append(replace(first, condition="verified"))
                continue
            system, user, docs = self.build_prompt(exp, grounded=True)
            user += REVISION.format(previous=first.raw.strip(), problems="\n".join(f"- {p}" for p in problems))
            try:
                out = self._parse(self.client.chat(system, user), "verified", docs, self.client.last_latency)
            except Exception as e:
                out = LLMExplanation("verified", retrieved=[d.id for d in docs], llm=self.name,
                                     error=f"{type(e).__name__}: {e}")
            out.revised, out.problems = True, problems
            results.append(out)
        return results

    def explain(self, exp, grounded: bool) -> LLMExplanation:
        """exp: src.xai.gradcam.Explanation"""
        return self.explain_batch([exp], grounded)[0]
