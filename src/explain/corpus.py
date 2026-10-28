"""Load the forensic evidence cards (corpus/*.md).

Each file holds cards for one topic. A card is a `## <id>` heading, `key: value` lines, a blank line and
1-3 sentences of text:

    ## bb-face-xray
    artifact: blending_boundary         one of the artifact labels, `none` (authentic cues) or `-`
    kind: artifact                       artifact | scope | limitation | authentic
    regions: left jawline, hairline      atomic facial regions the card applies to, or `all`
    pattern: outer                       central | outer | any (where on the face the evidence lies)
    families: swap_autoencoder           manipulation families the claim is about (see FAMILIES), or `any`
    confidence: moderate, borderline     optional, limitation cards: preferred for these confidence levels
    source: Li et al., "Face X-Ray ...", CVPR 2020
    checked: no                          yes once the claim has been checked against the source

    Most face manipulation methods ...

Kinds: an *artifact* card says where an artefact appears and supports that artifact label for its regions; a
*scope* card says when an artefact does not apply; a *limitation* card says what the evidence cannot show; an
*authentic* card explains evidence for a REAL verdict. Text before the first card is ignored.

Regions are written as atomic parts (teeth, mouth, hairline, forehead, chin, left/right jawline, face boundary,
background, plus the eyes, nose and cheeks), each listed only when the card's claim is about it. The demo uses
them as they are (load_corpus(atomic=True)); phases 3-5 map each to its combined region ("teeth" -> "mouth /
teeth"), which gives exactly the cards those phases were run with.
"""
import re
from dataclasses import dataclass, replace
from functools import lru_cache

from src.config import CORPUS_DIR
from src.xai.gradcam import COMBINED, all_regions

ATOMIC_REGIONS = tuple(all_regions(atomic=True))    # the names the cards are written with
KINDS = ("artifact", "scope", "limitation", "authentic")
PATTERNS = ("central", "outer", "any")
FAMILIES = {
    "swap_autoencoder": "face swap with an autoencoder (Deepfakes, Celeb-DF)",
    "swap_graphics": "graphics-based face swap (FaceSwap)",
    "swap_gan": "GAN-based face swap (FaceShifter)",
    "reenact_graphics": "graphics-based reenactment (Face2Face)",
    "reenact_neural": "neural-rendering reenactment (NeuralTextures)",
    "synthetic": "fully synthetic GAN face (e.g. StyleGAN)",
}
# Manipulation names in the datasets -> family; used only for evaluation, never shown to the LLM
METHOD_FAMILY = {"Deepfakes": "swap_autoencoder", "CelebDF": "swap_autoencoder", "FaceSwap": "swap_graphics",
                 "FaceShifter": "swap_gan", "Face2Face": "reenact_graphics", "NeuralTextures": "reenact_neural"}


@dataclass(frozen=True)
class Doc:
    id: str
    artifact_type: str        # artifact label, "none" for authentic cards, "" for limitation cards
    kind: str
    regions: tuple
    pattern: str
    families: tuple           # ("any",) or family keys
    citation: str
    text: str
    confidence: tuple = ()
    checked: bool = False
    file: str = ""

    def applies_to(self, family: str) -> bool:
        return "any" in self.families or family in self.families


def _split_list(value: str) -> tuple:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _parse_card(block: str, file: str) -> Doc:
    head, _, text = block.partition("\n\n")
    lines = head.strip().splitlines()
    card_id, meta = lines[0].strip(), {}
    for line in lines[1:]:
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{file} [{card_id}]: expected 'key: value', got {line!r}")
        meta[key.strip()] = value.strip()
    missing = {"artifact", "kind", "regions", "pattern", "families", "source"} - set(meta)
    if missing:
        raise ValueError(f"{file} [{card_id}]: missing {sorted(missing)}")
    regions = ATOMIC_REGIONS if meta["regions"] == "all" else _split_list(meta["regions"])
    doc = Doc(id=card_id, artifact_type="" if meta["artifact"] == "-" else meta["artifact"], kind=meta["kind"],
              regions=regions, pattern=meta["pattern"], families=_split_list(meta["families"]),
              citation=meta["source"], text=" ".join(text.split()),
              confidence=_split_list(meta.get("confidence", "")), checked=meta.get("checked", "no") == "yes",
              file=file)
    problems = [f"kind {doc.kind!r}" for _ in [0] if doc.kind not in KINDS]
    problems += [f"region {r!r}" for r in doc.regions if r not in ATOMIC_REGIONS]
    problems += [f"pattern {doc.pattern!r}" for _ in [0] if doc.pattern not in PATTERNS]
    problems += [f"family {f!r}" for f in doc.families if f != "any" and f not in FAMILIES]
    if (doc.kind == "authentic") != (doc.artifact_type == "none") or (doc.kind == "limitation") != (not doc.artifact_type):
        problems.append("artifact does not match kind")
    if not doc.text:
        problems.append("no text")
    if problems:
        raise ValueError(f"{file} [{card_id}]: invalid {', '.join(problems)}")
    return doc


@lru_cache
def load_corpus(atomic: bool = False) -> tuple:
    """Every card. atomic=False (phases 3-5): regions mapped to the combined 9, in the card's order."""
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        blocks = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
        docs += [_parse_card(b, path.name) for b in blocks]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate card ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    if not atomic:
        docs = [replace(d, regions=tuple(dict.fromkeys(COMBINED[r] for r in d.regions))) for d in docs]
    return tuple(docs)


def cards_by_id(atomic: bool = False) -> dict:
    return {d.id: d for d in load_corpus(atomic)}


def artifact_taxonomy(atomic: bool = False) -> dict:
    """artifact label -> set of facial regions where it is expected (union of its artifact cards' regions;
    `none` -> the authentic cards' regions). The labels are the fixed list the LLM chooses from."""
    taxonomy = {}
    for d in load_corpus(atomic):
        if d.kind in ("artifact", "authentic"):
            taxonomy.setdefault(d.artifact_type, set()).update(d.regions)
    return taxonomy
