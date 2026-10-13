"""Load the curated forensic-artifact corpus (corpus/*.md with a small front-matter header)."""
from dataclasses import dataclass
from functools import lru_cache

from src.config import CORPUS_DIR


@dataclass(frozen=True)
class Doc:
    id: str
    artifact_type: str
    regions: tuple
    citation: str
    text: str


def _parse(path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    return Doc(id=path.stem, artifact_type=meta["artifact_type"],
               regions=tuple(r.strip() for r in meta["regions"].split(",")),
               citation=meta["citation"], text=body.strip())


@lru_cache
def load_corpus() -> tuple:
    return tuple(_parse(p) for p in sorted(CORPUS_DIR.glob("*.md")))


def artifact_taxonomy() -> dict:
    """artifact_type -> set of facial regions where that artifact is expected to appear."""
    return {d.artifact_type: set(d.regions) for d in load_corpus()}
