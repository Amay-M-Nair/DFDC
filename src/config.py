import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path):
    """NAME=value lines from a local .env file (git-ignored, e.g. GROQ_API_KEY) into the environment.
    Variables already set win; empty values and # comments are skipped."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if sep and name and value:
            os.environ.setdefault(name, value)


load_env(ROOT / ".env")
# Override with env vars to read data in place (e.g. read-only /kaggle/input) and write elsewhere
DATA_DIR = Path(os.environ.get("DFD_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("DFD_OUTPUT_DIR", ROOT / "outputs"))
CHECKPOINT = Path(os.environ.get("DFD_CHECKPOINT", OUTPUT_DIR / "checkpoints" / "best.pt"))
CORPUS_DIR = ROOT / "corpus"

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Label convention: 0 = real, 1 = fake (model outputs a single "fake" logit)
CLASSES = ("real", "fake")

# LLM backend for explanations: see src/explain/providers.py (LLM_PROVIDER, LLM_MODEL, ...)
