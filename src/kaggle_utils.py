"""Helpers for the Kaggle notebooks in kaggle/.

Each notebook saves its results to /kaggle/working; the next notebook adds that output as an input,
where it appears somewhere under /kaggle/input. `setup()` finds those files and points the project's
paths at them. Call it BEFORE importing anything else from src (paths are read at import time).
"""
import glob
import os
from pathlib import Path

INPUT = "/kaggle/input"
WORKING = Path("/kaggle/working")


def find(pattern: str, max_depth: int = 6):
    """Shallowest match for `pattern` under /kaggle/input (avoids walking huge image folders)."""
    for depth in range(max_depth + 1):
        hits = sorted(glob.glob(os.path.join(INPUT, *(["*"] * depth), pattern)))
        if hits:
            return Path(hits[0])
    return None


def find_all(pattern: str, max_depth: int = 6):
    """Every match for `pattern` under /kaggle/input up to `max_depth` folders deep."""
    hits = []
    for depth in range(max_depth + 1):
        hits += glob.glob(os.path.join(INPUT, *(["*"] * depth), pattern))
    return sorted(Path(h) for h in set(hits))


def find_datasets() -> dict:
    """Face-crop folders attached as inputs: {'FF++': path, 'CelebDF': path} (by the methods in manifest.csv)."""
    found = {}
    for manifest in find_all("manifest.csv"):
        with open(manifest) as f:
            methods = {line.split(",")[2] for line in list(f)[1:] if line.count(",") >= 5}
        found.setdefault("CelebDF" if "CelebDF" in methods else "FF++", manifest.parent)
    return found


def checkpoint_roots() -> list:
    """Every 'checkpoints' folder from attached training notebooks."""
    return [p for p in find_all("checkpoints", max_depth=4) if p.is_dir()]


def setup(data: bool = True, checkpoint: bool = False, data_dir: str = None):
    os.environ.setdefault("DFD_OUTPUT_DIR", str(WORKING / "outputs"))
    if data_dir:
        os.environ["DFD_DATA_DIR"] = data_dir
    elif data:
        ffpp = find_datasets().get("FF++")
        if ffpp is None:
            raise FileNotFoundError("FF++ face crops not found: add the output of 01_extract_faces as an input.")
        os.environ["DFD_DATA_DIR"] = str(ffpp)
    if checkpoint:
        ckpt = find(os.path.join("checkpoints", "best.pt"))
        if ckpt is None:
            raise FileNotFoundError("Model not found: add the output of 02_train as an input.")
        os.environ["DFD_CHECKPOINT"] = str(ckpt)
    for key in ("DFD_DATA_DIR", "DFD_CHECKPOINT", "DFD_OUTPUT_DIR"):
        if key in os.environ:
            print(f"{key:15} = {os.environ[key]}")


def load_secret(name: str, quiet: bool = False) -> bool:
    """Copy a Kaggle Secret (Add-ons -> Secrets) into an environment variable."""
    if os.environ.get(name):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ[name] = UserSecretsClient().get_secret(name)
        return True
    except Exception as e:
        if not quiet:
            print(f"Secret {name} not available ({type(e).__name__}).")
        return False


def setup_llm(provider: str = "anthropic", model: str = None) -> bool:
    """Configure the explanation LLM from notebook settings + Kaggle Secrets.

    Store whichever key you use as a secret: LLM_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY or OPENAI_API_KEY.
    """
    os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model
    for name in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL"):
        load_secret(name, quiet=True)
    from src.explain import providers

    ok = providers.available()
    print(f"LLM: {providers.describe()} ({'ready' if ok else 'missing API key'})")
    return ok


def start_ollama(model: str = "llama3.1:8b"):
    """Install and start Ollama inside the Kaggle session and pull `model` (runs on the session's GPU)."""
    import subprocess
    import time
    import urllib.request

    if subprocess.run("command -v ollama", shell=True, capture_output=True).returncode:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434")
            break
        except Exception:
            time.sleep(1)
    subprocess.run(["ollama", "pull", model], check=True)
    print(f"Ollama serving {model} at http://localhost:11434")
