# Explainable Deepfake Detector

Real/fake face classifier with visual and natural-language explanations, plus a measurable check on whether
those explanations are grounded or hallucinated.

| Phase | What | Where |
|---|---|---|
| 1 | EfficientNet-B0 classifier (PyTorch, timm, AMP) | `src/classifier/` |
| 2 | Grad-CAM++ localisation + facial-region attribution | `src/xai/gradcam.py` |
| 3 | Ungrounded LLM explanation (Anthropic, OpenRouter or local LLM) | `src/explain/llm.py`, `src/explain/providers.py` |
| 4 | RAG-grounded explanation over a curated forensic corpus | `corpus/`, `src/explain/retriever.py` |
| 5 | Faithfulness IoU, pointing game, hallucination rate | `src/eval/` |
| 6 | Local demo UI (Gradio, localhost) | `app.py` |

## Setup
```bash
pip install -r requirements.txt
```

## Data: FaceForensics++
FF++ is a video dataset, so first extract face crops into `data/{train,val,test}/{real,fake}`:
```bash
python -m src.classifier.ff_extract --src path/to/FaceForensics++ --frames 10
```
- Works with the official layout (`original_sequences/`, `manipulated_sequences/<method>/c23/videos`) and flattened mirrors (`original/`, `Deepfakes/`, ...).
- Methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter (whichever are present). The separate DeepFakeDetection/actors subset is skipped.
- `--compression c23` (default) picks one quality level when several are present.
- **Split by identity, not by frame**, so no face appears in both train and test. Use `--splits <dir>` with the official `train/val/test.json` to reproduce the paper split.
- Real videos get more frames than each fake video to balance the classes. `--max-videos N` limits videos per method for quick runs.
- Faces are detected with OpenCV YuNet (auto-downloaded; falls back to a Haar cascade) and cropped square at 1.3x the face box.

Any other dataset already in `data/{train,val,test}/{real,fake}` image folders also works (e.g. `python -m src.classifier.prepare_data` for 140k Real and Fake Faces).

## Train and evaluate
```bash
python -m src.classifier.train --epochs 1 --subset 500     # smoke test
python -m src.classifier.train --epochs 8
python -m src.classifier.evaluate                          # outputs/metrics/ (+ per-method and video-level AUC for FF++)
```

## Explain one image
```bash
python -m src.xai.gradcam --image path/to/face.jpg        # outputs/gradcam/
```

## Choosing the LLM
The explanation step works with Anthropic, OpenRouter, a local model, or any OpenAI-compatible server.
Set these environment variables (on Kaggle, the notebooks set them for you):

| Provider | `LLM_PROVIDER` | Key | Default `LLM_MODEL` |
|---|---|---|---|
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Ollama (local) | `ollama` | none | `llama3.1:8b` |
| LM Studio / vLLM / llama.cpp / OpenAI | `openai` + `LLM_BASE_URL` | `OPENAI_API_KEY` (not needed for local servers) | `gpt-4o-mini` |

`LLM_API_KEY` works as a key for any provider. Keep the same model for both conditions in a comparison run;
the model name is recorded in `outputs/eval/summary.json`.

## Grounded vs. ungrounded evaluation
```bash
export LLM_PROVIDER=openrouter OPENROUTER_API_KEY=...    # or ANTHROPIC_API_KEY=..., or LLM_PROVIDER=ollama
python -m src.eval.run_eval --n 50                          # outputs/eval/
```
Metrics (per condition):
- **iou**: IoU between the facial region the explanation names and the top-20% Grad-CAM++ pixels
- **pointing_hit**: the named region contains the heatmap peak
- **region_unsupported**: the named region is not among the heatmap's top-3 regions
- **artifact_unsupported**: the claimed artifact type is unknown, or never occurs in the highlighted regions (per corpus taxonomy)
- **verdict_mismatch**: the claimed artifact contradicts the verdict (an artifact for REAL, or `none` for FAKE)
- **citation_invalid**: cites a source that was not retrieved
- **any_hallucination**: any of the above

## Demo UI
```bash
python app.py     # http://localhost:7860
```

## Running on Kaggle
The `kaggle/` notebooks run the whole system one phase at a time. Each notebook's saved output is added as an
input to the next one (**Save Version → Save & Run All**, then **Add Input → Your Work**).

| Notebook | Inputs | Accelerator | Produces |
|---|---|---|---|
| `01_extract_faces` | FF++ video dataset | None (CPU) | face crops + `manifest.csv` |
| `02_train` | 01 | GPU T4 | `checkpoints/best.pt`, training curves |
| `03_evaluate` | 01, 02 | GPU T4 | metrics, ROC, per-method AUC, worst mistakes |
| `04_gradcam` | 01, 02 | GPU T4 | heatmap examples, region attribution per method |
| `05_explanations` | 01, 02 + LLM key | GPU T4 | side-by-side grounded vs. ungrounded examples |
| `06_faithfulness_eval` | 01, 02 + LLM key | GPU T4 | IoU / hallucination rates + significance tests |
| `07_demo` | 02 | GPU T4 | public `gradio.live` demo link |

`00_all_in_one` runs every phase in a single session instead.

Every notebook needs **Internet on**. The project code comes from the repo (`backup` branch). If the repo is
private, add a `GITHUB_TOKEN` secret, or upload a zip of the project as a Kaggle dataset and attach it.
In notebooks 05–07, set `LLM_PROVIDER` / `LLM_MODEL` in the first cell. API keys come from **Add-ons → Secrets**.
With `LLM_PROVIDER = "ollama"` the notebook installs Ollama and runs the model inside the Kaggle session, so no key is needed. Use a T4, not a P100 (recent PyTorch builds no longer support it).

## Layout
```
src/classifier/  model, FF++ frame extraction, dataset, training, evaluation
src/faces.py     face detection + cropping (extraction and UI)
src/kaggle_utils.py  locate earlier notebooks' outputs on Kaggle
kaggle/          Kaggle notebooks, one per phase
src/xai/         Grad-CAM++
src/explain/     corpus loader, retriever, LLM providers + explainer
src/eval/        metrics + evaluation runner
corpus/          curated forensic-artifact notes with citations
notebooks/       CNN learning notebooks
data/ outputs/   (git-ignored)
```
