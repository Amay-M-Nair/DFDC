# Explainable Deepfake Detector

Real/fake face classifier with visual and natural-language explanations, plus a measurable check on whether
those explanations are grounded or hallucinated.

| Phase | What | Where |
|---|---|---|
| 1 | EfficientNet-B0 classifier vs. 5 baselines; trained on FF++, tested on FF++ and Celeb-DF | `src/classifier/` |
| 2 | Grad-CAM++ localisation + facial-region attribution | `src/xai/gradcam.py` |
| 3 | Ungrounded LLM explanation (GPT-OSS-120B/20B and Qwen3.8-27B on Groq's free tier; other providers also work) | `src/explain/llm.py`, `src/explain/providers.py` |
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
python -m src.classifier.train --epochs 8                   # EfficientNet-B0 -> outputs/checkpoints/best.pt
python -m src.classifier.evaluate                          # outputs/metrics/ (+ per-method and video-level AUC for FF++)
```

### Model comparison and cross-dataset test
Protocol: **train on FaceForensics++**, test on the FF++ test split (in-dataset) and on the **Celeb-DF v2 official
test split** (cross-dataset, never seen in training). All models share the same data, 224px input, augmentation,
schedule and early stopping; each model's decision threshold is tuned on FF++ validation only.

| `--backbone` | Model |
|---|---|
| `efficientnet_b0` | EfficientNet-B0 (proposed) |
| `meso4` | MesoNet-4 (Afchar et al. 2018), trained from scratch |
| `legacy_xception` | Xception (FaceForensics++ baseline) |
| `resnet50` | ResNet-50 |
| `efficientnet_b4` | EfficientNet-B4 |
| `vit_small_patch16_224` | ViT-S/16 |
| `convnext_tiny` | ConvNeXt-T (modern CNN) |

```bash
python -m src.classifier.train --backbone legacy_xception --epochs 8     # -> outputs/checkpoints/legacy_xception/
python -m src.classifier.celebdf_extract --src path/to/Celeb-DF-v2 --out data_celebdf
python -m src.classifier.compare --roots outputs/checkpoints --datasets FF++=data CelebDF=data_celebdf
```
`compare` writes `outputs/compare/`: AUC with 95% CIs (bootstrap over videos), accuracy/F1, video-level AUC,
per-method AUC, params/GFLOPs/throughput/training time, paired significance tests for every pair of models, and figures
(ROC, accuracy vs. compute, in-dataset vs. cross-dataset AUC).

## Explain one image
```bash
python -m src.xai.gradcam --image path/to/face.jpg --checkpoint outputs/checkpoints/convnext_tiny/best.pt
```
Facial regions (eyes, nose, mouth, cheeks, forehead, chin) are placed from YuNet's 5 landmarks on each face.
Regions are ranked by **concentration** (mean heat inside the region relative to the image mean), so large
regions such as the background or forehead don't come first just for their size; `rank_by="share"` ranks by
share of total heat instead.

### Heatmap evaluation (all models)
```bash
python -m src.xai.evaluate_cam --roots outputs/checkpoints --datasets FF++=data CelebDF=data_celebdf --n 300
```
Deletion/insertion faithfulness (Grad-CAM vs. Grad-CAM++ vs. random, paired Wilcoxon tests) and per-region
attribution for every model and manipulation, in `outputs/gradcam/`.

## Choosing the LLM
The explanation step works with the Gemini and Groq APIs (both have free tiers), Anthropic, OpenRouter, Hugging
Face models on the local GPU, a local model, or any OpenAI-compatible server. Set these environment variables
(on Kaggle, the notebooks set them for you):

| Provider | `LLM_PROVIDER` | Key | Default `LLM_MODEL` |
|---|---|---|---|
| Google Gemini API (free tier) | `gemini` | `GEMINI_API_KEY` (aistudio.google.com) | `gemini-3.8-flash` |
| Groq (free tier) | `groq` | `GROQ_API_KEY` (console.groq.com) | `openai/gpt-oss-120b` (also used: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) |
| Hugging Face (local GPU, `transformers`) | `hf` | `HF_TOKEN` (only for gated models such as Llama) | `Qwen/Qwen2.5-7B-Instruct` |
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Ollama (local) | `ollama` | none | `llama3.1:8b` |
| LM Studio / vLLM / llama.cpp / OpenAI | `openai` + `LLM_BASE_URL` | `OPENAI_API_KEY` (not needed for local servers) | `gpt-4o-mini` |

`LLM_API_KEY` is used when the provider's own key variable is not set. Keep the same model for both conditions in
a comparison run; the model name is recorded in `outputs/eval/summary.json`.

For free tiers, `LLM_MIN_INTERVAL` spaces requests (seconds) to stay under per-minute limits, and API calls are
retried with backoff on rate limits and transient server errors. `LLM_REASONING_EFFORT` sets the effort for models
that reason before answering (Gemini and Groq default to `low`; `off` sends nothing) and `LLM_MAX_TOKENS` the
output limit (2048 for Gemini and 1024 for Groq, which leaves room for reasoning tokens; 600 otherwise).
Groq's free tier allows, per model, 30 requests and 8K tokens per minute and 1K requests and 200K tokens per day;
notebook 05 spaces requests 12 s apart to stay under the per-minute token limit. Qwen3.8-27B is also capped at 1K
output tokens per minute, and Groq refuses any request whose output limit is above that, so notebook 05 gives it a
900-token limit and 30 s spacing.

Temperature: Groq, OpenRouter, Ollama and other OpenAI-compatible servers use 0, so repeated runs give (nearly) the
same answers. Gemini keeps its default of 1.0, because Google advises against lowering it for Gemini 3 models,
and Anthropic uses the API default; their answers can vary between runs. For `hf`, decoding is greedy,
`LLM_LOAD_4BIT=1` loads models too big for the GPU in fp16 (needs `bitsandbytes`) and `LLM_BATCH` sets how many
prompts are generated together (default 8).

## Ungrounded explanations (phase 3)
The LLM never sees the image. It receives only an **evidence summary** filled from a fixed template: the
verdict, P(fake), the decision threshold, a confidence level (high / moderate / borderline, by how far the score is
past the threshold) and the top-3 regions with their concentration and share of heat. It must answer in a fixed
JSON schema: a `claimed_region` and an `artifact_type` from fixed lists (the region set and the corpus taxonomy),
plus 2–4 plain sentences. For a REAL verdict the artifact is `none` and the explanation says which regions the
classifier relied on. Because the labels come from fixed lists, every claim can be checked against the heatmap
automatically (phase 5).

Each answer records whether it is valid (`schema_ok`: JSON with allowed labels), the `provider:model`, the
latency and the raw reply. Phase 4 adds retrieved forensic passages to the same prompt; nothing else changes.

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
Current phase: **ungrounded LLM explanations** (notebook 05), after model training and evaluation (01–03),
Grad-CAM++ explainability (04) and choosing the main detector, EfficientNet-B0 (04b).
The faithfulness evaluation and the demo are parked in `kaggle/later/`.

Each notebook's saved output is added as an input to the next one (**Save Version → Save & Run All (Commit)**,
then **Add Input → Your Work → Notebooks**).

| Notebook | Inputs | Accelerator | Produces |
|---|---|---|---|
| `01_extract_faces` | FF++ video dataset | None (CPU) | FF++ face crops (train/val/test) |
| `01b_extract_celebdf` | Celeb-DF v2 video dataset | None (CPU) | Celeb-DF official test-set face crops |
| `02_train` | 01 | GPU T4 | all models' checkpoints + training curves |
| `03_evaluate` | 01, 01b, 02 (+ copies of 02) | GPU T4 | comparison tables, significance tests, figures |
| `04_gradcam` | 01, 01b, 02 (+ copies of 02) | GPU T4 | heatmap gallery, deletion/insertion faithfulness, region attribution |
| `04b_model_selection` | 03, 04 (committed outputs) | None | accuracy vs. explainability charts, pre-declared scorecard to pick the main model |
| `05_explanations` | 01, 01b, 02 (the copy with EfficientNet-B0); secret `GROQ_API_KEY` | GPU T4 (detector only) | evidence summaries + ungrounded explanations from GPT-OSS-120B, GPT-OSS-20B and Qwen3.8-27B (Groq free tier), with prompts and raw replies; worked examples and pilot consistency rates |

Training all seven models takes roughly 7–8 GPU-hours. Kaggle stops sessions at 12 h, so `02_train` skips models
that would not fit; to be safe, run two copies of it with the model list split between them. `03_evaluate` collects
checkpoints from every attached copy.

Every notebook needs **Internet on**. The project code is cloned from the repo (`dev` branch). If the repo is
private, add a `GITHUB_TOKEN` secret, or upload a zip of the project as a Kaggle dataset and attach it.
Use a T4, not a P100 (recent PyTorch builds no longer support it).

## Layout
```
src/classifier/  model, FF++ frame extraction, dataset, training, evaluation
src/faces.py     face detection + cropping (extraction and UI)
src/kaggle_utils.py  locate earlier notebooks' outputs on Kaggle
kaggle/          Kaggle notebooks for the current phase (kaggle/later/: parked phases)
src/xai/         Grad-CAM++
src/explain/     corpus loader, retriever, LLM providers + explainer
src/eval/        metrics + evaluation runner
corpus/          curated forensic-artifact notes with citations
notebooks/       CNN learning notebooks
data/ outputs/   (git-ignored)
```
