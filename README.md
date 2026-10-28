# Explainable Deepfake Detector

Real/fake face classifier with visual and natural-language explanations, plus a measurable check on whether
those explanations are grounded or hallucinated.

| Phase | What | Where |
|---|---|---|
| 1 | EfficientNet-B0 classifier vs. 5 baselines; trained on FF++, tested on FF++ and Celeb-DF | `src/classifier/` |
| 2 | Grad-CAM++ localisation + facial-region attribution | `src/xai/gradcam.py` |
| 3 | Ungrounded LLM explanation (GPT-OSS-120B/20B and Qwen3.8-27B on Groq's free tier; other providers also work) | `src/explain/llm.py`, `src/explain/providers.py` |
| 4 | RAG-grounded explanation: evidence cards, hybrid retrieval, cited answers, verifier | `corpus/`, `src/explain/retriever.py`, `src/eval/verify.py` |
| 5 | Held-out evaluation: ungrounded vs. grounded vs. verified, paired tests, no-LLM baselines | `src/eval/experiment.py`, `src/eval/stats.py` |
| 6 | Demo for the exhibition (Gradio): live analysis + offline gallery | `app.py`, `src/demo/` |

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
The explanation step works with the Groq API (free tier; used for phase 3), Anthropic, OpenRouter, a local
model, or any OpenAI-compatible server. Set these environment variables (on Kaggle, the notebooks set them for you).
Locally you can instead put them in a `.env` file in the project root, one `NAME=value` per line (git-ignored;
variables already set in the environment take precedence):

| Provider | `LLM_PROVIDER` | Key | Default `LLM_MODEL` |
|---|---|---|---|
| Groq (free tier) | `groq` | `GROQ_API_KEY` (console.groq.com) | `openai/gpt-oss-120b` (also used: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) |
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Ollama (local) | `ollama` | none | `llama3.1:8b` |
| LM Studio / vLLM / llama.cpp / OpenAI | `openai` + `LLM_BASE_URL` | `OPENAI_API_KEY` (not needed for local servers) | `gpt-4o-mini` |

`LLM_API_KEY` is used when the provider's own key variable is not set. Keep the same model for both conditions in
a comparison run; the model name is recorded in `outputs/eval/summary.json`.

For free tiers, `LLM_MIN_INTERVAL` spaces requests (seconds) to stay under per-minute limits, and API calls are
retried with backoff on rate limits and transient server errors. `LLM_REASONING_EFFORT` sets the effort for models
that reason before answering (Groq defaults to `low`; `off` sends nothing) and `LLM_MAX_TOKENS` the output limit
(1024 for Groq, which leaves room for reasoning tokens; 600 otherwise).
Groq's free tier allows, per model, 30 requests and 8K tokens per minute and 1K requests and 200K tokens per day;
notebook 05 spaces requests 12 s apart to stay under the per-minute token limit. Qwen3.8-27B is also capped at 1K
output tokens per minute, and Groq refuses any request whose output limit is above that, so notebook 05 gives it a
900-token limit and 30 s spacing.

Temperature: Groq, OpenRouter, Ollama and other OpenAI-compatible servers use 0, so repeated runs give (nearly) the
same answers; Anthropic uses the API default, so its answers can vary between runs.

## Ungrounded explanations (phase 3)
The LLM never sees the image. It receives only an **evidence summary** filled from a fixed template: the
verdict, P(fake), the decision threshold, a confidence level (high / moderate / borderline, by how far the score is
past the threshold) and the top-3 regions with their concentration and share of heat. It must answer in a fixed
JSON schema: a `claimed_region` and an `artifact_type` from fixed lists (the region set and the corpus taxonomy),
plus 2–4 plain sentences. For a REAL verdict the artifact is `none` and the explanation says which regions the
classifier relied on. Because the labels come from fixed lists, every claim can be checked against the heatmap
automatically (phase 5).

Each answer records whether it is valid (`schema_ok`: JSON with allowed labels), the `provider:model`, the
latency and the raw reply. Phase 4 adds retrieved evidence cards and the instruction to cite them; the evidence
summary, labels and output schema stay the same.

### Pilot results (notebook 05)
48 pilot images (24 FF++, 24 Celeb-DF; one frame per test video), explained by each model with reasoning effort
*low*. The detector got 39 of the 48 right (7 of the 9 errors are on Celeb-DF, 4 at high confidence); the
explanations describe its verdict, right or wrong.

| | GPT-OSS-120B | GPT-OSS-20B | Qwen3.8-27B |
|---|---|---|---|
| Valid output | 100% | 100% | 98% |
| Region in top-3 | 100% | 100% | 98% |
| Pointing hit | 65% | 63% | 73% |
| Region IoU (mean) | 0.19 | 0.21 | 0.20 |
| Artifact plausible for the regions | 94% | 96% | 88% |
| Verdict consistent | 100% | 100% | 98% |
| Mentions uncertainty on borderline verdicts | 1 of 5 | 0 of 5 | 5 of 5 |
| Describes the image as if seen (unhedged) | 0% | 4% | 19% |
| GAN-fingerprint claims (fake verdicts) | 0 of 31 | 2 of 31 | 8 of 30 |
| Seconds per explanation | 0.8 | 0.8 | 26 |

- **The artifact is a guess.** On the 30 images all three models called fake, they agree on the artifact type
  only slightly (Fleiss' κ = 0.16); 11 of the 30 get three different types. GPT-OSS-20B names teeth/mouth
  artefacts for 21 of 31 fakes.
- **The easy checks are nearly saturated.** Region-in-top-3, verdict consistency and artifact plausibility are
  close to 100% because the prompt lists the regions and most artefact types fit a central-face heatmap.
  Pointing hit and IoU mostly reflect how often a model names the large mouth region; they need trivial baselines
  (e.g. "always name the top region") before they say anything about the LLM.
- **Qwen invents visual detail** ("natural skin texture… consistent tooth alignment") and claims GAN fingerprints
  on face swaps and re-enactments, which the corpus describes as a cue for fully synthetic faces. It is also the
  only model that hedges on borderline verdicts. One FF++ NeuralTextures image gets an empty reply from Qwen in
  both runs (reasoning uses up the output limit).
- **Runs are not exactly repeatable.** Two runs with the same settings (temperature 0) differed by up to 6 points
  on some rates, so differences of that size between conditions are within run-to-run noise.

The visual-claim, uncertainty and GAN counts come from keyword matching on the explanations, so they are
approximate; the other rates are computed by notebook 05.

## Grounded explanations (phase 4)
Same images, evidence summaries, models and settings as phase 3; the prompt adds **evidence cards** and the
instruction to use and cite them. A third condition, **verified**, checks each grounded answer and asks once for a
revision when a check fails.

```
evidence summary ──► query (fixed template) ──► retriever ──► 4–7 cards ──► LLM (Groq) ──► JSON + citations ──► verifier
                                                  ▲                                                              │
                                     corpus/ (35 evidence cards)                         revision request if a check fails
```

**Evidence cards (`corpus/*.md`).** Each card makes one claim in 1–3 sentences and carries metadata: the artefact
label it is about, its kind, the facial regions and pattern (central/outer) it applies to, the manipulation
families it concerns, and its source. Kinds: *artifact* (where an artefact appears; supports that label for its
regions), *scope* (when an artefact does not apply, e.g. GAN fingerprints are a cue for fully synthetic images,
not face swaps), *limitation* (what the evidence cannot show, e.g. Grad-CAM shows where, not why) and *authentic*
(evidence for a REAL verdict). The labels are the same ten as in phase 3, so phase 3's prompt is unchanged; the
taxonomy (regions per label) is now the union of each label's artifact cards, which removes `nose` from the GAN
fingerprint label and leaves the others as before. The card format is described in `src/explain/corpus.py`.
Every card is marked `checked: no` until its claim has been compared with the cited paper.

**Retriever (`src/explain/retriever.py`).**
1. *Filter* by verdict and regions: FAKE → artifact cards that list one of the heatmap's top-3 regions; REAL →
   authentic cards.
2. *Rank* with three rankers fused by reciprocal rank fusion (no weights to tune): region overlap weighted by rank,
   BM25 keyword match, and cosine similarity of `BAAI/bge-small-en-v1.5` embeddings (run locally with
   `transformers`; the 35 card embeddings are computed once and kept in memory, no vector database).
3. *Diversify*: the best card of each of 3 artefact types plus their scope cards (REAL: the best 2 authentic
   cards), then one limitation card (`lim-generalisation` for moderate or borderline confidence).

The query is filled from the evidence by a fixed template, so retrieval is deterministic. Notebook 06 compares
the rankers alone and combined, and a plain top-3 without the filter and diversity.

**Verifier (`src/eval/verify.py`).** Deterministic checks against the cards the answer cites (citations provided,
artefact supported by a cited artifact card, region listed on it), the heatmap checks from phase 3, and
sentence-level NLI with `cross-encoder/nli-deberta-v3-base`: each sentence must be entailed by the evidence summary
plus the cited cards. NLI measures attribution, not truth: a correct statement from the model's own knowledge
counts as unsupported, and hedged interpretations are often judged neutral (on phase 3's answers it marks only
13–29% of sentences as supported). The verified condition therefore requests a revision on failed deterministic
checks and on *contradicted* sentences; adding `"neutral"` to `REVISE_ON` makes it stricter. Before quoting NLI
rates, label the exported sentence sample by hand and compare.

Notebook 06 also scores phase 3's answers with the same checks (with the new taxonomy) and reports agreement
between models on the artefact type (Fleiss' κ), **method consistency** (the claimed artefact is documented for
the image's true manipulation family, which the LLM never sees) and the rate of GAN-fingerprint claims.

## Evaluation (phase 5)
Notebook 07 runs the frozen phase 3 and phase 4 pipeline on a **held-out test sample**. The pilot was used for
tuning (output limits, citation parsing, when to revise), so its numbers are not reported as results.
- **Sample:** one frame per test video from videos the pilot and worked examples did not use, balanced by
  manipulation method. The default is 48 per dataset, 96 images in total.
- **Conditions:** every image is explained by all three models in all three conditions (ungrounded, grounded,
  verified), so every comparison is paired by image.
- **Runtime:** the sample runs in 3 daily batches of about 140K Groq tokens per model.

**Metrics** (`src/eval/experiment.py`):

| Group | Metrics |
|---|---|
| output | `valid` |
| heatmap faithfulness | `region_in_top3`, `pointing_hit`, `region_iou` |
| artefact | `verdict_consistent`, `artifact_plausible`, `method_consistent` (fits the true manipulation family, which the LLM never sees), `gan_claim` |
| attribution | `supported_share` / `contradicted_share` (NLI against the evidence summary + cited cards), `citation_valid`, `artifact_in_cited_card`, `region_in_cited_card` |
| language (keyword checks, approximate) | `hedges_when_uncertain` (moderate/borderline verdicts), `overclaims` (certainty words), `visual_claim` (describes the image as if seen) |
| agreement | Fleiss' κ between models on the artefact type, on the same images in every condition |

**Baselines** use no LLM and are scored with the same checks:
- *top region + first card*: the heatmap's top region and the retriever's best artefact card, i.e. the evidence
  pipeline alone
- *always mouth*: the most common pilot answer
- *random plausible*: the exact expected score of a random top-3 region with an artefact that fits it

They show whether a rate means anything, for example whether the LLM points better than simply naming the top
region.

**Statistics** (`src/eval/stats.py`):
- **Tests:** paired by image. Yes/no checks use McNemar's exact test; shares and IoU use the Wilcoxon signed-rank
  test with rank-biserial r.
- **Intervals:** 95% bootstrap intervals over images for every mean and difference.
- **Corrections:** Holm within each family of tests.
- **Primary hypotheses** (pre-registered): grounded vs. ungrounded, per model, on `supported_share`,
  `contradicted_share`, `method_consistent` and `gan_claim`, Holm over those 12 tests.
- **Secondary:** verified vs. grounded, the other metrics, LLM vs. baselines, and breakdowns by dataset,
  verdict correctness and confidence.
- **NLI check:** a stratified sentence sample is exported for checking the NLI model by hand, and any labelled
  sample is compared with it (agreement, Cohen's κ).

**What the pilot suggests about power.** On the 48 pilot images, the same primary tests are significant for
`supported_share` in all three models (+0.16, +0.11 and +0.32, Holm p ≤ 0.02). The fall in contradicted
sentences (−0.04 to −0.05) is significant before the correction (p = 0.016) but not after it. The changes in
`method_consistent` and `gan_claim` are too small or too rare to test at this size. The test set is twice the
pilot.

`python -m src.eval.run_eval --n 50` is an older quick check (one model, ungrounded vs. grounded, heatmap and
hallucination flags only).

## Demo (phase 6)
```bash
python app.py [--checkpoint path/to/best.pt] [--share]     # http://localhost:7860
```
A Gradio app (`app.py`, `src/demo/`) with two pages:
- **Live analysis.** Upload a face image or a video frame and pick the explanation model (GPT-OSS-120B,
  GPT-OSS-20B or Qwen3.8-27B on Groq). A progress strip under the upload box lights up Detect → Heatmap →
  Retrieve → Explain → Check, with each step's time, and the panels fill in as the steps finish:
  - the verdict with P(fake), the decision threshold and the confidence level
  - the face the detector saw, the Grad-CAM++ heatmap with the hottest area boxed, and every facial-region box
    coloured by its heat concentration (blue = cold, red = hottest) and labelled with it, next to a list of all 9
    regions ranked by heat. They show what made the detector call the face fake (or, for a REAL verdict, real)
  - the evidence summary, which is all the LLM reads. Unlike phases 3–5, which list only the top 3 regions, the
    demo's summary lists all 9 regions with their concentration and share of heat. Card retrieval still uses the
    top 3
  - the evidence cards retrieved from the corpus (cited cards highlighted)
  - the grounded explanation: region, artefact, the sentences, the cited card ids, and a one-line check that the
    citations were given to the model, the artefact is backed by a cited card and where the named region ranks
    by heat (flagged when it is cooler than the image average)
- **Gallery · offline.** Test images analysed in advance by every model, so this page works without internet or
  an API key. The real label (dataset, manipulation method, and whether the detector was right) is shown with
  each result. It includes one or two images the detector gets wrong.

The live page needs EfficientNet-B0's `best.pt` (default `outputs/checkpoints/best.pt`) and `GROQ_API_KEY` (set in
the environment or in `.env`).
The gallery (`outputs/demo/gallery/`) is built on Kaggle by notebook 08, or locally with
`python -m src.demo.pipeline --data FF++=<face crops> CelebDF=<face crops>`.

## Running on Kaggle
Done: model training and evaluation (01–03), Grad-CAM++ explainability (04), choosing the main detector,
EfficientNet-B0 (04b), and ungrounded LLM explanations (05; prompt and settings frozen as the phase 4 baseline).
RAG-grounded explanations (06) are done on the pilot. Current phase: **evaluation** (notebook 07, one batch per
day). The demo (08) builds the offline gallery and launches the app with a public link.

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
| `06_grounded_explanations` | 01, 01b, 02 (the copy with EfficientNet-B0), 05 (committed output); secret `GROQ_API_KEY` | GPU T4 (detector, embeddings, NLI) | retriever comparison; grounded and verified explanations from the same three models on the same images; all three conditions scored with the same checks; model agreement; a sentence sample for checking the NLI model by hand. One run uses up to ~180K Groq tokens per model, so run each model at most once per day (`RUN_TAG`); `RUN_MODELS` re-runs only some models and loads the others from an attached earlier output. Qwen gets a 1,000-token output limit and 60 s spacing (Groq caps it at 1K output tokens per minute) |
| `07_evaluation` | 01, 01b, 02 (the copy with EfficientNet-B0), 05 (committed output); from batch 1 on the previous output of 07; secret `GROQ_API_KEY` | GPU T4 (detector, embeddings, NLI) | held-out test sample (96 images) explained by the three models in all three conditions, run as `BATCH` 0, 1, 2 on separate days (each run carries the earlier batches forward); per-answer scores, no-LLM baselines, paired tests with bootstrap intervals and Holm correction, model agreement, breakdowns, a forest plot and an NLI check sample. `BATCH = None` re-runs only the analysis |
| `08_demo` | 01, 01b, 02 (the copy with EfficientNet-B0); secret `GROQ_API_KEY` | GPU T4 | the offline gallery (one correctly classified test image per manipulation method and dataset plus up to two the detector gets wrong, each explained by all three models), then the demo app with a temporary public link. Download `outputs/demo/gallery/` to run the demo on your own computer |

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
src/explain/     evidence-card loader, retriever, LLM providers + explainer
src/eval/        metrics, verifier (grounding checks + NLI), phase 5 experiment (sample, baselines, scoring) and statistics
src/demo/        demo pipeline (step by step, offline gallery) and panel HTML for app.py
corpus/          evidence cards (one claim each, with regions, families and source)
notebooks/       CNN learning notebooks
data/ outputs/   (git-ignored)
```
