# Retrieval-Grounded and Checkable Explanations for Deepfake Detection

Code for the paper *Retrieval-Grounded and Checkable Explanations for Deepfake Detection* (Abhishek Singh,
Himanshu Khalatkar, Amay M Nair, Samraddh Pratap Singh; School of Computer Science and Engineering, VIT-AP
University; capstone project CAP4001).

The pipeline keeps detection and explanation apart. A **frozen image classifier** decides whether a face crop is
real or fake, and **Grad-CAM++** localises its evidence over nine facial regions. A **text-only LLM**, which never
sees the pixels, then explains that verdict from a templated evidence summary and must answer in fixed JSON with
region and artefact labels drawn from closed sets, so every claim can be checked automatically. In the
**grounded** condition, short forensic evidence cards picked by a hybrid retriever must be cited; in the
**verified** condition, a verifier (rule checks + sentence-level NLI) returns any failing answer for one revision.
No metric uses an LLM judge.

**Main findings (paper).**
- Seven backbones trained on FaceForensics++ (FF++): six reach 0.977–0.991 frame-level AUC on FF++ test but only
  0.756–0.798 on unseen Celeb-DF v2 test, and the two datasets rank them differently.
- On a 48-image development set, grounding raised the share of evidence-supported sentences from 0.29 / 0.14 / 0.13
  to 0.47 / 0.35 / 0.45 (GPT-OSS-120B / GPT-OSS-20B / Qwen3.8-27B; Holm-corrected p ≤ 0.003 for each), left no
  sentence contradicted, and removed Qwen's unfounded GAN-fingerprint claims (27% → 0%).
- The development set was also used for tuning, so these explanation results are preliminary; the held-out test
  (phase 5, 96 fresh images) follows.

| Phase | What | Where |
|---|---|---|
| 1 | Seven detector backbones trained on FF++, tested on FF++ and Celeb-DF v2; EfficientNet-B0 is the detector whose verdicts are explained. (Classifier **v2**, a follow-up not in the paper, is described below.) | `src/classifier/` |
| 2 | Grad-CAM++ localisation + attribution over nine facial regions | `src/xai/gradcam.py` |
| 3 | Ungrounded LLM explanation (GPT-OSS-120B/20B and Qwen3.8-27B on Groq; other providers also work) | `src/explain/llm.py`, `src/explain/providers.py` |
| 4 | Grounded explanation: 35 evidence cards, hybrid retrieval, cited answers, verifier | `corpus/`, `src/explain/retriever.py`, `src/eval/verify.py` |
| 5 | Held-out evaluation: ungrounded vs. grounded vs. verified, paired tests, no-LLM baselines | `src/eval/experiment.py`, `src/eval/stats.py` |
| 6 | Interactive demo (Gradio): live analysis + offline gallery (not evaluated in the paper) | `app.py`, `src/demo/` |

## Setup
```bash
pip install -r requirements.txt
pip install --no-deps "mediapipe>=1.2"     # demo only: face mesh for the atomic regions
```
`--no-deps` keeps MediaPipe from installing `opencv-contrib-python` next to `opencv-python`; both provide `cv2` and
overwrite each other. Its face-landmarker model (~4 MB) is downloaded on first use.

## Data: FaceForensics++
FF++ is a video dataset, so first extract face crops into `data/{train,val,test}/{real,fake}`:
```bash
python -m src.classifier.ff_extract --src path/to/FaceForensics++ --frames 10
```
- Works with the official layout (`original_sequences/`, `manipulated_sequences/<method>/c23/videos`) and flattened mirrors (`original/`, `Deepfakes/`, ...).
- Methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter (whichever are present). The separate DeepFakeDetection/actors subset is skipped.
- `--compression c23` (default) picks one quality level when several are present.
- **Split by identity, not by frame**, so no face appears in two splits. A manipulated video such as `000_003` links identities 000 and 003; every connected group of identities goes to exactly one split, and the groups are divided 72/14/14 into train/val/test (the split used in our paper). `--splits <dir>` with the official FF++ `train/val/test.json` uses the original FF++ authors' split instead.
- Real videos get more frames than each fake video to balance the classes (10 per manipulated video, 50 per real video, i.e. 10 per method). The resulting test split has 140 real and about 700 manipulated videos, 13,988 frames. `--max-videos N` limits videos per method for quick runs.
- Faces are detected with OpenCV YuNet (auto-downloaded; falls back to a Haar cascade) and cropped square at 1.3x the face box.

Any other dataset already in `data/{train,val,test}/{real,fake}` image folders also works (e.g. `python -m src.classifier.prepare_data` for 140k Real and Fake Faces).

## Train and evaluate
```bash
python -m src.classifier.train --epochs 1 --subset 500     # smoke test
python -m src.classifier.train --epochs 8                   # EfficientNet-B0 -> outputs/checkpoints/best.pt
python -m src.classifier.evaluate                          # outputs/metrics/ (+ per-method and video-level AUC for FF++)
```

### Model comparison and cross-dataset test
Protocol: **train on FaceForensics++ (c23)**, test on the FF++ test split (in-domain) and on the **Celeb-DF v2
official test list** (518 videos, up to 32 frames each, 16,575 frames; cross-dataset, never used for training, model
selection or threshold tuning). All models share the same data, 224px input, augmentation, schedule and early
stopping:
- ImageNet-pretrained `timm` backbone without its classifier, then dropout (p = 0.3) and one linear "fake" logit;
  MesoNet-4 is trained from scratch. Binary cross-entropy.
- AdamW (weight decay 0.05), batch 32, one-cycle schedule with 10% warm-up, at most 8 epochs, early stopping after
  3 epochs without a rise in validation AUC. Learning rate 3e-4 (1e-3 for MesoNet-4, 1e-4 for ViT-S/16 and
  ConvNeXt-T).
- Augmentation: random resized crop (scale 0.8–1.0), horizontal flip, colour jitter, random JPEG (p = 0.5,
  quality 40–95), random erasing (p = 0.3).
- The decision threshold τ maximises Youden's J on FF++ validation and is never re-tuned on test data.

| `--backbone` | Model |
|---|---|
| `efficientnet_b0` | EfficientNet-B0 (the detector whose verdicts are explained) |
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

### Detector results (paper, v1)
Single T4, about 6.2 GPU-hours for all seven models. Frame-level AUC with 95% intervals from a class-stratified
cluster bootstrap over videos (1,000 resamples); the 21 pairwise tests per dataset are not corrected for
multiplicity.

| Model | Params (M) | GFLOPs | Img/s (T4) | FF++ AUC | Acc. (%) | F1 | FF++ video AUC | Celeb-DF AUC | Rank FF++ → CDF |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 27.8 | 8.91 | 642 | **0.9914** | 96.03 | 0.9596 | 0.9955 | 0.7787 | 1 → 2 |
| Xception | 20.8 | 9.10 | 542 | 0.9879 | 95.07 | 0.9496 | 0.9934 | 0.7597 | 2 → 5 |
| EfficientNet-B0 | 4.0 | 0.77 | 1,623 | 0.9845 | 94.14 | 0.9397 | 0.9919 | 0.7618 | 3 → 4 |
| ResNet-50 | 23.5 | 8.17 | 799 | 0.9789 | 93.17 | 0.9304 | 0.9890 | 0.7636 | 4 → 3 |
| ViT-S/16 | 21.7 | 9.20 | 1,014 | 0.9774 | 91.98 | 0.9159 | 0.9841 | **0.7978** | 5 → 1 |
| EfficientNet-B4 | 17.6 | 3.00 | 576 | 0.9766 | 92.67 | 0.9252 | 0.9879 | 0.7560 | 6 → 6 |
| MesoNet-4 | < 0.1 | 0.09 | 7,494 | 0.7537 | 68.38 | 0.6956 | 0.7814 | 0.6309 | 7 → 7 |

- **In-domain:** ConvNeXt-T leads on FF++ and on all five methods; its lead is significant but small (+0.0035 AUC
  over Xception, p = 0.018). NeuralTextures is the hardest method for every model.
- **Cross-dataset:** modern backbones lose 0.180–0.228 AUC on Celeb-DF and are reordered (ViT-S/16 5th → 1st).
  With intervals of about ±0.03, most Celeb-DF differences are not significant; in-domain AUC did not predict
  cross-dataset AUC.
- **Thresholds transfer worse than rankings:** at ConvNeXt-T's FF++ threshold (τ = 0.93), false alarms rise from
  2.4% (FF++) to 34.0% (Celeb-DF) and missed fakes from 5.6% to 25.4%. Averaging per video lifts Celeb-DF AUC by
  0.05–0.08 (ConvNeXt-T 0.779 → 0.851).
- **Why EfficientNet-B0 is explained:** 99.3% of ConvNeXt-T's FF++ AUC with 4.0 M parameters, 0.77 GFLOPs and
  2.5× the throughput, no significant Celeb-DF difference (p = 0.174), and a convolutional last stage for
  Grad-CAM++.

## Classifier v2: generalisation, calibration and a three-way verdict (follow-up, not in the paper)
The v1 models (FF++ only) scored well on FF++ but keyed on image quality. A mildly blurred real photo scored
P(fake) ≈ 1, sharp AI-edited faces passed as real, and the decision threshold sat around 0.9–0.98, so many real
photos were called fake. v2 changes the training data, the augmentation and how the score is read:

```
real crops (FF++, Celeb-DF, FFHQ, CelebA-HQ) ──┬────────────────── real (0) ──┐
                                               └── self-blend ──── fake (1) ──┤  same degradation for all
dataset fakes (FF++ methods, Celeb-DF) ─────────────────────────── fake (1) ──┘  (blur / resize / JPEG / noise)
                                                                      │
        02a: EfficientNet-B0 + 6 baselines          02b: frozen CLIP ViT-L/14 + logistic probe
                         │                               (reals vs. StyleGAN, diffusion, FF++, Celeb-DF fakes)
                         └──── 02c: temperature, three-way thresholds, OOD check, CNN + CLIP fusion
                                                                      │
                                         REAL / UNCERTAIN / FAKE  +  out-of-distribution flag
```

- **Self-blended images** (Shiohara & Yamasaki, CVPR 2022; `src/classifier/sbi.py`):
  - A real crop is blended with a slightly altered copy of itself through a soft face mask.
  - The fake has a blending seam and a colour/sharpness mismatch, but no generator-specific artefacts, so the model learns blending traces rather than FF++'s methods.
  - The mask landmarks come from MediaPipe's face mesh, cached once per dataset (`landmarks.py`).
  - `python -m src.classifier.sbi --demo` draws original / self-blended / mask / degraded pairs.
- **More real faces, same degradation** (`augment.py`):
  - FFHQ and CelebA-HQ photos are re-cropped with the same detector and 1.3x crop as the video frames (`image_extract.py`).
  - Every training image, real or fake, gets the same random blur, downscale-upscale, JPEG, noise and sharpening, so image quality stops predicting the label.
- **Mixed training** (`dataset.MixedTrainSet`, `train.py --data NAME=root ...`):
  - Each epoch is 40% real, 35% self-blended and 25% dataset fakes, and is `--samples-per-epoch` images long. Each dataset gets the same weight within its group.
  - Celeb-DF's non-test videos are now training data (`celebdf_extract.py --with-train`), so Celeb-DF test is in-domain.
  - **DFDC** (`dfdc_extract.py`) is the cross-dataset test. It is never trained, early-stopped or calibrated on.
- **CLIP probe** (`clip_probe.py`, after Ojha et al., CVPR 2023): a logistic regression on frozen CLIP ViT-L/14 features (loaded via timm). It is trained on reals vs. StyleGAN, diffusion, FF++ and Celeb-DF fakes, and covers fully generated faces, which the CNNs never see.
- **Calibration** (`calibrate.py`, validation data only):
  - Temperature scaling for each model.
  - Two thresholds: at most 5% of validation real faces score above `t_fake` (FAKE), and at most 10% of validation fakes score below `t_real` (REAL). Anything in between is UNCERTAIN.
  - A Mahalanobis out-of-distribution check on the pooled features.
  - A logistic-regression fusion of each CNN with CLIP.

  `verdict.Detector` applies all of this to one image.
- **Evaluation** (`compare.py`, `robustness.py`):
  - AUC with video-bootstrap CIs on the two-class sets.
  - Three-way rates on every set, including the real-only (FFHQ, CelebA-HQ) and fake-only (StyleGAN, diffusion) ones: coverage, accuracy on decided images, FPR, FNR, ECE and OOD rate.
  - Robustness curves for blur, JPEG and resizing.
  - The blur test: the share of real photos called FAKE, sharp vs. blur radius 1.5.
  - v1 vs. v2.

```bash
python -m src.classifier.celebdf_extract --src path/to/Celeb-DF-v2 --out data_celebdf --with-train
python -m src.classifier.image_extract --out data_hq --source "FFHQ=real:path/to/ffhq/*.jpg" --source "StyleGAN=fake:path/to/stylegan/*.jpg"
python -m src.classifier.dfdc_extract --src path/to/dfdc --out data_dfdc
python -m src.classifier.landmarks --dataset FF++=data --dataset CelebDF=data_celebdf --dataset FFHQ=data_hq/ffhq --out landmarks
python -m src.classifier.train --data FF++=data --data CelebDF=data_celebdf --data FFHQ=data_hq/ffhq --landmarks landmarks
python -m src.classifier.clip_probe --out outputs/clip --data FF++=data --data FFHQ=data_hq/ffhq --data StyleGAN=data_hq/stylegan
python -m src.classifier.calibrate --roots outputs/checkpoints --clip outputs/clip --out outputs/calibration
python -m src.classifier.compare --roots outputs/checkpoints --datasets FF++=data FFHQ=data_hq/ffhq DFDC=data_dfdc --calibration outputs/calibration --clip-probe outputs/clip/probe.pt
```
The explanation stages (phases 2–6) still read the two-way verdict from the checkpoint's `threshold` (Youden J on
the pooled validation set). Moving them to the three-way verdict is the next step.

### Quick fix before v2: fine-tuned, calibrated v1 (`neo` branch)
`kaggle/02d_finetune_b0` fine-tunes the v1 EfficientNet-B0 for about 15 min on FF++ plus the Celeb-DF crops, under
the same blur/JPEG/resize degradation for real and fake images, then folds a validation-fitted temperature into the
model's last layer (`calibrate.py --bake`) and stores the calibrated two-way threshold. The demo, Grad-CAM and the
LLM stages read it like any other checkpoint; `app.py` offers it in a Detector menu when
`outputs/checkpoints/ft/best_calibrated.pt` exists (v1 stays the default and the Gallery stays v1).

- Celeb-DF only has its official test list extracted, so `split_crops.py` splits those crops by identity into
  train/val/test (60/15/25). A score on that test part is in-domain and isn't comparable with v1's cross-dataset
  Celeb-DF number; the phase 3–6 results are all v1.

```bash
python -m src.classifier.split_crops --src data_celebdf --out data_celebdf_split
python -m src.classifier.train --init outputs/checkpoints/best.pt --out outputs/checkpoints/ft/best.pt     --epochs 3 --lr 5e-5 --samples-per-epoch 20000 --no-sbi --data FF++=data --data CelebDF=data_celebdf_split
python -m src.classifier.calibrate --bake outputs/checkpoints/ft/best.pt     # -> ft/best_calibrated.pt
```

## Explain one image
```bash
python -m src.xai.gradcam --image path/to/face.jpg     # default: outputs/checkpoints/best.pt (EfficientNet-B0)
```
Grad-CAM++ is computed for the predicted class (the FAKE logit, negated for REAL) at the last convolutional layer
(`conv_head` in EfficientNet-B0). Nine facial regions — left eye, right eye, nose, mouth/teeth, left cheek, right
cheek, forehead/hairline, chin/jawline and background/face boundary — are placed from YuNet's 5 landmarks (a fixed
grid as fallback); a pixel in two boxes belongs to the region listed first, and pixels outside all eight face boxes
form the background (the demo uses finer regions; see Demo).
Regions are ranked by **concentration** (mean heat inside the region divided by the mean heat of the crop), so
large regions don't come first just for their size, and the top three are passed on; `rank_by="share"` ranks by
share of total heat instead.

### Heatmap evaluation (all models)
```bash
python -m src.xai.evaluate_cam --roots outputs/checkpoints --datasets FF++=data CelebDF=data_celebdf --n 300
```
Deletion/insertion faithfulness (Grad-CAM vs. Grad-CAM++ vs. random, paired Wilcoxon tests) and per-region
attribution for every model and manipulation, in `outputs/gradcam/`.

## Choosing the LLM
The explanation step works with the Groq API (free tier; used for phases 3–6), Anthropic, OpenRouter, a local
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

`LLM_API_KEY` is used when the provider's own key variable is not set. Every answer records the `provider:model`
that produced it.

For free tiers, `LLM_MIN_INTERVAL` spaces requests (seconds) to stay under per-minute limits, and API calls are
retried with backoff on rate limits and transient server errors. `LLM_REASONING_EFFORT` sets the effort for models
that reason before answering (Groq defaults to `low`; `off` sends nothing) and `LLM_MAX_TOKENS` the output limit
(1024 for Groq, which leaves room for reasoning tokens; 600 otherwise).
Groq's free tier allows, per model, 30 requests and 8K tokens per minute and 1K requests and 200K tokens per day;
notebook 05 spaces requests 12 s apart to stay under the per-minute token limit. Qwen3.8-27B is also capped at 1K
output tokens per minute, and Groq refuses any request whose output limit is above that, so notebook 05 gives it a
900-token limit and 30 s spacing; notebooks 06–08 and the demo use 1,000 tokens (60 s spacing in the notebooks).

Temperature: Groq, OpenRouter, Ollama and other OpenAI-compatible servers use 0, so repeated runs give (nearly) the
same answers; Anthropic uses the API default, so its answers can vary between runs.

## Ungrounded explanations (phase 3)
The LLM never sees the image. It receives only an **evidence summary** filled from a fixed template: the
verdict, P(fake), the decision threshold, a confidence level (high / moderate / borderline, by how far the score is
past the threshold) and the top-3 regions with their concentration and share of heat. It must answer in a fixed
JSON schema: a `claimed_region` from the nine regions, an `artifact_type` from nine artefact labels or `none`, 2–4
plain sentences and a citation list. Confidence uses the margin m = (s − τ)/(1 − τ) above the threshold or
(τ − s)/τ below it, with s = P(fake): high for m ≥ 0.5, moderate for 0.15 ≤ m < 0.5, borderline below. The system
prompt frames the model as a forensics assistant for a non-expert, says the image is unavailable, and asks for
cautious language and more hedging at borderline confidence. For a REAL verdict the artifact is `none` and the explanation says which regions the
classifier relied on. Because the labels come from fixed lists, every claim can be checked against the heatmap
automatically (phase 5).

Each answer records whether it is valid (`schema_ok`: JSON with allowed labels), the `provider:model`, the
latency and the raw reply. Phase 4 adds retrieved evidence cards and the instruction to cite them; the evidence
summary, labels and output schema stay the same.

### Development-set results (notebook 05)
The 48-image development set: one frame from each of 48 test videos — 24 FF++ (4 real, 4 per manipulation method)
and 24 Celeb-DF (12 real, 12 fake). It guided the construction of the explanation step (output limits, citation
parsing, when to request a revision), so its results are provisional. Each model runs on Groq at temperature 0
with reasoning effort *low*. The detector got 39 of the 48 right (7 of the 9 errors are on Celeb-DF, 4 at high
confidence); the explanations describe its verdict, right or wrong, so scoring uses the detector's own evidence,
not the ground truth (method consistency is the one exception).

| | GPT-OSS-120B | GPT-OSS-20B | Qwen3.8-27B |
|---|---|---|---|
| Valid output | 100% | 100% | 98% |
| Region in top-3 | 100% | 100% | 98% |
| Pointing hit | 65% | 63% | 73% |
| Region IoU (mean) | 0.19 | 0.21 | 0.20 |
| Artifact plausible for the regions | 94% | 96% | 88% |
| Verdict consistent | 100% | 100% | 98% |
| Hedges (moderate/borderline verdicts) | 5 of 8 | 1 of 8 | 7 of 8 |
| Describes the image as if seen | 0% | 4% | 10% |
| GAN-fingerprint claims (fake verdicts) | 0 of 31 | 2 of 31 | 8 of 30 |
| Seconds per explanation | 0.8 | 0.8 | 26 |

- **The artifact is a guess.** On the 30 images all three models called fake, they agree on the artifact type
  only slightly (Fleiss' κ = 0.16); 11 of the 30 get three different types. GPT-OSS-20B names teeth/mouth
  artefacts for 21 of 31 fakes.
- **The easy checks are nearly saturated.** Region-in-top-3, verdict consistency and artifact plausibility are
  close to 100% because the prompt lists the regions and most artefact types fit a central-face heatmap.
  Pointing hit and IoU mostly reflect how often a model names the mouth region, whose box is roughly twice the
  nose box (mouth claims hit the hottest pixel 70–88% of the time, nose claims 38–53%); they need trivial
  baselines (e.g. "always name the top region") before they say anything about the LLM.
- **Qwen behaves unlike the GPT-OSS pair.** It hedges most often (7 of 8 moderate or borderline verdicts), yet one
  in ten of its answers describes the image as if it were visible ("natural skin texture", "consistent tooth
  alignment"), and it claims GAN fingerprints for 8 of its 30 fake verdicts, a cue that applies only to fully
  synthetic faces, none of which are in the test set. One FF++ NeuralTextures image gets an empty reply from Qwen
  (reasoning uses up the output limit).
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
(evidence for a REAL verdict). The corpus has 19 artifact, 6 scope, 6 limitation and 4 authentic cards. Every
source is a published paper except one card that documents our own detector. The nine artefact labels are
background and hair artefacts, blending boundary, compression artefacts, eye inconsistency, lighting
inconsistency, spatial warping, spectral GAN fingerprint, teeth and mouth artefacts, and texture smoothing. The
labels (plus `none`) are the same as in phase 3, so phase 3's prompt is unchanged; the
taxonomy (regions per label) is now the union of each label's artifact cards, which removes `nose` from the GAN
fingerprint label and leaves the others as before. The card format is described in `src/explain/corpus.py`.
Every card is marked `checked: no` until its claim has been compared with the cited paper.

Card regions are written as **atomic parts**, each listed only when the claim is about it: teeth, mouth, hairline,
forehead, chin, left/right jawline, face boundary and background, plus the eyes, nose and cheeks. The demo uses them
as they are. Phases 3–5 map each part to the combined region it belongs to (teeth and mouth → `mouth / teeth`,
hairline and forehead → `forehead / hairline`, chin and jawlines → `chin / jawline`, face boundary and background →
`background / face boundary`), which gives exactly the cards, taxonomy and prompts those phases were run with.

**Retriever (`src/explain/retriever.py`).**
1. *Filter* by verdict and regions: FAKE → artifact cards that list one of the heatmap's top-3 regions; REAL →
   authentic cards.
2. *Rank* with three rankers fused by reciprocal rank fusion (no weights to tune): region overlap weighted by rank,
   BM25 keyword match, and cosine similarity of `BAAI/bge-small-en-v1.5` embeddings (run locally with
   `transformers`; the 35 card embeddings are computed once and kept in memory, no vector database).
3. *Diversify*: the best card of each of 3 artefact types plus their scope cards (REAL: the best 2 authentic
   cards), then one limitation card (`lim-generalisation` for moderate or borderline confidence). FAKE verdicts
   get 4–7 cards (5 or 6 on the development set), REAL verdicts 3.

Under grounding the cards are the LLM's only forensic knowledge: an artefact may be chosen only through an artifact
card that lists the claimed region, scope and limitation cards must be obeyed, and every card used must be cited.

The query is filled from the evidence by a fixed template, so retrieval is deterministic. Notebook 06 compares
the rankers alone and combined, and a plain top-3 without the filter and diversity.

**Verifier (`src/eval/verify.py`).** Deterministic checks against the cards the answer cites (citations provided,
artefact supported by a cited artifact card, region listed on it), the heatmap checks from phase 3, and
sentence-level NLI with `cross-encoder/nli-deberta-v3-base`: each sentence must be entailed by the evidence summary
plus the cited cards. NLI measures attribution, not truth: a correct statement from the model's own knowledge
counts as unsupported, and hedged interpretations are often judged neutral (on phase 3's answers it marks only
13–29% of sentences as supported). The verified condition therefore requests one revision on failed deterministic
checks and on *contradicted* sentences; adding `"neutral"` to `REVISE_ON` makes it stricter. Before quoting NLI
rates, label the exported sentence sample by hand and compare.

The three conditions share the model, decoding parameters, prompt, schema and labels; the only difference is the
grounding block (cards plus the card instructions appended to the system prompt).

Notebook 06 also scores phase 3's answers with the same checks (with the new taxonomy) and reports agreement
between models on the artefact type (Fleiss' κ), **method consistency** (the claimed artefact is documented for
the image's true manipulation family, which the LLM never sees) and the rate of GAN-fingerprint claims.

### Development-set results (notebook 06)
The same 48 development images as phase 3. GPT-OSS outputs are capped at 1,024 tokens; Qwen at 900 tokens
(ungrounded) and 1,000 tokens (grounded), the most Groq's per-minute output limit allows for it. Even so, 9 of
Qwen's 48 grounded replies hit the cap and came back truncated (6) or empty (3).

| | GPT-OSS-120B | GPT-OSS-20B | Qwen3.8-27B |
|---|---|---|---|
| Supported sentences (NLI), ungrounded → grounded | 0.29 → **0.47** | 0.14 → **0.35** | 0.13 → **0.45** |
| Contradicted sentences | 0.04 → **0** | 0.05 → **0** | 0.03 → **0** |
| GAN-fingerprint claims (fake verdicts) | 0 → 0 | 0.06 → **0** | 0.27 → **0** |
| Artefact plausible for the regions | 0.94 → **1.00** | 0.92 → **1.00** | 0.75 → **1.00** |
| Artefact backed by a cited card (grounded) | 1.00 | 1.00 | 1.00 |
| Method consistent | 0.89 → 0.89 | 0.82 → 0.74 | 0.65 → 0.75 |
| Valid output, grounded → verified | 1.00 → 1.00 | 1.00 → 1.00 | 0.81 → 0.94 |

Plausibility differs a little from the phase 3 table because the ungrounded answers are rescored with the corpus
taxonomy.

- **Grounding makes explanations traceable.** Paired mean gains in supported share are +0.18, +0.21 and +0.31
  (95% bootstrap intervals [0.10, 0.26], [0.14, 0.27], [0.22, 0.39]), all significant (Wilcoxon signed-rank, Holm
  over the 12 primary tests; p = 0.003, < 0.001, < 0.001). No grounded answer has a contradicted sentence; that
  drop is significant before correction for both GPT-OSS models (p = 0.016) but not after, and not for Qwen.
- **Scope cards work.** Qwen's GAN-fingerprint claims fall from 27% of its fake verdicts to none, because the
  scope card confines that label to fully synthetic faces.
- **The verifier is a safety net, not a corrective.** No GPT-OSS answer failed a check, so verified equals
  grounded for them; for Qwen, revisions cut invalid replies from nine to three (the three blank ones).
- **Method consistency is mixed** (up for Qwen, flat for GPT-OSS-120B, down for GPT-OSS-20B, which blames teeth or
  mouth for 22 of 31 fakes), and the models still do not converge on the artefact (Fleiss' κ = 0.20 over the 22
  fakes every model explained). No change in method consistency or GAN-fingerprint claims survives correction.
- **Run-to-run noise.** An earlier run with Qwen capped at 900 tokens gave supported shares of 0.45, 0.25 and 0.47
  with the same behaviour on GAN claims and contradictions; differences of about 0.09 between runs are noise.

## Evaluation (phase 5)
Notebook 07 runs the frozen phase 3 and phase 4 pipeline on a **held-out test sample** (the "held-out test that
follows" in the paper). The development set was used for tuning (output limits, citation parsing, when to revise),
so its numbers are provisional.
- **Sample:** one frame per test video from videos the development set and worked examples did not use, balanced
  by manipulation method. The default is 48 per dataset, 96 images in total.
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
- *always mouth*: the most common development-set answer
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

**What the development set suggests about power.** On the 48 development images, the same primary tests are
significant for `supported_share` in all three models (+0.18, +0.21 and +0.31, Holm p ≤ 0.003). The fall in
contradicted sentences (−0.03 to −0.05) is significant before the correction for the GPT-OSS models (p = 0.016) but
not after it. The changes in `method_consistent` and `gan_claim` do not survive correction at this size. The test
set is twice the development set.

`python -m src.eval.run_eval --n 50` is an older quick check (one model, ungrounded vs. grounded, heatmap and
hallucination flags only).

## Demo (phase 6)
```bash
python app.py [--checkpoint path/to/best.pt] [--share]     # http://localhost:7860
```
A Gradio app (`app.py`, `src/demo/`) with two pages.

**Live analysis.** Upload a face image or a video frame and pick the explanation model (GPT-OSS-120B, GPT-OSS-20B
or Qwen3.8-27B on Groq). A progress strip under the upload box lights up Detect → Heatmap → Retrieve → Explain →
Check, with each step's time, and the panels fill in as the steps finish:
- the verdict with P(fake), the decision threshold and the confidence level
- the face the detector saw, the Grad-CAM++ heatmap with the hottest area boxed, and every facial-region box
  coloured by its heat concentration (blue = cold, red = hottest) and labelled with it, next to a list of all
  regions ranked by heat (the prominent ones above a divider)
- the evidence cards retrieved from the corpus, with the cited cards highlighted
- the grounded explanation: one **finding per prominent region**, hottest first, each with its own artefact type
  and the cards cited for it, then sentences that cover every finding. Each finding shows where its region ranks
  by heat and whether a cited card backs it (FAKE: an artefact card of that type listing the region; REAL: an
  authentic card listing it). A one-line check says how many prominent regions the answer covers, how many
  findings are backed, and whether it cited ids it was not given
- the evidence summary, the only input the LLM reads (a collapsed section under the results)

**Gallery · offline.** Test images analysed in advance by every model, so this page works without internet or an
API key. Each result shows the real label (dataset, manipulation method, and whether the detector was right); the
gallery includes one or two images the detector gets wrong.

**How the demo differs from phases 3–5.** The phases name a single region from the top 3 and use the 9 combined
regions; the demo explains every prominent region and uses finer regions:
- **Atomic regions:** eyes, nose, teeth, mouth, cheeks, hairline, forehead, chin, left/right jawline, face
  boundary and background. YuNet still finds and crops the face (the crop the detector was trained on), then
  MediaPipe's 478-point face mesh places every region on the crop. *Teeth* is the opening between the inner lips
  and exists only when the mouth is open (inner-lip gap above 8% of the mouth's width), so a closed mouth gets no
  teeth region and no teeth cards; the hairline is a band above the mesh, which stops at the upper forehead. If
  MediaPipe finds no face or is not installed, the boxes come from YuNet's 5 points (the combined boxes split into
  their parts), without teeth.
- **Prominent regions:** every region whose evidence is at least half as concentrated as the hottest one
  (`PROMINENT` in `src/xai/gradcam.py`). The evidence summary lists every region with its concentration and share
  of heat and names the prominent ones; retrieval searches for them (matched against the cards' atomic regions)
  and adds a card for any prominent region the others miss.

The live page needs EfficientNet-B0's `best.pt` (default `outputs/checkpoints/best.pt`) and `GROQ_API_KEY` (set in
the environment or in `.env`).
The gallery (`outputs/demo/gallery/`) is built on Kaggle by notebook 08, or locally with
`python -m src.demo.pipeline --data FF++=<face crops> CelebDF=<face crops>`.

## Limitations
- **Development set and sample size.** Tuning used the same 48 images, and repeat runs moved some ungrounded
  rates by up to 6 points and GPT-OSS-20B's grounded supported share by 9 points, so all explanation figures are
  provisional; the phase 5 test set (96 images) is the basis for conclusions.
- **Premise length.** With grounding, the cited cards are part of the NLI premise; without it, the premise is the
  evidence summary alone. Part of the rise in supported sentences may come from the longer premise rather than
  the cards' content; a length-matched placebo condition would settle this. NLI also measures attribution, not
  truth.
- **Detector shortcuts.** The FF++-trained models key partly on blur and low resolution: a mild Gaussian blur
  (radius 1.5) turns a real photo from P(fake) ≈ 0 into ≈ 1 for six of the seven models (all but MesoNet-4), while
  sharp AI-generated or AI-edited portraits stay below 0.3. EfficientNet-B0's threshold crept from 0.57 to 0.98
  over eight epochs (the stored checkpoint uses τ = 0.981), so every missed fake on the development set scored
  0.83–0.97, high but short of τ. Fully synthetic faces never appear in the training data. A faithful explanation
  of such a detector describes its shortcut; the detector's quality bounds what an explanation is worth.
  Classifier v2 (above) addresses this with blur/resize augmentation, diffusion-generated fakes (CLIP probe),
  calibration and an out-of-distribution warning; its results come from notebook 03 and are not in the paper.
- **Heatmaps look alike, so retrieval barely varies.** Grad-CAM++ maps tend to form one central blob: the nose or
  the mouth topped 45 of the 48 images, 14 of 17 REAL verdicts received the same pair of authentic cards, and only
  five card sets occurred across the 31 FAKE verdicts.
- **Other limits.** One training run per detector, so intervals reflect test sampling only, not seed variation.
  The language checks are keyword-based, and the 35 cards are marked `checked: no` until compared with their
  papers. By design the LLM cannot confirm any visual detail; its explanations support a human reviewer and are
  not forensic proof.

## Future work
- The held-out test on 96 fresh images with the twelve pre-declared primary tests, Holm correction and no-LLM
  baselines (phase 5).
- A length-matched placebo condition; a detector retrained without the blur shortcut and with a calibrated
  real/fake/uncertain verdict (classifier v2); comparison with the human artefact annotations in ExDDV; tests on
  the diffusion-generated faces of DF40; and a user study of whether people rightly overrule an incorrect verdict.

## Running on Kaggle
The notebooks run the whole project in order: face extraction (01a–01d), training, CLIP probe and calibration
(02a–02c), evaluation (03),
Grad-CAM++ (04), choosing the main detector with a pre-declared scorecard (04b, `src/xai/model_selection.py`),
ungrounded explanations (05), grounded explanations (06), the held-out evaluation (07, one batch per day) and the
demo (08), which builds the offline gallery and launches the app with a public link.

Each notebook's saved output is added as an input to the next one (**Save Version → Save & Run All (Commit)**,
then **Add Input → Your Work → Notebooks**).

| Notebook | Inputs | Accelerator | Produces |
|---|---|---|---|
| `01a_extract_ffpp` | FF++ video dataset | None (CPU) | FF++ face crops (train/val/test) |
| `01b_extract_celebdf` | Celeb-DF v2 video dataset | None (CPU) | Celeb-DF crops: the official test split, plus (v2, `WITH_TRAIN`) train/val from the other videos |
| `01c_extract_hq` | 140k Real and Fake Faces (FFHQ + StyleGAN), a CelebA-HQ set, a diffusion-face set; 01a, 01b | None (CPU) | `data_hq/{ffhq,celebahq,stylegan,diffusion}` crops + `landmarks/<dataset>.npz` for every real train/val crop |
| `01d_extract_dfdc` | DFDC videos with `metadata.json` | None (CPU) | DFDC test crops (cross-dataset test only) |
| `02a_train_classifiers` | 01a, 01b, 01c | GPU T4 | all models trained on the v2 mix, with `val_preds.npz` + `train_feats.npy` for calibration |
| `02b_train_clip` | 01a, 01b, 01c | GPU T4 | CLIP ViT-L/14 probe (`clip/probe.pt`) + validation scores |
| `02c_calibrate` | 02a (+ copies), 02b | None (CPU) | `calibration/calibration.json`: temperatures, three-way thresholds, OOD statistics, CNN + CLIP fusion |
| `03_evaluate` | 01a–01d, 02a (+ copies), 02b, 02c; optionally the v1 checkpoints | GPU T4 | leakage audit; two-way and three-way metrics on every test set incl. DFDC; robustness curves + blur test; v1 vs. v2 |
| `04_gradcam` | 01a, 01b, 02a (+ copies) | GPU T4 | heatmap gallery, deletion/insertion faithfulness, region attribution |
| `04b_model_selection` | 03, 04 (committed outputs) | None | accuracy vs. explainability charts, pre-declared scorecard to pick the main model (v2 adds DFDC AUC and real-photo false alarms) |
| `05_explanations` | 01a, 01b, 02a (the copy with EfficientNet-B0); secret `GROQ_API_KEY` | GPU T4 (detector only) | evidence summaries + ungrounded explanations from GPT-OSS-120B, GPT-OSS-20B and Qwen3.8-27B (Groq free tier), with prompts and raw replies; worked examples and development-set consistency rates |
| `06_grounded_explanations` | 01a, 01b, 02a (the copy with EfficientNet-B0), 05 (committed output); secret `GROQ_API_KEY` | GPU T4 (detector, embeddings, NLI) | retriever comparison; grounded and verified explanations from the same three models on the same images; all three conditions scored with the same checks; model agreement; a sentence sample for checking the NLI model by hand. One run uses up to ~180K Groq tokens per model, so run each model at most once per day (`RUN_TAG`); `RUN_MODELS` re-runs only some models and loads the others from an attached earlier output. Qwen gets a 1,000-token output limit and 60 s spacing (Groq caps it at 1K output tokens per minute) |
| `07_evaluation` | 01a, 01b, 02a (the copy with EfficientNet-B0), 05 (committed output); from batch 1 on the previous output of 07; secret `GROQ_API_KEY` | GPU T4 (detector, embeddings, NLI) | held-out test sample (96 images) explained by the three models in all three conditions, run as `BATCH` 0, 1, 2 on separate days (each run carries the earlier batches forward); per-answer scores, no-LLM baselines, paired tests with bootstrap intervals and Holm correction, model agreement, breakdowns, a forest plot and an NLI check sample. `BATCH = None` re-runs only the analysis |
| `08_demo` | 01a, 01b, 02a (the copy with EfficientNet-B0); secret `GROQ_API_KEY` | GPU T4 | the offline gallery (one correctly classified test image per manipulation method and dataset plus up to two the detector gets wrong, each explained by all three models), then the demo app with a temporary public link. Download `outputs/demo/gallery/` to run the demo on your own computer |

v2 run order: 01b (with train) → 01c → 01d → 02a (two copies) → 02b → 02c → 03, then 04 onwards on the new
checkpoints. A v2 epoch is `SAMPLES_PER_EPOCH = 60000` images, about twice a v1 epoch, so training all seven
models takes roughly 12–14 GPU-hours (EfficientNet-B0 ~80 min, B4 ~3.3 h). Kaggle stops sessions at 12 h, so
`02a_train_classifiers` skips models that would not fit. Run two copies with the model list split between them
(copy A = B0, Meso, ResNet, ViT, ~6 h; copy B = Xception, B4, ConvNeXt, ~7.5 h). `02c_calibrate` and
`03_evaluate` collect checkpoints from every attached copy. The whole v2 sequence uses about 15–17 GPU-hours
(02b ~45 min, 03 ~1.5–2 h) and 2–4 CPU-hours.

Every notebook needs **Internet on**. The project code is cloned from the repo (`dev` branch). If the repo is
private, add a `GITHUB_TOKEN` secret, or upload a zip of the project as a Kaggle dataset and attach it.
Use a T4, not a P100 (recent PyTorch builds no longer support it).

## Layout
```
src/classifier/  model, extraction (FF++, Celeb-DF, HQ photos, DFDC), landmarks, self-blending, augmentation, training,
                 CLIP probe, calibration + three-way verdict, model comparison, robustness, leakage audit
src/faces.py     face detection + cropping (extraction and UI)
src/kaggle_utils.py  locate earlier notebooks' outputs on Kaggle
kaggle/          Kaggle notebooks 01a–08 (kaggle/later/00_all_in_one: an older single-notebook version)
src/xai/         Grad-CAM++, heatmap evaluation, model selection scorecard
src/explain/     evidence-card loader, retriever, LLM providers + explainer
src/eval/        metrics, verifier (grounding checks + NLI), phase 5 experiment (sample, baselines, scoring) and statistics
src/demo/        demo pipeline (step by step, offline gallery) and panel HTML
app.py           the demo app (Gradio)
corpus/          evidence cards (one claim each, with regions, families and source)
data/ outputs/   (git-ignored)
```
