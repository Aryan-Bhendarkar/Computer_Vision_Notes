# Computer Vision Mastery — Self-Study Notes

**~130,000 words.** The primary self-study material for the Computer Vision Mastery Program: every concept of the 6-module syllabus, the gaps that syllabus omitted, and the interview and engineering context around it.

**Currency:** written September 2026. Fast-moving version numbers (SAM, DINO, YOLO, DETR variants, VLMs, diffusion) are accurate as of then and flagged where they will age. Mechanisms are durable; version numbers are not.

---

## Start here

1. **Read [`Interview-Readiness.md`](Module-07-Supplements/Supplement-Interview-Readiness.md) first.** It changes how you should spend your time, and reading it last would waste weeks.
2. Skim [`NOTATION.md`](NOTATION.md) — one page, and it prevents the most common confusions (three different meanings of $H$, $K$ and $t$).
3. Open [`PROGRESS.md`](PROGRESS.md) and use it. Coverage without recall is the failure mode this material is built to prevent.
4. Then start on the modules in the order given under **The plan**.

---

## Live site

**Deployed:** https://aryan-bhendarkar.github.io/CV-Notes/ *(after the one-time setup below)*

`docs/index.html` is the built study tool — Read / Practice / Map / Rapid-fire —
committed as the deploy artifact, the same pattern as the ML-Notes site. GitHub
Actions (`.github/workflows/pages.yml`) packages `docs/` and serves it on every
push to `main` that touches `docs/**`; there is no server-side build step.

One honest difference from the Claude-hosted copy: this static version has no
backend, so review progress is saved to **this browser only** (`localStorage`),
not synced across devices. The Claude artifact version keeps a synced copy —
use that if you study from more than one device, and treat this one as the
public, always-available mirror.

### One-time setup (only needed once)

```
gh repo create CV-Notes --public --source=. --remote=origin --push
```

or, without the GitHub CLI: create an empty repo named `CV-Notes` at
github.com/new, then:

```
git remote add origin https://github.com/Aryan-Bhendarkar/CV-Notes.git
git branch -M main
git push -u origin main
```

Then in the repo's Settings → Pages, set Source to **GitHub Actions** (not
"Deploy from a branch"). The first push triggers the workflow; the site is
live at the URL above a minute or two later.

### Publishing an update after editing the notes

```
cd _app
python parse.py --root .. --out content.json
python build.py
cp index.html ../docs/index.html
cd ..
git add docs/index.html
git commit -m "Update site"
git push
```

---

## Research, projects, and the system itself

Three documents sit outside the module sequence:

- **`Module-07-Supplements/Supplement-Research-Frontiers.md`** — where the field's open problems actually are as of September 2026, ranked research directions scoped to a laptop GPU, where undergraduates realistically publish, and a method for finding a gap rather than re-solving a solved problem. Read the "How to read this" section before forming any research plans; it contains the honest version.
- **`Module-07-Supplements/Supplement-Capstone-Projects.md`** — the final project, ranked by what actually produces hiring signal. Includes an explicit list of traps.
- **`LEARNING-SYSTEM-SPEC.md`** — the design for the web study tool built on top of these notes, and the argument for why it should be a practice loop rather than a reader.

Each module also now ends with **`## Going Deeper — Papers, Sources and Research Scope`**: the canonical papers in reading order (including the *re-examinations* that later corrected them), one best source per hard concept, reference implementations worth reading line by line, and open research questions scoped to a small compute budget.

---

## How the explanations are written

Every concept in the six module files is written to one standard, set out in full in `_reviews/STYLE-GUIDE.md`. The short version — a self-learner has no one to ask *"but why?"*, so:

- each concept **opens with the question it answers**, not with a definition;
- formulas are **derived, not presented** — and where a derivation is genuinely out of scope, the notes say so rather than letting you think you missed a step;
- worked examples run **in sentences**, so you can do the next one unaided;
- the likely misconception is **named** ("You might expect… / The thing that trips people up here is…");
- a **Pause:** mid-section asks you to predict before the answer arrives — a wrong prediction is what makes the right one stick;
- new ideas are **anchored to what you already know** (deep learning, Transformers, and earlier modules);
- the teaching closes with **In your own words:** — if you can't restate it without notation, you haven't got it yet.

---

## How each concept is structured

Every numbered concept follows the same shape:

1. **Intuition** — plain language, usually with a real-world or industry anchor.
2. **The math** — the formal treatment, with derivations worked rather than asserted.
3. **Connections** — what it builds on and where it reappears. (Threaded through the prose from Module 2 onward rather than sitting under its own heading — the cross-references are there, they just aren't labelled.)
4. **🎯 Top-1% distinction** — what separates a surface answer from a strong one. Read this section twice.
5. **✅ Mastery check** — a probing question with a collapsed answer sketch. **Answer before expanding.** Practical/capstone sections carry a deliverable instead.
6. **🔨 Build + read** — one scoped build task and one or two specific sources.

**Priority tags:** 🔴 Critical (load-bearing, full depth) · 🟡 Important · 🟢 Enrichment · *(untagged)* = core syllabus.

---

## The files

### Core modules

| Module | Notes | Drills | Focus |
|---|---|---|---|
| **01 — Classical CV** | [Notes](Module-01-Classical-CV/Module-01-Notes.md) · [**Appendix: Image Processing**](Module-01-Classical-CV/Module-01-Appendix-Image-Processing.md) | [Drills](Module-01-Classical-CV/Module-01-Drills.md) | formation, filtering, edges, **scale-space 🔴**, Harris, SIFT/ORB, matching, RANSAC, homography · *appendix:* thresholding, morphology, connected components, Hough, watershed/GrabCut/SLIC, resampling |
| **02 — DL for Vision** | [Notes](Module-02-DL-for-Vision/Module-02-Notes.md) | [Drills](Module-02-DL-for-Vision/Module-02-Drills.md) | CNN priors, receptive fields, **normalisation 🔴**, architectures, ResNet, efficient convs, augmentation, transfer |
| **03 — Representations** | [Notes](Module-03-Visual-Representations/Module-03-Notes.md) | [Drills](Module-03-Visual-Representations/Module-03-Drills.md) | embeddings, retrieval, Siamese, triplet + mining, contrastive SSL, **ANN search 🔴** |
| **04 — Detection & Segmentation** | [Notes](Module-04-Detection-Segmentation/Module-04-Notes.md) · [**Appendix: Pose & Anomaly**](Module-04-Detection-Segmentation/Module-04-Appendix-Pose-and-Anomaly.md) | [Drills](Module-04-Detection-Segmentation/Module-04-Drills.md) | R-CNN lineage, **FPN 🔴**, one-stage, **focal loss 🔴**, NMS/mAP, DETR, segmentation, metrics · *appendix:* keypoint/pose, industrial anomaly detection |
| **05 — Geometry & 3D** | [Notes](Module-05-Geometry-3D/Module-05-Notes.md) · [**Appendix: 3D Deep Learning**](Module-05-Geometry-3D/Module-05-Appendix-3D-Deep-Learning.md) | [Drills](Module-05-Geometry-3D/Module-05-Drills.md) | camera model, calibration, **epipolar 🔴**, stereo, flow, **bundle adjustment 🔴**, SLAM, NeRF/3DGS · *appendix:* PointNet/PointPillars/BEV, monocular depth, MVS/ICP, sensors |
| **06 — Modern & Foundation** | [Notes](Module-06-Modern-Foundation-Models/Module-06-Notes.md) | [Drills](Module-06-Modern-Foundation-Models/Module-06-Drills.md) | ViT, MAE, CLIP, DINO, SAM, open-vocab detection, VLMs, GANs, diffusion, latent diffusion, discrete tokenisers |

### Supplements

| File | What |
|---|---|
| [**Interview Readiness**](Module-07-Supplements/Supplement-Interview-Readiness.md) | **read first** — the four gates, LLM/GenAI question bank, ML system design *with answer sketches*, project-narrative and behavioural prep, noisy labels, semi-supervised, domain shift, OCR/document AI, evaluation engineering |
| [Robustness, Scale & Governance](Module-07-Supplements/Supplement-Robustness-Scale-and-Governance.md) | adversarial/corruptions/OOD, uncertainty & conformal prediction, long-tail, distillation & pruning, distributed training, ethics & licensing, the tooling ecosystem |
| [Beyond the Syllabus](Module-07-Supplements/Supplement-Beyond-The-Syllabus.md) | video understanding, explainability & debugging, deployment & quantisation |

### Reference

| File | Use it when |
|---|---|
| [`NOTATION.md`](NOTATION.md) | a symbol surprises you — includes the cross-module collision table |
| [`FORMULA-SHEET.md`](FORMULA-SHEET.md) | revising, or the night before an interview. **If you can reproduce it from memory, you know the material** |
| [`GLOSSARY.md`](GLOSSARY.md) | you hit a term and can't remember which module defined it |
| [`PROGRESS.md`](PROGRESS.md) | every session |
| [`_reviews/`](_reviews/) | the review record — arithmetic verification, coverage audit, usability audit, and what each changed |

Each drill file contains: rapid-fire questions, whiteboard problems, a **traps table** (designed to catch memorisers), Anki-ready cards, a self-assessment rubric, and exit criteria.

---

## Before you start — environment

```bash
python -m venv .venv && source .venv/bin/activate     # or conda
pip install torch torchvision timm opencv-python \
            albumentations faiss-cpu scikit-learn matplotlib \
            open3d pycocotools fiftyone jupyter
```

**GPU expectations.** Roughly half the build tasks run on a CPU in minutes (all of Module 1 and its appendix, RANSAC/homography, FAISS, ICP, classical segmentation). The rest want a GPU; where one is needed the build task says so and gives a rough time. **A free Colab/Kaggle T4 is enough for everything except the SSL pretraining ablations** (3.6, 6.4), which want a few hours on a better card or a reduced-scale substitute.

**Datasets** you will actually download: CIFAR-10, Oxford-IIIT Pets or Flowers-102, the Oxford VGG affine sequences (graf/boat/bikes/leuven), a small COCO subset or Pascal VOC, MVTec-AD, ModelNet40, and 30–40 photos you take yourself for the COLMAP labs. All are free and none is large except COCO.

**One habit worth adopting now:** after loading any annotated dataset, **visualise the annotations on the image before training**. Box-format confusion (COCO abs-xywh vs YOLO normalised-centre vs VOC corners) is silent — the loss goes down and the boxes are simply in the wrong place. See the tooling section of the Robustness supplement.

---

## Two different questions: what's load-bearing, and what pays

These are not the same list, and conflating them is the easiest way to spend nine weeks badly. A hiring-manager review of this material was blunt: *"The mistake isn't studying CV — it's studying all of CV at uniform depth."*

### A. Load-bearing for understanding the subject

The six 🔴 concepts. Skip these and later modules stop making sense.

| # | Concept | The one derivation to own |
|---|---|---|
| **1.4** | Scale-space & DoG | $\partial G/\partial\sigma = \sigma\nabla^2 G \Rightarrow \text{DoG} \approx (k{-}1)\sigma^2\nabla^2 G$ |
| **2.4** | Batch/Layer Normalisation | $L(aW)=L(W) \Rightarrow \nabla_{aW}L = \frac1a\nabla_W L$ ⇒ self-decaying effective LR |
| **3.7** | ANN search | PQ + ADC lookup tables; the recall/latency/memory triangle |
| **4.4** | Feature Pyramid Networks | resolution from bottom-up, semantics from top-down; $k = \lfloor 4 + \log_2(\sqrt{wh}/224)\rfloor$ |
| **4.6** | Focal Loss | $100{,}000\times0.01 \gg 10\times0.69$; $\gamma{=}2 \Rightarrow$ 100× down-weight at $p_t{=}0.9$; prior bias init |
| **5.5 / 5.8** | Epipolar geometry & bundle adjustment | $E = [\mathbf{t}]_\times R$ from coplanarity; the Schur complement marginalising 3-D points |

### B. Highest return for a GenAI / ML-engineering job search

| Rank | Section | Why it pays |
|---|---|---|
| 1 | **All of Module 3** | retrieval *infrastructure* — the load-bearing skill in most GenAI products, and it turns "I built a RAG system" into demonstrable competence |
| 2 | **6.10, 6.5, 6.11, 6.15b** | multimodal is where LLM engineering is going; 6.10 is the most job-relevant section here and one of the shortest |
| 3 | **2.7 + Deployment (S.3) + Robustness supplement** | FLOPs≠latency, roofline, quantisation, serving, evaluation — systems skills wearing a CV costume, and the MLOps on-ramp |
| 4 | **2.8–2.11, 4.7, 4.13, S.2, A4.2** | applied judgement: regularisation, metrics, calibration, error analysis, debugging, anomaly detection |
| 5 | **2.4, 2.6, 4.4, 4.6** | the four classic "do you actually understand this?" filter questions |
| 6 | **6.2–6.4, 6.7, 6.13–6.14** | the working vocabulary of the field |

### C. Study by choice, not by default

**All of Module 5 and its appendix, most of Module 1, 4.2, the GAN lineage, most of 6.15.** These make you *educated in computer vision*, and they are genuinely beautiful. They are also asked almost exclusively in robotics / autonomous-vehicle / AR loops — overwhelmingly on-site and hardware-adjacent, not the remote GenAI roles this material's reader is targeting.

Study them because you want to understand vision, or because your course examines them. **That is the honest trade, stated here so the choice is deliberate.**

> ⚠️ **Interviewers probe exactly what you volunteer.** A rehearsed line like *"Gram anchoring preserves dense features at long schedules"* that collapses on the first follow-up scores **worse** than "I don't know." Only volunteer depth you can defend two questions deep — which is what the mastery checks are for.

---

## The plan

**Order:** `2 → 3 → 4 → 6 → 1 → 5`, with the supplements woven in. Modules 2, 3, 4 and 6 carry most of the interview value; Module 1's appendix is worth doing early despite the ordering, because it is cheap and immediately useful.

**Allocation — this is the corrected split, not a CV-only one:**

| Track | Share | What |
|---|---|---|
| **CV modules** | ~40% | in the order above, weighted per §B |
| **Coding** | ~25% | DSA to clear screens + timed ML-implementation drills (MHA, KV cache, NMS, focal loss, a training loop with a planted bug) |
| **One flagship artefact** | ~20% | shipped, measured, publicly linkable |
| **LLM systems + evaluation** | ~15% | the question bank and one real eval harness |

**Nine-week shape**, at ~5 h/day with that split applied *within* each week:

| Weeks | CV focus | Alongside |
|---|---|---|
| 1–2 | Module 2 + Module 3 | DSA daily; start the flagship project |
| 3–4 | Module 4 (+ pose/anomaly appendix) | ML-implementation drills; eval harness |
| 5–6 | Module 6 (+ 6.15b) | flagship project push; LLM question bank |
| 7 | Module 1 + its appendix | project write-up; rehearse the 5-minute narrative |
| 8 | Module 5 (+ 3D appendix), or skip per §C | mock interviews |
| 9 | Robustness supplement + full drill pass | all exit criteria; ship the artefact |

Compress or extend freely. **Do not skip the mastery checks to go faster**, and **do not let the CV track consume the other three.**

---

## The through-lines

Six ideas recur across every module. Volunteering them unprompted is what reads as mastery.

1. **Multi-scale never goes away.** Gaussian pyramids (1.4) → receptive fields (2.3) → FPN (4.4) → Swin/ViTDet (6.2) → SlowFast (S.1).
2. **Aliasing / Nyquist keeps biting.** Blur-before-decimate (1.4) → strided-conv shift-variance and BlurPool (2.1) → transposed-conv checkerboards (2.2) → StyleGAN3's texture sticking (6.12) → the resize bug (A1.5).
3. **Skip connections are universal.** FCN → U-Net (4.11) → ResNet (2.6) → FPN (4.4) → transformers → diffusion U-Nets (6.13). And "start as the identity" recurs: ResNet's zero-init BN, ControlNet's zero-init convs, LoRA's zero-init $B$.
4. **Retrieve then verify.** BoVW + RANSAC (3.2) → ANN + rerank (3.7) → RAG's bi-encoder + cross-encoder → detect then classify (6.11) → PatchCore anomaly scoring (A4.2) → OOD detection (S.11).
5. **Optimise the metric you evaluate.** IoU losses over L1 (4.7) · YOLOv2's IoU-based k-means (4.5) · reprojection error as the MLE (5.8) · scale-invariant depth loss (A5.2) · ScaNN's anisotropic quantisation (3.7).
6. **The learned front-end sits on a classical back-end.** SuperPoint/LightGlue feeding RANSAC + PnP + bundle adjustment (1.10, 5.8); 3D Gaussian Splatting initialised by COLMAP (5.10); ControlNet steered by a **Canny edge map** from 1986 (6.15).

---

## The build track

~50 build tasks are specified. These ten are the ones worth doing properly — each is a portfolio artefact and each answers a class of interview question with *your own numbers*.

| # | Build | Cost | Answers |
|---|---|---|---|
| 1 | Classical object counter (threshold→morphology→watershed) | 1 sitting, CPU | "count objects without a network" |
| 2 | DoG blob detector reporting correct radii ($R=\sqrt2\sigma$) | 1 sitting, CPU | "explain scale invariance" |
| 3 | SIFT vs ORB on Oxford VGG affine | 1 sitting, CPU | "when would you use ORB over SIFT?" |
| 4 | Plain-vs-ResNet training-error curves | ~4 h, GPU | "why does ResNet work?" |
| 5 | BN vs GroupNorm across batch sizes {128,32,8,2} | ~4 h, GPU | "what breaks at small batch?" |
| 6 | **Recall@10 vs QPS across 5 FAISS indexes on 1M vectors** (annotate memory; needs ~8 GB RAM) | ~half a day, CPU | "design a vector search system" |
| 7 | Detector fine-tune + **TIDE** decomposition + int8 latency | ~1 day, GPU | "how do you improve a detector?" |
| 8 | PatchCore from scratch on MVTec-AD | 1 sitting, GPU (no training) | "detect defects with 40 examples" |
| 9 | Stereo depth error vs distance, fitted to $Z^2/(fB)$ | 1 sitting + a tape measure | "explain stereo accuracy" |
| 10 | **Capstone (6.16)** — multimodal document retrieval + QA, hosted | 1–2 weeks | "tell me about a project" |

**Every one should produce a baseline, an ablation, a latency number, and a documented failure.** A measured project beats an ambitious unmeasured one in every interview.

---

## Sources

- **Classical & geometry:** Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., free PDF) · Hartley & Zisserman, *Multiple View Geometry* · Shree Nayar, "First Principles of Computer Vision" (YouTube) · Stanford CS231A
- **Deep learning for vision:** UMich EECS 498-007/598-005, Justin Johnson (YouTube) · CS231n notes · d2l.ai
- **Modern/foundation:** Hugging Face CV and Diffusion courses · Umar Jamil's from-scratch implementations · Meta AI blog for SAM/DINO version specifics · Roboflow blog for the detector landscape
- **Retrieval:** FAISS wiki · ann-benchmarks.com
- **Papers:** cited inline per concept in each "Build + read" section
