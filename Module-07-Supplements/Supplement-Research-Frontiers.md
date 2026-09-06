# Supplement — Research Frontiers, Scope, and Where the Gaps Actually Are

*Written September 2026. The dated claims in §1 and §5 have a short shelf life — re-check them before you rely on them.*

---

## How to read this

Most "research ideas" that fall out of a curriculum are already done. A curriculum teaches you the settled part of a field by construction: it is a compression of what survived. So if you finish Module 6 and think *"what if we combined CLIP with SAM?"* — that is a 2023 paper, and there are forty of them.

The value of this document is not a list of ideas. It is **calibration**: knowing where the frontier physically is, so that when you have an idea you can tell within ten minutes whether it is open, closed, or closed-but-nobody-wrote-it-down (the third category is where undergraduate papers actually come from).

Two honest framings before you start.

**First: research and employability are different games, and this program is optimised for the second.** A CVPR paper is worth enormously more for a PhD application than for a GenAI engineering role — most hiring managers at product companies will not read it. If your goal is the high-paying role, the capstone in `Supplement-Capstone-Projects.md` is a better use of four months than a paper attempt. Do research because you want to, or because you are keeping the PhD door open — not because you think it is the efficient path to a job. It isn't.

**Second: the field has moved, and the curriculum's shape reflects where CV *was*, not only where it is.** Of the papers highlighted at CVPR 2026, multimodal LLMs and vision-language models are the single largest and fastest-growing category (4.9% → 10.6% of highlights year over year), video generation and world models roughly doubled (3.8% → 8.8%), and embodied AI grew from 2.9% to 6.2%. Meanwhile **detection, segmentation and tracking collapsed from 3.8% to 1.2%**, and depth/geometry halved to 1.2%. One observer's summary — that CVPR "is closer to an applied-generative-AI conference now than the perception-centric venue it was five years ago" — is uncomfortable but accurate.

This does **not** mean Modules 4 and 5 were wasted. Detection and geometry are underrepresented *in research* precisely because they are largely *solved and deployed* — which is exactly why they are still asked about in interviews and still run in production at every autonomy, retail and robotics company. But it does mean: **do not look for a research gap in Module 4 or Module 5 territory.** You will be competing against fifteen years of accumulated effort for a shrinking slice of attention. If you want to do research, the open ground is in Module 6's direction and past it.

---

## 1. The frontier, area by area

Each subsection has the same four parts: what is **settled**, what is **open**, the papers that **define the current state**, and what an undergraduate on a laptop GPU could actually move.

### 1.1 Vision-language models & multimodal reasoning

**Settled.** That contrastive image-text pretraining produces transferable representations (CLIP, 2021). That a frozen vision encoder + a learned projector + a frozen-or-tuned LLM is a sufficient VLM recipe (LLaVA, BLIP-2, 2023) — the architecture question is closed; nobody is still asking whether you need cross-attention or a linear projector, because the linear projector works. That scaling instruction-tuning data improves benchmark scores. That VLMs can caption, VQA, and OCR at useful quality.

**Open, as concrete questions:**

- **Where does grounding actually live?** When a VLM answers "the mug is to the left of the laptop," is the spatial fact computed by the vision encoder, encoded by the projector, or reconstructed by the LLM from priors? Ablation evidence is thin and mostly indirect.
- **Why do VLMs fail at composition?** CLIP-family models are famously near-chance at distinguishing "the horse is eating the grass" from "the grass is eating the horse". The diagnosis (contrastive objectives don't require binding, because no negative in the batch differs only by binding) is widely accepted; the *fix* is not.
- **Spatial and counting reasoning.** VLMs remain poor at relative position, counting past ~5, and metric estimation. HiSpatial and related CVPR 2026 work is attacking this; it is not solved.
- **Evaluation is broken and everyone knows it.** Benchmarks leak into training sets; multiple-choice VQA rewards priors over perception; "the model scores 82%" tells you nothing about *which* 18%.

**Defining papers.** CLIP (Radford et al., ICML 2021, arXiv 2103.00020) · BLIP-2 (Li et al., ICML 2023, arXiv 2301.12597) · LLaVA (Liu et al., NeurIPS 2023, arXiv 2304.08485) · SigLIP (Zhai et al., ICCV 2023, arXiv 2303.15343 — sigmoid loss removes the global-batch dependency, which is the practical reason to prefer it) · Winoground and ARO for the compositionality failure · *Vision Language Models: A Survey of 26K Papers* (arXiv 2510.09586) — read this one first; it is the map.

**What you could actually move.** This is the most accessible area on a small budget, because the expensive part (pretraining) is already done and released. Ablating a *frozen* pipeline is cheap. Training a projector on 1% of LLaVA's data costs single-digit GPU-hours. A careful negative result about where grounding lives is publishable at a workshop.

### 1.2 Generative vision & video

**Settled.** DDPM's forward/reverse formulation and the noise-prediction parameterisation. That latent-space diffusion is the right cost/quality trade (Stable Diffusion). That classifier-free guidance works. That flow matching / rectified flow gives you the same sample quality with straighter probability paths and fewer steps — this has largely displaced the ε-prediction DDPM framing in new work, which is the single biggest "the curriculum teaches DDPM but the field moved" gap you should be aware of.

**Open:**

- **CFG's diversity collapse.** At guidance scale 7.5, the unconditional score carries weight −6.5 — you are extrapolating outside the region the model was trained on, and you pay for it in mode collapse and saturation. Fixes exist (dynamic thresholding, guidance intervals, autoguidance) but none is principled and free.
- **Few-step generation without distillation.** Consistency models and adversarial distillation get you to 1–4 steps, but by training a *second* model. Whether a single model can be trained few-step-native without quality loss is open.
- **Video: temporal consistency and controllability.** Video generation doubled in CVPR share because the useful-model transition is happening now — the open questions are long-horizon consistency, physical plausibility, and whether "world model" means anything beyond a well-conditioned video model.
- **Evaluation.** FID is a bad metric that the field cannot replace. This is a genuine, unglamorous, open problem.

**Defining papers.** DDPM (Ho et al., NeurIPS 2020, 2006.11239) · DDIM (Song et al., ICLR 2021, 2010.02502) · Latent Diffusion (Rombach et al., CVPR 2022, 2112.10752) · Classifier-Free Guidance (Ho & Salimans, 2207.12598) · Flow Matching (Lipman et al., ICLR 2023, 2210.02747) · Rectified Flow (Liu et al., 2209.03003) · Consistency Models (Song et al., ICML 2023, 2303.01469).

**What you could move.** CFG behaviour is measurable on a laptop with a pretrained SD checkpoint — you are doing inference-time analysis, not training. A careful, well-controlled study of *which* guidance schedule preserves diversity at fixed fidelity, on a fixed prompt set, with a metric better than FID, is a real contribution and costs almost nothing. This is the highest-feasibility research direction in this document.

### 1.3 3D, spatial understanding & world models

**Settled.** Classical multi-view geometry, entirely — epipolar geometry, bundle adjustment, SfM. COLMAP is the reference and has been for a decade. NeRF's core formulation. That 3D Gaussian Splatting beats NeRF on the speed/quality frontier for most reconstruction use cases.

**Open:**

- **Feed-forward 3D.** The DUSt3R → MASt3R → VGGT line replaced "run SfM, then optimise" with "one forward pass predicts geometry". How far this generalises, and whether it obsoletes bundle adjustment or merely initialises it better, is live.
- **Monocular metric depth.** Depth Anything v2 made relative depth a solved commodity. *Metric* depth from a single image, without camera intrinsics, remains genuinely hard and genuinely useful.
- **World models.** Whether a video model that predicts the next frame has learned physics, or has learned a very good renderer with no physics, is the field's most-argued and least-settled question.

**Defining papers.** NeRF (Mildenhall et al., ECCV 2020, 2003.08934) · 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023, 2308.04079) · DUSt3R (Wang et al., CVPR 2024, 2312.14132) · MASt3R (2406.09756) · VGGT (CVPR 2025, 2503.11651) · Depth Anything v2 (2406.09414) · SAM 3D (Meta, Nov 2025).

**What you could move.** Honestly: little, on a laptop. 3D is compute-hungry and the labs working on it have clusters. The exception is **evaluation and failure-mode characterisation** — nobody has carefully mapped where feed-forward 3D breaks (textureless scenes? repeated structure? wide baselines?), and that study is cheap because you run inference on released checkpoints against controlled synthetic scenes you generate yourself in Blender.

### 1.4 Self-supervised representation learning

**Settled.** That contrastive (SimCLR/MoCo) and masked-reconstruction (MAE) both work, and that they produce *different* representations — contrastive gives better linear probes, MAE gives better fine-tuning. That DINOv2/v3 features are good enough to be used frozen for dense tasks. That the projection head is discardable and that discarding it is essential.

**Open:**

- **Why does the projection head help?** The "it absorbs augmentation-specific information" story is a story, not a proof.
- **Do SSL features encode anything a supervised model doesn't?** DINO's emergent segmentation maps suggest yes; controlled evidence is weaker than the vibe.
- **Small-data SSL.** Every result is at ImageNet scale or above. What SSL does at 10k images — the regime an actual startup is in — is under-studied and eminently testable on a laptop.

**Defining papers.** SimCLR (2002.05709) · MoCo (1911.05722) · BYOL (2006.07733) · MAE (2111.06377) · DINO (2104.14294) · DINOv2 (2304.07193) · DINOv3 (Meta, 2025) · **A Metric Learning Reality Check** (Musgrave et al., ECCV 2020, 2003.08505) — read this one even though it's about metric learning, because its *method* (showing that a decade of claimed gains vanish under fair comparison) is the single most transferable research skill in this document.

**What you could move.** The small-data question. It is cheap, it is under-studied, it has an obvious industrial motivation, and a negative result ("SSL does not beat supervised pretraining below N images") is as publishable as a positive one.

### 1.5 Efficiency, quantisation & edge inference

**Settled.** Post-training INT8. Structured pruning basics. Depthwise-separable convolutions and the MobileNet/EfficientNet design space. Knowledge distillation.

**Open:**

- **Sub-4-bit vision.** LLM quantisation went to 4-bit and below; vision models mostly did not follow, and why is not fully understood.
- **Quantising diffusion.** The iterative structure means quantisation error compounds across steps in ways single-forward-pass analysis misses.
- **Latency-accuracy Pareto fronts are almost never reported honestly** — papers report FLOPs, which correlate poorly with wall-clock on real hardware.

**What you could move.** A lot, actually — this is the most laptop-friendly research area after §1.2, because measurement *is* the contribution and measurement is cheap. It is also the area most directly aligned with an MLOps/AI-infra pivot.

### 1.6 Robustness, OOD & evaluation

**Settled.** That models degrade under distribution shift. That adversarial examples exist. Standard benchmarks (ImageNet-C, -R, -A, ObjectNet).

**Open:** Basically all of it. Benchmark contamination in the foundation-model era is severe and under-measured. Whether robustness interventions transfer across shift types is unclear. Calibration under shift is poor and the fixes are ad hoc.

**What you could move.** Contamination auditing of a public VLM benchmark is a *fantastic* undergraduate project: cheap, uncomfortable, useful, and the kind of thing labs avoid because the result embarrasses everyone.

### 1.7 Data-centric vision & synthetic data

**Settled.** That data quality beats model tweaks at fixed budget. That synthetic data helps for detection and pose. That dataset distillation works at small scale.

**Open:** How much synthetic data before it hurts; whether generative-model-produced training data causes model collapse in vision the way it does in language; principled data-selection criteria that beat CLIP-score filtering.

**What you could move.** The model-collapse question in vision, at small scale, with a controlled generation loop. Cheap, topical, and the negative result is interesting.

### 1.8 Embodied & robotic vision

**Settled.** Very little, which is why it grew from 2.9% to 6.2% of CVPR highlights.

**Open:** Vision-language-action models, sim-to-real, whether perception should be a separate module at all.

**What you could move.** Nothing, without hardware. Be honest with yourself about this one. Simulation-only work exists but the reviewing community discounts it heavily.

---

## 2. Research directions, ranked for your constraints

Constraints assumed: one laptop GPU (~8–16 GB), Colab/Kaggle free or cheap tier, a few hundred rupees a month of cloud if pushed, and 8–12 hours a week alongside coursework. Ranked by **feasibility × signal**, not by how exciting they sound.

| # | Direction | Min. viable experiment | Compute | Venue realism | Verdict |
|---|---|---|---|---|---|
| 1 | **CFG diversity/fidelity trade-off, measured properly** | Fixed prompt set, sweep guidance scale and schedule, measure diversity with a metric that isn't FID (e.g. pairwise CLIP-embedding dispersion + a human-free coverage proxy). Compare 3–4 published "fixes". | Inference only. Laptop. | Workshop (diffusion/generative) | **Best first project.** Cheap, self-contained, unambiguous. |
| 2 | **Where does VLM grounding live?** | Take LLaVA-style model. Ablate: shuffle patch order, mask spatial positional info, swap encoder, train projector on 1%/10%/100%. Measure spatial-QA vs object-naming separately. | Projector-only training. ~10–30 GPU-hr. | Workshop (MLLM / vision-in-the-wild) | **Best second project.** High topicality, clean question. |
| 3 | **SSL below 10k images** | Pretrain SimCLR/MAE from scratch on 1k/5k/10k/50k image subsets, compare against supervised and against ImageNet-transfer. Find the crossover. | Real training but small. 20–50 GPU-hr. | Workshop | Solid. Negative result publishable. |
| 4 | **Benchmark contamination audit of a public VLM eval** | N-gram / near-duplicate image search between a benchmark's images and a public pretraining corpus (LAION subsets are indexable). Report contamination rate and score delta on clean subset. | CPU-heavy, little GPU | Workshop (evaluation/datasets) | Uncomfortable, cheap, valuable. |
| 5 | **Quantisation of diffusion, honestly measured** | INT8/INT4 the U-Net (or DiT), measure error accumulation *per denoising step*, not just final FID. Report wall-clock on real hardware, not FLOPs. | Laptop | Workshop (efficient vision) | Aligns with your MLOps pivot. |
| 6 | **Where feed-forward 3D breaks** | Generate controlled Blender scenes varying texture density, baseline, repetition. Run VGGT/DUSt3R/COLMAP. Map the failure boundary. | Inference + Blender | Workshop (3D) | Good, but 3D reviewers are demanding. |
| 7 | **Vision model collapse under synthetic-data loops** | Train classifier on real, generate synthetic with SD, retrain on mix, iterate 5 generations, measure degradation vs synthetic fraction. | Moderate | Workshop | Topical; risk: several papers already exist. Check first. |
| 8 | ~~Novel detection architecture~~ | — | — | — | **Trap.** You will not beat DINO-DETR. Do not try. |
| 9 | ~~"Apply SAM + CLIP to domain X"~~ | — | — | — | **Trap.** Done to death. This is a *project*, not research — and as a project it's fine (see the capstone doc). |
| 10 | ~~New attention variant for ViT~~ | — | — | — | **Trap.** Requires pretraining-scale compute to show anything. |

**The honest ranking:** do #1. If it goes well, do #2. Those two are the only ones I would tell you to start this year.

---

## 3. Where undergraduates actually publish

Main-conference CVPR/ICCV/ECCV as sole undergraduate first author, without a lab, is rare enough that you should not plan around it. The realistic ladder:

| Venue | What it is | Timing (verify before relying on it) |
|---|---|---|
| **CVPR workshops** | 4–8 page papers, real reviewing, real proceedings, indexed. The realistic target. | CVPR 2026 ran June 3–4 (workshops) / June 5–7 (main) in the 2026 cycle; workshop *proposal* deadline was Nov 3, 2025. Individual workshop paper deadlines are set per workshop — typically **March**, ~3 months before the conference. |
| **CVPR Findings Track** | A newer track (there is an OpenReview group for CVPR 2026 Findings) for solid-but-not-headline work. Worth investigating — this is exactly the tier undergraduate work lands in. | Follows main-conference cycle: abstract Nov 7, paper Nov 13, decisions Feb 20 for the 2026 edition. |
| **WACV** | Applications-focused, notably more forgiving than CVPR, two-round system. **This is the best main-conference target for you.** | WACV 2027: Round 1 submission **June 26, 2026**; Round 2 **Aug 28, 2026**; conference Jan 4–8, 2027. |
| **BMVC** | British Machine Vision Conference. Respectable, less brutal than CVPR. | Typically ~May deadline, September conference. Verify. |
| **NeurIPS / ICLR workshops** | Very accessible; often non-archival, which is a feature — you can submit the full version elsewhere later. | Workshop CFPs appear ~2 months before the conference. |
| **ICVGIP** | India's flagship CV/graphics venue, ACM-sponsored, biennial-ish. Genuinely realistic for an Indian undergraduate and worth targeting. | Check `icvgip.in` — 2025 and 2026 editions both have pages. |
| **NCVPRIPG** | National conference (NCVPRIPG 2026 at LNMIIT Jaipur). Lower bar, real experience, good for a first submission. | Check the 2026 site. |

**Strategy:** first submission to NCVPRIPG or a NeurIPS/ICLR workshop to learn the mechanics without stakes. Second, once you have a result you believe in, WACV Round 1 or a CVPR workshop.

---

## 4. How to find a gap — the actual method

Three techniques, in order of how much they're worth.

**4.1 Read the Limitations section first.** Modern venues mandate one. Authors write them defensively, which means they name the weakness they most fear a reviewer will find — which is the weakness they could not fix. That is a gap, stated by the people best positioned to know. Read fifteen Limitations sections in one subfield and the recurring one is your research question.

**4.2 Follow the citations forward, not backward.** Backward citation (reading a paper's references) teaches you history. **Forward** citation — who cited this, and what did they say about it — teaches you the frontier. Semantic Scholar and Connected Papers both do this. The pattern to look for: a paper cited 400 times where every citation says "following [X], we assume Y" and nobody has checked Y.

**4.3 Look for the reality check that hasn't been written.** The highest-leverage undergraduate paper genre is not a new method; it is *fair re-evaluation of existing ones*. Musgrave et al.'s "A Metric Learning Reality Check" showed that a decade of claimed metric-learning improvements largely vanished under a fair protocol. That paper required no new idea and no cluster — only rigour and the willingness to be unpopular. Ask, of any subfield you have just studied: **have the comparisons in this literature been fair?** If the answer is "probably not, and nobody has checked", you have a paper.

**The three-pass read.** Pass 1 (5 min): title, abstract, figures, conclusion — what is the claim? Pass 2 (1 hr): method and experiments, skipping proofs — is the claim supported? Pass 3 (4+ hrs, rare): reproduce the core derivation or result — is it *true*? Do pass 3 on perhaps five papers a year. Those five are what make you good.

---

## 5. Sequenced reading list

Read alongside the modules, not before them. Tags map to curriculum modules.

**Foundations you should read even though they're old (M1–M2)**
1. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 — M1. The scale-space argument, in the author's own words.
2. Fischler & Bolles, *Random Sample Consensus*, CACM 1981 — M1. Two pages, still correct.
3. Ioffe & Szegedy, *Batch Normalization*, ICML 2015 (1502.03167) — M2.
4. Santurkar et al., *How Does Batch Normalization Help Optimization?*, NeurIPS 2018 (1805.11604) — M2. **Read immediately after #3.** It refutes #3's stated explanation. This pairing is the single best lesson in the list about how the field self-corrects.
5. He et al., *Deep Residual Learning*, CVPR 2016 (1512.03385) — M2.
6. Bello et al., *Revisiting ResNets*, NeurIPS 2021 (2103.07579) — M2. Most of ResNet-vs-newer gaps are training recipe, not architecture.
7. Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022 (2201.03545) — M2/M6. The strongest argument that ViT's win was partly recipe.

**Representation (M3)**
8. Schroff et al., *FaceNet* / triplet loss, CVPR 2015 (1503.03832).
9. Musgrave et al., *A Metric Learning Reality Check*, ECCV 2020 (2003.08505). **Essential.**
10. Chen et al., *SimCLR*, ICML 2020 (2002.05709).
11. He et al., *MoCo*, CVPR 2020 (1911.05722).
12. Grill et al., *BYOL*, NeurIPS 2020 (2006.07733) — negatives turn out to be optional, which broke everyone's mental model.
13. Malkov & Yashunin, *HNSW*, 2016 (1603.09320) — M3. Directly reusable in your RAG work.
14. Jégou et al., *Product Quantization for NNS*, PAMI 2011 — M3.

**Detection & segmentation (M4)**
15. Ren et al., *Faster R-CNN*, NeurIPS 2015 (1506.01497).
16. Lin et al., *Feature Pyramid Networks*, CVPR 2017 (1612.03144).
17. Lin et al., *Focal Loss / RetinaNet*, ICCV 2017 (1708.02002).
18. Tian et al., *FCOS*, ICCV 2019 (1904.01355) — anchor-free.
19. Carion et al., *DETR*, ECCV 2020 (2005.12872).
20. Zhu et al., *Deformable DETR*, ICLR 2021 (2010.04159) — why plain DETR was unusable and what fixed it.
21. Bolya et al., *TIDE*, ECCV 2020 (2003.12237) — mAP is one number hiding six error types.
22. Ronneberger et al., *U-Net*, MICCAI 2015 (1505.04597); He et al., *Mask R-CNN*, ICCV 2017 (1703.06870).

**Geometry (M5)**
23. Hartley, *In Defense of the Eight-Point Algorithm*, PAMI 1997 — normalisation as conditioning. Short, and the argument is beautiful.
24. Triggs et al., *Bundle Adjustment — A Modern Synthesis*, 1999. Skim; use as reference.
25. Mildenhall et al., *NeRF*, ECCV 2020 (2003.08934); Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023 (2308.04079).
26. Wang et al., *DUSt3R*, CVPR 2024 (2312.14132) — the feed-forward turn.

**Foundation & generative models (M6)**
27. Dosovitskiy et al., *ViT*, ICLR 2021 (2010.11929); He et al., *MAE*, CVPR 2022 (2111.06377).
28. Radford et al., *CLIP*, ICML 2021 (2103.00020); Zhai et al., *SigLIP*, ICCV 2023 (2303.15343).
29. Oquab et al., *DINOv2*, TMLR 2024 (2304.07193).
30. Ho et al., *DDPM*, NeurIPS 2020 (2006.11239); Rombach et al., *Latent Diffusion*, CVPR 2022 (2112.10752); Ho & Salimans, *CFG* (2207.12598).
31. Lipman et al., *Flow Matching*, ICLR 2023 (2210.02747). **The curriculum teaches DDPM; the field now largely writes flow matching. Read this to close that gap.**
32. Liu et al., *LLaVA*, NeurIPS 2023 (2304.08485); Li et al., *BLIP-2*, ICML 2023 (2301.12597).
33. *Vision Language Models: A Survey of 26K Papers* (2510.09586) — read last, as a map of everything above.

---

## Sources

- [CVPR 2026 accepted-paper trends](https://www.bohrium.com/en/blog/research-notes/cvpr-2026-accepted-papers-highlights/)
- [CVPR 2026 dates and deadlines](https://cvpr.thecvf.com/Conferences/2026/Dates)
- [CVPR 2026 Findings Track (OpenReview)](https://openreview.net/group?id=thecvf.com%2FCVPR%2F2026%2FFindings_Track)
- [WACV 2027 dates and deadlines](https://wacv.thecvf.com/Conferences/2027/Dates)
- [Vision Language Models: A Survey of 26K Papers](https://arxiv.org/html/2510.09586v1)
- [ICVGIP](https://icvgip.in/2026/) · [NCVPRIPG 2026, LNMIIT Jaipur](https://ncvpripg2026.lnmiit.ac.in/)
- [Meta: new Segment Anything models (SAM 3, SAM 3D), Nov 2025](https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/) · [SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/)
- [Top CVPR 2026 papers (curated)](https://github.com/SkalskiP/top-cvpr-2026-papers)

*arXiv IDs above are given from memory where the paper is well known. If one 404s, search the title — do not cite an ID you have not opened.*
