# Supplement — The Capstone: Projects That Actually Move a Résumé

*What to build so that a senior engineer at a company you want to work for reads one line and wants to talk to you.*

---

## 0. First, the uncomfortable part

Almost every CV project on almost every résumé is one of four things:

1. A fine-tuned YOLO on a Roboflow dataset ("real-time PPE detection", "helmet detection", "drowsiness detection")
2. A Streamlit wrapper around a pretrained model
3. A notebook reproducing a tutorial with a new dataset
4. "SAM + CLIP for [domain]" — which was novel for about four months in 2023

These are not bad *learning* exercises. They are worth zero as hiring signal, because the reviewer has seen two hundred of them and none of them distinguished their author. The bottleneck was never "can you call `model.fit()`". Everyone can.

**What actually produces signal is one of exactly three things:**

- **You built the hard part yourself.** Not the model — the *system* around it: the data pipeline, the eval harness, the serving layer, the failure handling. Anyone can fine-tune. Very few people can tell you their p99 latency and why.
- **You measured something nobody had measured.** A benchmark, a reality check, a failure-mode map. This reads as scientific maturity, which is rarer than engineering skill.
- **Somebody who is not you uses it.** Real users, real traffic, or real GitHub stars. This is the strongest signal available and it is not close.

Everything below is designed to hit at least two of the three.

---

## 1. Ranking criteria

I scored each candidate on five axes, then ranked. I'm scoring on the projects' objective merits — technical depth, what a reviewer learns about you, and durability — not on what happens to be adjacent to your existing work.

| Axis | What it means |
|---|---|
| **Interview surface** | How many good questions does this let an interviewer ask you? A project you can talk about for forty minutes beats one you exhaust in five. |
| **Differentiation** | How many other candidates have something like it? |
| **Depth demonstrated** | Does finishing it require actually understanding the theory, or can it be assembled from tutorials? |
| **Shipping evidence** | Does it end with something running that someone else can use? |
| **Durability** | Will it still look good in 18 months, or does it depend on a model that will be superseded? |

---

## 2. The ranked list

### 🥇 #1 — A vision-model evaluation and regression-testing harness (`visdiff` / your name here)

**One line for the résumé:** *Open-source harness that catches silent quality regressions in vision models — slice-level metrics, failure-mode taxonomy, and CI integration; used by N projects.*

**What it is.** Not a model. Infrastructure. You take a detection/segmentation/VLM task and build the thing every team wishes it had and nobody builds: an eval system that goes beyond a single mAP number. Concretely:

- **Slice-based evaluation.** Overall mAP is a lie. Report per-slice: small/medium/large objects, crowded/sparse scenes, each lighting condition, each class. Surface the slice where the new model is *worse* even though the headline number improved.
- **A failure taxonomy**, implementing TIDE's six error types (classification, localisation, both, duplicate, background, missed) so a regression is diagnosable, not just detectable.
- **Calibration and confidence auditing** — reliability diagrams, ECE per slice. This is where focal-loss-trained detectors quietly fall apart, and you'll have derived exactly why in Module 4.
- **Golden-set regression tests that run in CI.** Model changes, CI fails, with a diff showing which 40 images got worse and how.
- **A report artefact** — an HTML page a non-ML teammate can read.

**Why it's #1.** Every axis at once. It requires real understanding (you cannot build a failure taxonomy without knowing what the failures *are*, which means Module 4 has to be genuinely absorbed). It is infrastructure, which is exactly the MLOps/AI-infra direction you want to be able to claim in 18 months. It is model-agnostic, so it does not rot when the SOTA changes — it's *more* useful as models proliferate. It is the rare project that other people will actually adopt, because the pain is universal and nobody enjoys solving it. And in an interview it opens onto metrics theory, statistics, systems design, and CI/CD — four different conversations.

**The differentiator that makes it credible:** ship it as a `pip install`-able package with real docs, and use it yourself on two or three of your other models so the README has genuine before/after screenshots showing a regression it caught. A tool with a real war story attached is unanswerable.

**Scope:** 6–8 weeks part-time. **Compute:** almost none — you're evaluating, not training. **Risk:** scope creep. Ship the slice-metrics core in week 3 and resist adding features until someone asks.

---

### 🥈 #2 — A production-grade multimodal retrieval service, built for scale you don't have

**One line:** *Multimodal (image + text) retrieval over 5M vectors — hybrid CLIP/SigLIP embeddings, quantised HNSW index, 40 ms p99, with a documented cost/latency/recall Pareto frontier.*

**What it is.** Image and text search over a genuinely large corpus, engineered as a service rather than as a notebook. The point is not that retrieval is novel. The point is that **the engineering decisions are all defensible and you can defend them**:

- Why SigLIP over CLIP (sigmoid loss removes the global-batch dependency, so batch size stops being a hyperparameter you can't afford)
- Why IVF-PQ at 5M vectors and flat below ~100k, with the crossover measured rather than assumed
- What `nprobe` and `efSearch` actually cost you, plotted as recall@10 vs latency vs index memory
- How you handle the cold-start / index-rebuild problem
- Where the reranker sits and whether it earns its latency

**Why it's #2 and not #1.** It is closer to something other candidates attempt, and it is downstream of your existing RAG experience, so it demonstrates breadth rather than a new capability. But it is still far above the median, *provided* you do the part everyone skips: **the measured Pareto frontier**. Anyone can stand up FAISS. Almost nobody can show you the curve and explain its shape.

**The differentiator:** the curve, plus honest cost accounting in rupees/month at each operating point. That combination — performance engineering with money attached — is precisely what an infra interview probes for.

**Scope:** 5–7 weeks. **Compute:** embedding 5M images is the expensive step; use a public precomputed embedding set (LAION subsets ship with them) rather than burning GPU-hours to prove a point you aren't making. **Risk:** turning into a demo app. Keep the emphasis on the measurement.

---

### 🥉 #3 — An honest reality check on a subfield (the research-flavoured capstone)

**One line:** *Re-evaluated N published [metric-learning / SSL / detection] methods under a single fair protocol; M of the claimed improvements do not survive.*

**What it is.** Pick a subfield where you suspect the comparisons have been unfair — and after Module 3, you'll suspect metric learning specifically, because Musgrave et al. already showed exactly this for the pre-2020 literature and *nobody has redone it for the post-2020 methods*. Reimplement or standardise 5–8 methods, fix the training budget, fix the augmentation, fix the architecture, fix the hyperparameter search protocol, and report what's left.

**Why it's #3.** The signal is enormous *if you finish* — this is genuinely research, it is publishable at a workshop, and it demonstrates the rarest quality in a junior candidate: willingness to check rather than assume. But the failure mode is severe. It is the only project on this list where six weeks of work can end with nothing presentable, and reimplementing other people's methods faithfully is miserable and slow.

**Take this one only if** you're keeping the PhD door open, or you already tried #1 and want something harder. Otherwise the risk-adjusted return is worse than #1.

**Scope:** 8–12 weeks. **Compute:** 40–80 GPU-hours. **Risk:** high, and the risk is total.

---

### #4 — Real-time CV on the edge, with the latency budget shown

**One line:** *[Task] at 30 FPS on a Raspberry Pi / Jetson Nano — INT8 quantised, with the full accuracy-vs-latency-vs-power trade-off measured on device.*

**What it is.** Take a real task, get it running on genuinely constrained hardware, and document every decision with numbers measured on the device rather than FLOPs counted on paper. Quantisation-aware vs post-training. Operator fusion. What the runtime (ONNX Runtime / TFLite / TensorRT) actually does to your graph. Where the frames are really going (usually: preprocessing, not inference — and demonstrating that you *found* that is worth more than the model choice).

**Why it's good but not top-three:** genuinely differentiated, because most candidates never touch hardware, and the "FLOPs lie, here's wall-clock" finding is a great interview story. But it needs a device (₹5–10k), and it's narrower — it argues you're an edge engineer specifically. Excellent as a *second* project alongside #1.

---

### #5 — A from-scratch implementation with a teaching artefact

**One line:** *Implemented [DDPM / DETR / 3D Gaussian Splatting] from scratch in PyTorch with an annotated walkthrough; N stars.*

**What it is.** Pick one hard thing and build it with no library help, then write the explanation you wished existed. The code is half of it; the writeup is the other half.

**Why it's mid-list.** It demonstrates depth convincingly and it's the single best thing you can do for *your own* understanding. But as hiring signal it's weaker than it feels, because reviewers can't easily distinguish "wrote it" from "transcribed it", and the genre is crowded (annotated-Transformer clones are everywhere). It converts to strong signal only if the writeup is genuinely better than what exists — which, given the self-study standard these notes are written to, is actually within reach for you.

**Best pick within it:** 3D Gaussian Splatting, because the CUDA rasteriser makes it substantially harder to fake than a DDPM, and far fewer people have done it.

---

### Traps — do not build these

| Project | Why not |
|---|---|
| "SAM + CLIP for medical/agricultural/retail X" | Every third portfolio. Zero differentiation. |
| Fine-tuned YOLO on a Roboflow dataset | Demonstrates you can follow a README. |
| Yet another Streamlit demo of a pretrained model | Reviewer learns nothing about you. |
| "Novel detection architecture" | You will not beat DINO-DETR, and losing to it silently is what will happen. |
| An LLM+vision chatbot with no eval | The absence of evaluation is the finding, and it's about you. |
| Anything whose README says "state of the art" without a table | Instantly discounted. |

---

## 3. My actual recommendation

**Build #1.** Then, if you have time, #2.

The reasoning: #1 is the only project on this list that simultaneously (a) requires you to have genuinely learned Module 4 rather than skimmed it, (b) points at infrastructure rather than modelling, which is where you've said you want to be in 18 months, (c) does not decay when the next foundation model lands, and (d) has a realistic path to being used by strangers — which is the strongest résumé signal that exists and the only one on this list you can't fake.

The counter-argument you should weigh: #1 is less visually impressive than a generative-model project, and if you are interviewing at product-led startups rather than infra teams, a demo people can *see* may open more doors. If that's the target, invert — build #2 first, because a working multimodal search over 5M images demos in ten seconds.

**How to present whichever you build.** A README with a results table, an architecture diagram, and an explicit "what doesn't work" section. That last section is the highest-signal paragraph in any portfolio: it proves you evaluated your own work honestly, and it is the thing nobody else writes.

---

## 4. Sequencing against the curriculum

| When | What |
|---|---|
| During M1–M3 | Small builds only — the per-concept 🔨 tasks. Do not start the capstone. |
| End of M4 | **Start #1.** You'll have just derived focal loss, mAP and TIDE's taxonomy; that's the whole intellectual content of the harness, fresh. |
| During M5 | Keep shipping #1. M5 is the least capstone-relevant module — treat it as reading. |
| During M6 | Ship #1 v1.0. Extend it to cover a VLM eval — this is where it stops being a detection tool and becomes generally useful. |
| After M6 | #2 if you want breadth, #4 if you want an infra angle, #3 only if research is the goal. |
