# Supplement — Robustness, Scale, and Governance

> **Why this file exists.** A coverage audit found three areas absent from the six modules that a comprehensive CV education must contain and that applied interviews probe directly: **robustness and uncertainty** (adversarial examples, corruptions, OOD detection, "how does your model say *I don't know*?"), **the scaling and compression toolkit** (distillation, pruning, distributed training, long-tail data), and **governance** (dataset licensing, biometric regulation, bias measurement) — the last being the area where a mistake is career-affecting rather than merely embarrassing.
>
> None of this is glamorous. All of it comes up.

---

## S.11 Adversarial Examples, Corruptions, and OOD Detection

Three different questions that are constantly conflated. Separating them is most of the value:

| Question | Threat model | The field |
|---|---|---|
| "Can someone *deliberately* fool this?" | an adversary who can perturb the input | adversarial robustness |
| "Does this survive *ordinary* real-world degradation?" | nature — rain, blur, JPEG, sensor noise | corruption robustness |
| "Is this input even the kind of thing I was built for?" | inputs outside the training distribution | OOD detection |

### Adversarial examples

An imperceptible perturbation that changes the prediction. **FGSM** (one step):

$$
x_{\text{adv}} = x + \varepsilon\cdot\operatorname{sign}\bigl(\nabla_x \mathcal{L}(f(x), y)\bigr)
$$

Take the gradient with respect to the *input* instead of the weights, and step in the direction that increases loss. The `sign` makes it an $\ell_\infty$-bounded attack — every pixel moves by exactly $\varepsilon$, so the perturbation is bounded in the max-norm and typically invisible at $\varepsilon = 8/255$.

**PGD** iterates FGSM with small steps, projecting back into the $\varepsilon$-ball each time, and is the standard strong first-order attack — "PGD-robust" is the accepted meaning of "adversarially robust".

**Why they exist** is more interesting than how to make them. The leading explanation (Ilyas et al., 2019) is that they are **not bugs but features**: datasets contain genuinely predictive but **non-robust** features — patterns that correlate with the label and are imperceptible to humans — and a model trained to maximise accuracy uses them, because nothing told it not to. Adversarial examples are then an artefact of a mismatch between what generalises statistically and what humans consider the object. **That framing is what separates a strong answer.**

**Defences, honestly.** Almost every proposed defence has been broken by an adaptive attack. The one that survives is **adversarial training** — train on PGD-generated examples — which works and costs you: 3–30× training compute and typically several points of clean accuracy. **The robustness/accuracy trade-off appears to be real, not an artefact.** Be sceptical of any defence evaluated only against FGSM or against attacks that don't know the defence exists; **gradient masking** (making gradients uninformative without making the model robust) is the classic way defences fool their own evaluation.

**Is this your threat model?** For most products, no — nobody is crafting $\ell_\infty$ perturbations against your retail shelf detector. It matters for face recognition and biometric spoofing, content moderation (adversaries are real and motivated), and any safety-critical system where physical adversarial patches (printed stickers on stop signs) are plausible. **Saying "adversarial robustness is not my threat model here, corruption robustness is" is a better answer than reciting PGD.**

### Corruption robustness

**ImageNet-C** applies 15 corruptions (Gaussian/shot/impulse noise, defocus/glass/motion/zoom blur, snow/frost/fog/brightness, contrast, elastic, pixelate, JPEG) at 5 severities, and reports **mCE** — mean corruption error, normalised against a baseline model so the number is comparable across architectures.

This is the robustness that actually matters in production, and the fixes are unglamorous and effective: **train with the corruptions you expect** (augment for your real domain gap, 2.9); **AugMix** and **DeepAugment** improve corruption robustness broadly without training on the test corruptions; larger models and more pretraining data help substantially; and **anti-aliased downsampling** (BlurPool, 2.1) improves consistency under small shifts.

### Out-of-distribution detection

**The question is different from "am I confident?"** — modern networks are confidently wrong on inputs from classes they have never seen, so softmax confidence alone is a weak detector.

| Method | Score | Note |
|---|---|---|
| **MSP** (max softmax probability) | $\max_k p_k$ | the baseline; weak but not useless |
| **ODIN** | temperature scaling + a small input perturbation | improves MSP substantially for little cost |
| **Energy score** | $-T\log\sum_k e^{z_k/T}$ | uses the logits' *magnitude*, not just their ratio; strictly better-motivated than MSP and usually better |
| **Mahalanobis** | distance to class-conditional Gaussians fitted to penultimate features | works in feature space, catches inputs that are far from all training data |
| **k-NN distance** | distance to the $k$-th nearest training feature | simple, strong, and **exactly Module 3's machinery again** |
| **Deep ensembles** | disagreement across independently trained models | strongest, most expensive |

**The practical framing:** OOD detection is a *retrieval* problem in disguise — "is this input near anything I was trained on?" — which is why feature-space distance methods work and why your Module 3 index can double as an OOD detector for free.

### 🎯 Top-1% distinction

1. **Separate the three questions** and say which is your actual threat model.
2. **"Adversarial examples are non-robust features, not bugs"** (Ilyas et al.).
3. **Adversarial training is the only defence that survives adaptive attacks**, and it costs clean accuracy — the trade-off is real.
4. **Be sceptical of defences evaluated against weak attacks**; name gradient masking.
5. **Softmax confidence is not an OOD detector**; energy scores and feature-space distances are better, and feature distance is retrieval.
6. **For most products, corruption robustness is the one that pays.**

### ✅ Mastery check

Your face-verification system for building access will be attacked. Your colleague proposes adversarial training.

(a) Is PGD adversarial training the right defence for this threat model? What is the actual attack?
(b) The system must reject people not enrolled in the database. Which of the three problems above is that, and what would you build?
(c) You add adversarial training and clean accuracy drops from 99.2% to 97.1%. Is that acceptable? Show the reasoning.

<details><summary>Answer sketch</summary>
(a) <b>Largely no — the threat model is wrong.</b> An $\ell_\infty$ digital perturbation assumes the attacker can modify the tensor reaching your model, but a physical access-control system's attacker stands in front of a camera. The real attacks are <b>presentation attacks / spoofing</b>: a printed photo, a phone screen, a video replay, a silicone mask, or a physical adversarial patch (printed glasses). The defence is <b>liveness detection / presentation-attack detection</b> — depth or IR sensing (an RGB-D or NIR camera defeats printed photos almost entirely), texture and moiré analysis, challenge-response (blink, turn), and rPPG pulse detection. Adversarial training defends a channel the attacker doesn't have; PAD defends the one they do. (Physical adversarial patches <i>are</i> a real concern, and adversarial training helps a little there, but it is not the primary defence.)
(b) That is <b>open-set recognition / OOD detection</b>, not classification — and it is the harder half of face recognition. Build it as a <b>verification</b> system, not an identification one: embed the face (ArcFace-family, 3.5), compare against the enrolled gallery, and <b>accept only if the similarity exceeds a threshold</b>; if the nearest enrolled identity is too far, reject. Calibrate that threshold against a large set of <b>non-enrolled</b> faces (impostor distribution), and report the operating point as FAR/FRR or as TAR at a fixed FAR — e.g. "98% true accept at 1-in-100,000 false accept", which is how the industry actually specifies it. Note this is exactly why the field uses metric learning rather than a softmax classifier: the gallery changes as people join and leave, and you cannot retrain a classifier for every new employee.
(c) <b>Show the arithmetic rather than judging in the abstract.</b> A 2.1-point clean-accuracy drop on an access system with, say, 2,000 badge-ins a day means roughly 42 additional failures a day — people standing at a door that won't open, who then call security, which is both a cost and a reason staff will demand the system be disabled. Against that, quantify what you bought: if the threat model is physical presentation attacks (per (a)), adversarial training bought you <b>almost nothing against the actual attack</b>, so the trade is 42 daily failures for a defence against an attack nobody is running — clearly unacceptable. If the threat model genuinely included digital perturbation (say the images arrive over a network from an untrusted client), the calculus changes and you would weigh it against the cost of a single successful unauthorised entry. <b>The method is the answer: convert both sides into incidents per day and cost per incident, then decide.</b> And note the third option — spend the same effort on liveness detection, which addresses the real threat at no clean-accuracy cost.
</details>

---

## S.12 Uncertainty Estimation

**Calibration (2.11) and uncertainty are different things.** Temperature scaling fixes *calibration* — making a stated 0.8 mean 80% correct on average. It does nothing about **epistemic** uncertainty — the model's ignorance about inputs unlike its training data.

- **Aleatoric** uncertainty: irreducible noise in the data (a blurry image genuinely is ambiguous). More data does not help. Modelled by predicting a variance alongside the mean.
- **Epistemic** uncertainty: uncertainty about the *model*, from limited data. More data does help. Requires a distribution over models.

**Methods, in ascending cost:**

| Method | Mechanism | Cost |
|---|---|---|
| **Temperature scaling** | one scalar fitted on validation | free; fixes calibration only |
| **MC dropout** | keep dropout on at test time, sample $N$ forward passes, use the predictive variance | $N\times$ inference; a crude posterior approximation |
| **Deep ensembles** | train $M$ models from different seeds, average and measure disagreement | $M\times$ everything; **consistently the strongest baseline** and the honest answer when accuracy of uncertainty matters |
| **Conformal prediction** | calibrate a score threshold on a held-out set to output a *prediction set* | ~free, and gives a **distribution-free coverage guarantee** |

**Conformal prediction deserves the emphasis.** Instead of a point prediction with a dubious confidence, it outputs a *set* of labels guaranteed to contain the truth with probability $1-\alpha$, under only an exchangeability assumption — no assumptions about the model or the data distribution. Procedure: on a calibration set, compute a nonconformity score (e.g. $1 - p_{\text{true class}}$) for every example, take the $\lceil(n{+}1)(1{-}\alpha)\rceil$-th smallest as the threshold $\hat q$, and at test time output $\{k : p_k \ge 1 - \hat q\}$. The set is **large when the model is uncertain and small when it is confident**, which is a far more honest interface than a number. It is increasingly the right answer for medical and safety applications, and very few candidates know it.

### 🎯 Top-1% distinction

- **Aleatoric vs epistemic**, and that only the second is reducible by more data.
- **Calibration ≠ uncertainty.** Temperature scaling does not make a model know what it doesn't know.
- **Deep ensembles remain the strongest baseline** despite years of cheaper proposals — say it, it is the honest state of the field.
- **Conformal prediction gives a distribution-free guarantee** and outputs sets, not scores.

---

## S.13 Long-Tail and Class Imbalance

Focal loss (4.6) solves **foreground/background** imbalance in detection. **Long-tail class imbalance is a different problem** and gets confused with it constantly.

Real datasets are Zipfian: a few head classes with thousands of examples, a long tail with a handful each. A model trained naively achieves good overall accuracy while being useless on the tail.

**The key empirical finding (Kang et al., "Decoupling Representation and Classifier"):** train normally with instance-balanced sampling and the **representation is fine** — it is only the **classifier** that is biased toward head classes. So the effective recipe is two-stage: learn the representation with ordinary sampling, then **retrain or rebalance only the classifier** (class-balanced sampling, or simply re-normalising the classifier weights). This is cheap and surprisingly effective, and it is the thing to know.

**Other tools:**
- **Effective-number reweighting** (Cui et al.): weight class $c$ by $\frac{1-\beta}{1-\beta^{n_c}}$ rather than by $1/n_c$ — because samples overlap in information, the marginal value of the $n$-th example decays, and naive inverse-frequency over-weights the tail.
- **Logit adjustment**: subtract $\tau\log \pi_c$ (the log class prior) from the logits at inference, or add it during training. Principled — it is the Bayes-optimal correction for a shifted prior — and free.
- **Re-sampling**: oversample the tail (risks overfitting to the few examples), undersample the head (discards data), or square-root sampling as a compromise.
- **Mixup and strong augmentation** disproportionately help the tail.

**Always report per-class or grouped (head/medium/tail) accuracy.** An aggregate number on a long-tailed dataset is close to meaningless — it measures the head.

---

## S.14 Model Compression: Distillation and Pruning

S.3 covers quantisation. The other two legs:

### Knowledge distillation

Train a small **student** to match a large **teacher**'s *softened* outputs:

$$
\mathcal{L} = \alpha\,T^2 \cdot \mathrm{KL}\Bigl(\sigma(z_t/T)\,\big\|\,\sigma(z_s/T)\Bigr) + (1-\alpha)\,\mathcal{L}_{\mathrm{CE}}(z_s, y)
$$

**Why the temperature $T$:** softening the teacher's distribution reveals the relative probabilities of the *wrong* classes — that a cat image scores higher on "dog" than on "truck". Hinton called this **dark knowledge**, and it is a much richer signal per example than a one-hot label: it tells the student about the *similarity structure* of the label space, not just the answer. The $T^2$ factor rescales the gradients so that $T$ can be changed without retuning the learning rate.

**Two things to connect:**
- **Label smoothing hurts distillation** (2.8) — precisely because it erases the relative wrong-class information that distillation transfers. Being able to link those two facts is a strong signal.
- **Adversarial distillation is how diffusion models get to 1–4 steps** (6.13) — distillation is not only a compression technique.

Variants: **feature/hint distillation** (match intermediate activations, not just logits), **attention transfer**, and **self-distillation** (a model teaching a copy of itself, which improves it — surprisingly).

### Pruning and sparsity

- **Unstructured (magnitude) pruning** — zero the smallest weights. Achieves very high sparsity (90%+) with little accuracy loss, but **gives no speedup without sparse kernels**, which most hardware and runtimes lack. It compresses storage, not latency. This is the point people miss.
- **Structured pruning** — remove whole channels, filters, or heads. Lower achievable compression, but the result is a *smaller dense model* that runs faster everywhere with no special support. **This is what you use in production.**
- **2:4 semi-structured sparsity** — exactly 2 of every 4 contiguous weights are zero, which NVIDIA Ampere+ tensor cores accelerate natively (~2×). The hardware-supported middle ground.
- **Iterative magnitude pruning** (train → prune → fine-tune → repeat) beats one-shot pruning substantially. The **lottery ticket hypothesis** — that a sparse subnetwork exists which trains to full accuracy from the original initialisation — is the well-known research framing; treat it as an interesting result rather than a practical recipe.

**The decision rule to state:** quantisation first (largest, most reliable win, best supported), then structured pruning if you need more, then distillation if you need a genuinely different architecture. Unstructured pruning only if your deployment stack actually exploits sparsity.

---

## S.15 Training at Scale

Directly relevant to an MLOps/infra pivot, and absent from the modules.

**Data parallelism — the default.** **DDP** replicates the model on every GPU, each processes a different shard of the batch, and gradients are averaged with an **all-reduce** before the optimiser step. Simple and near-linear up to the point where communication dominates.

**Sharding — when the model doesn't fit.** **ZeRO / FSDP** shard the optimiser state, gradients, and finally parameters across ranks, gathering each parameter only for the forward/backward that needs it. ZeRO stage 1 shards optimiser state (the biggest win for Adam, which stores 8 bytes/param), stage 2 adds gradients, stage 3 adds parameters. **Trades communication for memory** — which is the right trade when memory is the binding constraint.

**Large-batch training.** Scaling the batch without adapting the learning rate under-trains. The standard recipe (Goyal et al., "Accurate, Large Minibatch SGD"): **linear LR scaling** — multiply the LR by the same factor as the batch — plus a **gradual warmup** over the first few epochs, because the linear-scaling rule breaks down early in training when the loss landscape changes fast. Beyond ~8k batch size, layer-wise adaptive optimisers (**LARS**, **LAMB**) become necessary because different layers need different effective learning rates.

**SyncBatchNorm.** This follows directly from 2.4: with DDP each rank computes BN statistics over only *its* micro-batch, so at 8 images per GPU your BN sees 8 samples regardless of the global batch of 256. `SyncBatchNorm` all-reduces the statistics across ranks. **Essential for detection and segmentation**, which run small per-GPU batches — and note the cost is an extra communication per BN layer, which is why GroupNorm is sometimes preferred instead.

**Memory levers, in order of use:** mixed precision (fp16/bf16 — roughly halves activation memory and is nearly free); **gradient checkpointing** (recompute activations in the backward pass instead of storing them — ~30% more compute for a large memory saving, and the standard tool when activations dominate, which they do for high-resolution vision); gradient accumulation (larger effective batch without more memory — but remember it does *not* fix BN, 2.4); and finally sharding.

**The estimation question** ("how much memory to train this?") is in `Supplement-Interview-Readiness.md` §S.9 — ~16 bytes/parameter for fp32 Adam states, plus activations, which usually dominate in vision.

---

## S.16 Ethics, Licensing, and Governance

> **Read this section even though it isn't technical.** Shipping on a mis-licensed dataset or deploying face recognition without checking the jurisdiction is a *career-affecting* mistake, and it is one that competent engineers make routinely because nobody taught them to look.

### Dataset licensing and provenance

**"Publicly available" is not a licence.** The questions to ask before training on any dataset:

- **What is the licence, and does it permit commercial use?** Many academic datasets are explicitly **research-only** (e.g. numerous face datasets). Training a commercial model on a research-only dataset is a licence violation regardless of how the weights are distributed.
- **Did the dataset creator have the right to grant it?** Web-scraped datasets typically distribute *URLs and annotations*, not images — the underlying images remain under their original copyright, and the images' licences vary per item.
- **Has it been withdrawn or amended?** Several well-known datasets have been retracted or had classes removed over consent and content problems. Using a cached copy of a withdrawn dataset is a real risk, and "we downloaded it before" is not a defence.
- **Model licences propagate.** A model trained on AGPL-licensed code, or fine-tuned from weights with a restrictive licence, carries obligations downstream. This is exactly the Ultralytics-YOLO-is-AGPL point from 4.5, generalised.

**Practical discipline:** record the licence, source, download date, and version of every dataset in the model card, and get a legal read before commercial training on anything you did not create.

### Biometrics and regulated uses

Face and body data are regulated differently from other images, and the rules are **jurisdictional**:

- **GDPR Article 9** treats biometric data used for unique identification as a special category, requiring an explicit lawful basis; and Article 17's right to erasure raises the question of what "deleting" a person means once their face is in trained weights.
- **BIPA** (Illinois) requires informed written consent before collecting biometric identifiers and has produced very large settlements — it is the most commonly litigated statute in this space, and it applies to companies outside Illinois that process Illinois residents' data.
- The **EU AI Act** restricts real-time remote biometric identification in public spaces and imposes obligations on high-risk systems, with staged application dates.
- Sector rules (HIPAA for medical images, FERPA for students, sector-specific rules for finance and children's data) apply independently.

**The engineering consequences** are concrete: prefer on-device processing and template storage over raw biometric images; store irreversible embeddings rather than faces where possible; build deletion into the data pipeline from the start rather than retrofitting it; and be able to say what data a model was trained on.

### Measuring bias

Do not assert fairness — **measure it**, and report per-slice.

- **Gender Shades** (Buolamwini & Gebru, 2018) is the methodology to know: disaggregate error rates by intersectional subgroup rather than reporting a single accuracy. The finding — that commercial gender classifiers had error rates of ~1% for lighter-skinned men and ~35% for darker-skinned women, invisible in the aggregate — is the canonical demonstration that **aggregate metrics hide the failures that matter**.
- Choose a fairness criterion deliberately: **demographic parity** (equal positive rates), **equalised odds** (equal TPR and FPR), or **calibration within groups**. These are **mutually incompatible** except in degenerate cases (Kleinberg et al.) — so you must pick one and justify it rather than claiming "fair".
- Watch for **proxy variables**: removing a protected attribute does not remove it from the model if it is inferable from other features, which in images it almost always is.

### Disclosure artefacts

**Model cards** (Mitchell et al.) and **datasheets for datasets** (Gebru et al.) are the standard formats: intended use, out-of-scope uses, training data and its provenance, evaluation including **disaggregated** results, known limitations, and ethical considerations. They are increasingly required by procurement and regulation, and writing one is a good forcing function — it makes you notice what you don't know about your own system.

### Surveillance and dual use

Vision is disproportionately dual-use: a people-counter is a crowd-monitoring system; a re-identification model is a tracking system. **Decide your position before you're asked to build it**, and know that "I just build the models" has not been a durable professional stance in this field.

---

## S.17 The CV Tooling Ecosystem

Every build task in this curriculum assumes tooling the reader has to discover alone. Here it is.

| Need | Reach for | Note |
|---|---|---|
| Classical CV, I/O, geometry | **OpenCV** (`opencv-python`) | SIFT is in the main build since 4.4.0; SURF needs `opencv-contrib` + `OPENCV_ENABLE_NONFREE` |
| Differentiable classical ops | **Kornia** | classical CV as PyTorch ops — useful for putting geometry inside a network |
| Pretrained backbones | **timm** | the definitive collection; also the reference for modern training recipes |
| Augmentation | **Albumentations** | fast, and handles bboxes/masks/keypoints consistently — which torchvision transforms historically did not |
| Detection/segmentation training | **Ultralytics** (fast, AGPL), **Detectron2** (Meta, research-grade), **MMDetection** (huge model zoo, steep config learning curve) | pick by licence and by how much you need to modify |
| Foundation models | **Hugging Face** `transformers`, `diffusers`, `timm` | the default for anything from Module 6 |
| 3-D | **Open3D** (point clouds, ICP, reconstruction), **PyTorch3D**, **COLMAP** (SfM/MVS) | |
| Vector search | **FAISS**, **hnswlib**; **Qdrant**/**Milvus**/**pgvector** as databases | 3.7 |
| Dataset inspection and error analysis | **FiftyOne** | genuinely underused — visualise predictions vs ground truth, find label errors, slice by metadata |
| Annotation | **CVAT**, **Label Studio**, **Roboflow** | |
| Experiment tracking | **Weights & Biases**, **MLflow**, TensorBoard | |
| Deployment | **ONNX Runtime**, **TensorRT**, **OpenVINO**, **TFLite/LiteRT**, **Core ML** | S.3 |

**Annotation formats** — you will convert between these constantly:

- **COCO JSON** — the lingua franca. One file, `images`/`annotations`/`categories`; boxes as `[x, y, w, h]` absolute.
- **YOLO txt** — one `.txt` per image, one line per object: `class cx cy w h`, **normalised to $[0,1]$**, centre-based.
- **Pascal VOC XML** — one `.xml` per image, boxes as absolute `xmin ymin xmax ymax`.

**Three different box conventions in three formats** — absolute-corner, absolute-xywh, and normalised-centre-xywh. Conversion bugs between them are among the most common causes of "my model trains but detects nothing", and they are silent: the loss goes down, the boxes are just in the wrong place. **Always visualise your loaded annotations on the image before training.** That single habit prevents more wasted days than any other.

---

## Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Three robustness questions | adversarial (an adversary) vs corruption (nature) vs OOD (wrong input entirely) — **say which is your threat model** |
| FGSM / PGD | $x + \varepsilon\,\mathrm{sign}(\nabla_x\mathcal{L})$; PGD iterates and projects — the standard strong attack |
| Why adversarial examples exist | **non-robust but genuinely predictive features** (Ilyas et al.) — not bugs |
| Defences | only **adversarial training** survives adaptive attacks; costs 3–30× compute and clean accuracy; beware **gradient masking** |
| ImageNet-C | 15 corruptions × 5 severities, reported as **mCE**; this is the robustness that pays in production |
| OOD | softmax confidence is weak; **energy scores** and **feature-space distance** are better — and distance is Module 3 retrieval |
| Aleatoric vs epistemic | data noise (irreducible) vs model ignorance (reducible by data) |
| Calibration ≠ uncertainty | temperature scaling fixes the number, not the ignorance |
| Uncertainty methods | MC dropout (cheap, crude) · **deep ensembles (strongest, expensive)** · **conformal prediction (distribution-free coverage guarantee, outputs sets)** |
| Long-tail | **the representation is fine; only the classifier is biased** ⇒ decouple and retrain the classifier |
| Reweighting | effective-number $\frac{1-\beta}{1-\beta^{n_c}}$, not $1/n_c$; **logit adjustment** by $\tau\log\pi_c$ is free and principled |
| Distillation | match softened teacher logits; $T$ reveals **dark knowledge** (wrong-class similarity structure); $T^2$ rescales gradients; **label smoothing destroys it** |
| Pruning | unstructured = compression without speedup; **structured = real latency win**; 2:4 sparsity is the hardware-supported middle |
| Compression order | quantise → structured prune → distil. Unstructured only if your runtime exploits sparsity |
| DDP vs FSDP/ZeRO | replicate + all-reduce gradients vs shard optimiser/gradients/params — **trade communication for memory** |
| Large batch | **linear LR scaling + warmup** (Goyal); LARS/LAMB beyond ~8k |
| SyncBN | per-rank BN sees only the micro-batch — essential for detection/segmentation; **gradient accumulation does not fix BN** |
| Memory levers | mixed precision → **gradient checkpointing** (~30% compute for large memory savings) → accumulation → sharding |
| Licensing | "publicly available" ≠ licensed; research-only datasets are common; **model licences propagate** (AGPL) |
| Biometrics | GDPR Art. 9, **BIPA**, EU AI Act — jurisdictional; prefer on-device, store embeddings not faces, build deletion in |
| Bias | **disaggregate by intersectional subgroup** (Gender Shades); parity/equalised-odds/calibration are **mutually incompatible** — pick and justify |
| Disclosure | model cards + datasheets; forcing function as much as compliance |
| Box formats | COCO abs-xywh · YOLO **normalised centre**-xywh · VOC abs-corners — **always visualise loaded annotations before training** |

### 🔨 Build tasks

**1 — Attack your own model (one sitting, no GPU needed for a small model).** Implement FGSM and PGD against your Module 2 CIFAR classifier. Plot accuracy vs $\varepsilon$. Then adversarially train and re-plot both clean and robust accuracy — you will measure the robustness/accuracy trade-off yourself, which is worth more than any citation of it.

**2 — OOD detection as retrieval (one sitting).** Take the embedding index you built in 3.7 and use it as an OOD detector: score test images by their $k$-NN distance in the index. Evaluate on an in-distribution vs out-of-distribution split (CIFAR-10 in, SVHN out is the standard pair) and compare AUROC against max-softmax-probability and the energy score. **This makes the "OOD is retrieval" point concrete and reuses work you have already done.**

**3 — Compression trade study (half a day).** Take your Module 4 detector and produce one table: fp32 baseline, int8 quantised, structurally pruned to 50% channels, and a distilled smaller student — reporting AP, model size, and measured p50/p99 latency for each on your target hardware. **This is a portfolio artefact for an MLOps-track interview** and answers "how would you get this on-device?" with your own numbers.

**Read:** Ilyas et al., "Adversarial Examples Are Not Bugs, They Are Features" (NeurIPS 2019). Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions" (ICLR 2019). Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021) — genuinely gentle and the best entry point. Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition" (ICLR 2020). Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (2015). Goyal et al., "Accurate, Large Minibatch SGD" (2017). Buolamwini & Gebru, "Gender Shades" (FAT* 2018). Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).
