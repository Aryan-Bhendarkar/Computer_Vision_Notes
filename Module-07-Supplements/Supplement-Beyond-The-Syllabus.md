# Supplement — The Three Topics the Syllabus Omits Entirely

> Your gap analysis identified three areas that appear **nowhere** across all 16 sessions: **video understanding**, **explainability**, and **deployment**. This file closes them.
>
> Treat this as optional-but-strategic. None of it is on the exam. Two of the three (explainability, deployment) are asked about constantly in applied ML interviews, and deployment is directly on the path of your MLOps pivot. Video is the smallest gap for your goals — read it once, don't drill it.

---

## S.1 Video Understanding

### Intuition

An image tells you what is present. A video tells you what is *happening* — and "happening" is not visible in any single frame. A frame of someone standing near a door is ambiguous; the sequence tells you whether they are entering or leaving. Video understanding is the problem of extracting information that exists only in the *temporal* dimension.

Real-life anchor: a retail loss-prevention system can't tell "picked up an item and put it in a bag" from "picked up an item and put it back" from any single frame. The whole signal is in the ordering.

### The core problem: temporal modelling is expensive

A 10-second clip at 30 FPS is 300 images. Naive per-frame processing is 300× the cost of an image model, and it still misses the temporal signal. Every video architecture is a different answer to *"how do I model time without paying 300×?"*

### The architecture families

| Family | Idea | Representative | Trade-off |
|---|---|---|---|
| **Frame-level + pooling** | run an image model per frame, average the features | "CNN + average pooling" baselines | cheapest; **temporally blind** — shuffling the frames changes nothing |
| **Two-stream** | one RGB stream + one **optical flow** stream (5.7), fused late | Two-Stream (2014), **I3D** (2017) | flow explicitly encodes motion; but computing flow is expensive and it's a separate pipeline |
| **3-D convolution** | convolve over $(t, h, w)$ | C3D, **I3D** (inflate 2-D ImageNet kernels to 3-D), R(2+1)D, SlowFast | learns spatio-temporal features end to end; heavy |
| **Factorised (2+1)D** | a 2-D spatial conv followed by a 1-D temporal conv | R(2+1)D, S3D | **the separability trick from Module 1.2 applied to time** — far cheaper than full 3-D, and often *more* accurate because it adds a non-linearity between the two |
| **Two-rate** | a low-frame-rate "slow" pathway for semantics + a high-frame-rate lightweight "fast" pathway for motion | **SlowFast** (2019) | elegant: spatial semantics change slowly, motion changes fast — so sample each at its natural rate |
| **Video transformers** | attention over spatio-temporal tokens | TimeSformer, ViViT, **VideoMAE**, VideoSwin | **divided space-time attention** (attend over space, then over time, separately) cuts $O((THW)^2)$ to something affordable |
| **Foundation-model era** | image-language models extended to video | VideoCLIP, InternVideo, video-capable VLMs (Qwen-VL, Gemini-class) | video QA and captioning via a VLM with sampled frames + temporal position encoding |

**The single most useful design idea here is R(2+1)D's factorisation** — it is exactly the separable-convolution argument from 1.2 and 2.7, applied to the time axis, and it generalises: whenever you have an expensive $N$-dimensional operator, ask whether it factorises.

**SlowFast's insight is the second one worth carrying:** *different information changes at different rates, so sample each at its own rate.* That's a design principle, not just an architecture.

### Tasks and benchmarks

- **Action recognition** — classify a trimmed clip. Kinetics-400/600/700, Something-Something v2 (**deliberately requires temporal reasoning** — "pushing something left to right" vs "right to left" are different classes, so a frame-level model scores near chance; this dataset exists specifically to expose temporally-blind models).
- **Temporal action localisation** — *when* did it happen. ActivityNet, THUMOS.
- **Spatio-temporal detection** — where and when. AVA.
- **Video object segmentation (VOS)** — track a mask through a video. DAVIS, YouTube-VOS. **SAM 2 (6.8) is now the strong general answer here.**
- **Multi-object tracking** — covered in 5.7 (SORT/ByteTrack/HOTA).
- **Video QA / captioning** — increasingly handled by VLMs.

### 🎯 What separates a good answer

1. **"Check whether the task actually needs temporal modelling."** Many "video" problems are solved by a per-frame image model plus simple aggregation, and the Something-Something-vs-Kinetics contrast is the evidence: many Kinetics classes are recognisable from a single frame (context and objects give it away), which is why frame-level baselines are embarrassingly strong there and near-chance on Something-Something. **Diagnosing which regime you're in before building anything is the senior move.**
2. **The factorisation and multi-rate design principles** above.
3. **Frame sampling is a first-class design decision** — uniform, dense, or adaptive/keyframe sampling changes accuracy more than the architecture in many cases.
4. **Temporal consistency is often the real product requirement**, not accuracy: a detector that flickers between frames is worse for a user than a slightly less accurate one that's stable. Fixes are tracking, temporal smoothing, and hysteresis on class decisions.
5. **Cost reality**: video inference is usually gated by decode and I/O, not by the model. Measure the whole pipeline.

### 🔨 Build + read

**Build:** Take a per-frame CNN baseline and an R(2+1)D model, and evaluate both on a Kinetics subset **and** on Something-Something. The gap between the two datasets is the entire lesson of this section. Then implement frame-sampling ablations (1, 8, 16, 32 frames) and plot accuracy vs. cost.

**Read:** Carreira & Zisserman, "Quo Vadis, Action Recognition?" (I3D, CVPR 2017). Feichtenhofer et al., "SlowFast Networks" (ICCV 2019). Tong et al., "VideoMAE" (NeurIPS 2022) — MAE (6.4) applied to video, with a *90%+* masking ratio because video is even more redundant than images (the redundancy argument again, one step further).

---

## S.2 Explainability & Model Debugging

> Partially covered in 2.2 (Grad-CAM). This section completes it and — more importantly — frames it as a *debugging* discipline rather than a compliance checkbox, which is how it actually earns its keep.

### Why this matters for you specifically

In an applied ML interview, "how would you debug a model that has good metrics but bad real-world performance?" is one of the most common questions, and the strong answer is a *method*, not a tool list. This section is that method.

### The toolkit

| Method | What it gives | Cost | Caveat |
|---|---|---|---|
| **Grad-CAM / Grad-CAM++** | a class-discriminative heatmap over the last conv layer | one backward pass | coarse (last-layer resolution); undefined for pure transformers without adaptation |
| **Saliency / vanilla gradients** | per-pixel input sensitivity | one backward pass | very noisy |
| **SmoothGrad** | average saliency over noisy copies | $n$ passes | reduces noise, no new information |
| **Integrated Gradients** | attribution with a completeness axiom (attributions sum to the prediction difference from a baseline) | ~50 passes | **the baseline choice matters a lot** and is rarely justified |
| **Occlusion sensitivity** | slide a grey patch, measure the score drop | many passes | slow but assumption-free and very persuasive to non-ML stakeholders |
| **LIME / SHAP** | local surrogate / Shapley attributions | expensive | superpixel-dependent; SHAP's axioms are appealing but the vision approximations are loose |
| **Attention rollout / attention maps** (ViT) | which tokens the model attended to | cheap | **attention ≠ explanation** — weights don't reliably indicate causal importance |
| **Concept-based (TCAV, concept bottlenecks)** | does the model use a human-defined concept? | needs concept data | the most *actionable* form, and the closest to answering "why" |
| **Feature visualisation / activation maximisation** | what maximally excites a unit | optimisation per unit | beautiful, and of limited diagnostic value |

### The debugging method (this is the answer to the interview question)

1. **Look at the data first.** Sort the validation set by loss and inspect the worst 100. In practice a large fraction of "model bugs" are label noise, duplicates across splits, or a corrupted preprocessing path.
2. **Check the split.** Did you split by image when you should have split by patient / video / site / session? This is the most common silent invalidator of an entire evaluation.
3. **Run Grad-CAM on failures *and successes*.** "Right for the wrong reason" is invisible in the metrics and obvious in a heatmap. The canonical example: a pneumonia classifier reading the hospital-specific metal token in the corner of the X-ray rather than the lungs.
4. **Test for shortcut learning explicitly.** Train on a deliberately corrupted variant (blank out the object, keep the background; or blank the background, keep the object) and see how much accuracy survives. If the background-only model does well, your model is reading context, not content.
5. **Per-slice metrics.** Break accuracy down by every metadata axis you have — camera, site, lighting, class, object size, time of day. **Aggregate metrics hide the failure that will get you paged.**
6. **Calibration.** Reliability diagram + ECE (2.11). A model whose scores are consumed by a downstream system must be calibrated, and modern networks are systematically over-confident (and focal-loss detectors under-confident, 4.6).
7. **Robustness probes.** Common corruptions (ImageNet-C style: noise, blur, weather, digital), small translations (the shift-invariance failure of 2.1), and adversarial perturbations if the threat model warrants it.
8. **Ablate your own pipeline.** Turn each component off and measure. Components that contribute nothing are pure risk.

### 🎯 What separates a good answer

- **"Explainability is primarily a debugging tool for me, and secondarily a compliance artefact."** That framing is more useful and more honest than treating it as a reporting requirement.
- **The shortcut-learning test** (train on background only) — a concrete, falsifiable experiment, not a heatmap you squint at.
- **Attention is not explanation**, and Grad-CAM has known failure modes (it can be insensitive to the model's actual parameters in some sanity checks — see Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018, which showed several popular methods produce plausible-looking maps even for *randomised* models). **Citing that paper is a strong, slightly contrarian signal** — it shows you evaluate your evaluation tools.
- **Per-slice metrics before per-model tuning.**

### 🔨 Build + read

**Build:** On any classifier you've trained, implement Grad-CAM, occlusion sensitivity, and the background-only shortcut test. Then run the **sanity check from Adebayo et al.**: randomise the model's weights layer by layer and re-run your saliency method — if the maps barely change, the method is not telling you about your model.

**Read:** Selvaraju et al., "Grad-CAM" (ICCV 2017). Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018). Geirhos et al., "Shortcut Learning in Deep Neural Networks" (Nature MI, 2020) — a genuinely excellent paper and directly relevant to your interview answers.

---

## S.3 Deployment: Quantisation, Export, and Serving

> **This is the section most directly on your MLOps/infra pivot**, and the one that most reliably differentiates candidates in applied interviews — because most people can train a model and far fewer can say what happens to it afterwards.

### The pipeline

```
PyTorch model
   │
   ├─ graph capture:  torch.export / torch.compile / ONNX export / TorchScript
   │
   ├─ graph optimisation:  operator fusion, constant folding, BN FOLDING (2.4),
   │                       dead-code elimination, layout transforms (NCHW→NHWC)
   │
   ├─ precision:  fp32 → fp16/bf16 → int8 (PTQ or QAT) → int4 (rare in vision)
   │
   ├─ runtime:  TensorRT (NVIDIA) · ONNX Runtime · OpenVINO (Intel) ·
   │            TFLite/LiteRT + NNAPI (Android) · Core ML (Apple) · TVM
   │
   └─ serving:  Triton / TorchServe / a custom server — batching, concurrency,
                model versioning, warmup, health checks, autoscaling
```

### Quantisation — the part to actually understand

**Why it works:** neural network weights and activations have far less dynamic range than fp32 provides. Mapping them to int8 gives **4× smaller weights** and, on hardware with int8 tensor cores, **2–4× faster compute** — and for the memory-bandwidth-bound layers (2.7's depthwise convolutions), the win is close to linear in the bytes moved.

**Affine quantisation:**
$$
q = \text{round}\!\left(\frac{r}{S}\right) + Z, \qquad r \approx S\,(q - Z)
$$
with scale $S$ and zero-point $Z$. **Symmetric** ($Z=0$) is cheaper and standard for weights; **asymmetric** is used for activations after a ReLU, whose distribution is one-sided.

**Granularity matters:** **per-tensor** is cheapest; **per-channel** (a separate scale per output channel) is dramatically better for convolutions, because channel weight ranges vary by orders of magnitude. **Per-channel weight quantisation is usually the difference between int8 working and int8 destroying your accuracy** — and it costs nothing at inference.

**PTQ vs QAT:**
- **Post-Training Quantisation** — calibrate scales on a few hundred representative images. Minutes of work. Usually costs <1% accuracy for a well-behaved model.
- **Quantisation-Aware Training** — simulate quantisation during fine-tuning with a straight-through estimator for the gradient. Costs a training run; recovers most of the gap when PTQ fails.

**What breaks under quantisation** (know these, they're the real interview content):
- **Depthwise convolutions** — per-channel weight ranges vary enormously, so per-tensor quantisation is catastrophic. Per-channel is mandatory.
- **Layers with extreme activation outliers** — a few large values stretch the scale and crush the rest. This is the same phenomenon that drives LLM quantisation research (SmoothQuant-style outlier handling).
- **Anything after a BatchNorm that wasn't folded**, and non-standard activations (h-swish is fine and was designed to be; arbitrary custom ops often aren't supported by the runtime at all).
- **The first and last layers** — usually kept in higher precision by default, for good reason.
- **Detection post-processing** — NMS, box decoding, and score thresholds behave differently at reduced precision, and detection AP is more quantisation-sensitive than classification accuracy. **Always re-measure AP, not just latency.**

### Measuring latency honestly

- **Warm up** (CUDA kernel autotuning, JIT compilation, memory allocator) before timing. First-call latency is not your latency.
- **Synchronise** before and after on GPU (`torch.cuda.synchronize()`), or you're timing kernel *launches*.
- Report **p50 and p99**, not the mean. Tail latency is what users experience and what breaks SLAs.
- **Include preprocessing and postprocessing.** For detectors, letterboxing + normalisation + NMS is routinely 30–50% of wall-clock (4.13) and is invisible in FLOP counts.
- Measure at your **real batch size and real concurrency**, on the **real hardware**, under **sustained load** — mobile SoCs thermally throttle, and a 30-second benchmark won't show it.
- **Throughput ≠ latency.** Batching raises throughput and *raises* per-request latency. Know which one your product is bound by.

### Serving concerns

- **Dynamic batching** (Triton): accumulate requests for a few ms to fill a batch. Big throughput win, small latency cost — tune the queue delay against your p99 budget.
- **Model versioning and shadow deployment**: run the new model alongside the old on live traffic, compare outputs, and only then switch.
- **Monitoring**: input distribution drift (embedding statistics, image brightness/size histograms), prediction distribution drift, per-slice metrics, and — where you can get it — delayed ground truth. **Model quality degrades silently; only instrumentation catches it.**
- **The foundation-model serving pattern** (2.10, 6.7): one frozen backbone, many task heads. One artefact to version, quantise, cache, and monitor; heads are megabytes. This is the architecture-level answer that ties Modules 2, 3 and 6 together from an infra perspective.

### 🎯 What separates a good answer

1. **Per-channel quantisation for convolution weights**, with the depthwise-convolution reason.
2. **BN folding** (2.4) as a free 10–30% win.
3. **Measure end to end, including pre/post-processing**, with p99 and warmup.
4. **Detection AP is more quantisation-sensitive than classification accuracy** — re-measure the task metric, not a proxy.
5. **Throughput vs latency** as distinct objectives with opposite responses to batching.
6. **Shadow deployment and drift monitoring** — the answer to "how do you know the model is still good?"
7. **The FLOPs ≠ latency roofline argument** (2.7) — the single most reusable systems insight in this whole curriculum.

### 🔨 Build + read

**Build — do this once, properly, and it becomes an interview story.** Take your Module 2 classifier and your Module 4 detector and put both through the full pipeline: export to ONNX, verify numerical parity, fold BN, run PTQ int8 with per-tensor *and* per-channel weight quantisation, and produce a table of **accuracy/AP, model size, p50 latency, p99 latency** for {fp32, fp16, int8-per-tensor, int8-per-channel}, on CPU and GPU, with preprocessing and postprocessing broken out. Then, if int8 loses too much AP, do a QAT run and add the row.

**Read:** Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018) — the foundational paper, still the clearest. Then the PyTorch quantisation docs and NVIDIA's TensorRT best-practices guide. For the systems framing, Williams et al.'s roofline model paper.

---

## Where these connect back

| Supplement topic | Ties back to |
|---|---|
| R(2+1)D factorisation | separable convolution (1.2), depthwise-separable (2.7) |
| SlowFast multi-rate | scale-space's "process at multiple rates" idea (1.4) |
| VideoMAE's 90% masking | MAE's redundancy argument (6.4) |
| SAM 2 for VOS | 6.8 |
| Tracking metrics (HOTA) | 5.7; and PQ = SQ × RQ (4.12) — the same factorise-the-metric idea |
| Grad-CAM, shortcut learning | 2.2, 2.11 |
| Calibration | 2.11, and focal loss's cost (4.6) |
| BN folding | 2.4 |
| Per-channel quantisation of depthwise convs | 2.7 |
| FLOPs ≠ latency, roofline | 2.7 — the single most reusable idea in the curriculum |
| One frozen backbone, many heads | 2.10, 6.7 |
