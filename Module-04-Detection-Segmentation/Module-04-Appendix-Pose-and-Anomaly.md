# Module 4 — Appendix: Pose Estimation & Anomaly Detection

> **Why this appendix exists.** The main Module 4 covers the two famous dense-prediction tasks — detection and segmentation — and omits the third (**keypoint/pose estimation**) and the most commercially common non-obvious one (**anomaly detection**). Both use machinery that appears nowhere else in the curriculum, and both are asked about directly in interviews for applied roles.

---

## A4.1 Human Pose & Keypoint Estimation

### Intuition

Instead of a box around a person, predict the pixel location of each of their joints — wrists, elbows, knees — and connect them into a skeleton. It is detection where the "object" is a single point, repeated $K$ times per instance, with a known structural relationship between the points.

Real-life anchor: fitness apps counting reps, motion capture without markers, sports analytics, hand tracking on a VR headset, and the pose conditioning input to ControlNet (6.15).

### The design decision that defines the field: heatmaps, not coordinates

The obvious approach is to regress $2K$ numbers (the $(x,y)$ of each joint) with an L2 loss. **This works badly, and knowing why is the core insight of the whole area.**

Direct coordinate regression forces the network to collapse a spatial feature map into a single number through a fully-connected layer, which (i) destroys the spatial structure the convolutional features encode, (ii) provides a weak, non-local training signal — a small coordinate error gives no indication of *where* the evidence was, and (iii) cannot express uncertainty or multi-modality (two plausible wrist positions average to a point on neither).

**Heatmap regression** instead predicts, per joint, an $H\times W$ map whose target is a 2-D Gaussian centred on the ground-truth location, trained with MSE. The joint's position is the **argmax** at inference. This:
- keeps the problem **fully convolutional** and spatially structured;
- gives a **dense** supervisory signal at every pixel;
- naturally represents **uncertainty** (a broad heatmap = an uncertain joint) and multi-modality (two peaks);
- is trivially extended to multiple people (multiple peaks in one map).

**The cost is quantisation.** The argmax is integer-valued, so accuracy is bounded by the heatmap stride. Standard fix: shift the argmax a quarter-pixel toward the second-highest neighbour, or fit a local quadratic (**DARK** does this properly by modelling the Gaussian's log-likelihood). This is the same sub-pixel refinement idea as Harris corners (1.5) and SIFT's Taylor fit (1.6) — **and pointing that out is a good cross-module connection.**

### Top-down vs bottom-up

| | **Top-down** | **Bottom-up** |
|---|---|---|
| Pipeline | detect each person → crop → single-person pose per crop | detect **all** keypoints in the image → group them into people |
| Examples | Mask R-CNN keypoint head, HRNet, ViTPose | OpenPose, HigherHRNet, associative embedding |
| Accuracy | **higher** — each person is normalised to a consistent scale in their crop | lower, especially for small people |
| Cost | scales **linearly with the number of people** | **constant** in the number of people |
| Fails when | the detector misses a person, or in dense crowds where crops overlap heavily | grouping fails — the hard part |
| Use when | few people, accuracy matters | many people, real-time |

**The grouping problem is what makes bottom-up hard.** Two approaches worth naming: **Part Affinity Fields** (OpenPose) predict, for each limb, a 2-D vector field pointing along it, and a candidate limb connecting two keypoints is scored by the line integral of the field along it — then bipartite matching assigns limbs to people (**the Hungarian algorithm again**, 4.8). **Associative embedding** predicts a scalar "tag" per keypoint and groups keypoints with similar tags — the tag has no meaning except consistency within a person, which is a nice trick.

**HRNet** is worth knowing as an architecture: rather than downsample-then-upsample (U-Net style), it **maintains a high-resolution branch throughout** and repeatedly fuses information between resolutions. For a task whose output is a precise pixel location, never losing resolution is the right prior — and it beat encoder-decoder designs decisively.

### The metric: OKS

Pose AP uses **Object Keypoint Similarity** in place of IoU:

$$
\text{OKS} = \frac{\sum_i \exp\!\left(-\dfrac{d_i^2}{2s^2\kappa_i^2}\right)\delta(v_i>0)}{\sum_i \delta(v_i > 0)}
$$

where $d_i$ is the distance between predicted and true keypoint $i$, $s$ is the object scale (square root of the segment area), $v_i$ is the visibility flag, and $\kappa_i$ is a **per-keypoint constant** calibrated from human annotator variance.

Two details that matter: **normalising by object scale** makes a 5-pixel error on a distant person as bad as a 50-pixel error on a close one, which is right; and **$\kappa_i$ differs per joint** because humans agree closely on eye positions and poorly on hips — so the metric is calibrated to human labelling noise rather than treating all joints as equally locatable. AP is then computed by sweeping OKS thresholds exactly as 4.7 sweeps IoU.

### 🎯 Top-1% distinction

1. **Heatmaps beat coordinate regression**, and the three reasons (spatial structure preserved, dense signal, uncertainty representable). This is *the* question in this area.
2. **The quantisation cost of argmax and its sub-pixel fix** — connecting to Harris/SIFT refinement.
3. **Top-down cost is linear in people; bottom-up is constant** — that is the deployment trade, not accuracy.
4. **The grouping problem is bottom-up's hard part**, solved by PAFs (with a line integral and Hungarian matching) or associative embeddings.
5. **OKS normalises by object scale and uses per-keypoint human-variance constants** — a genuinely well-designed metric, and few candidates can say why.
6. **HRNet's argument**: never lose resolution for a task whose output is a pixel location.

### ✅ Mastery check

You are building rep-counting for a fitness app: one person, phone camera, must run at 30 FPS on-device.

(a) Top-down or bottom-up? Heatmap or coordinate regression? Justify both choices from the constraints.
(b) The user's knee is occluded by their own body at the bottom of a squat. What does your model output, and how should the app behave?
(c) Rep counts are accurate but the joint positions jitter visibly frame to frame. Give three causes and a fix for each.

<details><summary>Answer sketch</summary>
(a) <b>Top-down</b>, trivially — there is exactly one person, so the "linear in people" cost is a cost of one, and you get the accuracy benefit of a normalised crop for free. In fact you can often skip re-detection every frame and track the crop instead, which is cheaper still. <b>Heatmaps</b> for accuracy — but note the real constraint here: on-device at 30 FPS, a full-resolution heatmap per joint is expensive, so the practical answer is a lightweight top-down model at modest heatmap resolution with sub-pixel refinement, or a hybrid (MoveNet/BlazePose-style) that regresses coordinates from heatmap-derived features. Saying "heatmaps, but the resolution is the compute knob and sub-pixel refinement recovers most of what I lose" is the strong version.
(b) A well-trained heatmap model outputs a <b>low, diffuse response</b> for the occluded knee rather than a confident wrong peak — the uncertainty is representable, which is exactly the advantage over coordinate regression, and this is why per-keypoint confidence should be part of the output. The app should <b>not</b> place a skeleton joint at a low-confidence argmax and render it. It should either (i) hold the last confident position and mark it as inferred, (ii) infer it from the kinematic chain (a knee is constrained by hip and ankle), or (iii) use a temporal filter that trusts the motion model when the observation is weak. Crucially, the <b>rep-counting logic should not depend on a joint that is predictably occluded at the extreme of the movement</b> — count from the hip/torso trajectory instead. Designing around the predictable occlusion is better than trying to see through it.
(c) 1. <b>Heatmap argmax quantisation</b> — the peak snaps between integer positions across frames. Fix: sub-pixel refinement (quarter-offset or DARK's quadratic fit). 2. <b>No temporal model</b> — each frame is estimated independently, so independent noise appears as jitter. Fix: a Kalman filter or a One-Euro filter per joint (the One-Euro filter is the standard choice for interactive pose because it adapts its cutoff to speed, smoothing when still without lagging when moving). 3. <b>Crop instability</b> — if the person detector re-runs each frame, the crop box jitters, and since the model sees a slightly different normalised input each time, its output jitters even for a static pose. Fix: smooth the crop box, or track it rather than re-detecting. (Fourth: genuine model uncertainty on ambiguous joints — mitigate with test-time augmentation averaging, or accept and filter.)
</details>

---

## A4.2 Anomaly Detection & Industrial Inspection

### Intuition

You have 10,000 photos of correctly-manufactured parts and eleven photos of defects — and the twelfth defect will be a kind you have never seen. Supervised classification is the wrong frame entirely. **Train on what "normal" looks like and flag deviation.**

This is the most commercially common applied-vision job that nobody teaches: every production line, every PCB inspection, every pharmaceutical fill line, every textile mill.

### Why supervised detection fails here

- **Defects are rare** — you cannot collect a balanced dataset.
- **Defects are open-set** — a scratch, a void, a contamination, a colour deviation, and a new failure mode nobody has seen yet. A classifier trained on the first four will confidently mislabel the fifth.
- **Defects are tiny** relative to the image, and often low-contrast.
- **The cost asymmetry is extreme** — missing a defect may mean a recall; a false alarm means one human glance.

### The dominant approach: feature-memory methods

**PatchCore** is the method to know, and its mechanism is a direct application of Module 3.

1. Pass every **normal** training image through a **frozen ImageNet-pretrained backbone** (mid-level layers — early layers are too generic, late layers too semantic and too ImageNet-specific).
2. Store the **patch-level** feature vectors in a memory bank.
3. **Coreset-subsample** the bank (greedy farthest-point selection) to keep it small while preserving coverage — this is what makes it tractable and is the "core" in the name.
4. At test time, for each patch of the query image, compute the **distance to its nearest neighbour in the memory bank**. Large distance ⇒ this patch looks like nothing in the normal set ⇒ anomaly. The image score is the max over patches; the patch scores form a **segmentation map for free**.

**Notice what this is: nearest-neighbour retrieval in an embedding space** — Module 3.1 and 3.7 applied to inspection. No training, no labels beyond "these are normal", a pixel-level localisation map, and near-SOTA accuracy. **That connection is the thing to say in an interview**, because it shows you see the shared machinery rather than a list of methods.

**Other families:**
- **Reconstruction-based** — train an autoencoder or diffusion model on normals; anomalies reconstruct badly. Intuitive, but the classic failure is that a sufficiently powerful autoencoder learns to reconstruct anomalies too.
- **Normalising flows** (FastFlow, CFlow) — learn an explicit density over normal features; anomalies have low likelihood. Principled, gives calibrated scores.
- **Student–teacher** (STFPM) — a student trained to mimic a frozen teacher on normals only; on anomalies the two disagree.
- **Zero-shot / VLM-based** (WinCLIP and successors) — prompt with "a photo of a normal X" vs "a photo of a damaged X". Useful when you have *no* training images at all, weaker than PatchCore when you do.

### Evaluation

**MVTec-AD** is the standard benchmark (15 categories, texture and object classes). Report **image-level AUROC** (did you flag the defective image?) *and* **pixel-level AUROC or AUPRO** (did you localise the defect?) — a method can score well on the first while pointing at the wrong place, which matters because a human operator has to act on the localisation.

**And know AUROC's limitation here:** with a defect rate of 0.1%, ROC-AUC is optimistic because the false-positive rate is computed against an enormous negative pool. Report **AUPRC** and, for the deployed system, **the false-alarm rate per shift at the recall you require**, because that is the number that determines whether operators will keep using it.

### 🎯 Top-1% distinction

1. **Frame it as open-set, not classification** — and say why the cost asymmetry drives the design.
2. **PatchCore is k-NN in a frozen embedding space** — the Module 3 connection.
3. **Coreset subsampling** is what makes the memory bank practical.
4. **The reconstruction-method failure**: a strong autoencoder reconstructs anomalies too, defeating the premise.
5. **Report pixel-level localisation, not just image-level** — and know AUROC is optimistic under extreme imbalance.
6. **Operational metric:** false alarms per shift at the required recall. This is the number the customer cares about.

### ✅ Mastery check

A client inspects injection-moulded parts. 50,000 good images, 40 defect images across 6 defect types, and they say new defect types appear "a few times a year".

(a) Design the system. Why not fine-tune a classifier on the 40?
(b) The line changes to a new part colour. What breaks, and what is the operational fix?
(c) They ask for 99.9% defect recall. What do you tell them?

<details><summary>Answer sketch</summary>
(a) <b>PatchCore (or a normalising-flow method) on the 50,000 normals</b>, with the 40 defects held out purely as a <b>test set</b> to measure recall and to calibrate the threshold — never as training data. Why not fine-tune a classifier: 40 examples across 6 types is ~7 per class, which will overfit to those specific instances; and more fundamentally, a supervised classifier learns the boundary between "normal" and <i>these six</i> defect types, so the seventh type — which they have told you will arrive — falls on the normal side of a boundary that was never asked to exclude it. <b>The open-set requirement is stated in the problem, and it dictates the method.</b> The deliverable should include the pixel-level heat map, because an operator needs to see <i>where</i>, and a defect-type classifier can be added later as a second stage on the flagged patches once enough examples accumulate.
(b) The memory bank is a set of features of the <b>old colour</b>, so every part of the new colour is far from every stored normal patch — the system flags <b>100% of production as anomalous</b>. This is not a subtle degradation; it is a total, immediate failure, and it is the characteristic risk of memory-based methods. Operational fix: treat "collect a few hundred normals and rebuild the bank" as a <b>standard, documented changeover procedure</b> taking minutes, exactly like a mechanical tooling change — and make it a first-class feature of the product rather than a maintenance incident. Also add drift monitoring on the mean anomaly score so the system detects the change itself and alerts rather than silently alarming, and consider colour-invariant preprocessing or per-variant banks if changeovers are frequent.
(c) Push back, with numbers rather than a refusal. 99.9% recall is achievable by lowering the threshold, but recall and precision trade along a fixed PR curve, so the honest question is <b>"what false-alarm rate do you accept at 99.9% recall?"</b> Measure it: if the line runs 10,000 parts a shift and 99.9% recall costs a 5% false-alarm rate, that is 500 parts a shift sent to manual review — which may be more inspection labour than they had before automating, and operators will start ignoring the alerts. Then frame the real decision as an <b>expected-cost</b> one: cost of a missed defect × miss rate versus cost of a review × false-alarm rate, and pick the operating point that minimises it. Also note the ceiling: with 40 defect examples your recall estimate has a confidence interval of roughly ±5 points, so you <b>cannot even measure</b> 99.9% recall reliably — validating that claim needs on the order of thousands of defect examples, which is itself a project. Saying that is the senior answer.
</details>

---

## Appendix — Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Pose: heatmaps vs coordinates | heatmaps keep spatial structure, give a **dense** signal, and can express **uncertainty and multi-modality**; coordinate regression through an FC layer does none of these |
| Heatmap cost | argmax **quantises** to the heatmap grid ⇒ sub-pixel refinement (quarter-offset / DARK), the same idea as Harris and SIFT refinement |
| Top-down vs bottom-up | top-down is more accurate but **linear in the number of people**; bottom-up is **constant-cost** but must solve grouping |
| Grouping | **Part Affinity Fields** (line integral along a limb + Hungarian matching) or **associative embedding** (per-keypoint tags) |
| HRNet | maintain **high resolution throughout** rather than encode-decode — right prior for pixel-location outputs |
| OKS | IoU's replacement: **normalised by object scale**, with **per-keypoint constants calibrated to human annotator variance** |
| Anomaly detection framing | **open-set**, extreme imbalance, asymmetric cost ⇒ model "normal" and score deviation, don't classify defects |
| PatchCore | frozen backbone → patch features of normals → **coreset**-subsampled memory bank → test patch's **nearest-neighbour distance** = anomaly score. It is **Module 3 k-NN retrieval applied to inspection** |
| Reconstruction methods' flaw | a powerful autoencoder learns to reconstruct anomalies too |
| Memory-bank fragility | any global appearance change (new colour, new lighting, new lens) invalidates the bank ⇒ make bank rebuild a documented changeover procedure |
| Evaluation | image-level **and** pixel-level (AUROC/AUPRO); AUROC is optimistic under extreme imbalance ⇒ AUPRC, and **false alarms per shift at the required recall** |

### 🔨 Appendix build tasks

**1 — Pose (one sitting).** Fine-tune a pretrained keypoint model (torchvision's Keypoint R-CNN, or an HRNet from `mmpose`) on a small subset of COCO-keypoints. Then run the experiment: implement naive argmax versus quarter-offset sub-pixel refinement and measure the OKS-AP difference. It is small, real, and free — and it makes the quantisation argument concrete.

**2 — Anomaly detection (one sitting, no training).** Implement PatchCore from scratch on **MVTec-AD** — frozen ResNet-50 mid-layer features, patch memory bank, greedy coreset subsampling, nearest-neighbour scoring. Report image-level and pixel-level AUROC per category and compare against the paper. **This is a genuinely impressive portfolio piece for the effort involved**: no training, a few hundred lines, near-SOTA numbers, and it directly demonstrates the embedding/retrieval machinery from Module 3 in an applied setting.

**Read:** Sun et al., "Deep High-Resolution Representation Learning" (HRNet, CVPR 2019). Cao et al., "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields" (OpenPose, CVPR 2017) §3. Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore, CVPR 2022) — short and very clear. The MVTec-AD dataset paper for the benchmark design.
