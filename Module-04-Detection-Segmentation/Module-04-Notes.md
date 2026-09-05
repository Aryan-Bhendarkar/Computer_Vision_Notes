# Module 4 — Object Detection & Segmentation

> **The single highest-yield module for CV interviews.** Detection is where every idea in Modules 1–3 gets combined under real engineering constraints, and it is the sub-field interviewers use to test whether you can reason about design trade-offs rather than recite architectures. Two concepts here — **FPN** and **Focal Loss** — were flagged 🔴 in your gap analysis and are the two most commonly asked "do you actually understand detection?" questions.

**Concept map**

```
4.1 Problem framing ──> 4.2 Classical (HOG/DPM/Viola-Jones)
        │
        ├──> 4.3 Two-stage: R-CNN → Fast → Faster ──┐
        │                                            │
        ├──> 4.4 FPN 🔴 ─────────────────────────────┼──> 4.9 Synthesis + 2026 landscape
        │                                            │
        ├──> 4.5 One-stage: YOLO/SSD, anchor-free ───┤
        │            │                               │
        │            └──> 4.6 Focal Loss 🔴 ─────────┤
        │                                            │
        ├──> 4.7 NMS / IoU / AP / mAP ───────────────┤
        │                                            │
        └──> 4.8 DETR: set prediction 🟡 ────────────┘
                                │
                                v
        4.10 Segmentation types ──> 4.11 FCN/U-Net/Mask R-CNN ──> 4.12 IoU/Dice ──> 4.13 Practical
```

---

## 4.1 Classification vs. Localisation vs. Detection — Problem Framing

### Intuition

Start with a question you can answer from Module 2 knowledge alone: **you have a working image classifier. What is the smallest change that turns it into an object detector?**

The obvious answer is "add four more output units for the box." That works — and it is exactly the classification+localisation setup. But try it on a photo with three people in it and the whole thing falls apart, and the reason it falls apart is the entire subject of this module.

Classification asks "what is in this image?" — one answer. Localisation asks "where is the one object?" — one box. Detection asks "what objects are here and where?" — and critically, **you don't know how many there are.** That last property is what makes detection structurally different from everything in Module 2, and it drives every architectural decision that follows.

Notice what has actually broken. In Module 2, every network you built had a fixed output shape known at compile time: $K$ logits, or $K$ logits plus 4 coordinates. Backprop needs that. A tensor has a shape. But the *answer* to a detection question is a set whose size depends on the image — and worse, a set with no natural order. This is not a difficulty of scale or accuracy; it is a type mismatch between what a network can emit and what the task requires.

### The output-structure problem

| Task | Output | Structure |
|---|---|---|
| Classification | $c \in \{1..K\}$ | fixed size |
| Classification + localisation | $c$, $(x,y,w,h)$ | fixed size (assumes exactly one object) |
| **Detection** | $\{(c_i, \text{box}_i, s_i)\}_{i=1}^{n}$, **$n$ unknown** | **variable-size set** |
| Semantic segmentation | per-pixel class map $H\times W$ | fixed size |
| Instance segmentation | variable set of masks | variable-size set |

A neural network produces a fixed-size tensor. Detection needs a variable-size, **unordered set**. Every detector is, at bottom, a different answer to *"how do I make a fixed-size output represent a variable-size set?"*

There are only two structurally different ways to close that gap, and — remarkably — the entire literature consists of these two:

- **Two-stage / one-stage detectors:** produce a large fixed set of candidates (proposals or anchors), score them all, then **prune with NMS**. The set-ness is handled by post-processing.
- **DETR:** produce a fixed set of $N$ slots and use **bipartite matching** so the loss itself is permutation-invariant. The set-ness is handled by the loss.

**Framing detection as a set-prediction problem in your first sentence is one of the highest-signal things you can do in a detection interview** — it makes DETR look inevitable rather than exotic, and it explains why NMS exists at all. Hold onto the ordered-list-versus-unordered-set complaint; when we reach DETR in 4.8, bipartite matching is precisely the machinery for making a loss ignore order, and it will only feel necessary if you remember why order is the problem.

### Box parameterisation

Common formats: `(x1,y1,x2,y2)` corners (Pascal VOC), `(x,y,w,h)` top-left+size (COCO), `(cx,cy,w,h)` centre+size (YOLO, usually normalised to $[0,1]$).

Now the more interesting question: **what should the regression head actually predict?** The naive answer is the box coordinates themselves. Let us see it fail, and let the fix fall out of the failure.

Suppose you regress absolute pixel coordinates with an L2 loss. Two problems appear immediately. First, a 5-px error on a 20-px face is a wrecked detection; a 5-px error on a 400-px bus is imperceptible. Your loss scores them identically, so the network spends its capacity on the easy large boxes where absolute errors are naturally larger. Second, the target range is the image size — hundreds of pixels — while the head's initialised outputs are near zero, so early training is one long crawl toward the right magnitude.

Both complaints point the same way: **predict relative to something**. R-CNN's answer is to regress an offset from a reference box (an anchor or proposal) $(x_a,y_a,w_a,h_a)$, and to measure that offset in units of the reference box's own size:

$$
t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}
$$

Take the centre terms first. Dividing by $w_a$ says "how far off is the centre, expressed as a fraction of the box's width" — so a half-width error costs the same whether the box is 20 px or 400 px wide. That is exactly the scale-invariance the raw coordinates lacked, and it costs one division.

Now the size terms, where the choice of $\log$ is doing real work. **Pause:** before reading on — suppose you used the same normalisation trick and predicted $t_w = (w - w_a)/w_a$ instead. What goes wrong?

Two things. First, at inference you invert to $w = w_a(1 + t_w)$, and nothing stops the network emitting $t_w < -1$, giving a **negative width** — a degenerate box that produces garbage IoU and NaNs downstream. Second, and subtler: the encoding is asymmetric in scale. Doubling the box means $t_w = 1$; halving it means $t_w = -0.5$. The same *perceptual* error — off by a factor of two in either direction — costs twice as much in one direction as the other, so an L2 loss systematically biases the network toward under-predicting size. The $\log$ fixes both at once: $w = w_a e^{t_w}$ is positive for every real $t_w$, and doubling/halving become exactly $\pm\log 2$. Scale errors are multiplicative, so the natural coordinate for them is logarithmic — the same reason you plot ratios on a log axis.

So, three reasons for this specific form, now earned rather than asserted:
1. **Dividing by $w_a, h_a$** makes the target scale-invariant — a 5-px error on a 20-px object and a 5-px error on a 400-px object are not equally bad, and this normalisation encodes that.
2. **$\log$ for width/height** guarantees positivity after the inverse transform ($w = w_a e^{t_w}$) and makes the target symmetric — halving and doubling the size become $\pm\log 2$, equally penalised. Without the log, a network could predict negative widths and the loss would be asymmetric in scale.
3. Targets end up roughly zero-mean and unit-variance, which is what regression heads want. (This is the same normalisation instinct you already apply to inputs and to activations with BatchNorm in 2.4 — applied here to *targets*.)

### The multi-task loss

Every detector optimises

$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda\,\mathcal{L}_{\text{loc}}
$$

with $\mathcal{L}_{\text{loc}}$ computed **only on positive samples** (there is no box to regress for background). The restriction is not a tuning choice — it is forced. A background anchor has no ground-truth box, so there is no target to regress against; inventing one (zeros, say) would inject hundreds of thousands of meaningless regression terms alongside a handful of real ones. Balancing $\lambda$, and deciding which samples are positive, are the two decisions that dominate detector performance — see label assignment in 4.5.

**In your own words:** why is a variable-length *set* harder for a neural network to emit than a variable-length *sequence* would be?

### 🎯 Top-1% distinction

- **Lead with "detection is set prediction with an unknown cardinality."**
- **Explain the log in the box encoding** — most candidates recite the formula; few can say why.
- **Name the three fundamental sub-problems** every detector must solve: (i) **where to look** (proposals/anchors/queries), (ii) **how to assign ground truth to predictions** (label assignment), (iii) **how to remove duplicates** (NMS, or matching). Structuring your answer around those three makes any detector comparison crisp.

### ✅ Mastery check

(a) Why can't you just have a network output 100 boxes and train with MSE against the ground-truth boxes in order?
(b) Why is the box-regression loss computed only on positives, and what would break if you included negatives?
(c) Predict what happens if you drop the $\log$ from $t_w$.

<details><summary>Answer sketch</summary>
(a) Because the ground truth is an <b>unordered set</b> and your output is an ordered list. "In order" is undefined — any permutation of the GT is equally correct, so an order-sensitive loss punishes correct predictions for being in the wrong slot. The gradient then teaches the network an arbitrary ordering convention rather than the objects, and training is unstable. This is exactly the problem bipartite matching (4.8) solves.
(b) Background samples have no associated ground-truth box, so the regression target is undefined. If you forced a target (say zeros), you'd (i) inject a massive number of meaningless regression terms that dominate the loss — with ~100k anchors and ~10 positives, the background terms are 4 orders of magnitude more numerous — and (ii) teach the network to predict "no offset" everywhere, which corrupts the regression head for real objects.
(c) Without the log, $w = w_a + t_w \cdot w_a$ (or $w = t_w$ directly), so the network can predict <b>negative or zero widths</b>, producing degenerate boxes and NaNs in IoU computation. The loss also becomes asymmetric in scale: predicting $2\times$ too large costs $|t_w| = 1$ while predicting $2\times$ too small costs $|t_w| = 0.5$, so the network is systematically biased toward under-predicting size. The log makes those symmetric at $\pm\log 2$.
</details>

### 🔨 Build + read

**Build:** Write `encode_boxes(gt, anchors)` and `decode_boxes(deltas, anchors)` and verify they round-trip to numerical precision. Then visualise the target distribution of $t_x,t_y,t_w,t_h$ on a real dataset (COCO) with and without the log — you'll see the log version is near-Gaussian and the other is heavily skewed.

**Read:** Girshick et al., R-CNN (CVPR 2014), Appendix C (the bounding-box regression parameterisation). Zou et al., "Object Detection in 20 Years: A Survey" §1–2 for the framing.

---

## 4.2 Classical Detection: Sliding Windows, HOG, and the Cascade

### Intuition

If you only had an image *classifier* and had to detect with it, what would you do? Almost everyone reaches the same answer within a few seconds: crop a window, classify it, move the window, repeat. That reflex is correct, and it was the state of the art for a decade.

So: slide a fixed-size box over the image at every position and every scale, and ask a classifier "is the object here?" That works, and it is catastrophically expensive — which is why the entire history of detection is the history of **not** evaluating every window. Every technique in this section is a different answer to "how do I avoid paying for windows that obviously contain nothing?", and it is worth reading them that way rather than as three unrelated systems.

### The three classical systems worth knowing

**Viola–Jones (2001) — real-time face detection.** Three ideas, all still relevant:
1. **Haar-like features** — sums of rectangular regions, computed in $O(1)$ using an **integral image** $II(x,y) = \sum_{x'\le x, y'\le y} I(x',y')$, so any rectangle sum is 4 lookups.
2. **AdaBoost** for feature selection — from ~180,000 candidate features, greedily select a few thousand that matter.
3. **The attentional cascade** — a sequence of increasingly expensive classifiers; the first rejects ~50% of windows with 2 features. Most windows die in stage 1, so *average* cost is tiny.

The cascade is worth a moment of arithmetic, because the idea recurs constantly and the intuition is cheap. Suppose stage 1 costs 2 feature evaluations and rejects half the windows, stage 2 costs 10 and rejects half of what remains, stage 3 costs 50. A monolithic classifier costing 62 evaluations pays 62 on every window. The cascade pays $2 + \tfrac12(10) + \tfrac14(50) = 19.5$ on average — and the asymmetry is far more extreme in practice, because virtually every window in a real image is background and dies almost instantly. **The cascade exploits the fact that the classes are wildly imbalanced, by making the common class cheap.** Keep that sentence: focal loss (4.6) attacks the very same imbalance from the loss side rather than the compute side.

**The cascade principle is the ancestor of two-stage detection** and, more broadly, of every "cheap filter then expensive verifier" design in vision — including the two-stage retrieval of Module 3.

**HOG + linear SVM (Dalal & Triggs, CVPR 2005) — pedestrian detection.**
- Compute gradients; divide into $8\times8$-pixel **cells**; build a 9-bin *unsigned* orientation histogram per cell, weighted by gradient magnitude with trilinear interpolation.
- Group cells into overlapping $2\times2$ **blocks** and **L2-Hys normalise** each block (L2 normalise, clip at 0.2, renormalise — **the same trick as SIFT's descriptor**, and for the same reason: illumination robustness).
- Concatenate over the detection window (e.g. $64\times128$ → 3780-d) and feed a linear SVM.

**HOG is dense SIFT without keypoints.** The relationship — same orientation histograms, same block normalisation, applied on a regular grid instead of at detected keypoints — is a clean Module 1 callback.

**Deformable Parts Model (Felzenszwalb et al., 2010)** — the pre-CNN state of the art, and the last great hand-designed detector. A root HOG filter plus $n$ part filters at twice the resolution, with a **deformation cost** penalising each part's displacement from its anchor position:

$$
\text{score} = \underbrace{F_0 \cdot \phi(H, p_0)}_{\text{root}} + \sum_{i=1}^{n}\Bigl[\underbrace{F_i\cdot\phi(H,p_i)}_{\text{part}} - \underbrace{d_i\cdot\psi(\Delta p_i)}_{\text{deformation}}\Bigr] + b
$$

Trained with a **latent SVM** (part positions are latent variables). Won PASCAL VOC repeatedly. The idea — objects are compositions of parts with flexible spatial relations — is exactly what CNN feature hierarchies learn implicitly.

### Why sliding windows had to die

Count them. A $640\times480$ image scanned with a $64\times128$ window at 8-px stride gives roughly $(640/8)\times(480/8) = 80\times60 = 4800$ positions at one scale; over ~30 scales that is ~$10^5$, and finer strides or larger images push it toward $10^6$. A HOG+SVM evaluation per window is a dot product and survives this. A CNN forward pass per window does not: at even 1 ms per crop, $10^5$ windows is 100 seconds an image.

**Pause:** given that count, which of the two obvious escapes would you reach for — reduce the number of windows, or reduce the cost per window?

Both were taken, and they define the next two decades. **Region proposals** cut the count: evaluate only ~2000 plausible windows instead of $10^5$ (4.3, and R-CNN is exactly this). **Fully-convolutional evaluation** cuts the cost: because convolution is shift-equivariant, running the conv stack once over the whole image computes every overlapping window's features simultaneously, sharing all the redundant arithmetic between neighbouring windows — the insight behind OverFeat, Fast R-CNN, and every modern detector. The second escape turned out to be the deeper one, because it makes the number of windows almost free again, which is why modern one-stage detectors happily score $10^5$ locations per image (4.5) and then have to deal with the imbalance that creates (4.6).

**In your own words:** why does a cascade help even though the last stage is just as expensive as before?

### 🎯 Top-1% distinction

- **The cascade / early-rejection principle is alive**: it's what two-stage detectors do, what objectness scores do, and what cascade R-CNN revived explicitly.
- **HOG ≡ dense SIFT** — including the L2-Hys clip at 0.2.
- **DPM's parts-with-deformation is what CNNs learn implicitly** — and note that **deformable convolution** (v1–v4) is the explicit modern re-introduction of that idea.
- **Fully-convolutional evaluation is the real killer of sliding windows** — a conv layer applied to a whole image *is* a sliding-window classifier evaluated everywhere with shared computation. Say that and the R-CNN→Fast R-CNN story becomes obvious.

### ✅ Mastery check

Explain why applying a $224\times224$ CNN classifier convolutionally to a $448\times448$ image is equivalent to running it as a sliding window, and compute how many windows you get for a network with total stride 32. What is the catch?

<details><summary>Answer sketch</summary>
A conv layer is shift-equivariant, so running the conv stack on a larger input produces a feature map where each spatial position corresponds to the classifier applied to a shifted crop of the input. If the FC head is reinterpreted as a $1\times1$ (or $k\times k$) conv, the whole network becomes fully convolutional and the output map has one score vector per window. With total stride 32, a $448\times448$ input gives a $14\times14$ output = <b>196 windows</b>, evaluated for barely more than the cost of one forward pass instead of 196 — because all the overlapping windows share their convolutional computation.
<b>Catches:</b> (i) the effective stride between windows is 32 px, which is coarse — you can't localise better than that without a regression head or a finer stride; (ii) it only covers <b>one scale</b>, so you still need an image pyramid or a feature pyramid; (iii) boundary effects and padding make edge windows not exactly equal to the cropped equivalent; (iv) the receptive field, not the nominal window, determines what each output actually sees (2.3) — and the <i>effective</i> receptive field is smaller still.
</details>

### 🔨 Build + read

**Build:** Implement HOG from scratch (cells, orientation binning with interpolation, block L2-Hys normalisation) and train a linear SVM for pedestrian detection on INRIA. Then implement the sliding-window + image-pyramid + NMS inference loop and time it. Feeling how slow it is is the point.

**Read:** Dalal & Triggs (CVPR 2005) — short and clear. Viola & Jones (CVPR 2001) for the cascade.

---

## 4.3 Two-Stage Detectors: R-CNN → Fast R-CNN → Faster R-CNN

### The story is about progressively sharing computation

Read this section as one experiment repeated four times: **build the thing, profile it, find that one component now dominates the runtime, remove that component, repeat.** Each architecture in the lineage is not a new idea so much as the obvious response to the previous system's profiler output. Told that way, you will never need to memorise the order — you can reconstruct it.

| | Proposals | CNN passes | Training | Test time |
|---|---|---|---|---|
| **R-CNN** (2014) | Selective Search (~2000) | **2000 per image** | 3 separate stages (CNN, SVMs, box regressors) | ~47 s |
| **SPPnet** (2014) | Selective Search | 1, + spatial pyramid pooling per region | still multi-stage | ~2 s |
| **Fast R-CNN** (2015) | Selective Search | **1**, + RoI pooling | **single-stage, multi-task loss** | ~0.3 s (+2 s proposals) |
| **Faster R-CNN** (2015) | **RPN (learned)** | 1, shared with RPN | end-to-end | **~0.2 s** |

**Round 1 — R-CNN, and its profiler output.** R-CNN takes the second escape route from 4.2's dilemma only halfway: it uses Selective Search to cut $10^5$ windows down to ~2000, then runs AlexNet on each. 47 seconds an image, of which nearly all is the 2000 forward passes. Now look at *why* that is wasteful rather than merely slow. Two overlapping proposals — say boxes differing by 10 px — are warped to $227\times227$ and pushed through the same convolutional stack independently, so the network computes almost identical early feature maps twice. Across 2000 heavily-overlapping proposals, essentially the entire convolutional computation is being redone thousands of times over the same pixels.

That diagnosis names its own fix, and it is 4.2's fully-convolutional insight applied one level up: **convolution is shift-equivariant, so compute the feature map once for the whole image and take each proposal's features by cropping the shared map rather than re-running the backbone.** That is exactly what SPPnet and Fast R-CNN do. Test time falls from 47 s to ~2 s. Note the ratio: roughly 2000× less convolutional work, but only ~25× faster overall — which is the tell that something *else* is now the bottleneck.

**Round 2 — profile Fast R-CNN.** The backbone now costs ~0.3 s. Selective Search still costs ~2 s on CPU, so it is now 85% of the runtime. The proposal generator, which was previously an optimisation, has become the problem.

**Pause:** you need proposals, they must be cheap, and you have a feature map already sitting on the GPU. What do you do?

**Round 3 — Faster R-CNN.** Make proposal generation a small convolutional head that reads the *same shared feature map* the detector uses. The proposals then cost ~10 ms rather than 2 s, they run on GPU, and — an unplanned bonus — they are *learned* for your data rather than being a fixed hand-designed segmentation heuristic. Total: ~0.2 s. Every element of the pipeline now shares one backbone pass, which is where the "progressively sharing computation" framing pays off: there is nothing left to share, and the lineage after this point stops being about speed and starts being about accuracy (Cascade R-CNN, Mask R-CNN).

### RoI Pooling and RoI Align

Cropping from a shared feature map raises a problem R-CNN never had. R-CNN warped every crop to $227\times227$ *in pixel space*, so every region arrived at the FC layers with an identical shape. Crop from the feature map instead and regions have whatever shape their box happens to be — but an FC head needs a fixed-size input. Something must convert an arbitrary $w\times h$ region into a fixed tensor.

**RoI Pooling** (Fast R-CNN) is the simple answer: given a proposal box on the feature map, divide it into a fixed $7\times7$ grid and max-pool within each cell → fixed-size output regardless of box size, so a single FC head works for all proposals. Big regions get big cells, small regions get small cells; the output is always $7\times7$.

**The quantisation problem.** Follow one coordinate all the way through and the flaw shows itself. A proposal edge at image coordinate $x=127$ lands on a stride-16 feature map at $127/16 = 7.94$ — not an integer, and there is no feature vector at position 7.94. RoI Pooling **rounds it to 7**, which in image terms has just moved the box edge from 127 to 112, a 15-pixel shift. Then it happens *again*: dividing the (already-shifted) region into a $7\times7$ grid gives bin boundaries at fractional positions, which are rounded a second time. Two independent rounding steps, each up to half a stride. Total misalignment can reach half a stride ≈ 8 image pixels, and — this is the part that matters — the size of the error depends on the *fractional part* of the box coordinates, so it varies unpredictably from region to region.

**RoI Align** (Mask R-CNN, 2017): **do not quantise**. If the value you need sits at 7.94, do not round to 7 — *interpolate*. Sample 4 points per bin at exact floating-point locations using **bilinear interpolation**, then average/max. Everything stays differentiable and the extra cost is a handful of interpolations per bin, i.e. negligible.

The gain, however, is **much larger for masks than for boxes** (+~10% relative mask AP vs +~1% box AP), and the asymmetry is the interesting part. Why should the same fix be ten times more valuable for one head than the other? Because the box head outputs four numbers *relative to the proposal*, and it sits behind a regression layer that can learn to compensate for a systematic offset — if features are consistently shifted 8 px left, the head simply learns a +8 px bias. The mask head cannot do that. It outputs a $28\times28$ per-pixel grid that must correspond to specific image locations, and the misalignment is not systematic: it depends on the fractional part of each box's coordinates, so it differs from region to region and no fixed bias corrects it. **A head that can absorb a constant error is barely hurt by one; a head that needs per-pixel spatial fidelity against a varying error is crippled.** That is worth stating in exactly those terms.

### The Region Proposal Network

A small head on the shared feature map: a $3\times3$ conv, then two sibling $1\times1$ convs producing, per spatial location, $2k$ objectness scores and $4k$ box deltas for $k$ **anchors**.

Why anchors at all, rather than having each location regress one box freely? Because a single location would have to predict *any* box shape — tall, wide, huge, tiny — from one feature vector, which is an ill-posed one-to-many mapping (a location containing both a pedestrian and a bus edge has two right answers and a free regressor averages them). Anchors resolve this by discretising the shape space: each location gets $k$ reference boxes of fixed size and aspect ratio, each with its own output slot, and each slot only has to answer "does an object of roughly *this* shape sit here, and by how much is it off?" A hard multi-modal regression becomes $k$ easy unimodal residual regressions.

**Anchors:** at each location, $k = 9$ boxes — 3 scales ($128^2, 256^2, 512^2$) × 3 aspect ratios (1:1, 1:2, 2:1). Anchors are a *hand-designed prior over object shape*; the network only has to predict a residual from the nearest one — and this is precisely the residual encoding you derived in 4.1, with the anchor playing the role of $(x_a,y_a,w_a,h_a)$.

**Label assignment (the classic IoU rule).** Having created ~27,000 anchors, you now owe each of them a label. The obvious rule — "positive if it overlaps a ground-truth box well" — needs three refinements, each patching a specific failure:
- Positive if IoU with any GT $\ge 0.7$, **or** if it is the highest-IoU anchor for some GT. The second clause exists because the first can produce **zero** positives for an object: a 20-px object against anchors starting at $128^2$ has no anchor anywhere near IoU 0.7, so without a fallback it contributes no supervision at all and the model simply never learns to detect it. The "best available anchor is always positive" rule guarantees every GT gets at least one gradient.
- Negative if IoU $< 0.3$ with all GT.
- **Ignored** if in between — neither term contributes. This is the design choice most people skip past. Anchors at IoU 0.5 are genuinely ambiguous: calling them positive teaches the box head to regress from a bad starting point, calling them negative teaches the classifier to suppress a box that is half-right. Rather than guess, Faster R-CNN declines to supervise them at all. Assignment is not binary, it is ternary, and saying so is a cheap signal of having read the paper.
- Sample a mini-batch of 256 anchors per image with a target 1:1 positive:negative ratio (pad with negatives if too few positives). Note what this sampling step is really for — with ~27,000 anchors and perhaps 20 positives, an unsampled loss would be ~99.9% background. Hold that thought; it is the entire subject of 4.6, and Faster R-CNN's answer is "throw most of the data away," which focal loss will later improve on.

**Smooth L1 (Huber) loss for box regression.** Which regression loss should the box head use? Try the two obvious candidates and watch each fail at opposite ends of the error range.

*L2 first.* Its gradient is $\propto x$, so it grows without bound as the error grows. Early in training — or whenever an anchor is assigned to a poorly-matching object — you get a delta of 5 or 10, hence a gradient 10× larger than a typical one, from a single box. One outlier can dominate the batch's update and blow up training. L2 is well-behaved near the optimum and dangerous far from it.

*L1 next.* Its gradient is $\pm1$ everywhere, so outliers are bounded — good. But that also means the gradient never shrinks as you approach the optimum: at error $10^{-6}$ the step is the same size as at error 10, so the model oscillates around the minimum instead of settling, and the gradient is undefined at exactly zero. L1 is well-behaved far from the optimum and bad near it.

Each is good precisely where the other is bad, so take each where it wins and stitch them together at $|x| = 1$:

$$
\text{smooth}_{L_1}(x) = \begin{cases}0.5x^2 & |x| < 1\\ |x| - 0.5 & \text{otherwise}\end{cases}
$$

The $-0.5$ is not a tuning constant; it is forced. For the two pieces to meet continuously at $x=1$ you need $0.5(1)^2 = 1 - c$, giving $c = 0.5$. (The derivatives also agree there: $x = 1$ from the left, $1$ from the right — so the loss is $C^1$, and the optimiser sees no kink.) Huber is L2 near zero for smooth convergence and L1 far away for bounded, outlier-robust gradients — a pattern you will meet again whenever a loss must be sensitive in one regime and robust in another.

**In your own words:** if someone handed you a detector that was 30× too slow, what would the R-CNN lineage tell you to do first?

### The lineage after Faster R-CNN

- **Mask R-CNN** (2017): + a mask branch (FCN per RoI), + RoI Align. Instance segmentation for free.
- **Cascade R-CNN** (2018): a sequence of detection heads trained at increasing IoU thresholds (0.5, 0.6, 0.7). Fixes the mismatch between the IoU threshold used for training and the one used at evaluation, and improves high-IoU AP substantially. Explicitly the cascade idea again.
- **Sparse R-CNN** (2021): fixed set of learned proposals, no dense anchors, no NMS — two-stage design converging on DETR's ideas.

### 🎯 Top-1% distinction

1. **Tell the story as "progressively sharing computation"**, not as a list of papers.
2. **The RoI Align quantisation explanation, with the two rounding steps and the box-vs-mask asymmetry.**
3. **The "highest-IoU anchor is always positive" rule** and why it exists.
4. **The "ignore" band** — most candidates describe assignment as binary.
5. **Smooth L1's justification** from both directions (L2's outlier explosion, L1's non-smoothness at zero).
6. **Cascade R-CNN's IoU-threshold mismatch argument**: a head trained at IoU 0.5 produces proposals whose distribution doesn't match what a 0.7-threshold head needs, so you train a sequence, each on the output distribution of the previous. Genuinely elegant.

### ✅ Mastery check

(a) Faster R-CNN on a $800\times1000$ image with a stride-16 feature map and 9 anchors per location. How many anchors total? What fraction are typically positive if the image has 5 objects?
(b) Explain why RoI Align helps masks far more than boxes.
(c) Your detector misses small objects badly. Give three architectural changes and say which you'd try first.

<details><summary>Answer sketch</summary>
(a) Feature map $\lfloor 800/16\rfloor\times\lfloor 1000/16\rfloor = 50\times62 = 3100$ locations × 9 anchors = <b>~27,900 anchors</b>. With 5 objects and the IoU ≥ 0.7 rule, typically ~10–50 anchors are positive — so roughly <b>0.05–0.2%</b>. That ratio is the whole motivation for Faster R-CNN's fixed 1:1 sampled mini-batch, and later for focal loss (4.6).
(b) A box is 4 numbers; a half-stride (≈8 px) systematic misalignment shifts a box slightly and, because the regression head can learn to compensate for a <i>consistent</i> bias, costs little AP. A mask is a per-pixel prediction on a $28\times28$ grid warped back to the box — every pixel's spatial correspondence to the image matters, and an 8-px misalignment is a large fraction of a small object's extent. The mask head has no way to compensate for a misalignment that varies with the fractional part of the box coordinates. Hence ~+10% relative mask AP vs ~+1% box AP.
(c) 1. <b>FPN</b> (4.4) — attach detection heads to high-resolution, semantically-strong feature levels. Try this first; it is the single largest AP_small gain available and is nearly free. 2. <b>Increase input resolution</b> (or use higher-resolution feature levels / reduce total stride) — small objects are literally sub-pixel at stride 32. Cheap to test, expensive at inference. 3. <b>Add smaller anchors / switch to anchor-free with better label assignment</b> — with anchors starting at $128^2$, a 20-px object has no anchor with IoU ≥ 0.7, so it gets supervision only via the "highest-IoU" fallback rule. Also worth naming: <b>copy-paste augmentation</b> for small objects, and checking whether the evaluation metric (AP_small) is even the thing the product cares about.
</details>

### 🔨 Build + read

**Build:** Implement anchor generation, IoU-based label assignment (with the ignore band and the highest-IoU rule), and the encode/decode round trip. Then visualise, on real images, which anchors are positive for each GT box — seeing that a small object gets 1 positive anchor while a large one gets 40 makes the imbalance problem visceral.

**Read:** Ren et al., "Faster R-CNN" (NeurIPS 2015) §3. He et al., "Mask R-CNN" (ICCV 2017) §3 on RoI Align. Cai & Vasconcelos, "Cascade R-CNN" (CVPR 2018) §1–3.

---

## 4.4 Feature Pyramid Networks (FPN) 🔴

> Flagged critical. FPN is used in essentially every modern detector and segmenter, and it is a direct descendant of Module 1's Laplacian pyramid.

### Intuition

Here is the problem in one sentence: **a self-driving car must detect a distant pedestrian 20 px tall and a nearby truck 600 px tall in the same frame, in the same forward pass, with one network.** A 30× range of object sizes, and one backbone.

Why is that hard? Trace the pedestrian through a ResNet. At stride 32 — the last stage, the one with all the semantic power — a 20-px object occupies **less than one feature-map cell**. Whatever the network knows about pedestrians lives in that stage, and the pedestrian has been erased before it gets there. Now trace the truck: at stride 4 it is 150 cells across, beautifully resolved — but a stride-4 unit has a small receptive field and sees only edges and textures, so nothing at that level is capable of answering "is this a truck?"

That is the tension, and it is structural rather than accidental: **in a standard CNN, spatial resolution and semantic abstraction are traded against each other by construction, because the same operation (striding) that builds the receptive field destroys the resolution.** Deep layers know *what* but not *where*; shallow layers know *where* but not *what*. FPN's idea is to stop treating that as a trade-off: take the semantics from the top and push them back down to the high-resolution layers, so every scale gets both.

### The four architectures for multi-scale

There are four ways people have tried to resolve this, and they are best read as four *attempts* — each one fixes the previous attempt's flaw and introduces a new one, until the fourth has nothing left to fix.

```
(a) Featurised image pyramid      (b) Single feature map       (c) Pyramidal feature hierarchy   (d) FPN
    resize image ×N,                  predict from the             predict from each backbone       backbone stages
    run backbone on each              deepest layer only           stage independently (SSD)        + top-down + lateral
    ┌───┐ →                           ┌───┐                        ┌───┐→                           ┌───┐→─┐
    ┌──┐  →   predictions             │   │                        ┌──┐ →   predictions             ┌──┐→─┼→ predictions
    ┌─┐   →                           └───┘→ predictions           ┌─┐  →                           ┌─┐ →─┘
  accurate, N× cost                 fast, terrible for small     fast, shallow levels have        fast AND semantically
                                                                  no semantics                     strong at every level
```

**Attempt (a) — resize the image, not the network.** If the network only detects well at one scale, present the object at that scale: build an image pyramid, run the whole backbone on each level, merge the results. This is exactly what classical CV did (Module 1.4), and it is genuinely *correct* — every object is eventually seen at the size the network likes, and every level is fully semantic because every level is a full backbone pass. Its flaw is purely economic: $N\times$ the compute for $N$ levels, at both training and inference. Infeasible.

**Attempt (b) — one feature map, and live with it.** Fast/Faster R-CNN's original design: predict from the deepest layer only. Fast, one backbone pass, maximal semantics. Its flaw is the pedestrian from the opening paragraph: at stride 16 or 32, small objects have been resolved away and are nearly invisible. This fixes (a)'s cost and reintroduces the original problem.

**Attempt (c) — the hierarchy is already free, so use it.** This is SSD's move, and it is a good one: a CNN *already* computes feature maps at strides 4, 8, 16, 32 on its way to the top. Attach a detection head to each and predict small objects from the fine levels, large ones from the coarse levels. Zero extra backbone compute. So why isn't this the end of the story?

**Pause:** attempt (c) gives you high resolution *and* costs nothing extra. Predict what goes wrong before reading on.

The answer is the second half of the opening tension, and it is easy to overlook because (c) looks like it solved everything. **Resolution is not the only thing a small object needs — it needs semantics too, and the fine levels have none.** A stride-4 unit has a small receptive field and low-level features; asking it "is this a pedestrian?" is asking a question it structurally cannot answer, because it has never seen enough context and has not been through enough non-linearities. So (c) gives small objects a high-resolution map that cannot classify. SSD worked around it by starting its pyramid at conv4_3 and above — i.e. by refusing to use the *really* shallow levels — which quietly reintroduces the small-object problem it was meant to solve.

**Attempt (d) — FPN.** Take (c)'s free hierarchy and repair exactly the one thing wrong with it. The fine levels lack semantics; the coarse levels have semantics in abundance. So *move the semantics down*: upsample the deep, semantically-rich map and add it into the shallow, high-resolution one. Now the stride-4 level is high-resolution (it kept its own features) *and* semantically strong (it received the top's). Attempt (a)'s accuracy at attempt (b)'s cost. Nothing is left to fix, which is why FPN — unlike (a), (b) and (c) — is still in essentially every detector nine years later.

### The architecture, precisely

Attempt (d) is a one-line idea — "push semantics downward" — but there are four or five implementation choices hidden in it, and each one has a reason. Build the architecture by asking, at each step, what the constraint forces.

**Bottom-up pathway:** the backbone's natural stages. For ResNet, take the last layer of each residual stage: $C_2, C_3, C_4, C_5$ at strides 4, 8, 16, 32. Nothing is added here — this is the backbone you were already running.

**Now the merge.** You want to combine $C_4$ (stride 16, high resolution, weak semantics) with the already-enriched $P_5$ (stride 32, low resolution, strong semantics). Two things stop you from just writing $C_4 + P_5$. They are different *spatial sizes* — $P_5$ is half the resolution — and different *channel counts*, because a ResNet doubles channels at every stage (256, 512, 1024, 2048 for $C_2..C_5$). So the merge needs exactly two repairs: an upsample to fix the spatial mismatch, and a projection to fix the channel mismatch. That is the whole architecture:

**Top-down pathway + lateral connections:**

$$
P_5 = \text{conv}_{1\times1}(C_5)
$$
$$
P_l = \text{conv}_{3\times3}\Bigl(\underbrace{\text{conv}_{1\times1}(C_l)}_{\text{lateral}} + \underbrace{\text{upsample}_{2\times}(P_{l+1})}_{\text{top-down}}\Bigr), \quad l = 4,3,2
$$

Read the recursion as an instruction rather than a formula: *start at the top with a projected $C_5$; then for each level going down, project that level's backbone features, add in the upsampled version of the level above, and smooth the result.* Every term in it is one of the repairs above.

- The **$1\times1$ lateral conv** projects every stage to the same channel count (256 everywhere) so the addition is well-defined, and lets the network re-weight what the shallow features contribute. Note it is $1\times1$, not $3\times3$: its job is purely channel bookkeeping, not spatial reasoning, so a spatial kernel would be wasted parameters at the highest-resolution levels where they cost the most.
- **Upsampling is nearest-neighbour $2\times$** — deliberately the cheapest thing that works; the paper found no benefit from learned upsampling here. That negative result is informative rather than lazy: the top-down path's job is to carry *semantics*, which are spatially smooth, not to reconstruct fine detail — the detail is already sitting in the lateral branch.
- **Element-wise addition**, not concatenation — keeps channels constant and cost low.
- The **final $3\times3$ conv** on each merged map reduces the aliasing introduced by nearest-neighbour upsampling. Nearest-neighbour duplicates each value into a $2\times2$ block, which injects a hard step edge at every block boundary — high-frequency content that was never in the signal. A $3\times3$ smoothing conv is the standard cure. (**Aliasing again** — Module 1.4 and 2.1.)
- RetinaNet adds $P_6$ (and $P_7$) by strided conv on $P_5$ for very large objects.

**All levels share the same detection head** (same weights), which is only sensible *because* all levels now have the same channel count and comparable semantic strength. That weight sharing is itself a strong regulariser and a hint that the levels are genuinely comparable. Notice it also runs the causality backwards: the "256 channels everywhere" choice is not an aesthetic preference, it is what the shared head *requires*. Hold onto that — it is the whole reason FPN adds where U-Net concatenates (4.11).

### RoI-to-level assignment

For a two-stage detector, which pyramid level should a proposal of size $w\times h$ be pooled from? You now have four maps and one box; something has to choose.

**Derive the rule rather than memorising it.** The pyramid is geometric: consecutive levels differ by a factor of 2 in stride, so the natural coordinate for "which level" is $\log_2$ of size — the same reasoning that put a $\log$ in the box encoding in 4.1. You need one anchor point to pin the scale, and there is an obvious one: the backbone was pretrained on ImageNet at $224\times224$, so a $224\times224$ region is the size those features were tuned for, and it belongs at whatever level the original single-scale detector used — $P_4$, stride 16. Call that $k_0 = 4$. Everything else follows by counting halvings: a region half as wide is one level finer, a region a quarter as wide is two levels finer. "Half as wide" is $\log_2$ of the size ratio, and a region's linear size is $\sqrt{wh}$ (the side of the square with the same area, which is how you reduce a two-number box to one scale). Put those three pieces together and the formula writes itself:

$$
k = \left\lfloor k_0 + \log_2\!\left(\frac{\sqrt{wh}}{224}\right)\right\rfloor, \qquad k_0 = 4
$$

Read it back: a $224\times224$ RoI gives $\log_2(1) = 0$, so $k = 4$ → $P_4$ (stride 16). Halving the RoI's linear size subtracts 1 from the $\log_2$, sending it one level *up* the pyramid to a finer stride. So a $112\times112$ RoI → $P_3$, a $56\times56$ RoI → $P_2$. The floor just discretises a continuous scale into the four levels that exist, and in practice $k$ is clamped to $[2,5]$ so a huge or tiny box doesn't index off the end. **Being able to write and interpret this formula is a specific, checkable signal.**

**Pause:** the rule sends *small* boxes to *fine-stride* levels. That is the opposite of what a naive "small object, small feature map" instinct suggests. Say why the rule is right before reading on.

Because the point is to keep the RoI's *resolution in feature cells* roughly constant. A $56\times56$ box pooled from stride 16 covers $3.5\times3.5$ cells — you are asking RoI Align to produce a $7\times7$ grid out of twelve cells, so most of the grid is interpolated fiction. Pooled from stride 4 it covers $14\times14$ cells, and the $7\times7$ grid is a genuine downsample of real evidence. The level assignment exists so that every RoI, large or small, arrives at the head with a comparable amount of actual information — which is also precisely why one shared head can serve all of them.

### Why it works, stated sharply

The core insight is a **separation of concerns**: *resolution* comes from the bottom-up pathway, *semantics* come from the top-down pathway, and the lateral connections let each level combine both. Before FPN you had to choose one; after FPN you don't.

And it is **nearly free**: FPN adds a handful of $1\times1$ and $3\times3$ convs at 256 channels — a few percent of backbone FLOPs — for a large AP gain, concentrated in AP_small (the original paper: +8.0 AP over a single-scale Faster R-CNN baseline on COCO, with the largest relative gain on small objects).

### Connection to Module 1

**FPN is a learned Laplacian pyramid, run in reverse.** Where the Laplacian pyramid (1.4) *decomposes* an image into band-pass levels by subtracting an upsampled coarse level, FPN *composes* a feature pyramid by adding an upsampled coarse level. Same structural primitive — upsample the coarse level and combine it with the fine level — used constructively instead of analytically. Even the sign flip is meaningful: subtraction *removes* what the coarse level already explains, leaving the residual detail; addition *donates* what the coarse level knows to a level that lacks it. Volunteering this connection in an interview is exactly the kind of cross-module synthesis that reads as depth.

There is a second connection, to Module 2 rather than Module 1, and it is worth having ready. The top-down path is a **skip connection with a change of resolution**: information from a deep layer is routed back to a shallow one so the shallow layer does not have to learn what the deep one already knows. ResNet (2.6) skips forward across a few layers at constant resolution; U-Net (4.11) skips across the encoder–decoder at matching resolutions; FPN skips downward across the scale hierarchy. Three instances of one idea — *let a path exist that does not have to go through the whole stack* — which is why the three architectures look like variations on a theme once you see it.

**In your own words:** in one sentence, what does the top-down pathway give a shallow level that it could not get on its own?

### Variants worth naming

| Variant | Change | Motivation |
|---|---|---|
| **PANet** (2018) | adds a second, **bottom-up** path on top of FPN | in a deep backbone, low-level localisation information travels 100+ layers to reach $P_5$; the extra short path preserves it. Used in YOLOv4+ necks |
| **BiFPN** (EfficientDet, 2020) | bidirectional, repeated blocks, **learnable fusion weights** $\frac{w_i}{\epsilon + \sum w_j}$, removes single-input nodes | different levels should not contribute equally; let the network learn the weights |
| **NAS-FPN** (2019) | searched fusion topology | shows the hand-designed topology isn't optimal (but the searched one isn't interpretable) |
| **Recursive-FPN / DetectoRS** | feeds the FPN output back into the backbone | "look twice" |
| **ViTDet** (2022) 🟢 | **no hierarchical backbone at all** — a plain ViT with a simple pyramid built by deconv/pooling from the *last* layer only | shows the multi-stage backbone hierarchy may be unnecessary if the backbone is strong enough; strong currency signal |

### 🎯 Top-1% distinction

1. **State the separation of concerns** (resolution from bottom-up, semantics from top-down) — that's the whole idea in one sentence.
2. **Explain why (c) alone fails** — the semantic gap in shallow layers — because that's the question FPN answers.
3. **Know the level-assignment formula** and be able to interpret it.
4. **Name the aliasing role of the final $3\times3$ conv.**
5. **FPN as an inverted Laplacian pyramid.**
6. **FPN is nearly free**, which is why it became universal — the AP/FLOP ratio, not the raw AP, is the reason.
7. **ViTDet's counterpoint** — the field is currently re-examining whether the hierarchy is necessary. That's the 2026-aware answer.

### ✅ Mastery check

(a) A proposal is $64\times128$. Which FPN level does it go to?
(b) Your colleague proposes concatenating instead of adding in the lateral merge. Discuss.
(c) You replace nearest-neighbour upsampling with a transposed convolution. What might improve and what might get worse?
(d) You are detecting 10-px objects at stride 4 ($P_2$). AP_small is still bad. Give two reasons FPN alone may not fix it.

<details><summary>Answer sketch</summary>
(a) $\sqrt{64 \times 128} = \sqrt{8192} = 90.5$. $\log_2(90.5/224) = \log_2(0.404) = -1.31$. $k = \lfloor 4 - 1.31 \rfloor = \lfloor 2.69 \rfloor = \mathbf{2}$ → $P_2$ (stride 4).
(b) Concatenation preserves both signals separately and lets the following $3\times3$ conv learn an arbitrary combination, so it is <i>strictly more expressive</i>. Costs: channel count doubles at every merge, so either memory and FLOPs grow down the pyramid, or you need an extra $1\times1$ to project back — and then you've spent parameters to learn something addition gives you for free. Empirically addition matches concatenation here, and addition keeps the invariant that <b>all levels have identical channel counts</b>, which is what allows the detection head to be shared across levels. (U-Net, notably, <i>does</i> concatenate — because there the decoder needs to preserve fine detail exactly, and there's no shared-head constraint. Contrasting the two is a good answer.)
(c) <b>Might improve:</b> a learned upsample can produce sharper, better-aligned features than nearest-neighbour, potentially helping localisation. <b>Might get worse:</b> (i) <b>checkerboard artefacts</b> from uneven kernel overlap when kernel size isn't divisible by stride (2.2) — and these propagate down every pyramid level; (ii) more parameters and FLOPs at the highest-resolution levels, where they're most expensive; (iii) more capacity in the top-down path can overfit on small datasets. The FPN paper tested this and found no gain, which is itself the informative result: the top-down path's job is to <i>carry semantics</i>, not to reconstruct detail, so a dumb upsampler suffices.
(d) 1. <b>Receptive field vs. resolution is only half the problem — information is genuinely absent.</b> A 10-px object at input resolution has ~100 pixels of evidence; no architecture creates information that isn't there. The fix is <b>higher input resolution</b> (or tiling), not a better pyramid. 2. <b>Label assignment.</b> With anchors, a 10-px object may have no anchor exceeding the IoU threshold, so it's supervised only by the fallback rule and its positive/negative ratio is catastrophic — switch to anchor-free with centre-based assignment (FCOS) or an adaptive assigner (ATSS/SimOTA). 3. <b>The evaluation itself</b>: COCO's AP_small is dominated by boxes under $32^2$, and localisation error of 2–3 px already drops IoU below 0.5 for a 10-px box — so AP_small is intrinsically harsh, and you should check whether a lower IoU threshold or a different metric matches the product need.
</details>

### 🔨 Build + read

**Build:** Implement FPN on top of a torchvision ResNet-50 (extract $C_2..C_5$ with `create_feature_extractor`, build the top-down path). Then run the ablation on a small detection dataset: single-scale $C_5$ head vs. SSD-style independent-level heads vs. full FPN, reporting AP, AP_small, AP_large, and FLOPs for each. That three-row table *is* the answer to the FPN interview question, in your own numbers.

**Read:** Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017) — read the whole thing, it's 9 pages and unusually clear. Then Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection" (ViTDet, ECCV 2022) §3 for the counterpoint.

---

## 4.5 One-Stage Detectors: YOLO, SSD, Anchor-Based vs Anchor-Free

### Intuition

Ask the question 4.3 left open. Faster R-CNN's whole design exists to avoid evaluating a classifier everywhere — the RPN's job is to shortlist. But by 4.2 you already know the cost of "everywhere" collapsed once evaluation became fully convolutional: a conv layer applied to a feature map *is* a classifier run at every location, and it costs one pass. So: **if scoring every location is nearly free, why keep the shortlist at all?**

That is the one-stage bet. Two-stage detectors ask "where might objects be?" and then "what is in each of those places?" One-stage detectors skip the first question: densely predict a class and a box at every location on the feature map, all at once. Fewer moving parts, one forward pass, much faster.

The thing worth noticing is what you gave up. The RPN was not only a compute filter — it was also, silently, a **balance filter**, cutting ~28,000 anchors down to ~2,000 mostly-plausible proposals before any classifier had to learn from them. Delete it and the classification head now trains on 100,000 locations of which perhaps ten are objects. You removed a stage and inherited a statistics problem, and that problem is the entire subject of 4.6.

### YOLO v1 → v3: the core ideas

**YOLOv1 (2016).** Divide the image into an $S\times S$ grid ($S=7$). Each cell predicts $B=2$ boxes (with confidence) and one set of $C=20$ class probabilities → output tensor $7\times7\times(5B + C) = 7\times7\times30$. One network, one loss, 45 FPS. Weaknesses: coarse grid (each cell predicts one *class*, so overlapping objects of different classes in a cell are impossible), poor localisation (direct coordinate regression), bad on small and clustered objects.

**YOLOv2/v3.** The important changes:
- **Anchors from k-means on the dataset's box dimensions** — using IoU as the distance ($d = 1 - \text{IoU}$) rather than Euclidean, so the clustering optimises the thing that matters. A nice, concrete example of matching the objective to the metric.
- **Constrained centre prediction** $b_x = \sigma(t_x) + c_x$ — the sigmoid keeps the predicted centre inside its own grid cell, which stabilises early training (unconstrained centre regression lets any anchor claim any object and training diverges).
- **Multi-scale prediction** at 3 feature levels × 3 anchors each (an FPN-like neck).
- **Independent logistic classifiers** per class instead of softmax, so multi-label ("woman" and "person") works.

**SSD (2016).** Predicts from **multiple feature maps** of decreasing resolution, with per-level default boxes. Trains with **hard negative mining** at a fixed 3:1 negative:positive ratio — an explicit patch for the imbalance problem that focal loss later solved properly.

### Anchor-based vs anchor-free

**What anchors buy:** a prior over plausible box shapes, turning a hard regression problem into a small residual regression. **What they cost:**
- Hyperparameters (number, scales, aspect ratios) that must be tuned per dataset — a detector tuned for COCO transfers badly to, say, aerial imagery with very different aspect ratios.
- **Massive foreground/background imbalance** — ~100k anchors, ~10 positives (4.6).
- Extra memory and IoU computation during assignment.

**Anchor-free designs:**

| Method | Formulation |
|---|---|
| **CornerNet** (2018) | detect top-left and bottom-right **corners** as heatmaps, group with associative embeddings |
| **CenterNet / Objects as Points** (2019) | predict a **centre-point heatmap** + size regression; peak extraction replaces NMS |
| **FCOS** (2019) | per-pixel regression of distances $(l,t,r,b)$ to the box sides, plus a **centre-ness** branch that down-weights predictions far from the object centre (suppressing the low-quality boxes that would otherwise survive NMS) |

FCOS's **centre-ness** is worth knowing precisely, and it is much easier to remember once you see what it is measuring rather than treating it as an arbitrary expression.

**The problem it solves.** FCOS makes *every* pixel inside a ground-truth box a positive, each regressing its own $(l,t,r,b)$ to the four sides. A pixel near the object's centre has a symmetric, easy job. A pixel just inside the box's top-left corner has to predict a tiny $l$ and $t$ and an enormous $r$ and $b$ — a long-range extrapolation from features that mostly see background. Those corner predictions are systematically bad boxes, and because they are trained as positives they carry confident classification scores. At inference they survive NMS and displace the good centre prediction. So the head needs a way to say "trust this location less" *without* refusing to train on it.

**Now build the measure.** You want a number that is 1 at the centre and 0 at the edge. Take the horizontal direction alone: at the centre $l = r$, at the left edge $l = 0$. The ratio $\min(l,r)/\max(l,r)$ is exactly that — 1 when the two distances are equal, 0 when either collapses, and scale-free because it is a ratio of two lengths from the same box. Do the same vertically with $t$ and $b$. You now have two independent scores in $[0,1]$, and a location is only good if it is central in *both* directions, so multiply rather than average — one bad axis should sink the score, which a product does and a mean does not. Finally take the square root, because multiplying two numbers below 1 pushes the result toward 0 too aggressively; the root restores the scale so a location that is 0.5-central in each axis scores 0.5 rather than 0.25. That gives

$$
\text{centerness} = \sqrt{\frac{\min(l,r)}{\max(l,r)} \times \frac{\min(t,b)}{\max(t,b)}}
$$

It is trained with binary cross-entropy against this computed target, and at inference it is **multiplied into the classification score** so off-centre predictions rank lower and NMS discards them first. It is the anchor-free answer to "which of these dense predictions should I trust?" — and note the shape of the answer: rather than removing bad candidates from training, it keeps them and learns to *down-rank* them. Exactly the move focal loss makes in 4.6, one section later, applied to a different problem.

### Label assignment — the real modern battleground

**This is the most important thing to understand about modern one-stage detectors**, and most candidates have never heard of it. Given GT boxes and thousands of candidate locations, *which candidates are positive?*

It sounds like a bookkeeping detail. It is not, and the reason is worth stating plainly: **label assignment is the only place in a detector where you decide what the network is being asked to learn.** The architecture determines what it *can* express; the assignment determines what the loss actually rewards. Get it wrong and no amount of backbone helps, because you are supervising the wrong locations. You have already seen two assignment rules without them being named as such — Faster R-CNN's IoU ≥ 0.7 / ≤ 0.3 / ignore band (4.3) and FCOS's "every interior pixel, down-weighted by centre-ness" above — and noticing that those are two answers to the *same* question is most of the insight.

**Pause:** Faster R-CNN uses a fixed IoU threshold of 0.7. Name one kind of object for which that threshold is structurally unfair, before reading the table.

A very small one, and a very elongated one. IoU is extremely sensitive to shift when the box is small — a 3-px offset on a 20-px box already costs you most of the overlap — so a small object may have *no* anchor above 0.7 while a large object has forty. Similarly a 10:1 aspect-ratio object (a lamppost, a pen) overlaps poorly with every anchor in a 1:1/1:2/2:1 set. A fixed global threshold therefore hands different objects wildly different amounts of supervision purely as a function of their geometry, which is what "adaptive" in ATSS is reacting against.

| Assigner | Rule |
|---|---|
| **Static IoU threshold** | fixed IoU ≥ 0.5/0.7 (Faster R-CNN, RetinaNet). Simple; badly suited to objects of unusual size or shape |
| **ATSS** (2020) | compute per-GT the mean + std of IoUs of the $k$ closest candidates and use $\mu+\sigma$ as an **adaptive** threshold. The paper's headline result: **once you fix the assignment, anchor-based and anchor-free detectors perform identically** — i.e. the anchor-vs-anchor-free debate was really an assignment debate |
| **SimOTA** (YOLOX, 2021) | treat assignment as an **optimal transport** problem — cost = classification + regression loss, dynamic $k$ per GT based on IoU mass |
| **Task-Aligned Assigner** (TOOD/YOLOv8+) | score candidates by $s^\alpha \cdot \text{IoU}^\beta$, aligning the classification and localisation objectives so the highest-scoring box is also the best-localised |

ATSS's rule repays a second's thought, because $\mu + \sigma$ looks like a magic number and is not. Take the $k$ candidates closest to a GT box and look at their IoU distribution. If the object matches the anchor set well, those IoUs are high and tightly clustered, so $\mu+\sigma$ lands just above the pack and only the genuinely best few become positive. If the object matches badly — the small or elongated case from the Pause — the IoUs are all low, so $\mu+\sigma$ is *also* low and the best of a bad set still becomes positive. The threshold is derived from each object's own statistics instead of imposed globally, which is exactly the fairness the fixed rule lacked. It is the same instinct as normalising by $w_a$ in the box encoding (4.1): when a quantity's natural scale varies per object, measure it in units of that object.

**ATSS's finding is the single best thing to quote here**: the anchor-based/anchor-free distinction largely dissolves once label assignment is controlled for.

**In your own words:** what does a label assigner actually decide, and why does changing it move AP more than changing the backbone?

### NMS-free one-stage detection 🟢

**YOLOv10 (2024) / YOLO26 (Jan 2026)** use **consistent dual assignments**: a one-to-many assignment head for rich training supervision, plus a one-to-one head used at inference, trained to be consistent with it. Result: **no NMS at inference** — lower latency and no NMS hyperparameter. This is the one-stage family adopting DETR's end-to-end property while keeping YOLO's speed.

### The 2026 landscape 🟢

| Model | Type | Notes |
|---|---|---|
| **YOLO26** (Jan 2026) | one-stage, **NMS-free end-to-end** | edge-optimised; ~40.9 mAP nano at ~1.7 ms (T4), ~57.5 mAP XL; ~43% faster CPU inference than YOLO11-n. **AGPL-3.0** |
| **YOLOv12** (Feb 2025) | attention-centric one-stage | Area Attention + R-ELAN blocks; ~40.6 mAP-n at 1.64 ms, ~52.5 mAP-m |
| **RF-DETR** (Roboflow) | DETR-family, real-time | **first real-time model past 60 mAP on COCO**; DINOv2 backbone, deformable attention, anchor-free and NMS-free; strong domain transfer (~60.6 mAP on RF100-VL). **Apache-2.0** |
| **RT-DETR** | DETR-family, real-time | efficient hybrid encoder + IoU-aware query selection; the model that made DETR real-time |
| **RTMDet** | one-stage throughput specialist | ~52.8 AP at 300+ FPS; tiny variant >1000 FPS. **MIT** |
| **Grounding DINO / YOLO-World** | open-vocabulary | text-prompted detection, no fixed class list (6.9) |

**The licensing point is a genuine practical differentiator in interviews:** Ultralytics YOLO models are **AGPL-3.0**, which is a serious constraint for a commercial closed-source product (it can require releasing your source), while RF-DETR (Apache-2.0) and RTMDet (MIT) are permissive. Engineers who have shipped detection know this; candidates who have only benchmarked don't.

### 🎯 Top-1% distinction

1. **"Anchor-based vs anchor-free is mostly a label-assignment question"** — cite ATSS.
2. **Explain FCOS's centre-ness and what problem it solves.**
3. **Know YOLOv2's sigmoid centre constraint and the IoU-based k-means anchors** — two specific, checkable details.
4. **NMS-free detection via consistent dual assignments** (YOLOv10/YOLO26) — currency.
5. **Licensing (AGPL vs Apache/MIT)** — the practitioner's tell.
6. **Frame the one-stage/two-stage difference correctly**: it's not "speed vs accuracy" any more (one-stage detectors now match or beat two-stage on COCO). It's about *where the imbalance is handled* — two-stage handles it structurally via the proposal cascade, one-stage handles it in the loss (focal) or in the assignment.

### ✅ Mastery check

(a) Why does YOLOv2 pass $t_x$ through a sigmoid, and what goes wrong without it?
(b) Your detector on aerial imagery (mostly tiny, square objects) performs badly with COCO-default anchors. Give a principled fix and one that removes the problem entirely.
(c) Explain why a one-stage detector needs focal loss but Faster R-CNN doesn't.

<details><summary>Answer sketch</summary>
(a) $b_x = \sigma(t_x) + c_x$ constrains the predicted centre to lie <b>within its own grid cell</b>. Without the constraint, any anchor at any location can predict a centre anywhere in the image, so early in training (random weights) the assignment between predictions and objects is unstable — many anchors compete for the same object, gradients conflict, and training diverges or takes far longer to stabilise. The sigmoid makes the prediction a <i>local refinement</i> of a fixed prior, which is the same reason the R-CNN delta encoding works (4.1).
(b) <b>Principled fix:</b> re-run k-means on <i>your</i> dataset's box dimensions with IoU distance (YOLOv2's method) to get anchors matched to your object statistics; also add smaller scales and reduce the aspect-ratio spread since the objects are square. <b>Removes the problem entirely:</b> switch to an <b>anchor-free</b> detector (FCOS/CenterNet) with an <b>adaptive assigner</b> (ATSS or SimOTA), which has no shape prior to mistune. Also worth saying: increase input resolution / tile the imagery, because tiny objects in aerial data are usually resolution-limited rather than prior-limited.
(c) Faster R-CNN's <b>RPN acts as a cascade</b>: it reduces ~28,000 anchors to ~2,000 proposals, most of them plausible, and then the second stage samples a fixed mini-batch with a bounded 1:3 positive:negative ratio. So by the time the classification head sees data, the imbalance has been handled <i>structurally</i>, twice. A one-stage detector has no such filter — its classification head is trained on all ~100k dense locations, of which ~10–100 are positive, a ratio of ~1000:1. The aggregate loss from the vast number of easy negatives swamps the positives' gradient. Focal loss handles that imbalance <i>in the loss function</i> instead of in the architecture. (Modern nuance: with a good adaptive assigner the imbalance is smaller, and focal loss matters somewhat less than it did in 2017.)
</details>

### 🔨 Build + read

**Build:** Implement FCOS's head (per-pixel $l,t,r,b$ regression + centre-ness) and its centre-sampling label assignment. Then implement ATSS and swap it in, measuring the AP delta on a small dataset — reproducing ATSS's central claim yourself. Separately: run a modern detector (YOLO26 or RF-DETR) on your own images and profile it, including the NMS cost, so you can talk about real latency numbers.

**Read:** Zhang et al., "ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection" (CVPR 2020) — the most important detection paper most people haven't read. Tian et al., "FCOS" (ICCV 2019). Roboflow's "Best Object Detection Models" for the current landscape.

---

## 4.6 Focal Loss (RetinaNet) 🔴

> Flagged critical, and it is the most commonly asked "explain a loss function" question in CV interviews. The reason is that a good answer requires arithmetic, not recall.

### Intuition

Here is the question, and it is a question about *arithmetic*, not about architecture. **If 99.99% of your training examples are correct and contribute almost no loss each, can they still ruin training?**

The instinct is no — a solved example has a small gradient, so it should be harmless. That instinct is what makes this section worth doing carefully, because the answer is yes, and the reason is that "almost no loss" gets multiplied by a very large number.

A one-stage detector evaluates ~100,000 candidate locations per image, of which maybe 10 contain an object. Each of the 99,990 background locations is easy — the model is 99% sure there's nothing there — and contributes a tiny loss. But **tiny × 100,000 is bigger than large × 10**. The gradient is therefore dominated by examples the model has already solved, and the rare positives get drowned out. Focal loss fixes this by making the loss on easy examples *even smaller*, so the hard ones dominate.

Read that last sentence sceptically, because it is where people's mental model usually goes wrong. Focal loss does **not** make the model care more about positives — there is no term in it that mentions foreground versus background. It makes the model care less about *anything it is already confident and correct about*, which happens to be almost entirely background. The imbalance being fixed is not really foreground/background; it is **easy/hard**, and the class imbalance matters only because it is what produced the pile of easy examples.

### The arithmetic that makes the problem concrete

Standard cross-entropy for a binary detector: $\text{CE}(p_t) = -\log(p_t)$, where $p_t = p$ for positives and $1-p$ for negatives.

Take a well-trained-ish model: on each easy negative it predicts $p = 0.01$, so $p_t = 0.99$ and $\text{CE} = -\log(0.99) = 0.01$. On each positive it predicts $p=0.5$, so $\text{CE} = 0.69$.

$$
\underbrace{100{,}000 \times 0.01}_{\text{easy negatives}} = 1000
\qquad\text{vs}\qquad
\underbrace{10 \times 0.69}_{\text{positives}} = 6.9
$$

**The background contributes ~145× more loss than the objects.** The model's gradient budget is spent confirming that empty sky is empty sky. **Do this calculation on the whiteboard** — it is the whole motivation, and it converts a vague "class imbalance" statement into something quantitative.

Two details in that calculation are worth pausing on, because they are what make the number believable. First, $-\log(0.99) \approx 0.01$ uses $\log(1+x)\approx x$ for small $x$ — a confident-and-correct example's cross-entropy is *linear* in its residual error, so it never quite reaches zero; it just gets small. Second, the loss is a **sum over samples**, and gradients sum the same way, so the aggregate gradient pushing the classifier toward "background" is the sum of 100,000 small pushes all pointing the same direction, while the pull toward "foreground" is ten larger pushes pointing in ten different directions. Coherence matters as much as magnitude here. That is why the failure mode is not "the model learns slowly" but "the model collapses to predicting background everywhere and stays there."

### Prior approaches, and why they're worse

- **Fixed sampling ratio** (Faster R-CNN's 1:1, SSD's 3:1): throws away almost all the data, and the retained negatives are chosen by a heuristic.
- **OHEM (Online Hard Example Mining)**: sort negatives by loss and keep the top-$k$. Better, but it **completely discards easy examples**, and easy examples do carry a little useful signal. It also adds a sort and a hyperparameter.

Focal loss's framing: **don't discard, re-weight**. It is *soft* OHEM, differentiable, with no sampling step.

### The loss

**Derive the form before reading it.** You want to multiply cross-entropy by a weight $w$, and you can write down the requirements the weight must satisfy purely from the problem statement above:

1. $w$ must depend on **how well the example is already classified**, not on its class — the diagnosis was easy-vs-hard, not foreground-vs-background. The only thing available that measures "already classified well" is $p_t$, the probability assigned to the correct answer. So $w = w(p_t)$.
2. $w$ must **decrease** as $p_t \to 1$, and go to 0 there — a perfectly classified example should contribute nothing.
3. $w$ must be **1 (or near it) when $p_t$ is small** — a hard or misclassified example must keep its full loss. You are down-weighting the easy, not up-weighting the hard, and those are different: up-weighting the hard would amplify label noise.
4. It must be **smooth and differentiable**, since the whole complaint against OHEM was that its top-$k$ selection is a hard, non-differentiable cut.
5. It must have a **knob** so you can dial the aggressiveness rather than committing to one strength.

Now write the simplest function meeting all five. The quantity that is 0 when $p_t = 1$ and 1 when $p_t = 0$ is $(1 - p_t)$ — the model's residual error on that example. Raising it to a power $\gamma \ge 0$ gives you the knob and lets you make the decay as sharp as you like while keeping smoothness. There is essentially nothing simpler, and that is the answer:

$$
\boxed{\ \text{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\log(p_t)\ }
$$

The modulating factor is *the example's own error, raised to a power*. Said that way it is hard to forget, and $\gamma = 0 \Rightarrow$ cross-entropy is immediate rather than a memorised special case.

Two additions to cross-entropy:

**1. The modulating factor $(1-p_t)^\gamma$.** With $\gamma = 2$:

| $p_t$ | model is | $(1-p_t)^2$ | down-weighting |
|---|---|---|---|
| 0.5 | uncertain | 0.25 | 4× |
| 0.9 | confident, correct | 0.01 | **100×** |
| 0.968 | very confident | 0.00102 | **~1000×** |
| 0.1 | confidently **wrong** | 0.81 | 1.2× (barely touched) |

Read the table as a *ratio* rather than four independent rows, because the ratio is the mechanism. Compare the top and bottom lines: an example the model has right at $p_t = 0.9$ is suppressed 100×, while an example it has confidently *wrong* at $p_t = 0.1$ is suppressed 1.2× — barely at all. That 100:1.2 gap is focal loss in one number. Cross-entropy already treats those two differently (0.105 vs 2.303, a 22× gap); focal loss with $\gamma=2$ widens it to roughly 1800×.

So a well-classified example's contribution shrinks by two to three orders of magnitude, while hard and misclassified examples keep nearly their full weight.

**Pause:** before checking the arithmetic below — you suppressed the easy negatives by ~10,000× and the positives (at $p_t = 0.5$) by 4×. Does that leave the two groups roughly balanced, over-corrected, or still background-dominated?

Redo the earlier arithmetic with $\gamma=2$: easy negatives now contribute $100{,}000 \times 0.01 \times 0.0001 \approx 0.1$ against the positives' $10\times0.69\times0.25 \approx 1.7$ — **the balance has flipped.** It did not land on "balanced"; it landed on positives contributing ~17× more loss than background, having started at 145× the other way. That over-correction is not a rounding artefact — it is the thing $\alpha$ exists to undo, and you will see it explain the $\alpha = 0.25$ oddity in the next paragraph.

$\gamma$ controls the strength: $\gamma=0$ recovers cross-entropy; $\gamma=2$ is the paper's default; $\gamma>5$ over-suppresses and hurts. The failure at large $\gamma$ follows from requirement 3 above: push the decay hard enough and it stops discriminating between "easy" and "moderately hard", so the surviving training signal comes only from the extreme tail — which in real datasets is disproportionately label noise and ambiguous boxes.

**2. The $\alpha_t$ balancing factor.** A conventional class-frequency weight, $\alpha$ for positives and $1-\alpha$ for negatives. **The counter-intuitive part:** the paper uses $\alpha = 0.25$ — i.e. it weights the *rare foreground class down*, which looks backwards. The reason: $\gamma$ has already suppressed the easy negatives so aggressively that the balance has over-corrected toward positives; $\alpha$ pulls it back. **$\alpha$ and $\gamma$ interact, and the optimal $\alpha$ decreases as $\gamma$ increases.** Being able to explain that inversion is a strong signal — it shows you understand the two terms as coupled rather than as two independent knobs.

### The prior-initialisation trick — the detail that separates readers from implementers

At initialisation, weights are near zero and the bias is zero, so every logit is ~0 and the sigmoid gives $p \approx 0.5$ everywhere. Put a number on what that costs. Each of ~100,000 anchors is a negative sitting at $p_t = 0.5$, so each contributes $\alpha_t (0.5)^2 \log 2 \approx 0.13$, and the first batch's loss is on the order of $10^4$. Worse than its size is its *direction*: 100,000 coherent gradients all saying "lower the foreground logit" against ten saying "raise it." The first optimiser step slams the bias far negative, the head saturates, and the ten positives never recover — RetinaNet **diverges** without a fix.

Notice that this is a *starting-point* problem, not a loss problem. The model's initial belief — "an object is present at half of all 100,000 locations" — is absurd, and one large corrective step is the optimiser rationally responding to an absurd initialisation. So do not fix it in the loss; fix the initialisation, by starting the model at a belief that is approximately true.

**Derive the constant.** You want the head's initial output probability to equal a small prior $\pi$ (the paper uses $\pi = 0.01$, i.e. "one location in a hundred contains an object", which is the right order of magnitude for detection). At initialisation the weights contribute ~0, so the logit is just the bias $b$, and the output is $\sigma(b)$. Set $\sigma(b) = \pi$ and invert the sigmoid — the inverse of $\sigma$ is the logit function:

$$
\frac{1}{1+e^{-b}} = \pi \;\Longrightarrow\; 1 + e^{-b} = \frac{1}{\pi} \;\Longrightarrow\; e^{-b} = \frac{1-\pi}{\pi} \;\Longrightarrow\; b = -\log\!\left(\frac{1-\pi}{\pi}\right)
$$

With $\pi = 0.01$ that is $-\log(99) \approx -4.59$, and indeed $\sigma(-4.59) = 0.01$. So the constant is not tuned — it is the log-odds of the prior, and if you change $\pi$ you recompute it. Initialise the **bias of the final classification conv** to it (only that bias — the box head and every other layer are untouched) and the model starts out believing everything is background, which is (correctly) almost true. The initial loss drops by orders of magnitude and the first steps are corrections rather than a collapse.

The general principle is worth extracting, because it recurs: **initialise a head's bias to the base rate of the thing it predicts.** You have met it before without the detection framing — it is why a language model's output bias is often set to log unigram frequencies, and why BatchNorm's $\gamma,\beta$ (2.4) are initialised so a block starts as the identity. Start the model at the trivially correct answer and let gradients teach it the non-trivial part.

**This one line is the difference between RetinaNet training and RetinaNet diverging**, and it is almost never mentioned in blog posts. If you have implemented a one-stage detector, you know it; if you have only read about focal loss, you don't.

### RetinaNet

ResNet + **FPN** (4.4) + two small subnets applied to every pyramid level with shared weights: a **classification subnet** (4× conv $3\times3$ 256ch + a final conv producing $K\!\times\!A$ logits) and a **box regression subnet** (same shape, $4A$ outputs). Trained with focal loss on the classification branch. It was the first one-stage detector to beat two-stage detectors on COCO AP while remaining faster — and its entire contribution over previous one-stage designs was the loss plus the pyramid.

### Successors and the 2026 nuance

- **Generalized Focal Loss / Quality Focal Loss** (2020): merge classification score and localisation quality into a single *continuous* target (the IoU), because the standard setup trains them separately and then ranks by classification score at inference — a mismatch. QFL extends focal loss to continuous labels.
- **VariFocal Loss** (VarifocalNet): asymmetric — down-weight negatives but *not* positives, since positives are precious.
- **The nuance worth stating:** modern detectors with strong **adaptive label assignment** (ATSS, SimOTA, TaskAligned — 4.5) produce a far less extreme imbalance, so focal loss's marginal benefit is smaller than in 2017. It is still standard (YOLOX, YOLOv8, RTMDet, DETR-family all use focal or a variant), but the framing "focal loss solved detection imbalance" is a 2017 framing; the 2026 framing is "**assignment and loss re-weighting together** solve it, and assignment now does more of the work."
- **Calibration cost:** focal loss deliberately suppresses the loss on confident predictions, which means the resulting model's output probabilities are **poorly calibrated** — typically under-confident on positives. If a downstream system consumes detection scores as probabilities (risk scoring, sensor fusion, tracking gates), you need temperature scaling or an explicit calibration step. **Very few candidates mention this**, and it's a real production concern.

  The reason is worth being able to state, because it follows directly from the derivation rather than being an empirical surprise. Cross-entropy is a **proper scoring rule**: its minimiser is the true conditional probability, which is exactly why a CE-trained model's outputs mean something. Multiplying by $(1-p_t)^\gamma$ destroys that property — the weight depends on the prediction itself, so the loss is no longer minimised at the true probability. You bought better ranking (which is what AP measures) by giving up correct probabilities (which is what a threshold assumes). If your product reads the score as a number rather than as a rank, you have to buy the calibration back.

**In your own words:** state in one sentence what focal loss down-weights — without using the words "foreground" or "background".

### 🎯 Top-1% distinction

1. **Do the loss-mass arithmetic** (100k × 0.01 vs 10 × 0.69). Quantify the problem before describing the solution.
2. **Focal loss is soft OHEM** — re-weight rather than discard.
3. **Explain the $\alpha = 0.25$ inversion** and that $\alpha,\gamma$ are coupled.
4. **The prior bias initialisation $b = -\log((1-\pi)/\pi)$** and that training diverges without it.
5. **The 2026 nuance**: better assignment has absorbed part of focal loss's job.
6. **The calibration side-effect.**

### ✅ Mastery check

(a) With $\gamma = 2$, by what factor is the loss on an example with $p_t = 0.99$ reduced relative to CE? What about $p_t = 0.3$?
(b) You set $\gamma = 5$ and training collapses — the model predicts background everywhere. Explain.
(c) You use focal loss for a **balanced** binary image-classification task (50/50). Predict the effect.
(d) Your detector's confidence scores are used to decide whether to alert a human operator. What must you check?

<details><summary>Answer sketch</summary>
(a) $p_t = 0.99$: factor $(1-0.99)^2 = 10^{-4}$ — the loss is <b>10,000× smaller</b> than CE. $p_t = 0.3$: $(0.7)^2 = 0.49$, so about <b>2× smaller</b> — hard examples are barely affected. That ratio (10,000× vs 2×) is the mechanism in one line.
(b) With $\gamma=5$, $(1-p_t)^5$ suppresses <i>everything</i> the model is even moderately confident about, including <b>positives it has started to get right</b>. The effective training signal collapses to only the very hardest examples, which are disproportionately <b>label noise, ambiguous boxes, and outliers</b> — the same failure as hardest-negative mining in triplet loss (3.5). Gradients become high-variance and driven by bad data; the model retreats to the safe solution (predict background everywhere), which the suppressed loss no longer punishes enough. Also, the $\alpha$ that was tuned for $\gamma=2$ is now badly wrong for $\gamma=5$.
(c) Focal loss becomes roughly a <b>hard-example-mining regulariser</b> with no imbalance to correct. Effects: it will down-weight the (many) correctly classified examples, so training effectively focuses on the confusing minority — this can help slightly on hard datasets but generally gives little or no gain over CE, and it will <b>degrade calibration</b> and can amplify label noise. The honest answer is "it is the wrong tool — focal loss's benefit is proportional to the imbalance, and there isn't any here."
(d) <b>Calibration.</b> Focal loss produces systematically under-confident scores on positives, so a threshold chosen from intuition ("alert above 0.8") will have completely different real-world precision/recall than expected. Check: plot a <b>reliability diagram</b> and compute <b>Expected Calibration Error</b> on a held-out set; apply <b>temperature scaling</b> (or isotonic regression) fitted on validation; and — most importantly — choose the operating threshold from the <b>precision/recall curve at the operating point the product needs</b>, not from the raw score value. Also report the resulting per-image false-alarm rate, since operators experience alerts per hour, not AP.
</details>

### 🔨 Build + read

**Build:** Implement focal loss from scratch (binary and multi-class) and validate against `torchvision.ops.sigmoid_focal_loss`. Then run the two experiments that make it stick: (i) plot loss vs $p_t$ for $\gamma \in \{0,0.5,1,2,5\}$ on one figure; (ii) train a small RetinaNet **with and without** the prior bias initialisation and show the loss curve diverging in the second case. Bonus: measure ECE before and after temperature scaling on your trained detector.

**Read:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017) — read §3 and §4.1 (the prior initialisation is in §3.3 and §4.1, easy to miss). Then Li et al., "Generalized Focal Loss" (NeurIPS 2020) §3.

---

## 4.7 NMS, IoU, Precision/Recall, AP/mAP

### IoU

Start from the question the metric has to answer: **when is a predicted box "the same box" as a ground-truth box?** There is no crisp answer — boxes are continuous, so every prediction is wrong by some amount and you need a graded score, which you then threshold to force a binary decision. Everything awkward about detection evaluation descends from that one compromise.

What should the graded score be? A distance between corners is the obvious first try and fails immediately for the reason you met in 4.1: 5 px means something different on a 20-px face and a 400-px bus. So the score must be **relative to the boxes' own size**, which means it should be a ratio of areas. Intersection over the smaller box is one candidate, but it rewards a huge prediction that swallows the GT entirely. Intersection over union is the fix: the denominator counts every pixel either box claims, so over-predicting inflates the denominator and is punished just as under-predicting shrinks the numerator.

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{intersection area}}{\text{area}(A) + \text{area}(B) - \text{intersection}}
$$

The second form is the one you implement — you never compute a union directly, you add the two areas and subtract the intersection you double-counted. Properties: in $[0,1]$, scale-invariant (a 5-px error matters more on a small box — exactly the property coordinate-space L1 lacks), and **zero for any two non-overlapping boxes regardless of how far apart they are** — which is why IoU alone is a bad *loss* (no gradient when there's no overlap).

That last property deserves the emphasis. IoU is a fine *metric* and a treacherous *loss*: a prediction 5 px outside the GT and a prediction on the other side of the image both score exactly 0, so $\partial\text{IoU}/\partial b$ is identically zero across the entire region where the model most needs guidance. A loss that is flat wherever you are wrong teaches nothing. Every entry in the table below is a different way of adding a term that keeps varying once the overlap has vanished.

**The IoU-family losses**, each fixing the previous one's flaw:

| Loss | Adds | Fixes |
|---|---|---|
| **IoU loss** $1-\text{IoU}$ | optimises the evaluation metric directly | scale-dependence of L1/L2 |
| **GIoU** $\text{IoU} - \frac{|C \setminus (A\cup B)|}{|C|}$, $C$ = smallest enclosing box | a penalty for empty space in the enclosing box | **gives gradient when boxes don't overlap** |
| **DIoU** adds $\frac{\rho^2(\mathbf{c}_A,\mathbf{c}_B)}{d^2}$ | normalised centre distance | faster convergence; GIoU degenerates to slow expansion for enclosed boxes |
| **CIoU** adds an aspect-ratio consistency term | shape agreement | the last free degree of freedom |

Walk one row to see the pattern. **GIoU** takes the smallest axis-aligned box $C$ containing both $A$ and $B$ and subtracts the fraction of $C$ that neither box occupies. When the boxes overlap heavily, $C$ is barely bigger than the union and the penalty is near zero, so GIoU ≈ IoU and nothing changes. When they are disjoint and far apart, $C$ is enormous and mostly empty, so the penalty approaches 1 and GIoU approaches $-1$ — and crucially it keeps *changing* as the boxes move, because moving them closer shrinks $C$. That restores the gradient IoU lost. **DIoU** then notices GIoU's remaining slow case: if one box is entirely inside the other, $C$ equals the outer box no matter where the inner one sits, so GIoU's penalty is again constant and convergence stalls. Adding normalised centre distance $\rho^2/d^2$ gives an always-informative pull toward alignment. **CIoU** finally adds aspect ratio, which is the one degree of freedom neither position nor scale has pinned down. Each row is the previous row's blind spot, patched.

**The principle to state:** *optimise the metric you evaluate.* L1 on coordinates is not monotonic in IoU — two predictions with identical L1 error can have very different IoU depending on box size. This is the same principle as ScaNN's anisotropic quantisation (3.7) and YOLOv2's IoU-based k-means (4.5): **align the training objective with the evaluation objective.**

### NMS

The algorithm is three lines and the interesting content is entirely in *why it is needed at all*. Recall from 4.3 and 4.5 that label assignment marks **many** anchors positive for a single object — an object with forty overlapping anchors above the IoU threshold is trained to be detected forty times. Duplicates are therefore not a bug in the model; they are precisely what it was taught to produce. NMS is the post-hoc undo of a training decision, which is exactly why it feels like a hack: no part of the network knows it exists, no gradient flows through it, and its threshold is tuned against a metric the model never saw.

```
sort detections by score, descending
while boxes remain:
    take the highest-scoring box b, add to output
    remove every remaining box with IoU(b, ·) > threshold   # typically 0.5–0.7
```

Applied **per class** by default (a person box shouldn't suppress an overlapping handbag box). "Batched NMS" implements this efficiently by offsetting boxes of different classes into disjoint coordinate ranges and running one NMS.

**The failure mode: crowded scenes.** Notice what information NMS is working from: box overlap and score, nothing else. It has no way to tell "two boxes on one object" from "two boxes on two objects that happen to overlap" — those look identical in the only variables it can see. So two genuinely distinct objects that overlap by more than the threshold — a crowd of pedestrians, cars in dense traffic, cells in microscopy — and NMS deletes one of them. Raising the threshold keeps both but also keeps duplicates.

**Pause:** you have a single image containing both a sparse region (one isolated car) and a crowded one (twenty overlapping pedestrians). What is the best NMS threshold for that image?

There isn't one, and that is the point — the sparse region wants a low threshold to kill duplicates aggressively, the crowd wants a high one to preserve real objects, and NMS applies a single global value to both. **There is no threshold that solves both**, which is the structural argument for NMS-free detection. Everything below is either a softer version of the same guess (Soft-NMS decays instead of deleting, so a wrong decision is recoverable rather than fatal) or an attempt to smuggle in extra information (DIoU-NMS uses centre distance, on the reasoning that two boxes on *one* object share a centre while two occluding objects do not). The real fix is to stop creating duplicates in the first place, which is what one-to-one assignment does in 4.8 and what dual assignment does in YOLOv10.

**Variants:**
- **Soft-NMS**: instead of deleting, *decay* the score — linear $s_i \leftarrow s_i(1-\text{IoU})$ or Gaussian $s_i \leftarrow s_i e^{-\text{IoU}^2/\sigma}$. Neighbours survive with reduced confidence. Consistently ~+1 AP on COCO for one line of code, and much better in crowds.
- **DIoU-NMS**: suppress using DIoU rather than IoU, so boxes with distant centres survive even at high IoU — better for occluded objects.
- **Class-agnostic NMS**: when a detection must be unique per location regardless of class.
- **Matrix NMS / Cluster-NMS**: parallel formulations for speed.

### AP and mAP — compute it precisely

**First, why a curve rather than a number.** A detector's output depends on a confidence threshold you have not chosen yet: set it high and you get few, reliable detections (high precision, low recall); set it low and you get everything plus noise (high recall, low precision). Quoting precision and recall at one threshold therefore measures your threshold choice as much as your model. AP's answer is to refuse to choose: sweep the threshold across its whole range, trace out the precision–recall curve, and summarise the *curve*. That is why AP is threshold-free — and, as you will see below, why it can move in the opposite direction to your shipped product.

For one class:

1. Sort all detections across the dataset by confidence, descending.
2. Walk the list. Match each detection to an unmatched GT box with IoU ≥ threshold (greedily, highest IoU first). Matched → **TP**; unmatched, or matching an already-matched GT → **FP**.
3. Accumulate TP/FP down the list, computing $P = \frac{TP}{TP+FP}$ and $R = \frac{TP}{\#GT}$ at each step.
4. **AP = area under the precision–recall curve.** COCO uses 101-point interpolation with the precision monotonically smoothed ($p_{\text{interp}}(r) = \max_{r'\ge r} p(r')$); old VOC used 11 points.
5. **mAP** = mean of AP over classes.

**Walk it once on a tiny example and the mechanics stop being abstract.** Say one class, 2 ground-truth boxes in the dataset, and 4 detections sorted by score: D1 (0.9), D2 (0.8), D3 (0.6), D4 (0.3). Suppose D1 matches GT-A at IoU 0.85, D2 also overlaps GT-A at IoU 0.7, D3 matches GT-B at IoU 0.6, and D4 overlaps nothing.

Take them in order, because *sorted order is what makes step 2 well-defined* — the highest-scoring detection gets first claim on a GT, which is how the greedy matching resolves competition. D1 finds GT-A unmatched: **TP**. Running totals TP=1, FP=0, so $P = 1/1 = 1.0$ and $R = 1/2 = 0.5$. D2 clears the IoU threshold against GT-A, but GT-A is already spoken for and there is no other GT nearby: **FP** — a correct-looking box on a real object, counted as an error, purely because it is a duplicate. Now TP=1, FP=1, $P = 1/2 = 0.5$, $R$ still $0.5$. D3 matches GT-B, unmatched: **TP**. TP=2, FP=1, $P = 2/3 = 0.67$, $R = 2/2 = 1.0$. D4 matches nothing: **FP**. $P = 2/4 = 0.5$, $R = 1.0$.

So the PR points, in the order the walk generated them, are $(0.5, 1.0), (0.5, 0.5), (1.0, 0.67), (1.0, 0.5)$ as (recall, precision). Precision is not monotonic — it dipped to 0.5 and recovered to 0.67 — which is why step 4 smooths it: $p_{\text{interp}}(r) = \max_{r'\ge r} p(r')$ replaces each precision with the best precision achievable at *that recall or higher*, on the reasoning that you would never operate at a threshold that is dominated by a lower one. Here that raises the precision at $R=0.5$ from 1.0 (already maximal) and leaves $R=1.0$ at 0.67. Averaging the interpolated precision over the 101 recall points $0, 0.01, \ldots, 1.00$ gives AP ≈ $(51 \times 1.0 + 50 \times 0.67)/101 \approx 0.84$ — the first 51 points sit at recall ≤ 0.5 where interpolated precision is 1.0, the remaining 50 at 0.67.

The instructive part is D2. It was a good box on a real object and it cost you AP, and if NMS had removed it your AP would have been 1.0 instead of 0.84. **That single example is the whole reason detectors need duplicate removal**, and it is worth re-deriving on a whiteboard rather than asserting.

**COCO's headline metric is AP@[.50:.05:.95]** — AP averaged over 10 IoU thresholds from 0.50 to 0.95. This is why COCO AP is much lower than VOC mAP@0.5 for the same detector: it rewards *localisation quality*, not just detection. Also reported: AP50, AP75, and **AP_small / AP_medium / AP_large** (area $< 32^2$, $32^2$–$96^2$, $> 96^2$ px), plus AR@{1,10,100}.

**Three subtleties worth knowing:**

1. **Each GT can be matched only once.** A second detection of the same object is an FP. This is precisely why NMS matters for the metric, and why a detector that emits duplicates is penalised even though it "found" the object.
2. **The confidence threshold does not affect AP** — AP sweeps the entire curve. It absolutely affects your *deployed* precision and recall. Candidates who conflate "improving AP" with "improving the product" get caught here.
3. **mAP may not be your product metric.** AP integrates over the entire PR curve, including operating points you would never ship — a region at 99% recall and 5% precision contributes to your AP and would get an operator fired. For pedestrian detection, the field uses **log-average miss rate (MR⁻²)** over false-positives-per-image, because what matters is misses at a tolerable alarm rate. For counting, you care about count error. For safety systems, per-image recall at a fixed FPPI. **Always ask what the operating point is.**

**In your own words:** why is a second, well-localised detection of an object counted as a false positive rather than a second true positive?

### 🎯 Top-1% distinction

1. **Explain the one-GT-one-match rule** and connect it to why NMS exists.
2. **COCO AP averages over IoU thresholds** — so it measures localisation quality; VOC AP50 doesn't.
3. **The crowded-scene NMS dilemma has no threshold solution** → Soft-NMS, then NMS-free architectures.
4. **The IoU-family loss progression and the "optimise what you evaluate" principle.**
5. **AP is threshold-free; your product isn't.** Name a domain-appropriate alternative metric (MR⁻² for pedestrians).
6. **NMS is a latency cost too** — it's a sequential, data-dependent operation that doesn't batch well and can be a meaningful fraction of end-to-end latency at high detection counts. That's part of why NMS-free models matter for edge deployment.

### ✅ Mastery check

(a) Your model outputs, for one GT box, two detections at IoU 0.9 with each other, scores 0.95 and 0.90, both IoU 0.8 with the GT. Compute the contribution to AP with and without NMS.
(b) You raise the NMS threshold from 0.5 to 0.7. What happens to precision, recall, and AP?
(c) AP goes from 0.42 to 0.45 but your users report more misses. Give two explanations.

<details><summary>Answer sketch</summary>
(a) <b>With NMS</b> (threshold 0.5): the 0.90 box is suppressed. One detection remains, matches the GT at IoU 0.8 ⇒ 1 TP, 0 FP. Precision 1.0 at recall 1.0 for this object — clean contribution. <b>Without NMS</b>: both survive. Sorted by score, the 0.95 box matches the GT (TP). The 0.90 box now has <b>no unmatched GT available</b>, so despite having IoU 0.8 with a real object it counts as an <b>FP</b>. Precision drops from 1.0 to 0.5 at the recall level reached after both detections, dragging the PR curve down and reducing AP. This is exactly the mechanism by which duplicate predictions are punished.
(b) <b>Recall increases</b> (fewer true objects suppressed in crowded regions), <b>precision decreases</b> (more duplicates survive as FPs). <b>AP can go either way</b> and is empirically a shallow optimum — which is why 0.5–0.7 is the usual range and why the value is dataset-dependent (crowded datasets like CrowdHuman want a higher threshold; sparse ones want lower). The deeper point: NMS threshold trades the two error types and there is no setting that is right for both crowded and sparse regions of the same image — which is Soft-NMS's motivation.
(c) 1. <b>The AP gain came from the wrong place.</b> AP averages over all IoU thresholds and all classes; you may have improved localisation on already-detected large objects (raising AP75 and AP_large) while <i>losing</i> recall on small or rare ones. Check the AP breakdown (AP_small/medium/large, per-class AP) and the recall curve, not the single number. 2. <b>The operating threshold shifted.</b> AP is threshold-free, but your product ships one confidence threshold. If the new model's score distribution changed (very likely if you changed the loss — e.g. adopted focal loss, see 4.6's calibration point), the old threshold now corresponds to a different point on the PR curve, with higher precision and lower recall. Fix: re-tune the threshold to the target recall on a validation set, and report <b>recall at fixed precision</b> (or at fixed false-positives-per-image) as the tracked metric alongside AP.
</details>

### 🔨 Build + read

**Build:** Implement IoU, NMS, Soft-NMS, and COCO-style AP evaluation from scratch, then validate your AP against `pycocotools` on a real prediction file — matching it to 3 decimal places is a genuinely instructive debugging exercise (the greedy matching order and the interpolation are where bugs hide). Then plot AP vs NMS threshold on a crowded dataset (CrowdHuman) and a sparse one (COCO) to see the optimum move.

**Read:** Padilla et al., "A Survey on Performance Metrics for Object-Detection Algorithms" (2020) — the clearest treatment of AP's variants. Bodla et al., "Soft-NMS" (ICCV 2017), 4 pages. Rezatofighi et al., "GIoU" (CVPR 2019).

---

## 4.8 DETR: Object Queries, Set Prediction, Bipartite Matching 🟡

### Intuition

Go back to the question 4.1 opened with, because DETR is its answer. Detection's structural problem was that a network emits an **ordered, fixed-size tensor** while the task's answer is an **unordered, variable-size set**. Every architecture so far has closed that gap the same way: emit far too many candidates, then delete the extras with NMS. 4.7 showed what that costs — duplicates are trained *in* and hacked *out*, with a threshold no gradient ever sees.

DETR asks the other question: **what if you fixed the loss instead of the output?** The output can stay an ordered list of 100 slots — that is what tensors are. What has to change is that the loss must stop caring *which* slot holds which object. If slot 7 predicts the dog and slot 12 predicts the cat, that must cost exactly the same as the reverse. Make the loss permutation-invariant and the ordered list *becomes* a set, as far as training is concerned.

So: give the model 100 slots, tell it "fill these with the objects, one each, and put 'nothing' in the rest," and make the loss not care about the order. Then duplicates are penalised by construction and NMS is unnecessary. The rest of this section is what "make the loss not care about the order" actually requires — and it turns out to be a classical combinatorial-optimisation algorithm from 1955, sitting inside the training loop.

### The architecture

```
image → CNN backbone → feature map (C×H/32×W/32)
      → 1×1 conv to d=256 → flatten to HW tokens + fixed positional encoding
      → Transformer ENCODER (6 layers, self-attention over image tokens)
      → Transformer DECODER (6 layers):
             N=100 learned "object queries" (a learned embedding matrix)
             each layer: self-attention among queries + cross-attention to encoder output
      → shared FFN heads per query → (class logits over K+1, box (cx,cy,w,h) normalised)
```

**Object queries** are $N$ learned vectors, one per output slot. They are *not* anchors in the usual sense but they behave like a learned prior over where and what to look for — visualisations show each query specialises to a region of the image and a range of box sizes. (**DAB-DETR** later reinterpreted queries explicitly as **learned anchor boxes**, closing the circle: DETR's queries were anchors all along, just learned and in latent space.)

**Self-attention among the queries is what removes duplicates** — queries can see each other and learn "you take that object, I'll take this one." This is the mechanism that makes NMS unnecessary at inference; the matching loss is what *teaches* it.

### Bipartite matching with the Hungarian algorithm

**Build the mechanism from the requirement.** You need a loss $\mathcal{L}(\hat y, y)$ that gives the same value under any reordering of the slots. The standard trick for making a function permutation-invariant is to **minimise over all permutations**: define $\mathcal{L} = \min_\sigma \mathcal{L}_\sigma$, and since the set of permutations is closed under composition, relabelling the slots cannot change the minimum. That is the entire conceptual content of bipartite matching — you are not choosing an assignment because you want one, you are choosing it because minimising over assignments is how you delete order from a loss.

Two housekeeping steps make it well-posed. The GT set has $n$ objects and the prediction list has $N = 100$ slots, and a permutation needs both sides the same size — so **pad the GT with $N - n$ copies of $\varnothing$** ("no object"). Now every slot has a partner: $n$ of them get real objects, the other $N-n$ get "predict nothing", which is what turns "suppress duplicates" into an explicit, supervised target rather than a post-process. And you need a *cost* for pairing slot $j$ with GT $i$, which is just "how bad would this pairing be" — a class term plus a box term.

Formally: let $y = \{y_i\}$ be the GT set, padded with $\varnothing$ to size $N$, and $\hat{y} = \{\hat{y}_i\}_{i=1}^N$ the predictions. Find the permutation $\hat\sigma \in \mathfrak{S}_N$ minimising the total matching cost:

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}\bigl(y_i, \hat{y}_{\sigma(i)}\bigr)
$$

with, for a non-empty GT of class $c_i$,

$$
\mathcal{L}_{\text{match}} = -\hat{p}_{\sigma(i)}(c_i) + \mathcal{L}_{\text{box}}\bigl(b_i, \hat{b}_{\sigma(i)}\bigr)
$$

(note: the *probability itself*, not its log, is used in the matching cost — it makes the class and box terms commensurate). Be precise about why: $-\log \hat p$ is unbounded, blowing up to $+\infty$ as $\hat p \to 0$, while $\mathcal{L}_{\text{box}}$ lives on a bounded scale of order 1. Mix them and a single very-low-probability slot dominates the cost matrix and dictates the whole assignment for reasons that have nothing to do with box quality. Using $-\hat p \in [-1, 0]$ puts both terms on the same footing. Note this affects the **matching only** — the training loss below still uses $-\log \hat p$, because there you *want* the unbounded gradient. Choosing different functions for "decide who pairs with whom" and "compute the penalty" is deliberate, and spotting it is a strong read of the paper.

**Now: how do you actually find $\hat\sigma$?** $\mathfrak{S}_{100}$ has $100! \approx 10^{158}$ elements, so enumeration is not on the table, and the naive alternative — greedy, repeatedly taking the cheapest remaining pair — gives the wrong answer. See why with two slots and two objects: suppose pairing (slot 1, dog) costs 1, (slot 1, cat) costs 2, (slot 2, dog) costs 3, (slot 2, cat) costs 100. Greedy grabs the global cheapest cell, (slot 1, dog) at 1, and is then forced into (slot 2, cat) at 100, total 101. The optimum is (slot 1, cat) + (slot 2, dog) = 2 + 3 = 5. Greedy fails because taking a locally cheap pair *removes an option* another row needed, and it has no way to look ahead.

**Pause:** given that greedy is wrong and enumeration is impossible, would you expect the exact optimum to be reachable in polynomial time here?

It is, and the reason is the structure of the problem rather than a clever heuristic. This is a **linear assignment problem** — a linear objective over a permutation matrix — and its constraint polytope (doubly stochastic matrices) has permutation matrices as its *vertices*, so the continuous relaxation's optimum is automatically integral. Combinatorial problems that look exponential are often polynomial for exactly this kind of reason, and this one is solved exactly by the **Hungarian algorithm** in $O(N^3)$. The algorithm's own trick is worth one sentence: subtracting a constant from any row or column of the cost matrix leaves the *optimal assignment* unchanged (every permutation picks exactly one cell from that row), so you may repeatedly subtract to create zeros and then look for an assignment lying entirely on zeros. With $N=100$ that's $10^6$ operations per image — negligible next to the backbone, and it's done on CPU with `scipy.optimize.linear_sum_assignment`.

One consequence to note before moving on: the matching is a **non-differentiable, discrete decision computed under `no_grad`**, then held fixed while the loss and its gradients are computed on the chosen pairs. Nothing backpropagates through the Hungarian algorithm. This is the same pattern as label assignment in every other detector (4.5) — assignment is decided, then supervised — which is why DETR is best described as *replacing* label assignment rather than eliminating it.

**Then the actual training loss**, computed on the matched pairs:

$$
\mathcal{L}_{\text{Hungarian}} = \sum_{i=1}^{N}\Bigl[ -\log \hat{p}_{\hat\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \ne \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat\sigma(i)})\Bigr]
$$

with $\mathcal{L}_{\text{box}} = \lambda_{L1}\|b - \hat b\|_1 + \lambda_{\text{iou}}\mathcal{L}_{\text{GIoU}}(b,\hat b)$ (both, because L1 alone is scale-dependent and GIoU alone converges poorly), and the $\varnothing$ class log-probability **down-weighted 10×** to counteract the fact that most of the 100 slots are "no object" — DETR's own, much milder, version of the class-imbalance problem.

**Why matching removes the need for NMS.** The assignment is **one-to-one**: exactly one prediction is responsible for each GT object, and every other prediction is explicitly trained to say "no object" at that location. A duplicate is not a near-miss — it is directly punished as a $\varnothing$ prediction that emitted a confident box. Under standard dense detectors, by contrast, many anchors are positive for the same GT, so duplicates are *encouraged* during training and must be removed afterwards. **That contrast is the crisp answer to "why doesn't DETR need NMS?"**

The thing that trips people up here is assuming the matching *is* the duplicate-suppression mechanism at inference. It is not — Hungarian matching needs ground truth, so it exists only during training and is absent at test time. What the matching does is **create the training signal**; the mechanism that *acts* on that signal at inference is the decoder's self-attention among queries, which lets each query see what the others are claiming and back off. Loss teaches, attention executes. Say both halves and the answer is complete; say only one and it has a hole in it.

**In your own words:** why does minimising over all possible assignments make the loss permutation-invariant?

### DETR's two real problems, and the fixes

**1. Extremely slow convergence** — 500 epochs on COCO vs ~12–36 for Faster R-CNN. Causes:
- The Hungarian matching is **unstable early in training**: small changes in predictions flip the assignment, so a given query's target changes from step to step and it receives inconsistent supervision.
- Cross-attention starts nearly uniform over the whole image and must learn to become spatially selective from scratch — that takes a long time.

**Fixes:**
- **Deformable DETR** (2021): replace dense cross-attention with **deformable attention** — each query attends to only $K$ (=4) *learned sampling offsets* per feature level rather than all $HW$ positions. Consequences: (i) complexity drops from $O((HW)^2)$ to $O(HW\cdot K)$, so **multi-scale features become affordable** (fixing small-object AP), and (ii) the sampling locality gives the model a spatial prior, cutting training to ~50 epochs — **a 10× convergence speedup**.
- **DN-DETR / DINO-DETR** (2022): **query denoising** — feed in noised versions of the GT boxes as extra queries with *known* assignments, bypassing the matching instability for those queries and giving a stable auxiliary training signal. DINO adds **contrastive denoising** (positive and negative noised queries, teaching the model to reject near-duplicates) and **mixed query selection**. This family reached SOTA on COCO.
- **Auxiliary losses** at every decoder layer (in the original DETR already) — deep supervision that materially helps.

**2. Poor small-object AP** — single-scale features at stride 32, because full attention over a high-resolution map is quadratically expensive. Fixed by Deformable DETR's multi-scale sampling.

### The real-time DETR line 🟢

- **RT-DETR** (2023): an efficient hybrid encoder (intra-scale attention + cross-scale fusion, avoiding attention over all levels at once) plus **IoU-aware query selection** (initialise queries from high-scoring encoder features rather than from scratch). First DETR to beat YOLO on the speed/accuracy curve.
- **RF-DETR** (Roboflow, 2025–26): **DINOv2 backbone** + deformable attention; the **first real-time detector to exceed 60 mAP on COCO**, with notably strong domain transfer (≈60.6 mAP on RF100-VL). Apache-2.0.

The trajectory is worth stating: DETR's *idea* (end-to-end set prediction, no NMS) won; its *original implementation* was impractical, and five years of engineering — deformable attention, denoising, better query initialisation, and stronger backbones — made it the current frontier.

### 🎯 Top-1% distinction

1. **Explain why one-to-one matching removes NMS** (duplicates are explicitly trained as $\varnothing$), contrasted with one-to-many assignment in dense detectors.
2. **Know that the matching cost uses $\hat p$, not $\log \hat p$**, and why (commensurability with the box term).
3. **Diagnose the slow convergence correctly** — matching instability + cross-attention having to learn locality — and name denoising and deformable attention as the two fixes.
4. **Deformable attention's dual benefit**: linear complexity *enables* multi-scale, which is what fixes small objects. Most people mention only the speed.
5. **DAB-DETR's reinterpretation of queries as learned anchors** — it dissolves the "anchor-free vs anchor" framing again, the same way ATSS did (4.5).
6. **The $\varnothing$ down-weighting by 10×** — DETR's own imbalance patch, a nice callback to focal loss.

### ✅ Mastery check

(a) DETR uses $N = 100$ queries. What happens on an image with 150 objects? What does that imply for dense-scene applications?
(b) Why does DETR use *both* L1 and GIoU for boxes?
(c) Your DETR variant trains but produces many duplicate detections. Where would you look?
(d) Explain the compute difference between standard and deformable cross-attention for a $200\times200$ feature map with 100 queries.

<details><summary>Answer sketch</summary>
(a) It can detect at most 100 — the remaining 50 are simply unrecoverable, and worse, the Hungarian matching during training on such images assigns arbitrarily, giving noisy supervision. Implication: <b>$N$ must be set well above the maximum object count in your data</b> (DETR used 100 for COCO, whose images have ≤ ~93 objects). For dense-scene applications (crowds, cells, retail shelves with hundreds of items), you need a much larger $N$ — but attention cost and the $\varnothing$ imbalance both grow with $N$, so this is a genuine structural limitation of the vanilla design. Deformable DETR variants with 300–900 queries, or hybrid dense/query designs, are the practical answer.
(b) <b>L1 alone is scale-dependent</b>: the same absolute coordinate error is far more damaging for a small box than a large one, so an L1-only model under-serves small objects. <b>GIoU alone converges poorly</b>: it is scale-invariant and well-aligned with the metric, but its gradient is weak and can plateau, especially for boxes with little overlap. Combining them gives L1's strong, well-conditioned gradient signal <i>and</i> GIoU's scale-invariance and metric alignment. This is a specific instance of the general "optimise what you evaluate, but keep a well-conditioned gradient" trade-off (4.7).
(c) Three places, in order: (i) <b>the matching</b> — verify the Hungarian assignment is truly one-to-one and that you're not accidentally allowing many-to-one (a common bug when batching, if you match over the whole batch instead of per image); (ii) <b>decoder self-attention</b> — this is the mechanism that lets queries coordinate and avoid duplicating each other; if you removed it, reduced decoder depth, or broke the auxiliary losses at intermediate layers, duplicate suppression degrades; (iii) <b>the $\varnothing$ class weight</b> — if it's too low, predicting a spurious box is cheap. Also worth checking: whether you're evaluating an intermediate decoder layer's output (early layers legitimately have more duplicates — that's what the later layers refine).
(d) Feature map $200\times200 = 40{,}000$ tokens. <b>Standard cross-attention</b>: each of 100 queries attends to all 40,000 keys ⇒ $100 \times 40{,}000 = 4{,}000{,}000$ attention weights per head per layer (and the encoder's <i>self</i>-attention is far worse: $40{,}000^2 = 1.6\times10^9$, which is what actually makes high resolution impossible). <b>Deformable cross-attention</b>: each query samples $K=4$ points per level across, say, 4 levels ⇒ $100 \times 16 = 1600$ sampled values — roughly <b>2,500× fewer</b> attention computations, and linear rather than quadratic in the number of tokens. That is exactly why deformable attention makes multi-scale features affordable.
</details>

### 🔨 Build + read

**Build:** Implement the Hungarian matcher yourself (build the cost matrix, call `scipy.optimize.linear_sum_assignment`, and write the loss over the matched pairs) and unit-test it: construct a case where a greedy match and the optimal match differ, and verify yours picks the optimal one. Then fine-tune a pretrained Deformable DETR on a small custom dataset and compare the training curve to a YOLO fine-tune on the same data — the convergence difference is the lesson.

**Read:** Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) §3 — the matching and loss. Zhu et al., "Deformable DETR" (ICLR 2021) §3 — read the deformable-attention module carefully. Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes" (ICLR 2023).

---

## 4.9 Synthesis: Two-Stage vs One-Stage vs DETR 🟢

### The comparison that matters

The useful question is not "which family is best" — that question has no stable answer and interviewers know it. It is **"what are these three families actually disagreeing about?"** Read the table with 4.1's three sub-problems in mind (where to look, how to assign ground truth, how to remove duplicates); every row below is one of the three, and the families differ in almost nothing else.

**Pause:** cover the last four rows and predict them. Given only that DETR assigns one-to-one and dense detectors assign one-to-many, which family needs NMS, and which needs a class-imbalance patch?

One-to-many assignment creates duplicates by construction, so dense detectors need NMS and DETR does not; and it creates a huge negative pile, so dense detectors need focal loss or sampling while DETR needs only a mild 10× $\varnothing$ down-weight. Both columns fall out of the assignment row alone — which is the whole argument for treating assignment as the organising axis.

| Axis | Two-stage (Faster/Cascade R-CNN) | One-stage (YOLO, RetinaNet, FCOS) | DETR family |
|---|---|---|---|
| **How candidates are proposed** | learned RPN proposals | dense anchors / points | $N$ learned queries |
| **Label assignment** | IoU threshold, sampled mini-batch | IoU / adaptive (ATSS, SimOTA, TAL) | **Hungarian one-to-one** |
| **Imbalance handled by** | the proposal cascade (structural) | the loss (focal) + assignment | $\varnothing$ class weighting |
| **Duplicate removal** | NMS | NMS (or dual assignment → none) | none needed |
| **Speed** | slowest | fastest | now competitive (RT-DETR/RF-DETR) |
| **Small objects** | good with FPN | good with FPN | needs deformable multi-scale |
| **Training cost** | moderate | low | historically very high; now moderate |
| **Hyperparameters** | anchors, ratios, NMS | anchors, NMS, assignment | $N$, denoising settings |
| **Best for** | high accuracy, research baselines, mask tasks | edge, real-time, high throughput | end-to-end pipelines, strong transfer |

### The three framings that make you sound like a practitioner

1. **"The one-stage/two-stage speed-accuracy trade-off is largely historical."** Modern one-stage detectors match or beat two-stage on COCO. The surviving real difference is *where the class imbalance is handled* and whether you get instance masks easily.
2. **"The big open question moved from architecture to label assignment."** ATSS showed anchor-based ≈ anchor-free once assignment is controlled; DETR replaced assignment with matching; YOLOv10/YOLO26 use dual assignment to get end-to-end behaviour. **Assignment is the axis the field is actually optimising.**
3. **"Choose by constraint, not by leaderboard."** Edge with a hard latency budget → YOLO26/RTMDet. Commercial closed-source product → check the licence (AGPL vs Apache/MIT). Need masks → Mask R-CNN or Mask2Former. Domain far from COCO with few labels → RF-DETR (strong transfer) or an open-vocabulary model (6.9). Need calibrated scores → whatever you pick, budget for calibration.

**In your own words:** name the single design decision that most of this table is downstream of.

### ✅ Synthesis check

For each scenario, name a detector and defend it in three sentences: (a) counting cells in microscopy images, hundreds per image, offline; (b) pedestrian detection on an embedded automotive SoC at 30 FPS; (c) a retail shelf-audit product where new SKU classes are added weekly; (d) a research baseline where you need instance masks and the highest possible COCO AP.

---

## 4.10 Segmentation: Semantic vs Instance vs Panoptic

### The three tasks

Detection drew boxes; segmentation asks for the pixels. But "label the pixels" turns out to be three different requests, and the reason there are three is worth deriving rather than memorising.

Start with the obvious version: give every pixel a class. That is **semantic** segmentation, and it is a clean, fixed-size output — an $H\times W$ map, exactly the shape a CNN likes. Now ask it to count the people in a crowd and watch it fail: three touching people are one connected "person" region, and no amount of accuracy in the class map recovers the fact that there were three. The identity of *which* object a pixel belongs to was never part of the output.

So add it: label each pixel with an object identity too. That is **instance** segmentation — and notice it has quietly reintroduced 4.1's problem, because the number of identities is unknown and unordered, so instance segmentation is set prediction again, which is why Mask R-CNN is a detector with an extra head rather than a segmenter with an extra loss.

Now the third request. Instance segmentation only makes sense for things you can count, so it simply has nothing to say about sky, road or grass — those pixels come back unlabelled. Semantic segmentation labels them but can't count. **Panoptic** is the demand that one output do both, subject to the constraint that every pixel gets exactly one (class, instance) label.

**Pause:** why is that non-overlap constraint the hard part, rather than an obvious tidiness requirement?

Because it is what makes the task un-decomposable. Instance masks routinely overlap each other and disagree with the semantic map, so "run both and merge" needs an arbitrary tie-breaker at every contested pixel — see mastery check (b). The constraint is also what makes a single well-defined metric possible (PQ, 4.12), since every predicted segment can be matched to at most one ground-truth segment.

| Task | Question | Output | "Two people overlapping" |
|---|---|---|---|
| **Semantic** | what class is each pixel? | $H\times W$ class map | one "person" region — **cannot separate them** |
| **Instance** | which object does each pixel belong to? | a set of (mask, class, score); **only "things"** | two separate masks |
| **Panoptic** | both, for every pixel | $H\times W$ map of (class, instance id); every pixel labelled exactly once | two person instances + the labelled background |

**"Things" vs "stuff"** — the distinction that organises the field. *Things* are countable objects with instances (person, car, dog). *Stuff* is amorphous regions without instances (sky, road, grass). Semantic segmentation handles both but can't count; instance segmentation counts but ignores stuff; **panoptic unifies them** with the constraint that every pixel gets exactly one (class, instance) label — no overlaps, no gaps.

**Related task worth naming:** *referring expression segmentation* ("segment the man in the red hat") and *open-vocabulary segmentation* — where the class set is text, not a fixed list. This is the bridge to SAM and Grounding DINO in Module 6.

**In your own words:** what can panoptic segmentation express that semantic and instance segmentation cannot express even in combination?

### 🎯 Top-1% distinction

- **Use the things/stuff vocabulary** — it immediately signals familiarity with the literature.
- **State panoptic's non-overlap constraint** — it's what makes PQ (4.12) well-defined and what forces a different architecture than "run semantic and instance and merge."
- **Note that the three tasks converged.** Mask2Former (2022) showed a *single* architecture — mask classification with a transformer decoder — handles all three at SOTA. The framing shift: instead of "predict a label per pixel," predict a **set of binary masks each with a class label**. Semantic segmentation becomes mask classification where masks happen not to be instance-separated. Being able to state that unification is a strong currency signal.

### ✅ Mastery check

(a) A photo shows three overlapping people standing in front of a building, on grass. State exactly what semantic, instance, and panoptic segmentation each output for this image — including how many regions and what the "grass" gets.
(b) Why can't you produce a panoptic result by simply running a semantic model and an instance model and merging their outputs?
(c) Mask2Former handles all three tasks with one architecture. What reframing made that possible?

<details><summary>Answer sketch</summary>
(a) <b>Semantic:</b> one $H\times W$ class map. The three people become <b>one connected "person" region</b> — they cannot be separated, and counting is impossible. Building is one region, grass is one region, sky one. <b>Instance:</b> a set of three "person" masks with scores, plus nothing at all for building, grass or sky — instance segmentation covers only <b>things</b>, not <b>stuff</b>, so most of the image is unlabelled. <b>Panoptic:</b> every pixel gets exactly one (class, instance-id) pair — three distinct person instances, plus building, grass and sky each as a single stuff segment with no instance id. Grass is the clarifying case: semantic labels it, instance ignores it, panoptic labels it as stuff.
(b) Because the two outputs <b>conflict and the merge is ill-posed</b>. Instance masks overlap each other (two people's masks can claim the same pixel where they occlude) and they overlap the semantic map's boundaries; the semantic model may call a pixel "person" where no instance mask covers it, and vice versa. Panoptic's defining constraint is that <b>every pixel receives exactly one label — no overlaps, no gaps</b> — so you would need an arbitrary priority heuristic to resolve every conflict, and that heuristic is not learned, not principled, and different choices give materially different PQ. This is exactly why panoptic was posed as its own task with its own metric rather than as a post-processing step.
(c) The reframing from <b>"predict a label per pixel"</b> to <b>"predict a set of binary masks, each with a class label."</b> Under that formulation the three tasks stop being different problems and become different <i>post-processings</i> of the same output: semantic segmentation is mask classification where you merge all masks of the same class; instance segmentation keeps thing-masks separate; panoptic keeps thing-masks separate and merges stuff-masks, with a non-overlap resolution. One architecture — a transformer decoder producing $N$ mask embeddings, exactly the set-prediction idea from DETR (4.8) — therefore serves all three. It is the same conceptual move as DETR's: replace a dense per-location prediction with a set prediction, and the task's awkward structure dissolves.
</details>

---

## 4.11 FCN, U-Net, and Mask R-CNN

### FCN (Long, Shelhamer & Darrell, CVPR 2015) — the founding idea

The question in 2015 was: **you have a pretrained ImageNet classifier and you want a per-pixel class map. How much of the classifier can you keep?**

The conv stack is fine — it already produces a spatial map, just a coarse one. The problem is the fully-connected head, which flattens that map and destroys spatial structure, and which demands a fixed input size. Deleting it and training a new head from scratch throws away the pretrained weights that made the whole thing work. FCN's answer is that you do not have to choose, because an FC layer applied to a fixed-size feature map is *already* a convolution — one whose kernel is the size of the whole map. Reinterpret it as such and the pretrained weights carry over unchanged.

Take a classification CNN and **replace the fully-connected layers with $1\times1$ convolutions**. The network becomes fully convolutional, accepts any input size, and outputs a coarse class map. Then upsample back to input resolution.

Two contributions:
1. **FC → conv reinterpretation.** VGG's `fc6` (7×7×512 → 4096) becomes a $7\times7$ conv with 4096 outputs. The pretrained weights transfer directly. This is the same insight as 4.2's fully-convolutional sliding window.
2. **Skip connections for detail.** The stride-32 output is far too coarse (FCN-32s). FCN-16s adds a prediction from `pool4` (stride 16); FCN-8s adds `pool3`. Each fusion recovers finer boundaries. **This is the ancestor of both U-Net's skips and FPN's lateral connections.**

### U-Net (Ronneberger et al., MICCAI 2015) — the workhorse

A symmetric encoder–decoder with **concatenative skip connections** at every resolution.

```
     ┌──────────────── concat ────────────────┐
 conv,conv ──pool──> conv,conv ──pool──> ... ──> bottleneck
     ^                                              │
     │                                          upconv
 conv,conv <──concat── conv,conv <──concat── ... <──┘
```

**Pause:** U-Net and FPN both merge a coarse semantic feature map with a fine high-resolution one. U-Net concatenates; FPN adds. Before reading on — which constraint, present in one and absent in the other, forces the difference?

**Why concatenate rather than add (contrast with FPN, 4.4).** The decoder's job is to *reconstruct precise boundaries*, and the encoder's high-resolution features hold exactly the localisation information the bottleneck destroyed. Concatenation hands the decoder both signals intact and lets it learn the combination; addition would force a fixed 1:1 blend. FPN adds because it must keep channel counts identical across levels for a shared head; U-Net has no such constraint, so it can afford the channels.

State the asymmetry the other way round and it is easier to keep straight: **addition is a lossy merge that preserves the channel budget; concatenation is a lossless merge that spends it.** FPN's shared detection head (4.4) means every level must present the same 256 channels, so the budget is fixed and addition is the only option — and this is affordable because the top-down path is carrying smooth semantics, not fine detail. U-Net has a separate decoder block at every resolution, no shared head, and a job that lives or dies on fine detail, so it spends the channels. Neither is "better"; each follows from what the architecture next to it requires.

U-Net's other design notes: originally trained on ~30 images with heavy elastic deformation augmentation (biomedical data is scarce), used a **weighted loss** that up-weights pixels on the thin boundaries *between* touching cells — a beautifully task-specific trick, and still the standard answer for "how do I separate touching instances with a semantic model."

U-Net remains the default for medical imaging, satellite segmentation, and — importantly — it is **the backbone of diffusion models** (6.13), where the encoder–decoder-with-skips shape is used for denoising rather than segmentation.

### Mask R-CNN (He et al., ICCV 2017) — instance segmentation

Faster R-CNN + a third branch: a small FCN predicting a $28\times28$ binary mask per RoI, **per class** (K masks, select the one for the predicted class).

Two design decisions that matter:

1. **Decouple mask and class.** Predict $K$ binary masks with a **per-pixel sigmoid + binary cross-entropy**, not a per-pixel softmax over classes. Using softmax would make masks compete across classes, coupling segmentation to classification; with sigmoid, the mask branch only has to answer "is this pixel part of *the* object," and the classification branch independently says which class. The paper shows this is worth several mask AP points.
2. **RoI Align** (4.3) — without it, the mask branch is misaligned by up to half a stride and mask AP drops ~10% relative.

The mask branch adds ~20% compute for full instance segmentation — the "nearly free extra head" pattern. And notice *why* it is nearly free, because the reason generalises: the expensive part of the model is the shared backbone and FPN, which every head reads from, so a new task costs only its own head. That is the same "share the computation" logic that drove the whole R-CNN lineage in 4.3, applied to tasks rather than to proposals.

**In your own words:** what does a skip connection give a segmentation decoder that a deeper decoder could not?

### The rest of the landscape you should be able to name

| Model | Idea |
|---|---|
| **DeepLab v1–v3+** | **atrous/dilated convolution** to enlarge receptive field without losing resolution (2.3), **ASPP** (atrous spatial pyramid pooling: parallel dilations at multiple rates), and a CRF post-process in early versions |
| **PSPNet** | pyramid pooling module — global context at several scales |
| **SegFormer** (2021) | hierarchical transformer encoder + a very lightweight all-MLP decoder; efficient and strong |
| **Mask2Former** (2022) | **unified** semantic/instance/panoptic via mask classification + masked attention in a transformer decoder; SOTA across all three with one architecture |
| **SAM 1→3** (6.8) | promptable segmentation; SAM 3 (2025) adds **text-prompted concept segmentation** |

### 🎯 Top-1% distinction

1. **Explain the concatenate-vs-add difference between U-Net and FPN** with the reason for each. This is a favourite comparative question.
2. **Explain Mask R-CNN's sigmoid-not-softmax mask decision** and the decoupling argument.
3. **Trace the skip-connection lineage**: FCN's skips → U-Net's concatenative skips → FPN's lateral connections → diffusion U-Nets. One idea, four uses.
4. **Know dilated convolution's role** and its gridding artefact (2.3).
5. **The Mask2Former unification** — mask classification subsumes all three segmentation tasks.
6. **U-Net's boundary-weighted loss for touching instances** — a concrete, practical trick that shows applied experience.

### ✅ Mastery check

(a) You must segment overlapping cells in microscopy. Semantic U-Net or Mask R-CNN? Discuss both, then give a third option.
(b) Why does Mask R-CNN predict $K$ masks instead of one mask with $K+1$ channels?
(c) Your segmentation model has good IoU but ragged, jagged boundaries. Give three causes and fixes.

<details><summary>Answer sketch</summary>
(a) <b>U-Net</b> is semantic, so touching cells merge into one blob and you cannot count them — unless you use the classic workaround: predict <i>three</i> classes (background / cell interior / cell boundary) with a <b>boundary-weighted loss</b>, then run connected components or a watershed seeded by the interiors. That is cheap, trains on very little data, and is still the standard in microscopy. <b>Mask R-CNN</b> gives true instances directly and handles overlap, but needs box-level annotations, more data, and struggles when cells are densely packed (NMS suppresses genuinely overlapping cells — 4.7's crowded-scene failure). <b>Third option, and usually the best answer:</b> a <b>distance-transform / watershed-style</b> regression approach (StarDist, Cellpose) that predicts a per-pixel vector field or star-convex polygon toward each cell's centre and separates instances geometrically. These dominate biomedical benchmarks precisely because they sidestep both the merging problem and NMS. Naming Cellpose/StarDist here is strong domain-aware signal.
(b) Because $K+1$ softmax channels make the classes <b>compete per pixel</b>: the mask branch would have to simultaneously decide "is this foreground" and "which class," duplicating the classification branch's job and coupling two error sources. With $K$ independent sigmoid masks, the mask branch solves only the (easier, class-agnostic in effect) figure-ground problem, and the dedicated classification branch picks which mask to read out. He et al. measured the difference and it is several mask AP points. It also means a classification error degrades gracefully — you get a correct mask with a wrong label, rather than a corrupted mask.
(c) 1. <b>Resolution loss in the decoder</b> — predicting at stride 8 or 16 and bilinearly upsampling to full resolution guarantees blocky boundaries. Fix: deeper decoder with more skip levels, dilated convolutions to keep resolution (DeepLab), or predict at higher resolution. 2. <b>The loss doesn't weight boundaries</b> — cross-entropy and Dice are dominated by interior pixels, which are the easy majority, so the model has little incentive to get the thin boundary right. Fix: boundary-weighted CE (U-Net's trick), a boundary/Hausdorff loss term, or evaluate with <b>Boundary IoU</b> so the metric actually sees the problem. 3. <b>Label noise at boundaries</b> — human annotations are least reliable exactly at edges, so the model is learning from noisy targets there. Fix: check annotation quality, consider label smoothing near boundaries, or a soft-boundary target. (Bonus fourth: a CRF or guided-filter post-process snapped to image edges — the classical fix, still effective, and a nice Module 1 callback.)
</details>

### 🔨 Build + read

**Build:** Implement U-Net from scratch and train it on a small segmentation dataset. Then run the ablation that teaches the lesson: remove the skip connections and compare both mIoU *and* Boundary IoU — the mIoU drop will be modest and the Boundary IoU drop dramatic, which tells you exactly what the skips are for. Then fine-tune a pretrained Mask R-CNN on a custom instance dataset and compare mask quality with RoI Pool vs RoI Align.

**Read:** Long et al., "Fully Convolutional Networks" (CVPR 2015). Ronneberger et al., "U-Net" (MICCAI 2015) — 8 pages, read it all. He et al., "Mask R-CNN" §3. Cheng et al., "Masked-attention Mask Transformer" (Mask2Former, CVPR 2022) §3 for the unification argument.

---

## 4.12 Segmentation Metrics: IoU, Dice, PQ

### The two core metrics

A segmentation prediction is a set of pixels and so is the ground truth, so scoring one against the other is set comparison — the same problem 4.7 solved for boxes, minus the rectangles. Two conventions dominate, and the useful thing to know is that they are the same measurement in different units.

$$
\text{IoU (Jaccard)} = \frac{|A\cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}
$$

$$
\text{Dice (F1)} = \frac{2|A\cap B|}{|A| + |B|} = \frac{2TP}{2TP + FP + FN}
$$

The right-hand forms are worth reading carefully, since they are what you implement. Every pixel falls into one of three buckets: in both sets (TP), predicted only (FP), or ground truth only (FN). The union is all three, so IoU $= TP/(TP+FP+FN)$. Dice differs in exactly one place — it counts TP twice, in the numerator and again in the denominator — which is what makes it the harmonic mean of precision and recall, i.e. the F1 score.

**Relationship:** $\text{Dice} = \dfrac{2\,\text{IoU}}{1 + \text{IoU}}$, and $\text{IoU} = \dfrac{\text{Dice}}{2 - \text{Dice}}$. Derive the first in one line to convince yourself they carry no independent information: write $I = |A\cap B|$ and $U = |A\cup B|$, note $|A| + |B| = U + I$ (adding the two sets double-counts exactly the intersection), and substitute into Dice to get $2I/(U+I)$; divide top and bottom by $U$ and you have $2\,\text{IoU}/(1+\text{IoU})$. They are monotonically related, so they *rank* models identically — but Dice is always ≥ IoU and is more forgiving. Dice is the standard in medical imaging (it is the F1 score of the pixel classification); IoU is standard in general vision.

**Pause:** if the two metrics rank every pair of models identically, is there any situation where the choice between them actually matters?

Yes, and it is a real one: **thresholds and contracts are written in absolute numbers, not ranks.** "Dice above 0.9" and "IoU above 0.9" are very different requirements — IoU 0.9 corresponds to Dice 0.947, while Dice 0.9 is only IoU 0.818. A regulator, a clinical acceptance criterion or a paper's claimed number is meaningless unless the metric is named. The choice also matters as a *loss*, where the two have different gradients even though they induce the same ordering.

**mIoU** = IoU averaged over classes, which is what makes it a fair metric under class imbalance — a rare class counts as much as a common one. Contrast with **pixel accuracy**, which is nearly useless: a model predicting "road" everywhere on a driving dataset scores well.

### Dice as a loss

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

(using soft probabilities $p_i$, so it's differentiable). **Why use it instead of cross-entropy:** Dice is a *region-overlap* measure, so it is intrinsically robust to class imbalance — segmenting a tumour occupying 0.5% of the pixels, cross-entropy is minimised by predicting "background" everywhere, while Dice goes to 0 (worst possible). **Standard practice is a combination**, e.g. $\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{\text{Dice}}$, because CE gives well-conditioned per-pixel gradients and Dice gives the imbalance-robust region objective. Related: **Tversky loss** generalises Dice with separate FP/FN weights, letting you explicitly trade precision against recall — valuable in medical settings where a missed lesion costs far more than a false alarm.

**Dice's weakness:** it's unstable for very small objects (a few pixels wrong changes it a lot) and undefined when both prediction and ground truth are empty — hence the $\epsilon$, and hence the practice of reporting per-image Dice *and* the aggregate.

### Panoptic Quality

$$
\text{PQ} = \underbrace{\frac{\sum_{(p,g)\in TP}\text{IoU}(p,g)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{\text{Recognition Quality (RQ)}}
$$

Matching uses a **threshold of IoU > 0.5**, which — because segments in a panoptic map cannot overlap — guarantees a *unique* match. (Convince yourself: if two predicted segments each had IoU > 0.5 with the same ground-truth segment, they would each cover more than half of it, so they would have to share pixels — which panoptic's non-overlap constraint forbids. The 0.5 is not a tuning choice; it is the largest threshold-free guarantee available, and it is why panoptic needed its non-overlap rule before it could have a metric.)

RQ is exactly the F1 score of segment detection; SQ is the average IoU of the matched segments. The factorisation is the point: **PQ tells you separately whether you found the right things (RQ) and whether you outlined them well (SQ).** Two models with the same PQ can be very different products — which is precisely what mastery check (b) asks you to exploit.

**In your own words:** state what SQ and RQ each measure, without using the words "quality" or "IoU".

### Metrics that catch what mIoU misses

- **Boundary IoU** (Cheng et al. 2021): IoU computed only within a band of width $d$ around the boundary. mIoU is dominated by interior pixels, so a model with sloppy edges can score well; Boundary IoU exposes it. **Use it whenever boundary quality matters (medical, matting, robotics grasping).**
- **Hausdorff distance / 95% HD**: worst-case boundary deviation. Standard alongside Dice in medical segmentation challenges, because a single large excursion matters clinically even if the mean overlap is fine.
- **Per-image vs aggregate**: aggregating TP/FP/FN over the whole dataset before computing Dice hides per-case failures. Medical challenges report per-case Dice distributions, not just the mean.

### 🎯 Top-1% distinction

1. **Know the Dice↔IoU relation** and that they rank identically — so "which metric?" is about convention and interpretability, not ranking.
2. **Explain why Dice-as-a-loss handles imbalance** and why you still combine with CE.
3. **Know PQ's SQ×RQ factorisation** and what each half tells you.
4. **Name Boundary IoU** and why mIoU is blind to boundary quality.
5. **Name Tversky loss** for asymmetric FP/FN costs — the applied-medical answer.
6. **Per-case reporting** — the difference between a research number and a deployable one.

### ✅ Mastery check

(a) A model has IoU 0.6. What is its Dice?
(b) Two models both score mIoU 0.78. One has PQ 0.62 (SQ 0.85, RQ 0.73), the other PQ 0.62 (SQ 0.72, RQ 0.86). What is qualitatively different about them, and which would you ship for an autonomous forklift?
(c) Your tumour segmentation has Dice 0.91 but the surgeon says it's unusable. What metric were you not looking at?

<details><summary>Answer sketch</summary>
(a) $\text{Dice} = 2(0.6)/(1+0.6) = 1.2/1.6 = \mathbf{0.75}$.
(b) Model A (SQ 0.85, RQ 0.73): it <b>outlines well what it finds, but misses or hallucinates more segments</b>. Model B (SQ 0.72, RQ 0.86): it <b>finds nearly everything but outlines sloppily</b>. For an autonomous forklift, <b>missing an object is a safety failure and a slightly loose outline is not</b> — so recognition quality dominates and you ship <b>model B</b>, then spend engineering effort on boundary refinement (higher-resolution decoder, boundary loss). The general principle: decompose the metric until it maps onto the failure modes your application actually cares about.
(c) <b>Boundary accuracy</b> — specifically <b>Hausdorff distance (95% HD)</b> and/or <b>Boundary IoU</b>. Dice 0.91 on a compact tumour is dominated by the interior; the model can have a large excursion at one edge (spilling into an adjacent critical structure, or under-covering an infiltrating margin) while Dice barely moves. Surgical planning depends on the <i>margin</i>, which is exactly the boundary. Also check the <b>per-case distribution</b> rather than the mean — a mean Dice of 0.91 is compatible with several cases at 0.4, and those are the ones the surgeon saw.
</details>

### 🔨 Build + read

**Build:** Implement IoU, Dice, Tversky, PQ, and Boundary IoU from scratch. Train one segmentation model with CE only, one with Dice only, and one with CE+Dice on an imbalanced dataset, and report all five metrics for each — the disagreements between metrics are the whole lesson.

**Read:** Kirillov et al., "Panoptic Segmentation" (CVPR 2019) §4 for the PQ derivation. Cheng et al., "Boundary IoU" (CVPR 2021). Maier-Hein et al., "Metrics Reloaded" (Nature Methods 2024) — the definitive guide to choosing segmentation metrics; skim the decision trees.

---

## 4.13 Practical: Run and Fine-Tune a Pretrained Detector / Segmenter

### The workflow

Everything above told you how detectors work. This section answers a different question: **given a labelled dataset and a deadline, what is the order of operations that gets you a good model fastest?** The order is not arbitrary — each step exists to stop you wasting effort on a later one, so read the numbering as a dependency chain rather than a checklist.

The through-line is this: **almost every hour lost in applied detection is lost to changing the model when the problem was in the data, the split, or the metric.** Steps 1, 2 and 5 exist to find that out before you touch an architecture.

**1. Data**
- Annotation format: COCO JSON is the lingua franca. Convert everything to it.
- **Split by the right unit** — by video, by patient, by site, by capture session. Random image-level splits leak catastrophically when consecutive frames are near-duplicates.
- Audit the annotations: class distribution, box-size distribution (**plot it** — it tells you whether AP_small will dominate and whether default anchors fit), and instances per image.

**2. Baseline before anything else**
- Run a pretrained COCO model zero-shot on your data and look at the output. If your classes overlap COCO's, this is your floor.
- Then fine-tune the smallest reasonable model on a small subset. Get an end-to-end number in the first hour.

**3. Fine-tuning recipe**
- Freeze the first backbone stage; **freeze BN statistics** in the backbone (2.4, 2.10) — detection batches are small and this is the classic silent accuracy killer.
- Layer-wise LR decay; head LR ~10× backbone LR; warmup then cosine.
- Augmentation: horizontal flip, multi-scale training (randomly resize the short side in a range — one of the highest-value detection augmentations), **Mosaic / copy-paste** for small-object-heavy datasets. Check flip validity for your domain (2.9).
- Keep the loss weights from the reference implementation unless you have a reason; they're tuned together.

**4. Evaluation**
- COCO AP, **plus AP_small/medium/large**, plus per-class AP. The aggregate number hides everything.
- **Precision/recall at your deployed operating threshold**, not just AP.
- A confusion matrix built at the operating threshold (including a "background" row/column for missed and hallucinated detections) — this is where you discover that two of your classes are systematically confused.

**5. Error analysis — use TIDE**

**Pause:** your fine-tuned detector scores AP 0.41 against the pretrained baseline's 0.33. What is the next thing you should do — try a bigger backbone, tune the learning rate, or something else entirely?

Something else: find out *which kind of mistake* the remaining 0.59 consists of. AP is a single number summarising at least six distinct failure modes, and the fix for each is different — so acting on the aggregate is guessing. This is the applied version of 4.12's point about decomposing PQ into SQ and RQ, and 4.7's about reading AP_small separately from AP.

The **TIDE** toolkit (Bolya et al., ECCV 2020) does that decomposition, splitting AP loss into six error types: **classification, localisation, both, duplicate, background (FP on background), missed (FN)**. It works by asking, for each error type in turn, "how much would AP rise if I corrected *only* this?" — so the six numbers are attributions in AP units, directly comparable, and directly actionable. It tells you *which* error costs you the most AP, which converts "improve the model" into a specific action:
- Localisation-dominated → higher resolution, better box loss (CIoU), Cascade R-CNN.
- Classification-dominated → better backbone, more data for confused classes.
- Missed-dominated → check label assignment and anchors, check AP_small.
- Duplicate-dominated → NMS threshold or a dual-assignment/end-to-end model.

**Running TIDE on your model before touching the architecture is the single highest-leverage habit in applied detection** and it's a great thing to say in an interview.

**6. Deployment**
- Export to ONNX/TensorRT; **verify numerical parity** on a fixed batch.
- **NMS is often not exported** — decide whether it runs in the graph (TensorRT `EfficientNMS` plugin) or in your Python post-process, and measure it; at high detection counts it can be a large fraction of latency.
- Measure end-to-end latency including pre-processing (letterboxing, normalisation) and post-processing — these are routinely 30–50% of a detector's wall-clock time and are invisible in FLOP counts.
- int8 quantisation with a calibration set; re-measure AP, not just latency.

Notice how many of these six steps are traps you can now *predict* rather than discover: BN statistics in step 3 because detection batches are small (2.4), the AP breakdown in step 4 because a small-object problem is invisible in the aggregate (4.4, 4.7), the NMS export question in step 6 because NMS is a sequential post-process rather than part of the graph (4.7). The theory earns its keep here by telling you where to look before the bug costs you a week.

**In your own words:** why does running TIDE come before changing the architecture?

### Capstone deliverable

Fine-tune a detector on a custom dataset and produce a one-page report: baseline vs final AP with the full breakdown, a TIDE error decomposition before and after your main change, a PR curve with the chosen operating point marked and justified, five qualitative failure images with diagnoses, and measured latency (fp16 and int8) on your target hardware with the pre/post-processing cost broken out. **This is a portfolio artefact that will do more for you in interviews than three tutorial notebooks.**

---

## Going Deeper — Papers, Sources and Research Scope

*One honest framing note before the list. Of the papers highlighted at CVPR 2026, detection/segmentation/tracking fell from 3.8% to 1.2%. That is not because the problems are unimportant — it is because they are largely **solved and deployed**, which is exactly why they still dominate interviews and production systems. Read this module for depth and employability. Do not look for a research gap here; §D is deliberately short on genuinely open problems and long on measurement questions, which is the honest picture.*

### A. The canonical papers

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Dalal & Triggs, *Histograms of Oriented Gradients* | 2005, CVPR | The pre-deep baseline, and the ablation methodology is excellent. Read §6. | CVPR 2005 |
| Viola & Jones, *Rapid Object Detection using a Boosted Cascade* | 2001, CVPR | The cascade idea — most windows rejected in two features. Still how you should think about staged inference. | CVPR 2001 |
| Girshick et al., *R-CNN* | 2014, CVPR (1311.2524) | Detection as classification-of-proposals. Read to feel the 47-seconds-per-image problem the next two papers solve. | arXiv 1311.2524 |
| Girshick, *Fast R-CNN* | 2015, ICCV (1504.08083) | RoI pooling — share the convolution, crop the features. | arXiv 1504.08083 |
| **Ren et al., *Faster R-CNN*** | 2015, NeurIPS (1506.01497) | The RPN. Anchors as a discretisation of the box space, and the four-step alternating training that everyone later replaced. | arXiv 1506.01497 |
| **Lin et al., *Feature Pyramid Networks*** 🔴 | 2017, CVPR (1612.03144) | Fig. 1's four architectures are the paper — each is an attempt, and FPN is what survives. | arXiv 1612.03144 |
| **Lin et al., *Focal Loss / RetinaNet*** 🔴 | 2017, ICCV (1708.02002) | Class imbalance as *the* one-stage problem, and a loss designed against a stated spec. §3.3's prior-initialisation trick is the part people miss. | arXiv 1708.02002 |
| Redmon et al., *YOLO* | 2016, CVPR (1506.02640) | Detection as direct regression. Read v1 only; later versions are engineering, well documented elsewhere. | arXiv 1506.02640 |
| Tian et al., *FCOS* | 2019, ICCV (1904.01355) | Anchor-free, with the centre-ness branch built term by term. | arXiv 1904.01355 |
| Zhang et al., *ATSS* | 2020, CVPR (1912.02424) | Shows the anchor-based/anchor-free gap is **entirely** the label-assignment rule. One of the most clarifying ablations in detection. | arXiv 1912.02424 |
| **Carion et al., *DETR*** | 2020, ECCV (2005.12872) | Set prediction, object queries, Hungarian matching. NMS deleted. | arXiv 2005.12872 |
| **Zhu et al., *Deformable DETR*** | 2021, ICLR (2010.04159) | Why plain DETR needed 500 epochs and was bad at small objects, and what fixed both. Read directly after DETR. | arXiv 2010.04159 |
| Zhang et al., *DINO* (DETR with Improved deNoising anchOr boxes) | 2023, ICLR (2203.03605) | The DETR line becoming genuinely SOTA. Note: unrelated to DINO the SSL method — a real naming collision. | arXiv 2203.03605 |
| Zhao et al., *RT-DETR* | 2024, CVPR (2304.08069) | Real-time DETR — the first to beat YOLO on the speed/accuracy frontier without NMS. | arXiv 2304.08069 |
| **Bolya et al., *TIDE*** | 2020, ECCV (2003.12237) | **The essential critique.** mAP is one number hiding six distinct error types. This paper is the intellectual core of capstone project #1. | arXiv 2003.12237 |
| Ronneberger et al., *U-Net* | 2015, MICCAI (1505.04597) | Skip connections that carry *resolution* rather than gradient. Contrast with FPN deliberately. | arXiv 1505.04597 |
| **He et al., *Mask R-CNN*** | 2017, ICCV (1703.06870) | RoIAlign — and the demonstration that the quantisation in RoIPool was costing real accuracy. The decoupling of mask and class is the other key idea. | arXiv 1703.06870 |
| Kirillov et al., *Panoptic Segmentation* | 2019, CVPR (1801.00868) | Defines the task *and* the PQ metric. Read §4 on why PQ needed inventing. | arXiv 1801.00868 |
| Cheng et al., *Mask2Former* | 2022, CVPR (2112.01527) | One architecture for semantic, instance and panoptic — masked attention as the unifying mechanism. | arXiv 2112.01527 |

### B. The single best source, per hard topic

- **The two-stage lineage as one story (4.3).** Justin Johnson, UMich EECS 498/598, **Lecture 15 (Object Detection)** — R-CNN through Faster R-CNN in one hour, with the timing arithmetic that motivates each step.
- **Segmentation, same course, Lecture 16 (Detection and Segmentation).**
- **FPN (4.4).** The paper's Fig. 1 and §3, then Detectron2's implementation (below). The level-assignment formula's derivation is in §4.2 — one paragraph, easy to skim past, worth ten minutes.
- **Focal loss (4.6).** The paper's §3, then Lilian Weng's *Object Detection Part 4: Fast Detection Models* for the surrounding context.
- **NMS, IoU and the box-regression loss family (4.7).** The GIoU paper (1902.09630) §3 explains what plain IoU loss cannot do (zero gradient for disjoint boxes) more clearly than any blog.
- **mAP, computed by hand (4.7).** The COCO evaluation code itself — `cocoapi/PythonAPI/pycocotools/cocoeval.py`, function `accumulate`. Reading it once removes all mAP mystery permanently.
- **DETR and Hungarian matching (4.8).** *The Annotated DETR* walkthroughs, then `models/matcher.py` directly. The matching is ~40 lines and runs under `no_grad`.
- **Panoptic quality (4.12).** The panoptic paper §4. PQ = SQ × RQ, and the IoU > 0.5 rule that makes matching unique — that uniqueness proof is a good interview question.

### C. Reference implementations worth reading

- **`facebookresearch/detectron2` → `modeling/backbone/fpn.py`.** The lateral 1×1s, the top-down `interpolate`, and the 3×3 anti-aliasing convs. Then `modeling/poolers.py` → `assign_boxes_to_levels`, which is the level-assignment formula in six lines.
- **`facebookresearch/detr` → `models/matcher.py`.** `HungarianMatcher.forward`: the cost matrix built from three terms, then `scipy.optimize.linear_sum_assignment`. Note the `@torch.no_grad()`.
- **`facebookresearch/detectron2` → `layers/roi_align.py` and its CUDA kernel.** Look for the bilinear sampling at four points per bin — that is precisely what RoIPool's rounding destroyed.
- **`ultralytics/ultralytics` → `utils/tal.py`.** Task-aligned assignment, the modern label-assignment rule. Reading it beside ATSS shows how much of "YOLO progress" is assignment progress.
- **`cocodataset/cocoapi` → `cocoeval.py`.** Read `evaluateImg` and `accumulate` together. The 101-point interpolation and the maxDets logic are here and nowhere else clearly.
- **`dbolya/tide`.** The reference implementation of the error taxonomy. If you build capstone #1, start by reading this and deciding what it's missing.

### D. Open questions — mostly measurement, honestly

1. **Is mAP's ranking of detectors stable under TIDE decomposition?** *Why open:* two detectors at equal mAP can have completely different error profiles, and no one has systematically checked whether the mAP ordering survives when you weight error types by application cost. *Minimum experiment:* run TIDE across 6–8 released COCO checkpoints, cluster by error profile, ask whether mAP rank correlates with any application-weighted rank. Inference-only. **Very feasible, and it is literally the first experiment of capstone #1.**
2. **What is NMS actually costing at the tail?** *Why open:* NMS's failure in crowds is universally acknowledged and rarely quantified per-scene-density. *Minimum experiment:* stratify CrowdHuman by density, plot the NMS-threshold-optimal choice per stratum, and measure the loss from being forced to pick one global threshold. **Very feasible.**
3. **Do end-to-end (NMS-free) detectors actually win once latency is measured on the device, not in FLOPs?** *Why open:* RT-DETR claims the frontier, but comparisons are usually on an A100. *Minimum experiment:* fixed hardware (a Jetson or even a CPU), measure wall-clock including pre/post-processing, plot the real Pareto front. **Feasible and it doubles as capstone #4.**
4. **Is the detection literature's comparison fair?** *Why open:* the training-recipe confound that Bello et al. exposed for classification has never been cleanly run for detection. Longer schedules, better augmentation and stronger backbones move detectors a lot. *Minimum experiment:* two or three architectures under one identical, modern recipe on a small dataset. **Moderate — requires real training budget, but a partial result is still interesting.**
5. **Does segmentation quality degrade gracefully under quantisation, or catastrophically at boundaries?** *Why open:* quantisation is evaluated with mIoU, which averages over pixels and hides boundary-specific damage. *Minimum experiment:* boundary-IoU before and after INT8, per class. **Very feasible.**
6. ~~A new detection architecture.~~ **Trap.** You will not beat DINO-DETR, and the attempt teaches you less than experiment #1 above.

---

## Module 4 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Problem framing | detection = **set prediction with unknown cardinality**; three sub-problems: where to look, label assignment, duplicate removal |
| Box encoding | $t_x = (x-x_a)/w_a$, $t_w = \log(w/w_a)$ — normalise for scale invariance, log for positivity and scale symmetry |
| Classical | Viola-Jones **cascade** = early rejection (ancestor of two-stage); HOG ≡ dense SIFT with L2-Hys; DPM = parts + deformation cost |
| R-CNN lineage | a story about **sharing computation**: 2000 CNN passes → 1 pass + RoI pooling → learned RPN proposals |
| RoI Align | RoI Pooling quantises twice (≈ half-stride, ~8 px); bilinear sampling fixes it — **+10% mask AP, +1% box AP** |
| Label assignment (classic) | IoU ≥ 0.7 positive, ≤ 0.3 negative, **ignore in between**, plus "highest-IoU anchor is always positive" |
| Smooth L1 | L2 near 0 (smooth convergence), L1 far (bounded gradient, outlier-robust) |
| **FPN** | **resolution from bottom-up, semantics from top-down**, lateral $1\times1$ merges them; level $k=\lfloor 4+\log_2(\sqrt{wh}/224)\rfloor$; a learned **inverted Laplacian pyramid**; nearly free |
| SSD-style pyramid fails | shallow levels have high resolution but **no semantics** |
| One-stage | dense prediction everywhere; YOLOv2's sigmoid centre constraint; anchors from **IoU-based k-means** |
| Anchor-free | FCOS $(l,t,r,b)$ + **centre-ness** $\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}}$ |
| **Label assignment is the modern battleground** | **ATSS**: anchor-based ≈ anchor-free once assignment is controlled. SimOTA, TaskAligned |
| **Focal Loss** | $-\alpha_t(1-p_t)^\gamma\log p_t$; 100k×0.01 ≫ 10×0.69; $\gamma=2$ ⇒ 100× down-weight at $p_t{=}0.9$; **soft OHEM**; $\alpha{=}0.25$ because $\gamma$ over-corrects |
| Focal's hidden requirement | **prior bias init** $b=-\log((1-\pi)/\pi)$, $\pi{=}0.01$ — without it RetinaNet diverges |
| Focal's cost | poor **calibration** — under-confident positives |
| IoU losses | IoU → GIoU (gradient without overlap) → DIoU (centre distance) → CIoU (aspect ratio). Optimise what you evaluate |
| NMS | greedy, per class; **crowded scenes have no good threshold** → Soft-NMS (decay, +1 AP) → NMS-free architectures |
| AP | greedy match, **each GT matched once** (duplicates = FP); COCO AP averages IoU 0.5:0.05:0.95; AP is threshold-free, your product isn't |
| **DETR** | $N$ queries + **Hungarian one-to-one matching** ⇒ duplicates trained as $\varnothing$ ⇒ **no NMS**; matching cost uses $\hat p$ not $\log\hat p$; $\varnothing$ down-weighted 10× |
| DETR's problems | 500 epochs (matching instability + attention must learn locality) and bad small objects (single scale) |
| DETR's fixes | **Deformable attention** (K=4 samples/query ⇒ linear cost ⇒ multi-scale affordable ⇒ 10× faster convergence); **query denoising** (DN/DINO-DETR) |
| 2026 detectors | YOLO26 (NMS-free, edge, AGPL), YOLOv12, **RF-DETR** (DINOv2 backbone, first real-time >60 mAP, Apache-2.0), RTMDet (MIT). **Licensing matters** |
| Segmentation types | things vs stuff; panoptic = every pixel gets exactly one (class, instance); **Mask2Former unified all three as mask classification** |
| U-Net vs FPN | U-Net **concatenates** (decoder needs full detail); FPN **adds** (must keep channels equal for a shared head) |
| Mask R-CNN | $K$ **sigmoid** masks, not softmax — decouples mask from class; RoI Align is essential |
| Dice ↔ IoU | $\text{Dice} = 2\,\text{IoU}/(1+\text{IoU})$ — monotonic, so identical ranking; Dice-as-loss is imbalance-robust; use **CE + Dice** |
| PQ | $\text{SQ}\times\text{RQ}$ — average IoU of matches × F1 of segment detection; IoU > 0.5 gives a unique match |
| mIoU's blind spot | boundary quality → use **Boundary IoU** / 95% Hausdorff |
| Practical | run **TIDE** before changing anything; split by video/patient/site; freeze backbone BN; measure pre/post-processing latency |

---

*End of Module 4 notes. Drills in `Module-04-Drills.md`.*
