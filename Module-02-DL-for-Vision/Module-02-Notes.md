# Module 2 — Deep Learning, Specialised for Vision

> **Calibration:** you already know backprop, optimisers, and training loops. This module spends its entire depth budget on **what is specific to vision**: why the convolutional prior exists, what normalisation actually does, how architectures evolved and *why* each step happened, and the practitioner knowledge (efficiency, augmentation, transfer) that interviews for CV/ML roles actually probe.
>
> **Interview weight: very high.** Modules 2, 3, 4 and 6 are where 80% of ML/CV interview questions live. BatchNorm and ResNet in particular are near-guaranteed.

**Concept map**

```
2.1 Why CNNs ──> 2.2 Conv mechanics ──> 2.3 Receptive field & pooling
                        │                        │
                        v                        v
                 2.4 Normalisation 🔴  ────> 2.5 Architecture evolution ──> 2.6 ResNet
                                                      │                        │
                                                      v                        v
                                              2.7 Efficient convs 🟡    2.8 Regularisation
                                                                               │
                                                                               v
                                                        2.9 Augmentation ──> 2.10 Transfer learning ──> 2.11 Pipeline
```

---

## 2.1 Why CNNs Beat MLPs for Images

### Intuition

Ask the question the architecture is an answer to: **what does a network need to be told about images that it would otherwise have to discover from data?**

Here is one such fact. A cat in the top-left of a photo and the same cat in the bottom-right are the same cat. Obvious to you; not to a fully connected layer, which has a different weight for every pixel and therefore has to learn "cat" separately at every possible position, as if the top-left and bottom-right of an image were unrelated measurement devices. A convolution learns "cat detector" **once** and slides it everywhere. That single change — sharing weights across space — is the whole idea, and everything below is either a consequence of it or a limit on it.

Real-life anchor: a defect-detection camera on a production line sees a scratch that can appear anywhere on the part. An MLP would need training examples of the scratch at every position. A CNN needs a handful, anywhere.

### The three inductive biases

1. **Locality** — a pixel's meaning is determined by its neighbourhood, not by pixels 800 px away. So restrict each unit's input to a $k\times k$ window.
2. **Weight sharing / stationarity** — the statistics of natural images are approximately translation-stationary, so the same filter is useful everywhere.
3. **Hierarchical compositionality** — edges compose into textures, textures into parts, parts into objects. Stacking layers with growing receptive fields matches that structure.

### The parameter argument, with numbers

Input $224\times224\times3 = 150{,}528$ values.

- **FC layer** to 1000 hidden units: $150{,}528 \times 1000 \approx \mathbf{1.5\times10^8}$ parameters — *for one layer*. More parameters than ImageNet has training images.
- **Conv layer**, $3\times3$, 3 input channels, 64 output channels: $3\times3\times3\times64 + 64 = \mathbf{1792}$ parameters. Roughly **84,000× fewer**, and it produces a $224\times224\times64$ feature map rather than a 1000-vector.

But parameter count is the *symptom*, not the cause. The cause is **sample complexity**: the MLP's hypothesis space contains an enormous number of functions that are inconsistent with the translation structure of images, and it must use data to rule them all out. The CNN's hypothesis space excludes them by construction. The prior buys you data efficiency.

### Equivariance vs. invariance — get this distinction exactly right

- **Equivariance:** $f(T(x)) = T(f(x))$. Shift the input, the output shifts identically. **Convolution is translation-equivariant.**
- **Invariance:** $f(T(x)) = f(x)$. Shift the input, the output is unchanged. **Pooling (locally) and global average pooling (globally) provide invariance.**

A CNN is a stack of equivariant operations followed by an invariance-inducing readout. You want equivariance in the feature extractor (so spatial information survives — critical for detection and segmentation) and invariance only at the classification head. Saying "CNNs are translation invariant" without this distinction is a common tell.

**Pause:** you flip a cat image left-right and the classifier's output is unchanged. Is that equivariance or invariance, and which one did the convolutions provide?

Neither, from the convolutions. Horizontal flip is not a translation, so convolution gives you nothing for free here at all — the network is flip-robust only because you trained it on flipped images. **Convolution is *not* equivariant to rotation, reflection or scale.** That's why you need augmentation (2.9) — and why classical scale-space (1.4) had to be re-solved inside CNNs via feature pyramids (4.4). The convolutional prior is narrow: it covers translation, and translation only.

**In your own words:** why is a CNN's advantage over an MLP about *data*, not about what the two can represent?

### 🎯 Top-1% distinction

1. **CNNs are not actually shift-invariant in practice.** Strided convolution and max-pooling **downsample without low-pass filtering** — a direct Nyquist violation (1.4!). Zhang (ICML 2019, *"Making Convolutional Networks Shift-Invariant Again"*) showed classifier outputs can change dramatically from a **one-pixel shift**, and fixed it by inserting a blur (BlurPool) before every downsample — exactly the anti-aliasing step from Gaussian pyramids. This is a genuinely impressive thing to bring up: it connects Module 1 to Module 2 and shows you know a real, published failure of the standard architecture.
2. **The inductive bias is a data-efficiency prior, not a hard constraint.** ViT (6.2) with enough data (JFT-300M) beats CNNs, because with enough data you can *learn* locality rather than assume it. The right framing: **inductive bias trades ceiling for sample efficiency.** In the low-data regime the CNN prior wins; in the high-data regime it becomes a constraint.
3. **Weight sharing is a Bayesian prior**, formally: it is a hard constraint that the posterior over functions be translation-equivariant. Group-equivariant CNNs (Cohen & Welling) generalise this to rotation/reflection groups.

### ✅ Mastery check

You have 800 labelled X-ray images for a binary classification task. Your colleague proposes a 3-layer MLP on flattened pixels, arguing "images are just vectors and MLPs are universal approximators."

(a) Give the strongest version of *their* argument.
(b) Refute it precisely — not with "CNNs are better" but with the correct technical reason.
(c) Name one situation where the MLP is actually the right call.

<details><summary>Answer sketch</summary>
(a) Universal approximation says a sufficiently wide MLP can represent any continuous function, including the convolutional one. Weight sharing is a <i>restriction</i> of the hypothesis space, so the MLP's optimum is at least as good. With unlimited data and compute, the MLP is not worse.
(b) Universal approximation is about <b>representability</b>, not <b>learnability from finite data</b>. With 800 samples the MLP must use data to discover translation structure that the CNN gets for free; its effective sample complexity is vastly higher, and its variance dominates. The correct statement: the CNN's constrained hypothesis space has much lower Rademacher complexity / better generalisation bound for the same training error, because the constraint is <i>aligned with the true data-generating process</i>. Also, practically: 150M parameters on 800 images is a guaranteed memorisation regime.
(c) When the input has no spatial structure to exploit — e.g. a tabular feature vector, or an image whose pixel ordering is arbitrary (a permuted representation), or extremely low-resolution inputs where locality is meaningless. Also: after a frozen pretrained backbone, an MLP head is exactly right.
</details>

### 🔨 Build + read

**Build:** On CIFAR-10 with only 2000 training images, train (a) an MLP with matched parameter count to (b) a small CNN. Plot both learning curves. Then train both on translated test sets (shift every test image by 1–8 px) and plot accuracy vs. shift — you should see the CNN degrade gracefully and, if you implement BlurPool, degrade much less. That's Zhang's result reproduced in an afternoon.

**Read:** Justin Johnson, UMich EECS 498-007 Lecture 7 ("Convolutional Networks"). Then Zhang, "Making Convolutional Networks Shift-Invariant Again" (ICML 2019) — short and eye-opening.

---

## 2.2 Convolution & Feature Maps in a Trained Network

### The mechanics, precisely

A conv layer maps $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ to $Y \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}$ with weights $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$:

$$
Y[c_o, i, j] = b[c_o] + \sum_{c_i=1}^{C_{\text{in}}} \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} W[c_o, c_i, u, v]\; X[c_i,\; s\cdot i + d\cdot u - p,\; s\cdot j + d\cdot v - p]
$$

**Output size** (memorise this — it comes up constantly):

$$
H' = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1
$$

with padding $p$, stride $s$, dilation $d$. For the common case $d=1$: $H' = \lfloor (H + 2p - k)/s \rfloor + 1$. "Same" padding at $s=1$ needs $p = (k-1)/2$, which is why odd kernel sizes are standard.

**Cost accounting.** Rather than memorising two formulas, count what the layer physically contains. The weight tensor $W$ has shape $C_{\text{out}}\times C_{\text{in}}\times k\times k$, so it holds exactly that many numbers, and there is one bias per output channel. That is the whole parameter count. Notice what is *absent*: $H$ and $W$ do not appear. The same filter bank is reused at every spatial position, so a conv layer applied to a $7\times7$ map and to a $224\times224$ map has identical parameter counts. That is weight sharing (2.1) showing up in the arithmetic.

Compute is the opposite story. Producing *one* output number means running the sum in the equation above once: $k^2 C_{\text{in}}$ multiply-accumulates. There are $C_{\text{out}}\cdot H'\cdot W'$ output numbers to produce. Multiply the two:

$$
\text{Params} = k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}, \qquad
\text{MACs} = k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H' \cdot W'
$$

So $\text{MACs} = \text{Params}\times H'W'$, near enough. The spatial size is the *only* thing that separates the two quantities — which is the seed of the decoupling result below. (FLOPs $\approx 2\times$ MACs, since each MAC is a multiply plus an add. Papers are inconsistent about which they report — always check.)

**Worked example.** Take ResNet-50's `conv1`: a $7\times7$ kernel, 3 input channels → 64 output channels, stride 2, input $224^2$, so output $112^2$.

Parameters: the kernel is $7\times7=49$ numbers, and there is one such kernel for each (output, input) channel pair, of which there are $64\times3$. So $49\cdot3\cdot64 = 9408$. Tiny — this layer holds well under a ten-thousandth of the network's 25M parameters.

MACs: each of the $112\times112$ output positions, for each of the 64 output channels, sums over $49\cdot3$ terms. So $49\cdot3\cdot64\cdot112^2 = 118$M. That is roughly 12% of ResNet-50's total ~4.1 GFLOPs — from a layer holding **0.04% of its parameters**.

**Pause:** before reading on, why do those two percentages differ by a factor of three hundred?

Because $H'W'$ multiplies compute but not parameters, and $H'W'$ is *huge* early and *tiny* late. `conv1` runs at $112^2 = 12{,}544$ positions; ResNet-50's last conv stage runs at $7^2 = 49$ positions, 256× fewer. Meanwhile channel counts run the other way: 3→64 at the start, 512→2048 at the end, and parameters scale as $C_{\text{in}}C_{\text{out}}$, so the late layers hold hundreds of times more weights. The two trends are deliberately opposed — that is what "halve the resolution, double the channels" (2.5) means — and the consequence follows directly: **early layers are FLOP-heavy and parameter-light, late layers are the reverse.** Parameter count and compute are decoupled, and you have now derived that rather than been told it. It is one of the most useful practical facts in the module: it is why pruning parameters rarely speeds a network up much, and why a "small" model can still be slow.

**In your own words:** why does the input resolution change a conv layer's FLOPs but not its parameter count?

### The $1\times1$ convolution

It sounds like a null operation — what can a kernel with a single tap possibly compute? The answer is: everything in the channel direction, and nothing in the spatial one. Set $k=1$ in the convolution equation and the double sum over $u,v$ collapses, leaving $Y[c_o,i,j] = \sum_{c_i} W[c_o,c_i]\,X[c_i,i,j]$ — a matrix-vector product applied independently at each pixel. So a $1\times1$ conv is a **per-pixel fully connected layer across channels**: it cannot see spatial context, it only mixes channels. Once you see it that way, its uses are obvious, because mixing channels cheaply is exactly what you want before and after an expensive spatial operation:

- **Dimensionality reduction** — cut $C$ from 256 to 64 before an expensive $3\times3$ (the Inception/ResNet bottleneck). Params for $3\times3$ 256→256 = 590k; with a bottleneck (256→64 via $1\times1$, $3\times3$ 64→64, 64→256 via $1\times1$) = 16k + 37k + 16k = 69k. **8.5× cheaper.**
- **Adding non-linearity** without changing spatial extent (Network-in-Network, Lin et al. 2013).
- **Cross-channel feature recombination** — the pointwise half of depthwise-separable conv (2.7).
- **Head layers** in detection/segmentation (a $1\times1$ conv over a feature map = a classifier applied at every location).

### Other convolution variants worth naming

| Variant | What it does | Where used |
|---|---|---|
| **Dilated / atrous** ($d>1$) | inserts holes → larger RF at same cost & resolution | DeepLab semantic segmentation, WaveNet |
| **Transposed ("deconv")** | learnable upsampling | U-Net decoder, GAN generators. **Causes checkerboard artefacts** — prefer upsample+conv (Odena et al. 2016) |
| **Grouped** | split channels into $g$ groups, convolve independently | AlexNet (GPU memory), ResNeXt (cardinality), ShuffleNet |
| **Depthwise** | grouped with $g = C$ | MobileNet (2.7) |
| **Deformable (v1–v4)** | learn per-location sampling offsets | detection on deformed/irregular objects; the conv answer to attention |

### What filters actually learn

Empirically (Zeiler & Fergus 2014; feature visualisation via activation maximisation):

- **Layer 1:** oriented edge detectors and colour blobs. **They look like Gabor filters and derivative-of-Gaussians.** The network *rediscovers Module 1* from data. Say this in an interview — it lands.
- **Layers 2–3:** corners, junctions, textures, simple patterns.
- **Layers 4–5:** object parts (faces, wheels, text).
- **Final layers:** object-level, class-selective units.

**Explainability tools** (the syllabus omits these entirely — they are commonly asked and cheap to learn):

- **Grad-CAM** (Selvaraju et al. 2017): for class $c$, weight each feature-map channel $A^k$ of the last conv layer by its average gradient
  $$
  \alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \qquad
  L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)
  $$
  The ReLU keeps only *positive* evidence for the class. Cheap, architecture-agnostic, and the standard sanity check for "is my model looking at the object or the watermark?"
- **Saliency maps / integrated gradients / SmoothGrad** — input-gradient based, noisier.
- **Occlusion sensitivity** — slide a grey patch, watch the score drop. Slow but assumption-free and very convincing to non-ML stakeholders.

### 🎯 Top-1% distinction

1. **Params vs FLOPs are decoupled** — early layers are FLOP-heavy and parameter-light; FC/late layers are the reverse. VGG-16 has 138M params, ~90% of them in the first FC layer, yet most of its compute is in the conv stack.
2. **Convolution is implemented as a matrix multiply** (`im2col` + GEMM, or implicit GEMM in cuDNN). This is why GPUs love it and why a "cheaper" op with worse memory access patterns can be *slower* (see 2.7). A candidate who knows conv → GEMM is speaking the infra dialect.
3. **First-layer filters recapitulate classical CV filters.** Learned Gabor/DoG. Best one-line connection between Modules 1 and 2.
4. **Grad-CAM's ReLU is not incidental** — without it you get evidence for *and against* the class mixed together, and the map becomes uninterpretable.
5. **Transposed convolution's checkerboard artefacts** arise when kernel size isn't divisible by stride (uneven overlap). The fix is nearest-neighbour/bilinear upsample followed by a normal conv. This shows up again in GANs and diffusion U-Nets (6.12–6.14).

### ✅ Mastery check

A layer takes input $56\times56\times128$ and outputs $28\times28\times256$ using a $3\times3$ conv with stride 2.

(a) Padding needed? Params? MACs?
(b) Replace it with a bottleneck: $1\times1$ 128→64, then $3\times3$ 64→64 stride 2, then $1\times1$ 64→256. Params and MACs now?
(c) You measure the bottleneck version as only 1.4× faster on GPU despite the FLOP reduction. Give two reasons.

<details><summary>Answer sketch</summary>
(a) $p=1$ gives $\lfloor(56+2-3)/2\rfloor+1 = 28$. ✓ Params $= 9\cdot128\cdot256 + 256 = 295{,}168$. MACs $= 9\cdot128\cdot256\cdot28\cdot28 = \mathbf{231.2\text{M}}$.
(b) $1\times1$ 128→64 at $56^2$: params 8192, MACs $= 1\cdot128\cdot64\cdot56^2 = 25.7$M. $3\times3$ 64→64 stride 2 at output $28^2$: params 36,864, MACs $= 9\cdot64\cdot64\cdot784 = 28.9$M. $1\times1$ 64→256 at $28^2$: params 16,384, MACs $= 64\cdot256\cdot784 = 12.8$M. <b>Total: 61,440 params (4.8× fewer), 67.4M MACs (3.4× fewer).</b>
(c) (i) <b>Three kernel launches instead of one</b>, each with its own memory read/write of the intermediate tensor — launch overhead and DRAM traffic dominate at these sizes. (ii) <b>Arithmetic intensity</b>: the $1\times1$ convs do few FLOPs per byte moved, so they're memory-bandwidth-bound, not compute-bound; cutting FLOPs doesn't help a bandwidth-bound kernel. (iii, bonus) The $3\times3$ 128→256 dense conv maps to a large, well-shaped GEMM that saturates tensor cores, while the narrow 64-channel convs give poorly-shaped GEMMs with low occupancy. This is the roofline argument and it's the same reason MobileNet underdelivers on GPU (2.7).
</details>

### 🔨 Build + read

**Build:** Write a script that, given any `torchvision` model, prints per-layer params, MACs, activation memory, and cumulative percentages (or read `fvcore`/`thop` then re-derive one layer by hand to check). Then implement Grad-CAM from scratch (forward hook on the last conv block, backward hook for gradients) and run it on 20 ImageNet images — find at least one where the model is right for the wrong reason.

**Read:** Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks" (ECCV 2014). Selvaraju et al., "Grad-CAM" (ICCV 2017). Distill.pub's "Feature Visualization" (Olah et al.) — the best visual explanation that exists.

---

## 2.3 Receptive Field & Pooling

### Intuition

Here is the question to hold onto: **if a unit deep in the network fires, how much of the image was it allowed to look at before deciding?**

That region — the set of input pixels that can influence a given unit — is its **receptive field**. It is a hard information limit, not a soft preference. A unit with a 30-pixel receptive field cannot possibly recognise a 200-pixel object, because 170 pixels of that object never reached it by any path. No amount of training fixes that; it is a wiring fact.

So receptive field is the CNN's answer to "what scale am I operating at" — the direct descendant of Module 1's scale-space, except that where scale-space made the scale explicit and adjustable, a CNN's scale is an emergent consequence of its kernel sizes and strides. Which means you have to be able to compute it.

### The math

**Work out the two-layer case by hand first; the general rule falls out of it.**

One $3\times3$ conv: an output unit is fed by a $3\times3$ patch. Receptive field 3.

Now stack a second $3\times3$ on top. Its output looks at three neighbouring units of layer 1 — and *each of those* looks at 3 input pixels. The tempting answer is $3\times3 = 9$. That is the confusion worth naming: those three windows **overlap**. Neighbouring layer-1 units are one pixel apart, so their windows are offset by one, not by three. Counting the union: the leftmost covers pixels 1–3, the middle 2–4, the rightmost 3–5. Five pixels, not nine.

That overlap is the whole content of the formula. Each new layer extends the reach by $(k-1)$ — the amount the window sticks out past its centre on each side — **not** by $k$. Hence:

$$
r_l = r_{l-1} + (k_l - 1) \qquad (\text{stride-1 case, } r_0 = 1)
$$

**Stride is the second ingredient.** If a layer downsamples by 2, then units in the *next* layer are 2 input pixels apart rather than 1, so each step of that next kernel reaches 2 pixels further. We track that multiplier as the **jump** $j_l$ — how far apart, in input pixels, two adjacent units of layer $l$ are. It compounds, because a stride-2 layer on top of another stride-2 layer gives units 4 pixels apart:

$$
j_l = j_{l-1}\cdot s_l \qquad (\text{jump / effective stride, } j_0 = 1)
$$

Now the extension contributed by layer $l$ is $(k_l - 1)$ *jumps* rather than $(k_l-1)$ pixels, and the jump that matters is the one in force at the layer's **input**:

$$
r_l = r_{l-1} + (k_l - 1)\cdot j_{l-1} \qquad (r_0 = 1)
$$

With dilation, a kernel's taps are spaced $d_l$ apart instead of adjacent, so it sticks out $d_l$ times further: replace $(k_l - 1)$ with $d_l(k_l-1)$.

**Pause:** before reading on — three $3\times3$ stride-1 convs. What's the receptive field, and does it match a single $7\times7$?

Each layer adds $(3-1)\times1 = 2$, starting from 1: $1 \to 3 \to 5 \to 7$. **Seven** — exactly one $7\times7$ kernel's reach, with 45% of the parameters and two extra non-linearities. That is the entire argument behind VGG's all-$3\times3$ design (2.5), and you have now derived it rather than been told it.

**Now a worked example with stride,** where the jump does the interesting work: conv $3\times3$/s1 → pool $2\times2$/s2 → conv $3\times3$/s1.

The first conv behaves as before: it adds $(3-1)\cdot j_0 = 2$, so $r_1 = 3$, and since its stride is 1 the jump is unchanged, $j_1 = 1$. The pool has $k=2$, so it sticks out only $(2-1)\cdot j_1 = 1$ pixel beyond what layer 1 already saw: $r_2 = 4$. But its stride of 2 doubles the spacing of everything downstream, $j_2 = 2$. The third conv now pays that price: it adds $(3-1)\cdot j_2 = 2\cdot 2 = 4$, giving $r_3 = 8$ — twice what the same layer contributed before the pool.

That is the key asymmetry. A stride-2 layer barely enlarges the receptive field *itself*, but it doubles the contribution of **every layer after it**. Strides make the receptive field grow multiplicatively downstream, which is why RF explodes in deep nets and why a stride-32 backbone can have an RF of several hundred pixels while an early stride-4 layer sees almost nothing.

**In your own words:** why does each layer add $k-1$ rather than $k$, and why does a stride-2 layer affect later layers more than itself?

### The Effective Receptive Field — the thing most people don't know

Luo et al. (NeurIPS 2016) showed that the **influence of input pixels on a unit is not uniform over the theoretical RF — it is approximately Gaussian**, concentrated near the centre. Intuition: the number of paths from a centre pixel to the output is combinatorially larger than from a corner pixel; summing many random paths gives a Gaussian by the CLT.

Consequences:

- The **effective** receptive field is a small fraction of the theoretical one — and it grows as $O(\sqrt{L})$ with depth $L$, not $O(L)$.
- So "my theoretical RF is 400 px, my objects are 200 px, I'm fine" is **wrong reasoning**. In practice you want theoretical RF comfortably larger (2–3×) than the object.
- This is a major reason FPN (4.4) and dilated convolutions exist, and a reason ViT's global attention is genuinely different — attention has a *uniform*, full-image effective receptive field from layer 1.

### Pooling

- **Max pooling:** takes the max in each window. Provides local translation invariance and a mild non-linearity; sharpens feature response. Gradient flows only to the argmax.
- **Average pooling:** smoother, keeps all information but dilutes strong activations.
- **Global Average Pooling (GAP):** average each channel over all spatial positions → a $C$-vector. Introduced in Network-in-Network, adopted by ResNet. **Replaces the giant FC layers** (VGG's first FC is 102M params; GAP is 0), gives full translation invariance, and acts as a structural regulariser. It's also what makes Grad-CAM/CAM possible.
- **Adaptive pooling:** output a fixed grid regardless of input size → allows variable input resolution. `nn.AdaptiveAvgPool2d((1,1))`.
- **RoI Pooling / RoI Align:** pooling from an arbitrary box to a fixed grid — the core of Fast R-CNN and Mask R-CNN (4.3, 4.11).

**Why pooling largely disappeared.** Modern architectures use **stride-2 convolutions** instead: they downsample *and* learn what to keep, rather than applying a fixed hand-designed reduction. ResNet keeps exactly one max-pool (after conv1) and one GAP (at the end); everything else is strided conv. All-convolutional nets (Springenberg et al. 2015) showed max-pool can be dropped with no accuracy loss.

**But both strided conv and max-pool alias** (2.1's top-1% point) — hence BlurPool.

### 🎯 Top-1% distinction

1. **Effective ≠ theoretical receptive field**, ERF is Gaussian, grows as $\sqrt{L}$. This is a strong, specific, citable fact.
2. **Compute the RF for a given architecture on the spot.** Interviewers ask "you're detecting 16-px objects with a backbone whose stride-32 layer has RF 400 — what's wrong?" (Answer: the object occupies half a pixel at stride 32; you need shallow, high-resolution features → FPN.)
3. **GAP's triple role**: parameter elimination, translation invariance, and enabling class activation maps.
4. **Dilated convolution** grows RF exponentially with linear depth ($r$ grows by $d(k-1)$, with $d$ doubling), at constant resolution — but produces **gridding artefacts** because consecutive same-dilation layers sample disjoint lattices. Fix: hybrid dilation rates (e.g. 1,2,3 rather than 2,2,2).
5. **Max-pool's gradient is sparse** (only the argmax gets gradient), which is one reason very deep max-pool stacks trained poorly pre-ResNet.

### ✅ Mastery check

A backbone is: conv7×7/s2 → maxpool3×3/s2 → 3×[conv3×3/s1] → conv3×3/s2 → 3×[conv3×3/s1].

(a) Compute $j$ and $r$ after every layer.
(b) You must detect objects that are 24×24 px in a 640×640 input. Which layer's feature map would you attach a detection head to, and why?
(c) Your colleague says "just add 10 more 3×3 layers to grow the receptive field." Why is that a weaker fix than it looks?

<details><summary>Answer sketch</summary>
(a) L1 conv7/s2: $r=7$, $j=2$. L2 maxpool3/s2: $r = 7 + 2\cdot2 = 11$, $j=4$. L3–L5 conv3/s1 (three of them): each adds $2\cdot4=8$ → $r = 19, 27, 35$; $j=4$. L6 conv3/s2: $r = 35 + 2\cdot4 = 43$, $j=8$. L7–L9 conv3/s1: each adds $2\cdot8=16$ → $r = 59, 75, \mathbf{91}$; $j=8$.
(b) At $j=8$ a 24-px object spans 3×3 cells — workable but marginal. At $j=4$ (after L5) it spans 6×6 cells with $r=35$, comfortably larger than the object. <b>Attach to the $j=4$ / stride-4 map, or better, use both via an FPN</b> so small objects get high-resolution features and large objects get deep semantic ones. Key reasoning: a detection head needs (i) spatial resolution fine enough that the object covers several cells, and (ii) receptive field larger than the object plus context.
(c) Ten more stride-1 $3\times3$ layers add only $10 \times 2 \times j$ to the <b>theoretical</b> RF, and by Luo et al. the <b>effective</b> RF grows as $\sqrt{L}$ — so 10 extra layers give roughly a $\sqrt{}$-scale improvement in actual context, at full compute and memory cost. Dilated convolutions (exponential RF growth) or an explicit multi-scale design (FPN) buy far more context per FLOP. Also, more depth without residual connections risks the degradation problem (2.6).
</details>

### 🔨 Build + read

**Build:** Write `receptive_field(model)` that walks a `torch.nn.Sequential` and prints $(r_l, j_l)$ per layer. Validate it empirically: pick a unit in a deep layer, backprop a one-hot gradient to the input, and visualise which input pixels have non-zero gradient (theoretical RF) and the *magnitude* distribution (effective RF). You should see the Gaussian falloff directly.

**Read:** Luo et al., "Understanding the Effective Receptive Field in Deep CNNs" (NeurIPS 2016). Then the distill.pub article "Computing Receptive Fields of Convolutional Neural Networks."

---

## 2.4 Batch / Layer Normalisation 🔴

> **The gap analysis flagged this as critical, and it is the single most-asked "deep" question in ML interviews after backprop.** Budget real time here.

### Intuition

Start from a problem you already know from general deep learning. Every layer's output is the input to the next, and its scale depends on the current weights. Nudge an early layer's weights and every downstream layer's input distribution moves — by a factor that compounds through the stack. Push a fifty-layer network and activations either shrink toward zero or blow up, and the only defence anyone had before 2015 was to choose the initialisation scale so carefully that the growth factor sat near exactly 1, then keep the learning rate small enough that training never disturbed it.

Normalisation removes the need for that balancing act by brute force: **rescale the activations at each layer so that, whatever the layer before it did, the layer after it sees inputs with a predictable mean and variance.** If you enforce the scale rather than hoping for it, drift cannot compound, and you can take much larger steps without the training diverging.

Real-life anchor: it's the difference between a network you can train in 30 epochs and one that needs careful hand-tuned initialisation and 300 epochs — BatchNorm is why "just stack more layers" became a viable engineering strategy in 2015. That is the practical claim. *Why* it works turns out to be a much better story than the one originally told, and we get to it below.

### BatchNorm, precisely

For a conv feature map of shape $(N, C, H, W)$, BN computes statistics **per channel, over the $N\cdot H\cdot W$ elements of that channel**:

$$
\mu_c = \frac{1}{NHW}\sum_{n,h,w} x_{nchw}, \qquad
\sigma_c^2 = \frac{1}{NHW}\sum_{n,h,w}(x_{nchw} - \mu_c)^2
$$
$$
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad
y_{nchw} = \gamma_c\,\hat{x}_{nchw} + \beta_c
$$

Read the statistics line carefully, because *which axes are reduced* is the entire design. BN pools over $N$, $H$ and $W$ but **not** over $C$: every channel gets its own mean and variance. That is the right choice for a conv net for the same reason weight sharing was (2.1) — a channel is one filter's response, its statistics are the same wherever in the image it fires, so all $NHW$ of its values are samples of one distribution. Different channels detect different things and have no reason to share a scale.

$\gamma, \beta \in \mathbb{R}^C$ are **learnable**, and it's worth asking why, given that we just went to the trouble of fixing the scale. Because forcing every layer's output to be zero-mean unit-variance is a real restriction: it would, for instance, keep a sigmoid pinned to the near-linear part of its curve, throwing away the non-linearity. So we hand the scale and shift back to the network as two free parameters per channel. If the optimal thing is to undo the normalisation entirely, $\gamma = \sigma$ and $\beta = \mu$ does it exactly. The normalised network can therefore represent everything the unnormalised one could — **BN can only help representationally**; whatever it does, it does to the optimisation, not the hypothesis space. Hold that thought.

**Train vs. inference — the asymmetry that causes every BN bug.** Notice something unusual about $\mu_c$: it is computed over the batch, so *your prediction for one image depends on the other images that happened to be in the batch with it.* Every other layer you know is a function of a single sample. BN is not. That is fine, even useful, during training; it is unacceptable at inference, where you need a deterministic answer and may have only one image. So BN runs in two different modes.

At training time it uses the batch statistics. At inference it substitutes **running estimates** accumulated during training by an EMA:

$$
\mu_{\text{run}} \leftarrow (1-m)\,\mu_{\text{run}} + m\,\mu_B
$$

Make this vivid, because it is the thing to keep in mind: a BN network at test time is **not the same function** it was at train time. It is an approximation of the training-time function in which the batch-dependent constants have been frozen to their historical averages. If those averages are noisy (small batches), or were gathered on a different distribution (fine-tuning, domain shift), or you simply forget to switch modes — the test-time function drifts away from the one you actually trained, and the loss curve gives you no warning at all, because training looked perfect. Every failure in the table below is a version of this one sentence. **`model.eval()` is what switches the mode. Forgetting it is the most common bug in PyTorch inference code**, and the symptom — great training metrics, mysteriously bad deployed accuracy — looks like a hundred other problems.

**The bias before BN is redundant** — $\mu$ subtraction removes any constant, and $\beta$ re-adds a learnable one. Hence `nn.Conv2d(..., bias=False)` before a BN. Free parameter savings and a small clarity signal.

### Why does it work? (The part that separates candidates)

**This is a story about how the field corrected itself, and it is worth telling in that order — the wrong explanation, why it was believed, how it was falsified, what replaced it — because that sequence is far more memorable than the conclusion, and interviewers are testing whether you learned the conclusion or the reasoning.**

*The original story.* Ioffe & Szegedy (2015) named the disease **internal covariate shift**: as earlier layers update, the distribution of inputs to each later layer keeps moving, so every layer is forever chasing a target that shifts under it. BN, they argued, pins those distributions in place, so each layer faces a stationary problem and can be optimised properly.

It is worth appreciating *why* everyone believed this, because it is genuinely plausible. It matches the intuition above exactly. It uses a concept — covariate shift — that was already respectable in statistics. It explains the observed effect (faster training) via a mechanism that is obviously present (distributions do move). And it is what the paper that introduced the technique said. For three years it was the standard answer.

*The falsification.* Santurkar et al. (NeurIPS 2018, *"How Does Batch Normalization Help Optimization?"*) did the experiment the story invites. If BN helps by *removing* distribution shift, then putting the shift back should destroy the benefit. So they **injected random, time-varying noise after each BN layer** — deliberately re-introducing severe covariate shift, worse than the unnormalised network had. The network still trained fast. They also measured ICS directly and found that networks *without* BN don't exhibit much of it in the first place, and that BN doesn't reliably reduce what there is. The proposed cause was neither necessary nor present. The correlation was real; the causal story was wrong.

Note the shape of the correction: nobody disputed that BN works. What changed was the *mechanism*, and the new mechanism turns out to have nothing to do with distributions of activations and everything to do with the **geometry of the loss surface**.

**What's actually going on:**

1. **Loss-landscape smoothing.** BN makes the loss and its gradients more **Lipschitz** — bounds on $\|\nabla L\|$ and on the Hessian's effect are tightened.

   "The landscape is smoother" is easy to say and hard to picture, so make it concrete in terms of the only thing an optimiser can actually do. Gradient descent computes the gradient at your current point and then *extrapolates*: it assumes the direction it just measured remains roughly correct for a whole step of length $\eta$. That assumption is exactly what a Lipschitz bound on the gradient controls. In a rough landscape, the gradient a step away can point somewhere entirely different, so a large $\eta$ overshoots into a region the gradient never described, the loss goes up, and you must shrink $\eta$ until the extrapolation is safe. In a smooth landscape the gradient changes slowly, so the measurement stays valid over a much longer distance, and you can take much bigger steps.

   The concrete, measurable version from the paper: at any point during training, walk along the current gradient direction and plot the loss. Without BN this curve is jagged and the safe step size is tiny and varies wildly from step to step; with BN it is close to a well-behaved bowl, so the same learning rate is safe everywhere. **Larger stable learning rates are the mechanism, and every other benefit — speed, robustness, the ability to stack fifty layers — follows from being allowed to move faster without falling off a cliff.** That is the main answer.

2. **Scale invariance of the loss w.r.t. weights, and auto-tuning of the effective learning rate.**

   Why would anyone go looking for a scale invariance in the first place? Because of the observation made above: BN divides by the standard deviation of its own input. Anything that scales that input scales the standard deviation identically, so it cancels. Weight magnitude is precisely such a thing. And whenever a network is exactly invariant to some change in its weights, that is a strong signal — it means an entire direction in parameter space does nothing to the function, and you should ask what the optimiser is doing while it moves along a direction that cannot change the loss. That question has a surprisingly practical answer.

   Formally: with BN immediately after a linear layer, scaling the weights $W \to aW$ leaves the *output* unchanged (the scale cancels in $(x-\mu)/\sigma$), and therefore leaves the loss unchanged: $L(aW) = L(W)$ for every $a>0$.

   **Differentiate that identity with respect to $W$.** The left side needs the chain rule, which brings down a factor of $a$:
   $$
   \frac{\partial}{\partial W}L(aW) = a\,(\nabla L)(aW), \qquad \frac{\partial}{\partial W}L(W) = (\nabla L)(W)
   $$
   The two sides are equal, so $a\,(\nabla L)(aW) = (\nabla L)(W)$, i.e.
   $$
   \boxed{\ \nabla_{aW} L = \frac{1}{a}\nabla_W L\ }
   $$
   Read what that says: **gradients shrink in exact proportion as the weights grow.** Now push it one step further, because the useful consequence is the *square*, and it comes from comparing two quantities rather than looking at one. Scale the weights up by $a$. The step the optimiser takes is $\eta\|\nabla\|$, which has shrunk by $1/a$. But the thing that step is modifying, $\|W\|$, has grown by $a$. What matters to the function is not the absolute step but the **relative** change — moving 0.01 when $\|W\|=1$ is a large change of direction, moving 0.01 when $\|W\|=100$ is nothing. That relative change is $(1/a)/a = 1/a^2$.

   So the effective learning rate is $\propto 1/\|W\|^2$. During training $\|W\|$ tends to grow (there is nothing stopping it — growing costs no loss, by the invariance we just proved), so the **effective learning rate decays automatically**: BN gives you a built-in, self-tuning LR schedule that nobody wrote. And run the argument backwards for the other consequence: if you initialise the weights ten times too large, the effective learning rate for that layer is simply 100× smaller and the forward pass is unchanged — so the initialisation scale, the thing pre-2015 practitioners agonised over, stops mattering. **This derivation is the single highest-value thing to be able to state about BN.**

3. **Regularisation via batch noise.** Come back to the strange property from earlier: a sample's output depends on its batch-mates. $\mu_B, \sigma_B$ are therefore stochastic — reshuffle the data loader and the same image is normalised slightly differently on every epoch. That is data-dependent noise injected into the activations, which is precisely the mechanism dropout uses, arrived at by accident. It is real regularisation, and two non-obvious predictions follow that both hold in practice: BN networks often need less dropout (you are already injecting noise), and **reducing** the batch size acts like **more** regularisation, because fewer samples means noisier statistics. Note that this puts you in a bind — the same knob that controls regularisation strength also controls estimate quality, which is why very small batches don't give you a nicely regularised model, they give you the failure in the first row of the table below.

**In your own words:** what is the one property of BatchNorm that all of its odd behaviours descend from?

### Failure modes — where interviews actually probe

| Failure | Mechanism | Fix |
|---|---|---|
| **Small batch size** (detection/segmentation often run batch 1–2 per GPU) | $\mu_B, \sigma_B$ estimated from very few samples → high variance → train/test statistics mismatch → accuracy collapse | **GroupNorm**, **SyncBatchNorm** (all-reduce statistics across GPUs), or freeze BN and use running stats |
| **Batch dependence leaks information** | A sample's output depends on its batch-mates. In contrastive learning this lets the model "cheat" by using batch statistics to identify positives | **MoCo's shuffling BN** — shuffle sample order across GPUs before the key encoder so query and key see different BN statistics. (Direct link to 3.6.) SimCLR avoids it with global BN over very large batches |
| **Train/test distribution shift** | Running stats were estimated on the training distribution | Re-estimate BN statistics on the target distribution (a surprisingly effective, cheap domain-adaptation trick) |
| **Fine-tuning with tiny batches** | Updating BN stats on a small, unrepresentative batch destroys a good pretrained estimate | Freeze BN layers (`.eval()` on BN modules) during fine-tuning — standard practice in detection |
| **Sequence models / variable length** | Batch statistics over padded sequences are meaningless; and autoregressive inference has no batch | **LayerNorm** — this is exactly why transformers use LN |
| **BN + weight decay interaction** | WD on BN's $\gamma,\beta$ and on scale-invariant weights changes the effective LR rather than regularising | Exclude BN parameters (and biases) from weight decay — standard in modern recipes |
| **Adversarial / RL settings** | Non-stationary input distributions break running estimates | GroupNorm or no norm |

### The normalisation family — know the axes

For a tensor $(N, C, H, W)$, each method differs only in **which axes are reduced**:

| Method | Reduces over | Batch-dependent? | Typical use |
|---|---|---|---|
| **BatchNorm** | $(N, H, W)$ per channel | ✅ yes | CNN classification with large batches |
| **LayerNorm** | $(C, H, W)$ per sample | ❌ no | Transformers, ViT, RNNs |
| **InstanceNorm** | $(H, W)$ per sample per channel | ❌ no | Style transfer — removing per-image contrast/colour statistics *is* removing style |
| **GroupNorm** | $(C/G, H, W)$ per sample per group | ❌ no | Detection/segmentation at small batch; $G=32$ default |
| **RMSNorm** | like LN but **no mean subtraction**, divide by RMS | ❌ no | Modern LLMs (LLaMA); cheaper, works as well |
| **Weight Standardisation** | normalises the *weights*, not activations | ❌ no | Pairs with GN to match BN performance (BiT, NFNets) |

```
        N (batch) ──────────────>
      ┌──────────────────────────┐
  C   │  BatchNorm: one slice per channel, spanning all N and all HW
 (ch) │  LayerNorm: one slice per sample, spanning all C and all HW
      │  InstanceNorm: one slice per (sample, channel)
  v   │  GroupNorm: one slice per (sample, channel-group)
      └──────────────────────────┘
            HW (spatial)
```

**Why transformers use LN, not BN** — three reasons, give all three: (1) variable sequence length makes batch statistics ill-defined over padding; (2) autoregressive/streaming inference has batch size 1 and no future tokens; (3) NLP batch statistics are far noisier than vision's because token distributions are heavy-tailed. Bonus: **Pre-LN vs Post-LN** — the original Transformer put LN after the residual add (Post-LN), which needs learning-rate warmup to train; Pre-LN (inside the residual branch) trains stably without warmup and is now standard.

### Deployment: BN folding 🟡 (MLOps-relevant)

At inference, BN is an affine transform with fixed constants, so it can be **folded into the preceding convolution**:

$$
W_{\text{fold}} = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}} W, \qquad
b_{\text{fold}} = \beta - \frac{\gamma\,\mu}{\sqrt{\sigma^2+\epsilon}}
$$

Zero BN layers at inference, zero extra memory traffic, typically 10–30% latency reduction. Every deployment toolchain (TensorRT, ONNX Runtime, TFLite) does this automatically. **Mentioning BN folding signals deployment experience** and is directly on-track for an MLOps pivot.

### 🎯 Top-1% distinction

1. **"Internal covariate shift is not the explanation."** Cite Santurkar et al. and give the real one: loss smoothing / gradient Lipschitzness.
2. **Derive the scale invariance**: $\nabla_{aW}L = \frac{1}{a}\nabla_W L$ ⇒ automatic effective-LR decay ⇒ initialisation robustness.
3. **BN is the only common layer whose output depends on other samples in the batch.** Everything strange about it follows from that one sentence — the train/test gap, the small-batch failure, the regularisation, the contrastive-learning leak, the RL problems.
4. **Know MoCo's shuffling BN** as a concrete instance of the leak. Very few candidates connect BN to contrastive learning.
5. **BN folding at inference.**
6. **Exclude BN parameters from weight decay** and be able to say why.

### ✅ Mastery check

You fine-tune an ImageNet-pretrained ResNet-50 for a segmentation task. GPU memory limits you to **batch size 2**. Training loss looks fine but validation accuracy is terrible and unstable across epochs.

(a) Name the most likely cause and the precise mechanism.
(b) Give three fixes, ranked, with the trade-off of each.
(c) Separately: your teammate scales all the weights of a conv layer (followed by BN) by 10 "to increase capacity." Predict the effect on the forward pass, on the gradient, and on the effective learning rate.

<details><summary>Answer sketch</summary>
(a) <b>BatchNorm with batch size 2.</b> $\mu_B,\sigma_B$ are estimated from 2 samples (times $H\!\times\!W$ positions, but the $N$-direction variance dominates for correlated spatial statistics), so they're high-variance and differ wildly batch to batch. Training still "works" because each batch is self-consistently normalised. At validation the model uses <b>running statistics</b>, which are an EMA of those noisy, biased estimates and do not match what any layer saw during training. Hence the train/val gap and instability.
(b) Ranked:
 1. <b>Replace BN with GroupNorm</b> ($G=32$). Batch-independent, so train and test behave identically; costs a small amount of accuracy at large batch but is strictly better here. Standard in detection/segmentation (Mask R-CNN with GN).
 2. <b>SyncBatchNorm</b> across GPUs if you have several — makes the effective batch $2\times N_{\text{GPU}}$. Fixes the statistics but adds an all-reduce per BN layer (communication cost) and doesn't help on a single GPU.
 3. <b>Freeze BN</b> — set BN layers to `.eval()` so they use the pretrained ImageNet running statistics and don't update. Cheapest, keeps a good estimate, but the statistics are from the wrong domain; works well when the target domain is close to ImageNet.
 (Also valid: gradient accumulation does <b>not</b> fix this — accumulation increases the effective optimisation batch but BN still normalises over 2 samples. That distinction is a great thing to say.)
(c) <b>Forward pass: completely unchanged.</b> BN divides by the standard deviation, and scaling $W$ by 10 scales both the activations and their std by 10, which cancels. <b>Gradient w.r.t. the weights: scaled by $1/10$</b> (chain rule through the cancelled scale). <b>Effective learning rate: reduced by $\sim 1/100$</b> — the update $\Delta W$ is $10\times$ smaller while $\|W\|$ is $10\times$ larger, so the *relative* change per step drops by $100\times$. So the change does nothing for capacity and silently slows learning by two orders of magnitude for that layer. This is exactly the scale-invariance property, and it's why BN networks are robust to initialisation scale.
</details>

### 🔨 Build + read

**Build:** Implement BatchNorm2d from scratch (forward, running stats, and the backward pass by hand — the BN backward derivation is itself a classic interview question). Then run the experiment: train a small CNN on CIFAR-10 at batch sizes {128, 32, 8, 2} with BN, and again with GroupNorm. Plot final validation accuracy vs. batch size for both. You will reproduce the GroupNorm paper's headline figure and you will never forget the result. Bonus: implement BN folding and measure the latency change.

**Read:** Ioffe & Szegedy (2015) for the original framing, then **Santurkar et al., "How Does Batch Normalization Help Optimization?" (NeurIPS 2018)** for the correction — read this one properly. Then Wu & He, "Group Normalization" (ECCV 2018).

---

## 2.5 Hierarchical Features & CNN Architecture Evolution

### The story, and why each step happened

Architecture history is only worth memorising if you know **what problem each design was solving**. That's what interviewers are testing.

| Year | Model | Key idea | Problem it solved | ImageNet top-1 |
|---|---|---|---|---|
| 1998 | **LeNet-5** | conv + pool + FC, 60k params | proof that gradient-trained convs work | (MNIST) |
| 2012 | **AlexNet** | ReLU, dropout, GPU training, heavy augmentation, 8 layers, 60M params | sigmoid/tanh saturation killed deep nets; ReLU fixed it. Compute made it feasible | 63.3% (top-5 error 15.3% vs 26.2% runner-up) |
| 2014 | **VGG-16/19** | **stacks of $3\times3$ only**, uniform design, 138M params | showed depth is what matters; simplified design space | 71.6% |
| 2014 | **GoogLeNet / Inception-v1** | multi-branch parallel kernel sizes, **$1\times1$ bottlenecks**, GAP instead of FC, auxiliary classifiers | accuracy at 12× fewer params than AlexNet; multi-scale in one layer | 69.8% |
| 2015 | **ResNet-50/152** | **residual connections** + BN | the degradation problem: >20-layer plain nets got *worse* | 76.1% / 78.3% |
| 2016 | **Inception-v3/v4** | factorised convs ($n\times n \to 1\times n + n\times 1$), label smoothing, BN-aux | efficiency + regularisation | 78.8% |
| 2017 | **ResNeXt** | **grouped conv**; "cardinality" as a third scaling axis beside depth/width | more accuracy per FLOP than making ResNet deeper/wider | 78.8% |
| 2017 | **DenseNet** | concatenate all previous feature maps | feature reuse; strong gradient flow; parameter-efficient (but memory-hungry) | 77.9% |
| 2017 | **SENet** | **channel attention**: squeeze (GAP) → excite (2 FC + sigmoid) → rescale channels | let the network reweight channels per input; won ILSVRC 2017 | 82.7% (SE-ResNeXt) |
| 2019 | **EfficientNet-B0..B7** | **compound scaling** of depth/width/resolution together, NAS-found base | previous scaling was one-axis and suboptimal | 77.1% → 84.3% |
| 2020 | **RegNet** | design-space *search* yielding simple rules for width/depth | showed hand-design and NAS converge on simple linear width rules | 79–81% |
| 2021 | **NFNets** | **no normalisation** — adaptive gradient clipping + scaled weight standardisation | proved BN isn't necessary, just convenient | 86.5% |
| 2022 | **ConvNeXt** | a ResNet modernised with ViT's *recipe*: $7\times7$ depthwise, fewer activations, LN, inverted bottleneck, AdamW, heavy aug | **the controlled experiment**: showed much of ViT's advantage was training recipe, not architecture | 87.8% (ConvNeXt-XL) |

### The VGG argument, with numbers

You derived the receptive-field half of this in 2.3: because overlapping windows extend reach by $k-1$ per layer, two stacked $3\times3$ convs have the same receptive field as one $5\times5$, and three have the same RF as one $7\times7$. So the choice between them is not a choice about what the network can see. It is purely a question of cost and non-linearity, and both favour the stack. For $C$ input and output channels:

- One $7\times7$: $49C^2$ parameters.
- Three $3\times3$: $3 \times 9C^2 = 27C^2$ parameters — **45% fewer**, plus **three** non-linearities instead of one.

That is the entire argument for the $3\times3$-only design, and it held for eight years until ConvNeXt showed that with modern training, large depthwise kernels ($7\times7$ depthwise) are competitive again — because a *depthwise* $7\times7$ costs $49C$, not $49C^2$.

### Design principles distilled

1. **Depth > width**, up to the point where optimisation breaks (which residuals fixed).
2. **Bottleneck before expensive ops** ($1\times1$ reduce → $3\times3$ → $1\times1$ expand).
3. **Downsample spatially while increasing channels**, roughly preserving compute per stage (halve $H,W$ ⇒ double $C$).
4. **Replace FC heads with GAP.**
5. **Multi-scale processing** — Inception did it in-layer, FPN does it across layers (4.4).
6. **Attention as reweighting** — SE was the first widely-used attention in vision, three years before ViT.
7. **The training recipe matters as much as the architecture** — ConvNeXt's central lesson.

**In your own words:** pick any two consecutive rows of that table and say what the later one could do that the earlier one couldn't, and what forced the change.

### 🎯 Top-1% distinction

- **Auxiliary classifiers in GoogLeNet existed because BN and residuals didn't yet** — they injected gradient at intermediate depths to fight vanishing gradients. Once BN + ResNet arrived, they became unnecessary and were dropped. Knowing *why a design disappeared* is stronger than knowing it existed.
- **ConvNeXt as a controlled ablation.** The correct framing of "CNN vs ViT" (6.3) is not "transformers are better"; it's "when you equalise the training recipe, the gap narrows dramatically, and what remains is about scaling behaviour and data regime."
- **SENet is attention.** Channel attention predates and is orthogonal to spatial self-attention. CBAM adds spatial. This is the bridge from Module 2 to Module 6.
- **NFNets prove BN isn't load-bearing** — it's a convenience with a good cost/benefit ratio, not a necessity.

### ✅ Mastery check

You must pick a backbone for: (i) a mobile app doing on-device classification, (ii) a server-side detector on 4K images, (iii) a research project with 5000 labelled images and a frozen-backbone budget.

For each, name a specific architecture, and justify with one quantitative and one qualitative reason. Then: why would you *not* choose VGG-16 for any of them, despite it being conceptually the simplest?

<details><summary>Answer sketch</summary>
(i) <b>MobileNetV3-Small or EfficientNet-Lite0</b>. Quantitative: ~2–5M params, ~60–300M MACs — fits an ARM CPU/NPU budget at 30 FPS. Qualitative: designed with hardware-aware NAS and quantisation-friendly ops (h-swish, no exotic layers).
(ii) <b>ConvNeXt-T/S or a ResNet-50/101 + FPN</b>, depending on whether you need mature tooling. Quantitative: ResNet-50 is ~4.1 GFLOPs at $224^2$ and scales predictably; at 4K you tile or downsample anyway. Qualitative: hierarchical multi-scale features are required for detection heads — a plain ViT gives a single-scale feature map and needs adaptation (ViTDet/Swin).
(iii) <b>A frozen DINOv2 or DINOv3 ViT-B</b> with a linear head. Quantitative: 5000 images is far too few to fine-tune 86M params without overfitting; linear probing on a strong SSL backbone is the highest-accuracy-per-label option. Qualitative: DINOv3's features are explicitly designed to be strong *without* backbone fine-tuning (6.6).
<b>Why not VGG-16:</b> 138M parameters (90% of them in one FC layer) for 71.6% top-1 — it is dominated on every axis. ResNet-50 gets 76.1% with 25M params and fewer FLOPs. VGG survives only as a <i>perceptual loss</i> feature extractor (its features happen to correlate well with human perceptual similarity, which is why LPIPS and style transfer still use it) — that exception is a nice detail to add.
</details>

### 🔨 Build + read

**Build:** Implement, from scratch and trained on CIFAR-10: a VGG-style block stack, an Inception block, a ResNet basic block, and an SE block. Match parameter counts across all four and compare accuracy/epoch curves. Then take your ResNet and apply ConvNeXt's recipe changes one at a time (AdamW, cosine schedule, LN instead of BN, GELU, fewer activations, larger depthwise kernel), logging the delta from each — this is a miniature version of the ConvNeXt paper and it teaches you more than reading it.

**Read:** ConvNeXt paper (Liu et al., CVPR 2022) — read the "modernising a ConvNet" roadmap section; it's a beautifully written ablation. Skim the SENet paper. UMich EECS 498-007 Lecture 8 ("CNN Architectures").

---

## 2.6 ResNet — Skip Connections and the Degradation Problem

### Intuition

Before the explanation, sit with the observation, because it should genuinely bother you.

Take a 20-layer CNN, train it on CIFAR-10, note its **training** error. Now build a 56-layer version of the same architecture and train it the same way. Its training error is **worse**. Not its test error — its training error, the number measuring how well it fits data it is looking at directly.

Take a moment on why that is close to impossible. A 56-layer network *contains* the 20-layer one as a special case: copy the 20-layer network's weights into the first 20 layers, and set the remaining 36 to compute the identity function. That configuration exists in the 56-layer network's parameter space and achieves exactly the 20-layer training error. The deeper network is strictly more expressive — it can do everything the shallower one can, plus more. A more expressive model that is *worse at fitting the training set* is a contradiction, unless something is wrong with the search rather than the space.

And that is precisely the diagnosis. The good solution is provably there; **gradient descent cannot find it.** So this is an *optimisation* failure, not a capacity or generalisation failure — and the fix should not add capacity, it should make that solution easier to reach.

That is exactly what ResNet does. If the problem is that the identity is hard to find, stop asking the network to find it: **wire it in as the default**, and let the layers learn only the deviation from it.

### The degradation problem — state this correctly

He et al. (CVPR 2016) showed a 56-layer plain CNN has **higher training error** than a 20-layer one on CIFAR-10. Not test error — **training** error.

> **This is the single most common misstatement in interviews.** People say "deep nets overfit, ResNet fixes it." Wrong — and you can now see why it's wrong from the argument above: overfitting means training error keeps *falling* while test error rises, and here the training error itself went up. It's an **optimisation** failure, not a generalisation failure. Getting this right immediately separates you.

### The residual block

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

That one addition re-parameterises what the layers are asked to produce. Previously a block had to output the whole desired mapping $\mathcal{H}(\mathbf{x})$; now it outputs $\mathcal{F} = \mathcal{H} - \mathbf{x}$, the *residual* — the correction to apply to what it was already given.

Why that helps is an asymmetry in how easy the two targets are to reach. If the optimal mapping $\mathcal{H}(\mathbf{x})$ is close to identity — which is the regime the degradation experiment says we're in, since the extra layers should ideally do nothing — then $\mathcal{F} = \mathcal{H} - \mathbf{x}$ is close to zero. And **driving a stack of conv layers to output zero is easy**: shrink the weights toward zero, which is the direction weight decay is already pushing, and every configuration near it also outputs near-zero, so the target is a broad, stable basin. Driving that same stack to output exactly the identity is the opposite: it requires one precise arrangement of weights, with ReLU passing everything through unclipped, and any perturbation breaks it. Same function, two parameterisations, wildly different difficulty for gradient descent — which is exactly the right kind of fix for a problem we diagnosed as optimisation rather than capacity.

**Basic block** (ResNet-18/34): conv3×3 → BN → ReLU → conv3×3 → BN → **(+x)** → ReLU.
**Bottleneck block** (ResNet-50/101/152): conv1×1 (reduce) → BN → ReLU → conv3×3 → BN → ReLU → conv1×1 (expand ×4) → BN → **(+x)** → ReLU.

When shapes change (stride 2, or channel count changes), the shortcut needs a **projection**: a $1\times1$ conv with matching stride ("option B"). He et al. tested zero-padding the identity ("option A") and projecting everywhere ("option C"); B is the standard compromise.

### The gradient argument — derive this

**Start with the plain network, so the contrast is visible.** In a plain stack, layer $i+1$ is some function of layer $i$ and nothing else. Backprop is the chain rule applied down that chain, so the gradient reaching an early layer $l$ from the loss at the top is a **product** of every intermediate Jacobian:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\prod_{i=l}^{L-1}\frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i}
$$

A product of $L-l$ matrices is a fragile object. If those Jacobians have typical scale slightly below 1, the product decays geometrically and the early layers receive essentially nothing; slightly above 1, it explodes. There is no scale that is stable over 50 factors, and nothing in training holds them at exactly 1. This is the same disease you already know from RNNs through time, expressed through depth instead.

**Now redo it with the skip.** For the **pre-activation** (v2) formulation the block is exactly $\mathbf{x}_{i+1} = \mathbf{x}_i + \mathcal{F}(\mathbf{x}_i)$. Unroll that recursion — each step adds a term and passes $\mathbf{x}$ through untouched — and the sum telescopes:

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_i, W_i)
$$

That is already the key structural fact, and it is worth pausing on: the deep activation is the *early* activation plus a sum of corrections, rather than the early activation pushed through a chain of transformations. Differentiate it. The first term is $\mathbf{x}_l$ differentiated with respect to itself, which is the identity — **that is where the $\mathbf{1}$ comes from, and it is there for free, before any weights are involved.** The second term differentiates the sum of residual branches:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l}
= \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\cdot\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l}
= \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}\left(\mathbf{1} + \frac{\partial}{\partial \mathbf{x}_l}\sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_i, W_i)\right)
$$

**Read the "$\mathbf{1}$".** It means the gradient at layer $L$ reaches layer $l$ **undiminished** — not attenuated by depth, not scaled by any learned weight — added to whatever the residual branches contribute on top. Compare the two expressions side by side and the whole innovation is one substitution: **the additive identity path converts a product into a sum.** Products of many terms vanish or explode; a sum whose leading term is exactly 1 does neither.

**Pause:** the sum term could in principle cancel the $\mathbf{1}$. Why doesn't that reintroduce the problem?

Because cancellation would require the residual branches' Jacobian to equal exactly $-1$, and to do so simultaneously for every sample in the mini-batch and at every depth. Vanishing in the plain network is the *generic* case — it happens for almost any scale of Jacobian. Vanishing here is a measure-zero coincidence. So the gradient essentially never vanishes, and that is a qualitative difference, not a quantitative improvement. **That derivation is what you write on the whiteboard.**

### Pre-activation ResNet (v2)

He et al. (ECCV 2016) rearranged to **BN → ReLU → conv → BN → ReLU → conv → (+x)**, with *nothing* on the shortcut path. In v1 the post-block ReLU sits on the main path, so the identity isn't quite clean. v2's shortcut is a pure identity from input to output of the entire network, making the derivation above exact and enabling 1000-layer training. Small change, big theoretical tidiness.

### Two deeper interpretations

1. **Ensemble view** (Veit et al., NeurIPS 2016). Unrolling a network of $n$ residual blocks gives $2^n$ distinct paths from input to output (each block either taken or skipped). ResNets behave like an **ensemble of relatively shallow networks**: deleting a single block barely changes accuracy (unlike VGG, where deleting one layer is catastrophic), and gradient magnitude is dominated by short paths. So "ResNet-152 is 152 layers deep" is misleading — its *effective* depth is much smaller.
2. **Loss-landscape view** (Li et al., NeurIPS 2018). Visualising the loss surface along random directions shows plain deep nets have chaotic, non-convex surfaces with many bad minima; adding skip connections makes the surface dramatically smoother and more convex-like. Pair this with 2.4's BN smoothing argument — **both** of the two great 2015 innovations work by making the optimisation landscape more benign.

### Where skip connections went next

Skip connections are the most transferable idea in all of deep learning:

- **Transformers**: every attention and MLP sublayer is residual. Without it, transformers don't train.
- **U-Net (4.11)**: long skips from encoder to decoder preserve spatial detail.
- **DenseNet**: concatenative rather than additive skips.
- **Diffusion U-Nets (6.13)**: residual blocks throughout.
- **LSTM's cell state** is, in retrospect, an additive skip through time — the same medicine for the same disease.

**In your own words:** why is "output zero" an easier thing for a stack of layers to learn than "output your input unchanged"?

### 🎯 Top-1% distinction

1. **"Degradation is an optimisation problem, not overfitting."** Say it with the training-error evidence.
2. **Derive $\partial\mathcal{L}/\partial\mathbf{x}_l$ and point at the $\mathbf{1}$.**
3. **The ensemble interpretation** and the block-deletion experiment.
4. **Explain the bottleneck's economics**: $1\times1$–$3\times3$–$1\times1$ makes a 50-layer net cheaper than a 34-layer basic-block one, because the expensive $3\times3$ operates at 1/4 the channels.
5. **Pre-activation v2 and why it's theoretically cleaner.**
6. **Why identity is hard to learn but zero is easy** — this is the actual causal claim, and most candidates state the residual formula without it.

### ✅ Mastery check

(a) Why is representing the identity function hard for a stack of conv+BN+ReLU layers, but representing zero easy? Be specific about ReLU.
(b) In a ResNet-50, you replace every identity shortcut with a $1\times1$ conv (option C). Predict: parameters, accuracy, and gradient behaviour. Would you do it?
(c) You train a 100-layer plain CNN with BN and good initialisation, no residuals. It trains, but plateaus at higher training loss than a 30-layer version. Give the ResNet explanation and one *non*-ResNet intervention that would also help.

<details><summary>Answer sketch</summary>
(a) ReLU is <b>not surjective onto negatives</b> — it zeroes half its input domain. To pass a signal through unchanged, a conv+BN+ReLU stack would need weights that keep every pre-activation positive for every input, which is a measure-zero, unstable configuration; and two stacked ReLU layers can only represent the identity on the positive orthant. Meanwhile <b>outputting zero is trivial</b>: set $W \to 0$ (or let BN's $\gamma \to 0$), which is exactly the direction weight decay pushes. So $\mathcal{F}\to 0$ is on the easy path of the optimiser and $\mathcal{F} \to \text{id}$ is not. Bonus: modern ResNet recipes <b>initialise the last BN's $\gamma$ to 0</b> in each block, so every block starts as exactly the identity — a direct exploitation of this asymmetry.
(b) Parameters increase modestly (the $1\times1$ projections on every block, roughly +5–10% for ResNet-50). Accuracy in He et al.'s ablation was marginally better than option B but not worth the cost. <b>Gradient behaviour is the real problem</b>: the shortcut is no longer an identity, so the clean $\mathbf{1}$ in $\partial\mathcal{L}/\partial\mathbf{x}_l$ becomes a product of projection Jacobians and the "gradient highway" degrades — at very great depth this reintroduces the degradation problem. He et al. observed exactly this: option C hurts as depth grows. So: no, don't do it; use projections only where shapes change.
(c) ResNet explanation: without identity paths the gradient is a product of $\sim$100 Jacobians, so early layers receive poorly-scaled gradient and the optimiser cannot coordinate a 100-layer update; the loss landscape is chaotic and SGD lands in a poor region. Non-ResNet interventions that genuinely help: <b>(i) careful scaled initialisation + adaptive gradient clipping (the NFNet recipe)</b>, which keeps signal propagation well-conditioned without skips; (ii) <b>deep supervision / auxiliary losses</b> at intermediate depths (the GoogLeNet trick), injecting gradient directly; (iii) <b>LayerScale / ReZero</b>-style learnable per-branch scalars initialised near zero. Naming NFNets here is a strong close because it proves the point that skips are one solution to a signal-propagation problem, not the only one.
</details>

### 🔨 Build + read

**Build:** Train plain-20 vs plain-56 vs ResNet-20 vs ResNet-56 on CIFAR-10 and **reproduce Figure 1 of the ResNet paper** (training error, not test). Seeing plain-56 sit above plain-20 with your own eyes is the thing that makes the concept permanent. Then run the **block-deletion experiment**: delete one residual block at a time from a trained ResNet and plot accuracy drop; do the same for a VGG layer. That reproduces Veit et al.

**Read:** He et al., "Deep Residual Learning" (CVPR 2016) §3.1 and §4.1. He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016) for v2. Veit et al., "Residual Networks Behave Like Ensembles of Relatively Shallow Networks" (NeurIPS 2016).

---

## 2.7 Efficient Convolutions: Depthwise-Separable and the MobileNet/EfficientNet Lineage 🟡

> Directly relevant to an MLOps / edge-deployment track. Also the source of the most useful "FLOPs vs latency" interview answer in the module.

### Intuition

Look again at the conv cost formula from 2.2: $k^2 C_{\text{in}} C_{\text{out}} HW$. The expensive part is the **product** $k^2 \times C_{\text{in}}C_{\text{out}}$ — every output channel touches every input channel at every one of $k^2$ offsets. Ask why those two factors have to multiply.

They multiply because a standard convolution insists on doing two conceptually separate jobs in a single operation: it mixes information **across space** (the $k\times k$ window) and **across channels** (summing over $C_{\text{in}}$), and it does them jointly, learning a separate spatial pattern for every input–output channel pair. Depthwise-separable convolution asks whether that joint treatment is necessary, and answers no: do the two jobs as two cheap steps instead of one expensive one — filter each channel spatially on its own, then mix channels with a $1\times1$. The product becomes a sum.

### The math

**Count the two steps separately; the ratio then writes itself.**

The **depthwise** step gives each input channel its own $k\times k$ filter and no cross-channel summation at all. So per output element it does $k^2$ MACs instead of $k^2 C_{\text{in}}$, and there are $C_{\text{in}}$ channels of output (one per input channel, not $C_{\text{out}}$): $k^2 \cdot C_{\text{in}} \cdot H \cdot W$ MACs.

The **pointwise** step is a $1\times1$ conv, which by 2.2 is a per-pixel channel mixer: $C_{\text{in}}\cdot C_{\text{out}} \cdot H\cdot W$ MACs. Note that this one carries the full channel product but has lost the $k^2$.

The **standard conv** we are replacing costs $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$ — it carries *both* factors. That is the whole story in one line, and the ratio just makes it arithmetic. Divide, cancelling the common $C_{\text{in}}HW$:

$$
\frac{k^2 C_{\text{in}} + C_{\text{in}}C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{k^2 C_{\text{in}}}{k^2 C_{\text{in}} C_{\text{out}}} + \frac{C_{\text{in}}C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}
$$

Read the two terms as what they are, because that is what makes the result memorable rather than a formula to recall. Each term is one step's cost expressed as a fraction of the original: the depthwise step avoided the channel product, so it survives as $1/C_{\text{out}}$; the pointwise step avoided the spatial product, so it survives as $1/k^2$. Neither step pays for both.

That also tells you immediately which term dominates. $C_{\text{out}}$ is typically 128–1024, so $1/C_{\text{out}}$ is negligible, and the saving is governed by $1/k^2$ — **the pointwise $1\times1$ is where essentially all the remaining compute lives.** For $k=3$: $\frac{1}{C_{\text{out}}} + \frac{1}{9} \approx \frac{1}{9}$, so **8–9× fewer MACs and parameters**. For $k=5$: ~25×, since the saving is quadratic in kernel size — which is why depthwise convolutions are the only setting in which large kernels are affordable, a fact ConvNeXt's $7\times7$ depthwise later exploits (2.5).

**Pause:** predict the split before checking. $3\times3$, 512→512, at $14\times14$ — of the separable version's total cost, roughly what fraction is the depthwise step?

Standard: $9\cdot512\cdot512\cdot196 = 462$M MACs. Separable: depthwise $9\cdot512\cdot196 = 0.9$M, pointwise $512\cdot512\cdot196 = 51.4$M, total **52.3M MACs — 8.8× less**. The depthwise step is **under 2%** of it. The "efficient" operation everyone names the technique after is arithmetically almost free; the $1\times1$ you barely think about is 98% of the work. Remember that when you get to latency, because the operation doing 2% of the FLOPs will turn out to consume 40% of the time.

### The lineage

**MobileNetV1 (2017):** stack depthwise-separable blocks. Two global knobs: width multiplier $\alpha$ (scale all channel counts) and resolution multiplier $\rho$.

**MobileNetV2 (2018) — inverted residuals + linear bottleneck.** This is the design worth understanding properly.
- A standard ResNet bottleneck is **wide → narrow → wide** (compress, process, expand), with the skip connecting the wide ends.
- MobileNetV2 inverts it: **narrow → wide → narrow** (expand by factor $t=6$ with a $1\times1$, do the depthwise $3\times3$ in the expanded space, project back down with a $1\times1$), with the skip connecting the **narrow** ends. Why: skips over thin tensors mean less memory traffic, and the expensive depthwise happens where it's cheap.
- **Linear bottleneck:** *no ReLU after the final projection*. Reason: ReLU on a low-dimensional tensor destroys information irreversibly (if the manifold of interest doesn't fit in the positive orthant of a low-dim space, ReLU collapses part of it and it can't be recovered). In high dimensions ReLU is survivable because information is likely preserved in *some* subspace. **This is a genuinely non-obvious design insight and a great thing to be able to explain.**

**MobileNetV3 (2019):** platform-aware NAS (MnasNet) + NetAdapt fine-tuning, **h-swish** ($x\cdot\text{ReLU6}(x+3)/6$ — a piecewise-linear, quantisation-friendly swish), and squeeze-excite blocks. Note the *hardware-aware* objective: NAS optimised measured latency on a real phone, not FLOPs.

**ShuffleNet v1/v2:** grouped $1\times1$ convs + **channel shuffle** (permute channels between groups so information crosses group boundaries). ShuffleNetV2's paper is important for a different reason — it articulated four **practical guidelines** derived from measuring latency rather than FLOPs: equal channel widths minimise memory access cost; excessive group convolution increases MAC; network fragmentation (many small branches) reduces parallelism; element-wise ops are non-negligible.

**EfficientNet (2019) — compound scaling.** Rather than scaling depth, width, or resolution alone, scale all three together:

$$
\text{depth } d = \alpha^\phi, \quad \text{width } w = \beta^\phi, \quad \text{resolution } r = \gamma^\phi
\quad \text{s.t.}\quad \alpha\cdot\beta^2\cdot\gamma^2 \approx 2,\ \ \alpha,\beta,\gamma \ge 1
$$

The constraint keeps FLOPs scaling as $2^\phi$ (doubling $\phi$ doubles compute), because FLOPs scale linearly with depth and quadratically with both width and resolution. Grid search on B0 found $\alpha=1.2, \beta=1.1, \gamma=1.15$. B0→B7 is just increasing $\phi$.

**EfficientNetV2 (2021):** replaced depthwise convs in *early* layers with **fused-MBConv** (a regular $3\times3$ replacing the expand-$1\times1$ + depthwise pair) — because depthwise convs are slow on modern accelerators at high resolution. Plus progressive resizing during training. This is EfficientNet admitting the FLOPs-vs-latency problem.

**RepVGG / structural re-parameterisation (2021):** train a multi-branch block ($3\times3$ + $1\times1$ + identity), then **algebraically fuse it into a single $3\times3$ conv for inference**. You get multi-branch training dynamics and a plain-VGG inference graph, which is what GPUs are fastest at. Elegant and very deployment-relevant.

### Why the FLOP saving does not become a speed-up — arithmetic intensity from first principles

We have just proved an 8.8× reduction in arithmetic. Measure it on a GPU and you will find nothing like an 8.8× speed-up, and often the depthwise layers are *slower* than the standard conv they replaced. That is not a benchmarking mistake; it follows from a single quantity, and you can reason it out without ever having seen a roofline plot.

**The idea.** A processor does two different things when it runs a kernel: it moves numbers between memory and the chip, and it does arithmetic on them. Both take time, and they happen concurrently, so the kernel's runtime is set by whichever is slower — the same way a factory's output is set by whichever of "parts arriving" and "workers assembling" is the bottleneck. The single number that decides which one binds is **arithmetic intensity**: how many arithmetic operations you perform per byte you had to move. High intensity means you fetch a number and then do a lot of work with it, so the arithmetic units are the constraint. Low intensity means you fetch a number, do almost nothing with it, and go back for the next one, so the memory bus is the constraint.

Hardware sets the crossover. A modern GPU can do on the order of a hundred arithmetic operations in the time it takes to fetch one byte from DRAM. So any kernel below roughly that intensity spends its time waiting on memory, and **cutting its FLOP count buys nothing at all** — you have made the workers faster in a factory that was starved of parts. (A "roofline plot" is just this argument drawn as a graph: intensity on the x-axis, achievable throughput on the y-axis, a rising bandwidth-limited line on the left meeting a flat compute-limited ceiling on the right. The corner is the crossover. Nothing more is hiding in it.)

**Now compute the intensity of each convolution.** Take one input element. A standard conv uses it in the computation of $k^2 C_{\text{out}}$ different output values, so having paid to fetch it once you extract $k^2 C_{\text{out}}$ MACs of work — with $k=3$ and $C_{\text{out}}=512$, that is nearly 5000 operations per element. Far above the crossover: comfortably compute-bound, which is precisely why GPUs are so good at convolution. A depthwise conv uses that same element in only $k^2 = 9$ outputs, because there is no channel dimension to reuse it across. Nine operations per element fetched. Far *below* the crossover: memory-bound.

So the depthwise step didn't remove work from the bottleneck; it removed work from the part that wasn't the bottleneck, and moved the operation into a regime where the hardware's peak arithmetic throughput is irrelevant. This is the same phenomenon as the bottleneck-block puzzle in 2.2's mastery check, and the same reason the 2%-of-FLOPs depthwise layers can eat 40% of measured latency. It is the single most useful thing in this section, so the interview-grade version follows.

### 🎯 Top-1% distinction — the answer that lands

**"FLOPs are not latency."** Depthwise convolution has terrible **arithmetic intensity** (FLOPs per byte of memory traffic): it does $k^2$ MACs per input element, versus $k^2 C_{\text{out}}$ for a standard conv. On a roofline model, a standard conv sits in the **compute-bound** region where GPUs are efficient; a depthwise conv sits in the **memory-bandwidth-bound** region where the FLOP reduction buys you almost nothing. Empirically MobileNetV2 has ~9× fewer FLOPs than ResNet-50 but is often only ~2–3× faster on a V100, and can be *slower* per-FLOP than a plain conv.

Corollaries to have ready:
- Depthwise separable convs help most on **CPU and mobile NPUs** (bandwidth-rich relative to compute), least on **big GPUs**.
- **Structural re-parameterisation (RepVGG)** exists precisely because plain $3\times3$ convs map best to hardware.
- The right optimisation target is **measured latency on the target device**, which is exactly what MobileNetV3's platform-aware NAS did.
- Other latency factors FLOPs ignore: kernel launch overhead, memory access cost (MAC), fragmentation, and element-wise ops (ReLU, add) which are pure bandwidth.

**In your own words:** what does an 8.8× reduction in MACs actually buy you, and on what kind of hardware does it fail to buy you anything?

**Second discriminator: explain the linear bottleneck.** "No ReLU at the narrow end because ReLU on a low-dimensional representation loses information irrecoverably" is the kind of answer that shows you read the paper rather than the summary.

### ✅ Mastery check

You're deploying a classifier to an ARM Cortex-A76 phone CPU. Target: 25 ms per frame.

(a) MobileNetV3-Large (0.22 GFLOPs) vs ResNet-18 (1.8 GFLOPs). Which do you *expect* to be faster, and what would you actually do before committing?
(b) You profile and find the depthwise convs are 40% of latency but only 3% of FLOPs. Explain and give two mitigations.
(c) Your teammate proposes replacing every $3\times3$ in ResNet-18 with a depthwise-separable pair to "get MobileNet's efficiency." What's wrong with that plan?

<details><summary>Answer sketch</summary>
(a) Expect MobileNetV3 to win — on a <b>CPU</b>, depthwise separable convs realise most of their theoretical advantage because CPU compute is the bottleneck relative to its cache bandwidth, and V3 was NAS-optimised against measured phone latency. But: <b>measure, don't assume</b>. Benchmark both with the actual runtime (TFLite/NNAPI/XNNPACK), at the actual input size, with the actual quantisation, on the actual device — including thermal-throttled sustained performance, not a single cold run.
(b) Depthwise convs are <b>memory-bandwidth-bound</b>: they read and write a full feature map while doing only $k^2 = 9$ MACs per element, so arithmetic intensity is ~9 vs ~$9C_{\text{out}}$ for a standard conv. Time is dominated by DRAM/cache traffic, which FLOPs don't measure. Mitigations: (i) <b>operator fusion</b> — fuse depthwise + pointwise + activation into one kernel so the intermediate never hits DRAM; (ii) <b>quantise to int8</b>, halving or quartering the bytes moved (bandwidth-bound ops benefit almost linearly from lower precision); (iii) use <b>fused-MBConv</b> (EfficientNetV2's fix) in the early high-resolution stages where the feature maps are largest; (iv) channels-last memory layout so the depthwise kernel reads contiguously.
(c) Several things. (i) ResNet-18 uses <b>basic blocks with wide→wide $3\times3$s</b>; naively swapping them gives a network with far less capacity and no compensating width increase — MobileNet compensates with the $t{=}6$ expansion. (ii) You'd keep ResNet's <b>wide→narrow→wide</b> residual topology, whereas the whole point of V2 is the <b>inverted</b> topology with skips over thin tensors. (iii) You'd keep the ReLU after the projection, destroying the linear-bottleneck property. (iv) The result would likely be <i>both</i> less accurate and not much faster on GPU. The correct move is to use a purpose-designed efficient architecture, or to distil ResNet-18 into MobileNetV3, not to hand-edit ops.
</details>

### 🔨 Build + read

**Build:** Implement a depthwise-separable block and an inverted-residual block from scratch. Then run the experiment that teaches the lesson: benchmark standard conv vs depthwise-separable at matched output shape, on **CPU and GPU**, and plot measured latency against FLOPs for both devices. You will get two very different pictures and you will own the "FLOPs ≠ latency" answer forever. Bonus: implement RepVGG's train-time-multi-branch → inference-time-single-$3\times3$ fusion and verify the outputs match to numerical precision.

**Read:** MobileNetV2 (Sandler et al., CVPR 2018) §3 on linear bottlenecks — short and worth it. ShuffleNetV2 (Ma et al., ECCV 2018) §2 for the four practical guidelines. EfficientNet (Tan & Le, ICML 2019) §3 for the compound-scaling derivation.

---

## 2.8 Overfitting & Regularisation in a Vision Context

### The vision-specific truth

If you come to vision from general ML, you arrive with a ranked list of regularisers — weight decay, dropout, early stopping — and vision quietly reorders it. **In vision, data augmentation is the dominant regulariser**, and the classics are secondary; dropout has essentially left conv layers altogether.

The reason is worth having ready, because it is the justification, not just the fact. Weight decay and dropout are *generic* capacity constraints: they know nothing about your task, so they shrink the hypothesis space more or less indiscriminately and hope the good solutions survive. Augmentation is different in kind — it injects a *specific, true* fact about your domain ("a flipped cat is a cat"), which removes exactly the wrong hypotheses and leaves the right ones untouched. Vision is unusual in that we happen to know many such facts and can apply them cheaply. That's the headline, and it's different from tabular ML or NLP, where far fewer label-preserving transformations are available.

### The toolkit

| Technique | Mechanism | Modern status in vision |
|---|---|---|
| **Data augmentation** | expands the effective dataset along known invariances | **primary**; see 2.9 |
| **Weight decay** | shrinks weights; with SGD ≡ L2 penalty | standard, ~1e-4 (SGD) / 0.05 (AdamW recipes) |
| **Dropout** | random unit zeroing; approximate model averaging | **largely abandoned in conv layers**; still used in FC heads and in ViTs |
| **Stochastic depth / DropPath** | randomly drop whole residual branches | **standard in deep nets and ViTs** — dropout's successor for vision |
| **Label smoothing** | soften one-hot targets | standard, $\varepsilon = 0.1$ |
| **Early stopping** | halt at best val | still used, but modern recipes favour longer training + more augmentation |
| **Mixup / CutMix** | convex/spatial mixing of samples **and labels** | standard in strong recipes |
| **EMA of weights** | keep an exponential moving average of parameters for eval | cheap, reliably +0.2–0.5% |

### Details that matter

**Dropout + BatchNorm interact badly.** Li et al. (CVPR 2019, *"Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift"*): dropout changes the variance of activations between train and test, but BN's running statistics were estimated under the *train* variance. The mismatch degrades accuracy. Fixes: put dropout only *after* all BN layers (i.e. in the head), or drop dropout entirely (what modern CNNs do). **Knowing this specific interaction is a strong signal.**

**Weight decay ≠ L2 with adaptive optimisers.** With Adam, an L2 penalty added to the loss gets divided by the per-parameter gradient magnitude estimate, so it regularises large-gradient parameters *less* — the opposite of what you want. **AdamW** decouples it: apply $\theta \leftarrow \theta - \eta\lambda\theta$ directly, outside the adaptive scaling. Always use AdamW, not Adam + L2. Also: **exclude BN parameters and biases from weight decay** (2.4 — decaying scale-invariant weights just changes the effective LR).

**Label smoothing:**

$$
y^{LS}_k = (1-\varepsilon)\,y_k + \frac{\varepsilon}{K}
$$

It prevents the network from driving the correct logit to $+\infty$ (which is what CE with hard targets asks for), improving calibration and generalisation. **The subtlety (Müller et al., NeurIPS 2019):** label smoothing tightens the clusters in the penultimate representation and **erases the relative-similarity information between classes** — a smoothed teacher makes a *worse* knowledge-distillation teacher, because the "dark knowledge" (that a cat looks more like a dog than a truck) is what distillation transfers. Knowing that label smoothing helps classification but hurts distillation is a genuinely differentiating detail.

**Loss functions — the gap the syllabus assumes.** Cross-entropy for classification, not MSE. Why: with a softmax output, $\partial \mathcal{L}_{CE}/\partial z_k = p_k - y_k$ — clean, bounded, and non-vanishing when the prediction is badly wrong. With MSE + softmax/sigmoid, the gradient carries an extra $\sigma'(z)$ factor that **vanishes exactly when the model is confidently wrong**, which is the worst possible time to have no gradient. Also, CE is the negative log-likelihood of a categorical model — MSE assumes Gaussian residuals, which is the wrong likelihood for a categorical variable.

**Class imbalance** (vision-specific and a bridge to Module 4): re-weighting, re-sampling, and — the modern answer — **focal loss (4.6)**. Also: *effective number of samples* re-weighting (Cui et al. 2019), $w_c \propto (1-\beta)/(1-\beta^{n_c})$.

**Double descent** 🟢: test error can *decrease again* past the interpolation threshold as model size grows. Worth knowing exists — it's why "bigger model = more overfitting" is not a reliable rule in the overparameterised regime.

**In your own words:** what does augmentation give a model that weight decay cannot, no matter how well you tune it?

### 🎯 Top-1% distinction

1. **"In vision the main regulariser is augmentation"** — and be able to justify it: augmentation injects a *known, correct* invariance, whereas dropout/weight decay are generic capacity constraints with no task knowledge.
2. **Dropout + BN variance shift.**
3. **AdamW's decoupling** and *why* L2-with-Adam misbehaves.
4. **Label smoothing's downside for distillation.**
5. **Softmax+CE gradient is $p - y$**; the MSE vanishing-gradient argument.
6. **Exclude norm parameters and biases from weight decay.**

### ✅ Mastery check

You train a ResNet-50 on 20k images of manufacturing defects. Train accuracy 99.8%, val 71%.

(a) Rank five interventions by expected impact, with reasoning specific to this being a *vision* problem with *20k* images.
(b) Your teammate adds dropout 0.5 after every conv block. Predict the result and explain the mechanism.
(c) The dataset is 95% "no defect." What changes in your loss and your metrics?

<details><summary>Answer sketch</summary>
(a) 1. <b>Stronger augmentation</b> (RandAugment + RandomResizedCrop + CutMix) — largest expected gain, because it directly expands the effective dataset along invariances that are genuinely true for this domain. 2. <b>Transfer learning done properly</b> — start from ImageNet or a DINOv2 backbone, freeze early layers, use layer-wise LR decay; 20k images is small enough that pretraining dominates. 3. <b>Weight decay tuning + AdamW</b> and EMA — cheap, reliable. 4. <b>Smaller model or stochastic depth</b> — ResNet-50 (25M params) on 20k images is over-parameterised; ResNet-18 or a frozen backbone + linear head may beat it. 5. <b>Label smoothing + longer cosine schedule</b>. Also non-modelling: check for a train/val <b>distribution shift</b> (different production line, lighting, camera) — a 29-point gap that large often isn't overfitting at all, and Grad-CAM will tell you in ten minutes whether the model is looking at the defect or at a batch-specific artefact.
(b) It will make things <b>worse</b>, for two compounding reasons. (i) <b>Variance shift with BN</b>: dropout changes activation variance between train and test, but the BN running statistics were estimated with dropout active; at eval the variance is different and the normalisation is wrong. (ii) Dropout in conv layers is <b>structurally weak</b> anyway — adjacent pixels in a feature map are highly correlated, so zeroing individual activations barely removes information (this is why <b>DropBlock</b>, which drops contiguous regions, was invented). The right modern substitute is <b>stochastic depth (DropPath)</b>, which drops whole residual branches and composes cleanly with BN.
(c) Loss: move from plain CE to <b>class-weighted CE</b> or <b>focal loss</b> (4.6) — the 95% negatives otherwise dominate the gradient and the model converges to "always predict no defect" with 95% accuracy. Effective-number re-weighting is a principled alternative to inverse-frequency. Metrics: <b>accuracy is useless here</b> — report precision/recall per class, PR-AUC (not ROC-AUC, which is optimistic under heavy imbalance), and pick the operating point from the business cost of a miss vs. a false alarm. In manufacturing, recall on defects usually dominates, so tune the threshold for a target recall and report the resulting precision.
</details>

### 🔨 Build + read

**Build:** On a 5k-image subset of CIFAR-10, run a regularisation ablation: baseline, +augmentation, +label smoothing, +mixup, +stochastic depth, +all. Log val accuracy and the train–val gap for each. Then add dropout after conv blocks to a BN network and measure the degradation — reproducing the variance-shift result yourself makes it permanent.

**Read:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019). Müller et al., "When Does Label Smoothing Help?" (NeurIPS 2019). Li et al., "Understanding the Disharmony between Dropout and Batch Normalization" (CVPR 2019).

---

## 2.9 Data Augmentation

### Intuition

You know things about the world that your dataset doesn't say out loud: a flipped cat is still a cat, a slightly darker cat is still a cat. Your network has no way to learn those facts except by seeing enough examples to infer them — which is spending labelled data to buy knowledge you already had.

Augmentation is the shortcut. Rather than *stating* the invariance, you *demonstrate* it: show the network the flipped image with the same label, and the only way to fit the data is to become insensitive to flips. It's the cheapest way to inject domain knowledge into a model, and it explains the structure of everything below — the taxonomy is a list of transformations that preserve labels, and the table of failure cases is what happens when one of them doesn't.

### The taxonomy

**Geometric:** horizontal flip, random crop, **RandomResizedCrop** (crop a random area 8–100% with random aspect ratio, then resize — *the* single most important augmentation for ImageNet-scale training, because it teaches scale and translation invariance simultaneously), rotation, shear, perspective warp.

**Photometric:** brightness/contrast/saturation/hue jitter, grayscale conversion, Gaussian blur, noise, JPEG compression artefacts, solarise, posterise.

**Occlusion:** **Cutout / Random Erasing** — zero out a random rectangle. Forces the model to use distributed evidence rather than one discriminative part.

**Mixing** (these change the *label* too):
- **Mixup:** $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$, with $\lambda \sim \text{Beta}(\alpha,\alpha)$. Theoretical framing: **vicinal risk minimisation** — instead of training on the empirical distribution (delta functions at data points), train on a *vicinal* distribution that assumes linear behaviour between examples. Improves calibration and adversarial robustness.
- **CutMix:** paste a rectangular patch of image $j$ into image $i$; mix the labels **in proportion to the patch area**. Combines Cutout's occlusion with Mixup's label mixing, and avoids Mixup's unnatural ghosted images.

**Learned policies:**
- **AutoAugment** (2018): RL-searched policy of (operation, probability, magnitude) triples. Expensive to search (thousands of GPU hours), transfers across datasets.
- **RandAugment** (2020): drop the search entirely. Two hyperparameters — $N$ (ops per image) and $M$ (global magnitude) — sample $N$ ops uniformly from a fixed list. Matches AutoAugment at ~zero search cost. **This is the practical default.**
- **TrivialAugment** (2021): $N=1$, magnitude sampled uniformly at random. Zero hyperparameters, competitive. A useful humility lesson about augmentation search.

**Test-time augmentation (TTA):** average predictions over flips/crops/scales. Reliable +0.5–1% at $k\times$ inference cost. Standard for competitions, usually too expensive for production.

### Augmentation is task-dependent — and getting this wrong is a real bug

| Augmentation | Safe for | **Wrong for** |
|---|---|---|
| Horizontal flip | natural images, ImageNet | text/OCR, digit recognition (6↔9 issues in some fonts), **medical laterality** (left vs right lung, situs inversus), chirality in chemistry/microscopy |
| Vertical flip | satellite/aerial imagery, microscopy | natural scenes (gravity is a real prior) |
| Rotation (arbitrary) | cells, astronomy, aerial | road-scene detection (cars aren't upside down), face recognition |
| Colour jitter | natural images | anything where **colour is the label** — defect classification by discolouration, medical staining, agricultural disease |
| Heavy blur | robustness training | fine-grained texture tasks |
| Mixup/CutMix | classification | **detection/segmentation need adapted versions** (Mosaic in YOLO); regression targets need care |

**The governing principle:** augment along transformations that are **label-preserving in your domain**, and match the augmentation distribution to the **test-time** distribution shift you expect. If your production cameras are noisy and dim, augment for noise and low light — not because it's a generic best practice, but because it's your actual domain gap.

### The bridge to Module 3 and 6

**Augmentation is the supervisory signal in contrastive self-supervised learning.** SimCLR (3.6) creates two augmented views of the same image and trains the network to map them to nearby embeddings. The augmentation set *defines what invariances the representation learns* — SimCLR's ablation showed **random crop + colour jitter** is the critical pair, and that crop alone lets the network cheat by matching colour histograms. So augmentation goes from "a regularisation trick" to "the definition of the learning objective." Make this connection explicitly when you reach 3.6.

**In your own words:** you're told an augmentation "improved accuracy on the benchmark." What single question decides whether you should use it on your own problem?

### 🎯 Top-1% distinction

1. **RandomResizedCrop is the workhorse**, not flip. Name it specifically.
2. **Mixup as vicinal risk minimisation**, not just "blend images."
3. **Augmentation defines the invariances of a self-supervised representation** — the Module 3/6 bridge.
4. **The task-dependence table** — being able to name a case where flip is *harmful* (medical laterality) shows applied judgement.
5. **RandAugment reduced AutoAugment to two hyperparameters at no accuracy cost, and TrivialAugment to zero** — the trajectory says the search was mostly unnecessary, which is a useful, slightly contrarian point.
6. **The train/test augmentation gap:** training with RandomResizedCrop and testing with centre-crop creates a systematic object-size mismatch. "FixRes" (Touvron et al. 2019) showed that fine-tuning at test-time resolution gives a free accuracy gain — a lovely detail almost nobody mentions.

### ✅ Mastery check

You're building a model to grade diabetic retinopathy from fundus photographs. Images come from three different camera models with different colour rendition.

(a) Design an augmentation policy. Justify each choice, and name at least one commonly-used augmentation you would **exclude** and why.
(b) Your val set is from cameras A and B; production is camera C. What does that mean for your policy?
(c) You add Mixup and accuracy drops. Give two plausible mechanisms.

<details><summary>Answer sketch</summary>
(a) <b>Include:</b> random rotation (full 360° — a retina has no canonical orientation), horizontal <i>and</i> vertical flip (same reason; note left/right eye differ but the grading task is per-eye pathology, not laterality), random resized crop at a mild ratio (0.8–1.0, since the optic disc and macula must stay in frame), mild brightness/contrast jitter and gamma jitter (to span exposure differences across the camera fleet), and Gaussian blur / defocus (real fundus images vary in focus quality). <b>Exclude or restrict:</b> aggressive <b>hue/saturation jitter</b> — retinopathy grading depends on <i>haemorrhages and exudates</i>, whose colour (red vs yellow-white) is diagnostic; large hue shifts are label-corrupting. Also exclude heavy Cutout/Random Erasing at large scale, since a single small microaneurysm can be the entire evidence for a grade and erasing it makes the label wrong.
(b) Your validation set does not measure the thing you care about: <b>generalisation to an unseen camera</b>. Two consequences. (i) Your augmentation policy should explicitly simulate <i>inter-camera</i> variation — colour-constancy / white-balance jitter, per-channel gain jitter, sensor-noise and JPEG-artefact simulation — i.e. augment along the axis of the expected domain shift. (ii) You should change your <b>evaluation protocol</b> to leave-one-camera-out cross-validation so the validation number actually predicts production. This is the more important half of the answer: no augmentation policy fixes a validation set that can't see the failure.
(c) (i) <b>Label corruption on a fine-grained ordinal task.</b> DR grades are ordinal (0–4) and the visual evidence is small and localised; blending two fundus images produces an image whose true grade isn't the convex combination of the two labels, and the ghosted lesions are not realistic. CutMix would be a better fit than Mixup here. (ii) <b>Interaction with heavy class imbalance</b> — mixing a common grade-0 with a rare grade-4 mostly produces near-grade-0 images with a small grade-4 label component, which dilutes the already-scarce positive signal. (iii, bonus) Mixup's benefits are largest with long training schedules; on a short schedule it can simply under-train, since the effective task is harder.
</details>

### 🔨 Build + read

**Build:** Implement Mixup and CutMix from scratch (including the correct loss: $\lambda\mathcal{L}(f(\tilde x), y_i) + (1-\lambda)\mathcal{L}(f(\tilde x), y_j)$, which is equivalent to and simpler than mixing the one-hot targets). Then run a controlled sweep on CIFAR-10: none / flip+crop / +RandAugment(N,M grid) / +CutMix, and produce the accuracy table. Separately, reproduce the **FixRes** effect: train with RandomResizedCrop at 160 px, test at 160 vs 224 centre-crop, then fine-tune the classifier head at 224 and re-test.

**Read:** Cubuk et al., "RandAugment" (2020). Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018) — read §2 for the VRM framing. Chen et al., "SimCLR" (ICML 2020) §3 and Figure 5 (the augmentation ablation) — read it *now*, it makes 3.6 trivial later.

---

## 2.10 Transfer Learning: Feature Extraction vs. Fine-Tuning

### Intuition

Somebody already spent 10,000 GPU-hours teaching a network what edges, textures, and object parts look like. Ask what fraction of that is specific to *their* thousand ImageNet classes: the Gabor-like first layer (2.2) would look the same if they'd trained on X-rays, and so, largely, would the texture and part detectors above it. Almost none of it is task-specific. Transfer learning is refusing to pay for it twice.

The whole of this section is then one question — **how much of the pretrained network do you let move, and why?** — with the answer set by how far your domain sits from theirs and how much data you have to justify the movement.

### The two modes

| | **Feature extraction (linear probe)** | **Fine-tuning** |
|---|---|---|
| What trains | only the new head | backbone (all or part) + head |
| Data needed | very little (hundreds) | more (thousands+) |
| Compute | tiny — you can precompute features once | full backward pass |
| Overfitting risk | low | high on small data |
| Best when | target domain ≈ source domain, little data | domain gap is large, enough data |
| Serving | **one backbone, many heads** | one model per task |

### The decision matrix

| | **Similar domain** | **Different domain** |
|---|---|---|
| **Small data** | Linear probe / freeze most layers | Freeze early layers, fine-tune the last stage; heavy augmentation; consider a domain-closer pretrained model |
| **Large data** | Fine-tune everything, low LR | Fine-tune everything, or train from scratch (pretraining mainly buys convergence speed) |

### The practitioner details that get asked

**Layer-wise learning rate decay.** Early layers encode general features and should barely move; later layers are task-specific. Set $\eta_l = \eta_{\text{base}} \cdot \xi^{L-l}$ with $\xi \approx 0.65$–$0.9$. Standard in ViT fine-tuning (BEiT, MAE recipes) and consistently worth ~1%.

**Gradual unfreezing / discriminative fine-tuning (ULMFiT-style):** train the head first with the backbone frozen, then unfreeze progressively from the top. Prevents large early gradients from a randomly-initialised head destroying pretrained weights.

**Freeze BatchNorm statistics.** When fine-tuning with a small batch, BN's running statistics get overwritten by noisy estimates from your small dataset, destroying a well-estimated pretrained quantity. Standard practice in detection: `BN.eval()` on the backbone. (Direct 2.4 callback.)

**Warmup + low LR.** A randomly initialised head produces large gradients in the first steps; without warmup those propagate into the backbone and undo pretraining.

**What transfers, quantitatively.** Yosinski et al. (NeurIPS 2014) measured layer-by-layer transferability: layers 1–2 are almost perfectly general (Gabor filters and colour blobs — Module 1 again), transferability degrades with depth, and there's a *co-adaptation* effect where splitting a network mid-stack and freezing hurts even when transferring to the *same* task.

**The nuance that makes you sound senior.** He et al., *"Rethinking ImageNet Pre-training"* (ICCV 2019): for COCO detection, training from **random initialisation** matches ImageNet-pretrained performance — *if* you train long enough and use proper normalisation (GN/SyncBN). Pretraining buys **convergence speed**, not a higher ceiling, once target data is plentiful. It still matters enormously when target data is scarce. The right summary: **"pretraining is a data-efficiency and time-efficiency tool, not a magic accuracy source."**

### The 2026 shape of this

The foundation-model era changed the default:

- **Frozen backbone + lightweight head** is now the dominant production pattern, because you serve *one* backbone for many downstream tasks (huge MLOps win: one model to host, cache, quantise, and monitor).
- **DINOv2/DINOv3** (6.6) are explicitly designed so a frozen backbone + linear head is competitive — Meta's pitch is "no fine-tuning required."
- **PEFT for vision**: LoRA on ViT attention projections, adapters, visual prompt tuning, BitFit. Train 0.1–1% of parameters, keep one shared frozen base. Same economics as LLM serving.
- **Linear-probe accuracy** is the standard yardstick for self-supervised representation quality (3.6, 6.4) — worth knowing the protocol: freeze the backbone, train a linear classifier on ImageNet features, report top-1.

**In your own words:** if pretraining doesn't raise the ceiling once you have plenty of target data, why is the frozen-backbone pattern nonetheless taking over production?

### 🎯 Top-1% distinction

1. **The He et al. "Rethinking ImageNet Pre-training" result** — pretraining accelerates, doesn't raise the ceiling, given enough target data. Nuanced and citable.
2. **Freeze BN when fine-tuning at small batch**, with the mechanism (2.4).
3. **Layer-wise LR decay** with a concrete $\xi$.
4. **The serving argument for frozen backbones** — one backbone, many heads, one thing to quantise and monitor. This is an infra answer to an ML question and it's exactly the register for an MLOps track.
5. **Negative transfer exists.** If the source domain is sufficiently unlike the target (natural images → radar, seismic, or spectrogram data), pretrained features can be worse than random init. Don't present transfer learning as unconditionally free.
6. **Catastrophic forgetting** if you fine-tune a shared backbone per task — which is exactly the argument for adapters/LoRA.

### ✅ Mastery check

You have a pretrained ImageNet ResNet-50 and need three models: (i) classify 12 species of bird, 300 images each; (ii) classify satellite land-use, 200k images, 10 classes; (iii) detect defects on X-ray images of welds, 4000 images.

For each: freeze or fine-tune, which layers, what LR schedule, and one risk specific to that case. Then: for (ii), your colleague says "just fine-tune everything, we have plenty of data." Is he right?

<details><summary>Answer sketch</summary>
(i) <b>Birds, 3.6k images, similar domain</b> (ImageNet contains ~60 bird classes). Freeze the backbone, train a linear head — or fine-tune only stage 4 with a 10× lower LR than the head. Cosine schedule, heavy augmentation, label smoothing. Risk: <b>fine-grained</b> classification needs high-resolution detail and discriminative local regions; a global-average-pooled ImageNet feature may not separate similar species. Mitigation: higher input resolution, and consider a stronger backbone (DINOv2) whose features are known to be strong for fine-grained tasks.
(ii) <b>Satellite, 200k images, different domain</b> (overhead view, different statistics, often multispectral). <b>Fine-tune everything</b>, with warmup + cosine, layer-wise LR decay, and SyncBN or GN. Risk: the ImageNet prior includes a strong "gravity/up" orientation prior that is wrong for overhead imagery — so use full rotation and vertical-flip augmentation, and expect early layers to change more than usual. If the imagery has non-RGB bands, the pretrained first conv doesn't apply directly (inflate or re-initialise it).
(iii) <b>Weld X-rays, 4k images, very different domain</b>. Fine-tune, but freeze the first stage and use a low LR with layer-wise decay; <b>freeze BN statistics</b> (small effective batch on high-res images). Risk: severe class imbalance and small defect size — this is really a detection/segmentation problem in disguise, so consider a segmentation formulation with Dice loss (4.12) rather than whole-image classification, and evaluate with PR-AUC.
<b>Is the colleague right for (ii)?</b> Largely yes, and He et al. supports him — with 200k target images, training from scratch would <i>eventually</i> match pretrained performance too. But "fine-tune everything" is still the better choice because it converges several times faster and costs nothing extra, and the correct refinement is that <b>pretraining's value here is schedule length, not final accuracy</b>. The one caveat worth raising: with a large domain gap and plentiful data, an even better option may be <b>self-supervised pretraining on the unlabelled satellite imagery itself</b> (there is far more unlabelled overhead imagery than labelled), which beats both options.
</details>

### 🔨 Build + read

**Build:** On a small dataset (Oxford-IIIT Pets or Flowers-102), run all four cells of the decision matrix and produce a single plot of accuracy vs. number of training images (100 / 500 / 2000 / full) for: linear probe on ResNet-50, full fine-tune, linear probe on a frozen DINOv2 ViT-B/14, and training from scratch. The crossover points in that plot are the answer to every transfer-learning interview question you will ever get, in your own numbers.

**Read:** He et al., "Rethinking ImageNet Pre-training" (ICCV 2019). Yosinski et al., "How transferable are features in deep neural networks?" (NeurIPS 2014). Kornblith et al., "Do Better ImageNet Models Transfer Better?" (CVPR 2019).

---

## 2.11 Practical: A Full Image-Classification Pipeline

This is the module's integrative lab. Build it once, properly, and it becomes your template forever.

Read the recipe as a sequence of decisions rather than a checklist, because the ordering carries most of the value: each step exists to make the *next* one interpretable. You split the data first so that every number afterwards means something; you baseline before optimising so that later gains are measurable against something real; you do error analysis before more training so that you fix the actual defect rather than the one you assumed. Most candidates can recite the hyperparameters in step 4. Far fewer can say why step 6 comes before another round of step 4.

### The recipe

**1. Data**
- Splits: train / val / **test**, split by the right unit (by *patient*, *site*, *camera*, or *product batch* — never randomly by image when images are grouped, or you leak).
- Inspect the data by hand. Look at 50 images per class. Find the label noise before the model does.
- Class balance: report it. Decide on weighting/sampling now, not after.

**2. Baseline first**
- Train a small model with minimal augmentation for a few epochs. Get an end-to-end number before optimising anything. If your baseline is broken, everything after it is noise.

**3. Model**
- Default: `timm` pretrained ResNet-50 or ConvNeXt-Tiny; frozen DINOv2 + linear head as the small-data baseline.
- Input resolution is a first-class hyperparameter — often worth more than architecture.

**4. Training recipe** (the modern default)
- Optimiser: **AdamW**, lr 1e-3 (scaled with batch size), weight decay 0.05, **excluding BN/bias params**.
- Schedule: **linear warmup (3–5 epochs) + cosine decay**.
- Augmentation: RandomResizedCrop + flip + **RandAugment(N=2, M=9)** + Random Erasing + Mixup/CutMix.
- Label smoothing 0.1. Stochastic depth 0.1 for deep models.
- **Mixed precision** (`torch.autocast` + `GradScaler`) — ~2× speed, ~half the memory.
- **EMA** of weights for evaluation.
- Fix seeds; log everything (Weights & Biases or MLflow).

**5. Evaluation — beyond accuracy**
- Top-1, **per-class** precision/recall/F1, confusion matrix.
- **Calibration**: reliability diagram and **Expected Calibration Error**. Modern networks are systematically over-confident; temperature scaling on the validation set fixes most of it with one parameter. If a downstream system consumes your probabilities, this matters more than 1% accuracy.
- PR curves and PR-AUC for imbalanced problems.
- Latency and memory on the target hardware.

**6. Error analysis — the highest-value hour you will spend**
- Look at the 100 worst-loss examples. Categorise the failures. Half the time you'll find label noise, near-duplicates across splits, or a spurious correlation.
- Run **Grad-CAM** on failures *and successes*. "Right for the wrong reason" is common and invisible in the metrics.
- Check for **shortcut learning**: is the model reading a watermark, a hospital-specific ruler, a border artefact?

**7. Deployment sanity (the MLOps seam)**
- Export to ONNX; verify numerical parity with PyTorch.
- **Fold BN into conv** (2.4); measure the latency delta.
- Post-training int8 quantisation with a calibration set; measure the accuracy drop. If it's too large, do quantisation-aware training.
- Log input distribution statistics in production for drift detection.

### 🎯 Top-1% distinction

The differentiator here isn't recipe knowledge, it's **process**: baseline before optimisation, split by the correct grouping unit, error analysis before more training, calibration reported alongside accuracy, and a deployment measurement rather than a FLOP estimate. Interviewers for applied roles probe exactly this, and most candidates answer with hyperparameters instead of a method.

### ✅ Capstone check

Ship a classifier end to end on a dataset of your choice and write a one-page report containing: the baseline number, the final number, an ablation table showing the contribution of each recipe component, a confusion matrix, a reliability diagram before and after temperature scaling, five Grad-CAM examples of failures with your diagnosis, and measured int8 latency on CPU. **That report is a portfolio artefact** — it is far more persuasive in an interview than a GitHub repo full of notebooks.

---

## Going Deeper — Papers, Sources and Research Scope

*arXiv IDs given where available. If one 404s, search the title — never cite an ID you haven't opened.*

### A. The canonical papers

The pairings matter here more than in any other module. Several of these papers exist to correct the paper above them, and reading the original without its rebuttal leaves you holding a mental model the field abandoned.

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Krizhevsky et al., *AlexNet* | 2012, NeurIPS | Read it for the engineering, not the architecture: two-GPU model parallelism, ReLU justified empirically, dropout, and the augmentation scheme. | NeurIPS 2012 |
| Simonyan & Zisserman, *VGG* | 2015, ICLR (1409.1556) | The all-3×3 argument, stated as a parameter-count-vs-receptive-field trade you derived in 2.3. | arXiv 1409.1556 |
| **Ioffe & Szegedy, *Batch Normalization*** | 2015, ICML (1502.03167) | The method. Note the stated *explanation* — internal covariate shift. | arXiv 1502.03167 |
| **Santurkar et al., *How Does Batch Normalization Help Optimization?*** | 2018, NeurIPS (1805.11604) | **Read immediately after the above.** Shows ICS is not the mechanism (they inject noise *after* BN, ICS returns, BN still helps) and argues for loss-landscape smoothing instead. The most instructive original-plus-refutation pair in vision. | arXiv 1805.11604 |
| Ba et al., *Layer Normalization* | 2016 (1607.06450) | Why the batch dimension is a liability, which is the whole reason Transformers use LN. | arXiv 1607.06450 |
| Wu & He, *Group Normalization* | 2018, ECCV (1803.08494) | Fig. 1 — BN's error explodes below batch size 8, GN's is flat. Memorise that figure; it answers a very common interview question. | arXiv 1803.08494 |
| **He et al., *Deep Residual Learning*** | 2016, CVPR (1512.03385) | The degradation problem framed as a paradox (deeper is *worse on training error*), which is what makes the identity-shortcut answer inevitable. | arXiv 1512.03385 |
| He et al., *Identity Mappings in Deep Residual Networks* | 2016, ECCV (1603.05027) | The pre-activation ordering, with the clean gradient-flow derivation the original paper lacked. This is the version modern code implements. | arXiv 1603.05027 |
| Veit et al., *Residual Networks Behave Like Ensembles* | 2016, NeurIPS (1605.06431) | Reframes ResNet as an implicit ensemble of shallow paths. Deleting a layer barely hurts — which no "it helps gradients" story predicts. | arXiv 1605.06431 |
| Howard et al., *MobileNets* | 2017 (1704.04861) | Depthwise-separable convolution with the cost algebra worked out. | arXiv 1704.04861 |
| Sandler et al., *MobileNetV2* | 2018, CVPR (1801.04381) | Inverted residuals and the linear bottleneck. The argument for *why* you must not put a ReLU on the narrow layer is the part to understand. | arXiv 1801.04381 |
| Tan & Le, *EfficientNet* | 2019, ICML (1905.11946) | Compound scaling — the claim that depth, width and resolution must scale together at a fixed ratio. | arXiv 1905.11946 |
| **Bello et al., *Revisiting ResNets*** | 2021, NeurIPS (2103.07579) | **The essential re-examination.** A ResNet with a modern training recipe closes most of the claimed gap to EfficientNet. Much of a decade's "architecture progress" was training progress. | arXiv 2103.07579 |
| **Liu et al., *A ConvNet for the 2020s* (ConvNeXt)** | 2022, CVPR (2201.03545) | The same lesson pointed at ViT: modernise a ResNet step by step and it matches Swin. The ablation table is the paper. | arXiv 2201.03545 |
| Zhang et al., *mixup* | 2018, ICLR (1710.09412) · Yun et al., *CutMix*, 2019, ICCV (1905.04899) | The two augmentations that actually changed defaults. | arXiv |
| Kornblith et al., *Do Better ImageNet Models Transfer Better?* | 2019, CVPR (1805.08974) | Transfer is not monotone in ImageNet accuracy, and fine-tuning vs feature-extraction behave differently. Directly informs 2.10. | arXiv 1805.08974 |

### B. The single best source, per hard topic

- **Why convolution beats an MLP on images (2.1).** Justin Johnson, UMich EECS 498-007 / 598-005, **Lecture 7 (Convolutional Networks)**. The full playlist is on YouTube; Johnson's treatment of parameter sharing and equivariance is the clearest available.
- **Receptive field arithmetic (2.3).** Araujo, Norris & Sim, *Computing Receptive Fields of Convolutional Neural Networks*, Distill (distill.pub/2019/computing-receptive-fields) — interactive, and it covers the effective-vs-theoretical distinction most sources skip.
- **BatchNorm, the whole story (2.4).** Read Ioffe & Szegedy §2–3, then Santurkar §3–4, then Wu & He's Fig. 1. In that order, in one sitting. Nothing else is needed.
- **ResNet's degradation problem (2.6).** The original paper's Fig. 1 plus §3.1. Then Veit et al. for the ensemble reading.
- **Architecture evolution as one narrative (2.5).** Johnson, **Lecture 8 (CNN Architectures)** — a single hour that sequences LeNet → AlexNet → VGG → GoogLeNet → ResNet with the reason for each transition.
- **Depthwise-separable cost arithmetic (2.7).** MobileNetV1 §3.1 — half a page, and it's the derivation you'll be asked to reproduce.
- **Augmentation and regularisation in practice (2.8–2.9).** `timm`'s training recipes documentation, plus the ResNet-strikes-back paper (Wightman et al., 2110.00476) — this is where the modern recipe is actually written down.
- **Transfer learning decisions (2.10).** Kornblith et al. above; there is no better single source.

### C. Reference implementations worth reading

- **`huggingface/pytorch-image-models` (`timm`) → `timm/models/resnet.py`.** The reference modern ResNet. Look at how `downsample` is constructed and at the stem variants (`deep`, `deep_tiered`) — those "small" details are most of ResNet-strikes-back's gains.
- **`pytorch/pytorch` → `aten/src/ATen/native/Normalization.cpp`.** Read `batch_norm_cpu` to see the train/eval divergence and the running-statistics update in code. The `momentum` convention (PyTorch's is the *opposite* of the paper's) is a classic bug source and seeing it here inoculates you.
- **`facebookresearch/ConvNeXt` → `models/convnext.py`.** Under 200 lines. Read it beside a ResNet block and enumerate the seven changes.
- **`tensorflow/models` MobileNetV2 or `timm/models/efficientnet_blocks.py`.** Find `InvertedResidual` and confirm for yourself that there is no activation after the projection.
- **`timm/data/mixup.py`.** mixup and CutMix in ~150 readable lines, including the label-smoothing interaction most tutorials get wrong.

### D. Open research questions

1. **How much of the CNN-vs-ViT gap is architecture and how much is recipe — at *small* data scale?** *Why open:* Bello and ConvNeXt settled this at ImageNet scale; below ~50k images, where most real projects live, the comparison is rarely run fairly. *Minimum experiment:* ResNet-50 vs ViT-S vs ConvNeXt-T, identical augmentation, identical schedule, identical hyperparameter budget, on 5k/20k/100k subsets. ~30–50 GPU-hours. **Feasible on Colab, and directly publishable as a workshop paper if done carefully.**
2. **Is BatchNorm's regularisation effect separable from its optimisation effect?** *Why open:* the two are confounded in every standard training run, and the loss-smoothing account explains the second but not obviously the first. *Minimum experiment:* fix the effective learning rate explicitly (BN makes the loss scale-invariant, so LR is confounded — you derived this in 2.4), then compare BN / GN / no-norm at matched effective LR. **Feasible; the tricky part is the matching, which is where the contribution is.**
3. **Do the standard augmentations still earn their place under long schedules?** *Why open:* augmentation policies were tuned at 90–200 epochs; modern recipes run 300–600, and there is evidence the ranking changes. *Minimum experiment:* small model, CIFAR-100 or Imagenette, ablate each augmentation at 100 vs 600 epochs. **Very feasible.**
4. **Does compound scaling's fixed ratio survive outside EfficientNet's search space?** *Why open:* the ratio was found by grid search on one architecture family and then asserted generally. *Minimum experiment:* re-run the scaling search on a ResNet family at small scale and compare the discovered exponents. **Moderate — borderline on a laptop.**
5. ~~A new normalisation layer.~~ **Effectively closed at accessible scale.** Dozens exist; none has displaced BN/LN/GN, and demonstrating a real win requires pretraining-scale compute. Skip.

---

## Module 2 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| CNN vs MLP | weight sharing + locality = a prior aligned with image statistics ⇒ lower sample complexity; conv is **equivariant**, pooling gives **invariance** |
| Shift invariance | strided conv/pooling alias (Nyquist!); BlurPool fixes it (Zhang 2019) |
| Conv cost | params $=k^2C_{in}C_{out}$, MACs $=k^2C_{in}C_{out}HW$; params and FLOPs are decoupled across depth |
| $1\times1$ conv | per-pixel channel mixing; bottlenecks cut cost ~8×; enables NiN/Inception/ResNet designs |
| Receptive field | $r_l = r_{l-1} + (k_l-1)\prod_{i<l}s_i$; **effective** RF is Gaussian and grows as $\sqrt{L}$ |
| Pooling | max/avg for local invariance; **GAP** killed FC heads and enabled CAM; strided conv replaced pooling |
| BatchNorm | normalise per channel over $(N,H,W)$; **ICS is not the reason** — it smooths the loss landscape and makes the loss scale-invariant in $W$ so the effective LR self-tunes |
| BN failures | small batch, batch-dependence leaks (MoCo shuffling BN), train/test stats, sequence models ⇒ GroupNorm/LayerNorm |
| Norm family | BN over $(N,H,W)$; LN over $(C,H,W)$; IN over $(H,W)$; GN over channel groups; RMSNorm drops mean-centring |
| Architectures | AlexNet=ReLU+GPU, VGG=3×3 depth, Inception=1×1 bottleneck+multi-scale, ResNet=residual, SE=channel attention, EfficientNet=compound scaling, ConvNeXt=recipe matters |
| ResNet | degradation is an **optimisation** failure; $\partial\mathcal{L}/\partial x_l$ has an additive $\mathbf{1}$; behaves like an ensemble of shallow paths |
| Depthwise separable | cost ratio $1/C_{out} + 1/k^2 \approx 1/9$; **FLOPs ≠ latency** — depthwise is bandwidth-bound |
| MobileNetV2 | inverted residual + **linear bottleneck** (no ReLU at the narrow end — ReLU destroys low-dim information) |
| Regularisation | augmentation dominates in vision; dropout+BN clash (variance shift); AdamW decouples WD; exclude norms from WD |
| Label smoothing | improves calibration, **hurts distillation** (erases dark knowledge) |
| Augmentation | RandomResizedCrop is the workhorse; Mixup = vicinal risk minimisation; augmentation *defines* SSL invariances |
| Transfer learning | linear probe vs fine-tune matrix; freeze BN at small batch; layer-wise LR decay; pretraining buys speed not ceiling (He 2019) |
| 2026 default | frozen foundation backbone + light head/adapters — one backbone, many tasks, one thing to serve |

---

*End of Module 2 notes. Drills in `Module-02-Drills.md`.*
