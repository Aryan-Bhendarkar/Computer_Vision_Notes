# Module 1 — Appendix: The Classical Image-Processing Toolkit

> **Why this appendix exists.** A coverage audit of this curriculum found that the main Module 1 teaches the *features and geometry* half of classical CV thoroughly and skips the *image processing* half almost entirely — thresholding, morphology, connected components, Hough, histogram operations, and the classical segmentation family. That is a real hole for two reasons.
>
> First, **this is the most widely deployed computer vision in the world.** Every industrial inspection line, every OCR preprocessor, every microscopy counting pipeline, every document scanner runs some version of threshold → morphology → connected components. There is far more of it in production than there is of anything in Module 6.
>
> Second, **it is the easiest interview question to fail.** "Count the objects in this image without a neural network" is a warm-up, and a candidate who reaches for a detector reveals that they only know the fashionable half of the field. The right answer takes thirty seconds and no GPU.
>
> Everything here is fast, deterministic, debuggable, and needs no training data. Read it once properly; you will use it more than you expect.

---

## A1.1 Histogram Operations & Contrast

### Intuition

The histogram of an image tells you how its brightness is distributed. Most of the time it's badly distributed — everything crushed into a narrow band because the scene was dim, or backlit, or the camera metered for the sky. Fixing the distribution before doing anything else is often the difference between a pipeline that works and one that doesn't.

### Histogram equalisation

Use the image's own **cumulative distribution function as the transfer function**:

$$
s = T(r) = (L-1)\int_0^r p_r(w)\,dw \quad\longrightarrow\quad s_k = (L-1)\sum_{j=0}^{k} \frac{n_j}{N}
$$

The result has an approximately flat histogram, so contrast is spread across the full range. **Why the CDF specifically:** applying a random variable's own CDF to itself yields a uniform distribution (the probability-integral transform) — the equalisation is not a heuristic, it is that identity.

**Its failure mode is severe and worth knowing:** it is *global*, so a bright window in one corner determines the mapping for the whole frame, and it **amplifies noise in flat regions** because it stretches near-empty histogram bins across a wide output range.

### CLAHE — the one you will actually use

**Contrast-Limited Adaptive Histogram Equalisation:** equalise **per tile** (typically $8\times8$ tiles), then bilinearly interpolate between tile mappings to avoid visible tile seams. The "contrast-limited" part is the essential detail: **clip the histogram at a threshold before computing the CDF** and redistribute the clipped mass uniformly. Without the clip, adaptive equalisation amplifies noise catastrophically in smooth regions; the clip limit is what makes the method usable.

`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`. Standard in medical imaging, low-light preprocessing, and OCR on faded documents.

### Gamma and white balance

- **Gamma correction** $I' = I^{\gamma}$ — $\gamma < 1$ brightens shadows, $\gamma > 1$ darkens. Remember from 1.1 that images are *already* gamma-encoded; applying another gamma is a display adjustment, not a physical correction.
- **White balance** — estimate the illuminant and divide it out. Grey-world ("the average scene is grey": scale each channel so its mean matches), max-RGB/white-patch ("the brightest pixel is white"), or a learned/measured illuminant. This is a **prior**, not a computation, because colour constancy is ill-posed (metamerism, 1.1).

### 🎯 Top-1% distinction

- **CLAHE's clip limit is the whole method.** Anyone can say "adaptive histogram equalisation"; knowing *why* the clip exists (noise amplification in flat tiles) is the differentiator.
- **Equalisation is the probability-integral transform**, not a trick.
- **The domain-shift connection:** if your training data and your deployment cameras have different tone curves, per-image contrast normalisation is a cheap, real fix — and it belongs in *both* the training augmentation and the inference preprocessing, identically. A mismatch between the two is a classic silent bug.

---

## A1.2 Thresholding, Morphology & Connected Components

> **This is the pipeline.** Threshold → clean up with morphology → label with connected components → measure. Learn it as one unit, because that is how it is used.

### Thresholding

**Global thresholding** picks one value; **Otsu's method** picks it automatically by exhaustive search over all thresholds $t$, maximising the **between-class variance**:

$$
\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t) - \mu_1(t)\bigr]^2
$$

where $\omega_i$ are the class probabilities (histogram mass below/above $t$) and $\mu_i$ the class means. Maximising between-class variance is equivalent to minimising within-class variance, since the total is fixed — that equivalence is the elegant part and the thing to say.

**Otsu assumes a bimodal histogram.** On a unimodal or heavily skewed histogram it returns a meaningless threshold, confidently. Check the histogram before trusting it.

**Adaptive / local thresholding** computes a threshold per neighbourhood (mean or Gaussian-weighted mean, minus a constant) — essential under uneven illumination, which is the normal case for document scanning and microscopy. `cv2.adaptiveThreshold`.

### Morphological operations

Defined by a **structuring element** $B$ (the shape and size of the neighbourhood). On binary images:

| Operation | Definition | Effect |
|---|---|---|
| **Erosion** $A \ominus B$ | keep a pixel only if $B$ fits entirely inside $A$ | shrinks objects, removes small specks, breaks thin bridges |
| **Dilation** $A \oplus B$ | keep a pixel if $B$ hits $A$ anywhere | grows objects, fills small holes, joins nearby fragments |
| **Opening** $(A\ominus B)\oplus B$ | erode then dilate | **removes small objects, preserves the size of large ones** |
| **Closing** $(A\oplus B)\ominus B$ | dilate then erode | **fills small holes and gaps, preserves outer size** |
| **Morphological gradient** | dilation − erosion | object boundaries |
| **Top-hat** | $A$ − opening($A$) | bright details smaller than the structuring element — **the standard trick for uneven-illumination correction** |

The key property: **opening and closing are idempotent and approximately size-preserving**, which is why you use them instead of raw erosion/dilation. An erosion shrinks everything; an opening removes only what is smaller than the structuring element and leaves the rest at its original size.

Structuring element choice matters: a disk for isotropic cleanup, a horizontal line to remove horizontal rules from a form, a vertical line for table borders. **Choosing the structuring element to match the artefact you want to remove is the whole skill.**

### Connected components

Label each maximally-connected set of foreground pixels with a unique integer. Two connectivity conventions — **4-connected** (edge neighbours) and **8-connected** (edges + corners) — and they give different answers on diagonal structures, so state which you used. Two-pass algorithm with union-find, or the modern single-pass variants; `cv2.connectedComponentsWithStats` returns per-component area, bounding box, and centroid, which is usually all the measurement you need.

**The counting pipeline in full:**

```
grayscale → CLAHE (if illumination is uneven)
          → adaptive threshold (or Otsu, if lighting is controlled)
          → opening with a disk (remove specks)
          → closing (fill holes)
          → connectedComponentsWithStats
          → filter components by area / aspect / solidity
          → count, measure, done
```

That is a 15-line function, runs in milliseconds on a CPU, requires no data, and solves a very large fraction of real industrial vision problems.

### The distance transform

For each foreground pixel, the distance to the nearest background pixel. Uses:
- **Separating touching objects** — peaks of the distance transform are object centres; this is the standard seed generator for watershed (A1.4).
- **Skeletonisation** — ridges of the distance transform.
- **Measuring** minimum feature width, clearance, and erosion depth.

### 🎯 Top-1% distinction

1. **Otsu maximises between-class variance ≡ minimises within-class variance**, and it *assumes bimodality*. Naming the assumption is the differentiator.
2. **Opening removes small things at their original size; erosion shrinks everything.** Most people conflate them.
3. **Top-hat for uneven illumination** — a specific, non-obvious, very practical tool.
4. **4- vs 8-connectivity changes your count.** Anyone who has debugged a counting pipeline knows this; nobody who has only read about it does.
5. **Distance transform + watershed is the classical answer to "separate touching objects"** — the same problem U-Net's boundary-weighted loss solves in 4.11, and worth being able to solve both ways.

### ✅ Mastery check

You must count bacterial colonies on an agar plate photographed under a ring light. Colonies are roughly circular, vary 10× in size, some touch, and the plate edge is much darker than the centre.

(a) Write the pipeline, naming the operation and the reason at each step.
(b) Why would plain Otsu fail here, and what replaces it?
(c) Two colonies touch and merge into one component. Give the classical fix and the deep-learning fix, and say when you'd choose each.

<details><summary>Answer sketch</summary>
(a) 1. <b>Mask the plate</b> (Hough circle, A1.3, or a simple radial mask) so the background outside it never enters the statistics. 2. <b>Top-hat</b> with a structuring element larger than the biggest colony, or CLAHE — corrects the ring-light vignetting so one threshold can serve centre and edge. 3. <b>Adaptive threshold</b> (see (b)). 4. <b>Opening</b> with a small disk to remove dust specks and sensor noise. 5. <b>Closing</b> to fill the lighter centres of large colonies. 6. <b>Distance transform + watershed</b> to split touching colonies. 7. <b>connectedComponentsWithStats</b>. 8. <b>Filter by area and circularity</b> ($4\pi A/P^2$) to reject scratches, bubbles, and merged blobs that survived. 9. Count.
(b) Otsu picks a <b>single global</b> threshold from the whole-image histogram. With a bright centre and a dark edge, the histogram is not cleanly bimodal — it is the superposition of two different illumination regimes — so any single threshold either loses the dim edge colonies or floods the bright centre. Replace with <b>adaptive/local thresholding</b>, or fix the illumination first with top-hat and <i>then</i> use Otsu, which is often cleaner because it keeps a single interpretable threshold.
(c) <b>Classical:</b> distance transform → find its local maxima as markers → <b>watershed</b> flooding from those markers splits the merged blob along the constriction. Cheap, no data, works well for convex objects. <b>Deep learning:</b> an instance-segmentation model (Mask R-CNN) or, far better for this domain, a <b>distance/vector-field method like Cellpose or StarDist</b> that predicts a per-pixel flow toward each object's centre and separates instances geometrically. <b>Choose classical</b> when objects are convex and roughly blob-like, you have no labels, and you need determinism and auditability (regulated lab settings). <b>Choose learned</b> when colonies are irregular, heavily overlapping, or vary in appearance in ways morphology can't capture — and note you'll need a few hundred annotated images, which for this task is a day of work and probably worth it if the count matters.
</details>

---

## A1.3 The Hough Transform & Template Matching

### The Hough transform — properly

**Intuition.** You have a set of edge pixels and want to find the lines. Instead of testing candidate lines against pixels, let **each pixel vote for every line that could pass through it**. Lines are then peaks in the space of votes.

**The parameterisation matters.** Do *not* use $y = mx + c$ — vertical lines need $m = \infty$. Use the normal form:

$$
\rho = x\cos\theta + y\sin\theta
$$

with $\rho$ the perpendicular distance from the origin and $\theta \in [0,\pi)$ the normal's angle. Both are bounded, so the accumulator is a finite 2-D array.

**Algorithm:** for each edge pixel $(x,y)$, for each $\theta$ in the discretised range, compute $\rho$ and increment `accumulator[ρ][θ]`. Each pixel traces a **sinusoid** through parameter space; collinear pixels' sinusoids **intersect at one point**, which is the line's $(\rho,\theta)$. Threshold the accumulator for peaks.

**Why it is robust:** voting is a consensus mechanism, so it tolerates **gaps** (a dashed lane marking still votes), **occlusion** (a partially hidden line still peaks), and outliers (noise votes spread thinly and don't peak). This is the same "let the data vote" logic as RANSAC (1.8), and the comparison is worth making: Hough enumerates the parameter space exhaustively, RANSAC samples it randomly — Hough is better for many instances of a simple model, RANSAC for one instance of a higher-dimensional model.

**Extensions:** circles use a 3-D accumulator $(x_c, y_c, r)$ — much more expensive, so `cv2.HoughCircles` uses the gradient-direction trick to vote only along each edge pixel's normal. The **Generalised Hough Transform** handles arbitrary shapes with an R-table of boundary-point offsets relative to a reference point.

**Practical:** `cv2.HoughLinesP` (probabilistic Hough) samples edge pixels rather than using all of them and returns line *segments* with endpoints, which is what you usually want. Its parameters — `threshold`, `minLineLength`, `maxLineGap` — are what you'll actually tune.

### Template matching and NCC

Slide a template over the image and score the match. **Normalised cross-correlation** is the one to use:

$$
\text{NCC}(u,v) = \frac{\sum_{x,y}\bigl[I(x{+}u,y{+}v) - \bar{I}_{u,v}\bigr]\bigl[T(x,y) - \bar{T}\bigr]}{\sqrt{\sum \bigl[I - \bar{I}_{u,v}\bigr]^2 \sum\bigl[T - \bar{T}\bigr]^2}}
$$

Subtracting the means removes **brightness offset**; dividing by the standard deviations removes **contrast gain**. So NCC is invariant to affine illumination change $I \to aI + b$, which raw SSD is not — that is the entire reason to prefer it. Output is in $[-1,1]$, so the threshold is interpretable across images.

**What it cannot do:** rotation, scale, or perspective. Template matching is exact-appearance matching. If the object rotates, you need a template per rotation (and then you have reinvented a very slow detector). This is precisely the limitation SIFT's canonical orientation solved (1.6).

Still the right tool for: fixed-camera industrial alignment, fiducial and registration-mark finding, screen-content and GUI automation, and stereo block matching (5.6 uses NCC for exactly this reason).

### 🎯 Top-1% distinction

- **The $(\rho,\theta)$ parameterisation exists because $m$ is unbounded** — most candidates recite the algorithm without knowing why the polar form is used.
- **Each point is a sinusoid; collinear points' sinusoids concur.** That is the geometric picture.
- **Hough vs RANSAC** — exhaustive voting vs random sampling; many instances vs one instance; low- vs high-dimensional models.
- **NCC's mean-subtraction and variance-normalisation buy invariance to $aI+b$** — say the transformation, not just "it's normalised."

---

## A1.4 Classical & Interactive Segmentation

Before FCNs, segmentation was an optimisation over pixels. These methods are still the right answer when you have almost no labels, and two of them are load-bearing context for why the deep methods look the way they do.

### Watershed

Treat the image (usually a **gradient** image) as a topographic surface and flood it from **markers**. Water rising from different markers meets at ridges; those ridges are the boundaries.

**Unseeded watershed massively over-segments** — every local minimum becomes a basin. The method is only useful in its **marker-controlled** form: supply markers (from the distance transform's peaks, from user clicks, or from a coarse model's confident regions) and flood only from those.

**Its niche is splitting touching convex objects** — cells, coins, pills, grains — which is exactly the problem that morphology alone cannot solve (A1.2).

### Graph cuts and GrabCut

Model the image as a graph: pixels are nodes, plus a source (foreground) and sink (background) terminal. Minimise an energy

$$
E(\mathbf{L}) = \underbrace{\sum_p D_p(L_p)}_{\text{data: how well does pixel } p \text{ fit its label's colour model}} + \lambda \underbrace{\sum_{(p,q)\in\mathcal{N}} V_{pq}(L_p, L_q)}_{\text{smoothness: penalise label changes, less so across strong edges}}
$$

For two labels this is exactly a **min-cut / max-flow** problem and is solved **globally optimally in polynomial time** — a genuinely unusual guarantee in vision, and the reason graph cuts mattered so much.

**GrabCut** makes it interactive and iterative: the user drags a box; everything outside is definite background; Gaussian mixture models are fitted to the inside/outside colour distributions; min-cut is solved; the GMMs are refitted from the new labels; repeat. A few strokes correct the result. `cv2.grabCut`.

**When to reach for it in 2026:** you need a handful of high-quality masks *right now* with no model and no training — annotation bootstrapping, one-off asset cutouts, or generating seed labels to train something. It is also the honest baseline that a learned interactive segmenter (SAM, 6.8) must beat.

### Mean-shift and SLIC superpixels

**Mean-shift** is a mode-seeking algorithm: each point iteratively moves to the mean of its neighbours within a kernel bandwidth, and points converging to the same mode form a cluster. For segmentation, points live in joint colour+position space. **No need to specify the number of clusters** — the bandwidth determines it implicitly, which is both its appeal and its difficulty.

**SLIC** (Simple Linear Iterative Clustering) produces **superpixels**: k-means in a 5-D space of $(L, a, b, x, y)$ with a compactness parameter trading colour homogeneity against spatial regularity, and a search restricted to a local window (which is what makes it linear rather than quadratic).

**Why superpixels still matter:** they reduce an image from $10^6$ pixels to $10^3$ primitives that respect boundaries, which makes downstream graph methods, CRFs, and weakly-supervised labelling tractable. They are also a useful unit for **explainability** — LIME's image variant perturbs superpixels, not pixels.

### Conditional Random Fields — and why DeepLab looked the way it did

A CRF over pixel labels adds a pairwise term encouraging **nearby pixels with similar colour to share a label**. The **dense CRF** (Krähenbühl & Koltun, 2011) makes this tractable over *all* pixel pairs with Gaussian edge potentials and mean-field inference in a permutohedral lattice.

**The historical role is the point.** FCN and early DeepLab produced blobby, boundary-imprecise output because they predicted at stride 8–16 and upsampled. A dense CRF applied as a **post-process** snapped those predictions to image edges and was worth several mIoU — so from 2015 to about 2018, "FCN + dense CRF" was the standard segmentation pipeline, and DeepLab v1/v2 shipped with it.

**Then it disappeared**, because atrous convolution, better decoders, and higher-resolution training fixed the underlying resolution problem rather than patching it afterwards. **Knowing that arc — a post-hoc fix that became unnecessary once the architecture improved — is a good, specific piece of history**, and it explains a design (DeepLab's) that otherwise looks arbitrary.

### 🎯 Top-1% distinction

1. **Watershed is only useful marker-controlled** — unseeded, it over-segments hopelessly.
2. **Graph cuts give a global optimum for two labels via min-cut** — a rare guarantee, and the reason the method was important.
3. **GrabCut's loop is GMM ↔ min-cut**, alternating, seeded by one box.
4. **SLIC's compactness parameter is the colour-vs-spatial trade**, and superpixels are what make graph methods tractable.
5. **The dense-CRF story explains DeepLab** and illustrates a general pattern: post-processing patches that vanish when the architecture is fixed properly. The same pattern will repeat with NMS (4.7 → NMS-free detectors).

### ✅ Mastery check

You must produce 200 high-quality object masks to bootstrap a segmentation dataset. You have no labels and one afternoon.

(a) Rank three approaches by expected masks-per-hour, and name the failure mode of each.
(b) Why is a dense CRF unlikely to help a modern segmentation model's output the way it helped FCN's?
(c) Your interactive tool produces masks that leak into the background wherever the object and background are similar colours. Which term of the graph-cut energy is failing, and what would you change?

<details><summary>Answer sketch</summary>
(a) 1. <b>SAM with box prompts</b> (6.8) — fastest by far: one box per object, near-instant, and the image embedding is cached across prompts. Failure mode: ambiguous prompts on nested objects (does the box mean the shirt or the person?), and it has <b>no semantics</b>, so you still supply the class. 2. <b>GrabCut</b> — one drag plus a few correction strokes, maybe 20–40 masks/hour. Failure mode: it is purely a <b>colour-model</b> method, so it fails whenever foreground and background share colours, and on thin structures and fine boundaries (hair, wires). 3. <b>Polygon annotation by hand</b> — the baseline, ~10–20 masks/hour for complex shapes, and the failure mode is your own consistency and fatigue rather than the algorithm. The honest 2026 answer is SAM for the first pass, GrabCut or manual correction for the ones SAM gets wrong, and <b>always a human verification pass</b>, because unverified auto-labels propagate their errors into everything trained on them.
(b) The dense CRF's contribution was <b>recovering boundary precision that the architecture had thrown away</b> by predicting at stride 8–16 and bilinearly upsampling. Modern segmenters predict at much higher effective resolution — via atrous convolution, proper decoders with skip connections, or transformer decoders with per-pixel mask embeddings — so the boundary error the CRF was correcting largely isn't there any more. Applying one now mostly costs inference time and can actively hurt by snapping to <i>image</i> edges that aren't <i>object</i> edges (texture, shadows, specular boundaries — the "edges are not object boundaries" point from 1.3).
(c) The <b>data term</b> $D_p$ is failing: it scores how well a pixel fits the foreground versus background colour model, and when the two models overlap it provides almost no discrimination, so the segmentation is decided almost entirely by the smoothness term and leaks across the ambiguous region. Fixes, in order: (i) give the data term more to work with — add <b>texture or deep features</b> to the colour model rather than raw RGB (this is precisely what learned interactive segmenters do); (ii) add <b>user strokes</b> inside the leaking region as hard constraints, which is what GrabCut's interactive loop is for; (iii) raise $\lambda$ to lean harder on edges — but note this is a band-aid that will round off genuine fine structure. The general lesson: when a segmentation energy fails, identify <i>which term</i> lost its signal before tuning weights.
</details>

---

## A1.5 Frequency, Resampling, and the Bug That Costs You a Model

### The Fourier view, used rather than stated

The convolution theorem, $\mathcal{F}\{w*I\} = \mathcal{F}\{w\}\cdot\mathcal{F}\{I\}$, is stated in 1.2. Here is what it buys you:

- **Blurring is low-pass, sharpening is high-pass.** A Gaussian's transform is a Gaussian (strictly positive, no side lobes ⇒ no ringing); a box filter's is a **sinc** (side lobes ⇒ ringing and contrast inversion on fine texture). That is the rigorous version of "why not a box blur."
- **Unsharp masking** is $I + \alpha(I - G_\sigma * I)$ — the original plus a scaled high-pass residual. Every "sharpen" slider is this.
- **Periodic noise** (scanner banding, sensor pattern noise, moiré) appears as isolated peaks in the spectrum and is removed by notch filtering — something no spatial filter does cleanly.

### Resampling kernels, and the resize bug

Downsampling **must** be preceded by low-pass filtering or high frequencies alias into low ones (1.4). The interpolation kernel determines quality:

| Kernel | Support | Character |
|---|---|---|
| **Nearest** | 1 px | blocky; the only correct choice for label masks and index images |
| **Bilinear** | 2×2 | fast, slightly soft; the default almost everywhere |
| **Bicubic** | 4×4 | sharper, mild overshoot at edges (can exceed the input range) |
| **Lanczos** | 6×6 or 8×8 | windowed sinc, sharpest, most ringing |
| **Area / box** | variable | correct averaging for large downscales — use this, not bilinear, when shrinking a lot |

**The bug.** `cv2.resize`, `PIL.Image.resize`, `torchvision.transforms.Resize`, and `tf.image.resize` **do not agree**. They differ in default kernel, in whether they apply an anti-aliasing prefilter when downscaling, and in pixel-centre convention. If your training pipeline resizes with one and your inference service resizes with another, you have introduced a **train/serve skew** that shifts every input slightly out of distribution — and it presents as "accuracy is 2% lower in production and nobody knows why."

**This is one of the most common real production bugs in computer vision.** The fix is trivial and the discipline is what matters: **pin the resize implementation and its parameters, share the exact preprocessing code between training and serving, and write a test that asserts byte-identical output for a fixed input.** Note also that `torchvision`'s `Resize` gained an `antialias` argument whose default changed between versions — a version bump alone can silently move your input distribution.

### 🎯 Top-1% distinction

- **Name the resize-mismatch bug unprompted** when asked "what goes wrong between dev and prod?" — it is specific, common, and demonstrates production experience rather than reading.
- **Nearest-neighbour is mandatory for label masks.** Bilinearly interpolating a segmentation mask invents class indices that don't exist. People do this constantly.
- **Use `INTER_AREA` for large downscales** — bilinear over a 4× reduction samples sparsely and aliases.
- **Bicubic overshoots**, so it can produce values outside $[0,255]$ or $[0,1]$; clamp, or your normalisation is subtly wrong.

### ✅ Mastery check

Your classifier scores 94% offline and 91% in production. Preprocessing is "resize to 224 and normalise" in both.

(a) Give three specific ways that sentence can hide a discrepancy.
(b) How would you prove it is preprocessing rather than data drift, in one experiment?
(c) You resize a segmentation label mask from $512^2$ to $256^2$ with bilinear interpolation. What exactly goes wrong?

<details><summary>Answer sketch</summary>
(a) 1. <b>Different library or kernel</b> — training with PIL bicubic (which anti-aliases on downscale) and serving with `cv2.resize` default bilinear (which does not) produces measurably different images, softer in one and aliased in the other. 2. <b>Different resize semantics</b> — resize-shortest-side-then-centre-crop versus resize-to-square (which changes aspect ratio), or a different crop ratio. The object's scale in the frame differs, and after RandomResizedCrop training the model is sensitive to exactly that (2.9, FixRes). 3. <b>Different normalisation or channel order</b> — BGR vs RGB (OpenCV vs PIL), normalising before vs after conversion to float, or a different mean/std. Also: JPEG re-encoding in the serving path adding compression artefacts the model never saw.
(b) <b>Run the production preprocessing code on the offline evaluation set</b> and re-score. If accuracy drops to ~91%, it is preprocessing; if it stays at 94%, the images themselves differ and it is drift. This isolates one variable and takes ten minutes — and the stronger version is to assert byte-identical tensors: push the same source image through both paths and compare with `np.allclose`, which tells you not just <i>that</i> they differ but <i>by how much and where</i>.
(c) Bilinear interpolation <b>averages neighbouring label indices</b>, which are categorical, not numeric. Averaging class 3 and class 7 yields 5 — a class that was never present at that location, and possibly a class that means something entirely unrelated. You silently corrupt the boundaries of every object and introduce phantom thin regions of wrong classes along every edge. Always use <b>nearest-neighbour</b> for masks, index images, and any categorical raster. (If you need soft downsampling of masks — for a loss at lower resolution — one-hot encode first, resize each channel, then argmax; averaging in one-hot space is meaningful, averaging indices is not.)
</details>

---

## Appendix — Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Histogram equalisation | apply the image's own CDF — it's the probability-integral transform; global, and amplifies noise |
| CLAHE | per-tile equalisation with a **clip limit** (the clip is what stops noise amplification) + bilinear blending between tiles |
| Otsu | maximise between-class variance ≡ minimise within-class variance; **assumes a bimodal histogram** |
| Adaptive threshold | per-neighbourhood threshold — required under uneven illumination |
| Erosion/dilation | shrink/grow by a structuring element |
| **Opening vs closing** | opening removes small objects **at original size**; closing fills small holes. Erosion shrinks *everything* |
| Top-hat | image − opening ⇒ bright detail smaller than the SE; the uneven-illumination fix |
| Connected components | 4- vs 8-connectivity **changes your count**; `connectedComponentsWithStats` gives area/bbox/centroid |
| Distance transform | distance to nearest background; peaks = object centres = watershed markers |
| The pipeline | CLAHE → adaptive threshold → open → close → connected components → filter by area/shape |
| Hough | $\rho = x\cos\theta + y\sin\theta$ (polar because $m$ is unbounded); each point is a sinusoid, collinear points concur; robust to gaps and occlusion |
| Hough vs RANSAC | exhaustive voting vs random sampling; many instances vs one; low- vs high-dimensional models |
| NCC | mean-subtraction removes offset, variance-normalisation removes gain ⇒ invariant to $aI+b$; no rotation or scale invariance |
| Watershed | flood a gradient surface from **markers**; unseeded it over-segments; splits touching convex objects |
| Graph cuts | data term + smoothness term; two labels ⇒ **globally optimal via min-cut** |
| GrabCut | alternate GMM colour models ↔ min-cut, seeded by one box; fails when fg and bg share colours |
| SLIC | k-means in $(L,a,b,x,y)$ with a compactness parameter; superpixels make graph methods tractable |
| Dense CRF | the FCN-era boundary fix; **disappeared once decoders and atrous conv fixed resolution properly** |
| Resampling | nearest for **masks**, area for large downscales, bicubic overshoots, Lanczos rings |
| **The resize bug** | cv2 / PIL / torchvision disagree on kernel, anti-aliasing and conventions ⇒ **train/serve skew**; pin and share the preprocessing code |

### 🔨 Appendix build task

**One sitting, no GPU, no training data.** Write `count_objects.py` implementing the full classical pipeline (CLAHE → adaptive threshold → opening → closing → distance transform → marker-controlled watershed → connected components → shape filtering) and run it on a photograph of coins, pills, or seeds on a plain background. Report the count and overlay the labelled components.

Then run the ablation that teaches the lesson: remove each stage in turn and record the count error. You will discover that on real images the pipeline degrades gracefully without CLAHE, badly without adaptive thresholding, and catastrophically without watershed as soon as two objects touch — and you will have a concrete, defensible answer to "how would you count objects without a neural network?"

**Read:** Szeliski §3.1 (point operators), §3.3 (morphology), §4.3 (line detection). Otsu (1979) — four pages. Rother, Kolmogorov & Blake, "GrabCut" (SIGGRAPH 2004). Achanta et al., "SLIC Superpixels" (PAMI 2012). Krähenbühl & Koltun, "Efficient Inference in Fully Connected CRFs" (NeurIPS 2011).
