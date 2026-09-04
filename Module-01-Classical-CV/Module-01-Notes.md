# Module 1 — Classical CV Foundations

> **Prerequisite calibration:** You already know deep learning. Nothing here re-teaches gradients-as-in-backprop. "Gradient" in this module means *image gradient* — a spatial derivative of pixel intensity. The entire module is about what an image *is* geometrically and photometrically, and how to find repeatable structure in it without learning anything.
>
> **Why a DL-fluent person still must own this module:** every modern vision system that touches 3D — SLAM, AR, robotics, autonomous driving, photogrammetry, 3D Gaussian Splatting pipelines — still runs classical geometry at its core. Interviewers use Module 1 + Module 5 as the *filter* question set precisely because most candidates skipped them.

**Concept map for this module**

```
1.1 Image formation ──> 1.2 Filtering ──> 1.3 Edges ──┐
                                │                     │
                                └──> 1.4 Scale-space ─┼──> 1.5 Harris ──> 1.6 SIFT/ORB
                                                      │        │              │
                                                      └────────┘              v
                                                                       1.7 Matching
                                                                              │
                                                                              v
                                                                       1.8 RANSAC ──> 1.9 Homography ──> 1.10 Synthesis
```

**Priority legend:** 🔴 Critical (load-bearing, full depth) · 🟡 Important · 🟢 Enrichment · *(untagged)* = core syllabus.

---

## 1.1 Image Representation & Formation 🟡

### Intuition

Start with a question you have probably never had to ask, because in deep learning the image is just a tensor that arrives: **where did those numbers actually come from, and what physical quantity is each one measuring?** The answer decides which arithmetic on them is meaningful.

A digital image is not "a picture." It is a **grid of measurements of how much light landed on a tiny bucket during a fixed time window**. Everything that feels weird about computer vision — why shadows break your colour threshold, why brightening an image ruins your blending, why the same object looks like a different colour indoors — traces back to that sentence.

Real-life anchor: your phone's camera does not store red, green and blue at every pixel. Each physical photosite measures *one* colour behind a filter, and the other two are guessed by an interpolation algorithm. The "12 megapixel RGB image" you get back is roughly 4 MP of real measurement and 8 MP of educated guessing.

### The formation chain

We can trace the whole path from photons to `uint8`, and every stage on it is a place where information is either destroyed or distorted. Watch for those two, because they are what will bite you later.

$$
\text{Scene radiance } L \;\xrightarrow{\text{lens, aperture}}\; \text{Sensor irradiance } E \;\xrightarrow{\text{integrate over } \Delta t,\ \lambda}\; \text{charge} \;\xrightarrow{\text{ADC}}\; \text{raw} \;\xrightarrow{\text{ISP}}\; I(x,y)
$$

For one photosite with spectral sensitivity $s_c(\lambda)$ for channel $c$:

$$
I_c(x,y) \;=\; \gamma\!\left( \int_{\lambda} E(x,y,\lambda)\, s_c(\lambda)\, d\lambda \cdot \Delta t \right)
$$

Three things in that equation cause 90% of practical pain:

1. **The integral over $\lambda$ destroys information.** Two physically different spectra can produce identical $(R,G,B)$ — *metamerism*. Colour is a 3-D projection of an infinite-dimensional signal. This is why "match this exact paint colour from a photo" is an ill-posed problem.
2. **$\Delta t$ and aperture are yours to choose**, which is why exposure differences between two shots of the same scene are the norm, not the exception → why panorama stitching needs gain compensation (1.9).
3. **$\gamma(\cdot)$ is a non-linearity.** This is the one people get wrong.

### Gamma: the trap

Sensor output is (approximately) **linear in photons**. The 8-bit image you load with `cv2.imread` is **not**. It is sRGB-encoded, roughly

$$
I_{\text{sRGB}} \approx L_{\text{linear}}^{1/2.2}
$$

(the true sRGB curve is a linear segment near black plus $1.055\,L^{1/2.4}-0.055$ above it, but $1/2.2$ is the working approximation).

Why it exists: human brightness perception is roughly a power law (Stevens), so encoding with $\approx 1/2.2$ spends 8 bits where the eye can see differences, avoiding banding in shadows. It is a *perceptual compression*, not a display artefact.

**Now work out the consequence yourself rather than being told it.** Averaging, blurring, downsampling and alpha-blending are all *linear* operations — they compute weighted sums. A weighted sum is only physically meaningful on a quantity that adds physically, and what adds physically is photons, i.e. **linear** light. Applying a linear operator to $L^{1/2.2}$ does not give you the encoding of the linear average, because $f(\tfrac{a+b}{2}) \neq \tfrac{f(a)+f(b)}{2}$ for any non-linear $f$. Jensen's inequality even tells you the *direction* of the error: $x^{1/2.2}$ is concave, so averaging encoded values undershoots the encoding of the average. Blurring sRGB values darkens the result.

**Pause:** you average an sRGB `0` and an sRGB `255`. You get `128`. Is that the correct half-brightness pixel?

No. Half the *light* is $L = 0.5$, and its sRGB encoding is $255 \times 0.5^{1/2.2} \approx 186$ — noticeably brighter than 128. The 58-level gap is the whole bug. This is exactly why naive image resizing makes thin bright lines on dark backgrounds fade away: the bright line's photons get averaged in the wrong space and lose most of their weight.

### Colour spaces

| Space | Structure | Good for | Fails at |
|---|---|---|---|
| **RGB** | device additive primaries; channels highly correlated | display, CNN input | illumination changes shift all 3 channels together |
| **HSV / HSL** | cylinder: Hue (angle), Saturation (radius), Value (axis) | colour thresholding, shadow tolerance | Hue undefined at $S{=}0$, unstable at low $S$; Hue is *circular* |
| **LAB (CIELAB)** | $L^*$ lightness + $a^*$ (green–red), $b^*$ (blue–yellow) opponent axes | perceptual distance, colour-based clustering, colour transfer | needs a white-point assumption; more compute |
| **YCbCr** | luma + 2 chroma | JPEG / video codecs, 4:2:0 chroma subsampling | not perceptually uniform |

**RGB → grayscale.** Not a mean. Rec.601 luma: $Y = 0.299R + 0.587G + 0.114B$; Rec.709 (HD): $Y = 0.2126R + 0.7152G + 0.0722B$. Green dominates because the human photopic luminosity function peaks near 555 nm and the eye has ~2× more L/M cones than S cones. OpenCV's `COLOR_BGR2GRAY` uses the 601 weights.

**RGB → HSV.** The formula below looks arbitrary until you see what it is trying to do. Picture the RGB cube and look down its main diagonal, the grey axis from black to white. Every colour projects onto a hexagon. *How far along the grey axis* you are is brightness ($V$), *how far out from the axis* you are is colourfulness ($S$), and *which way round* you are is the hue ($H$). So all three quantities are read off the max, the min, and their difference — because the max fixes how bright the brightest primary is, and the spread $C = M-m$ measures how far the colour departs from grey. With $M = \max(R,G,B)$, $m = \min$, $C = M - m$:

$$
V = M, \qquad S = \begin{cases}0 & M = 0\\ C/M & \text{else}\end{cases}, \qquad
H = 60^\circ \times \begin{cases}
\frac{G-B}{C} \bmod 6 & M = R\\
\frac{B-R}{C} + 2 & M = G\\
\frac{R-G}{C} + 4 & M = B
\end{cases}
$$

The three cases just say "measure the angle starting from whichever primary is currently winning," and the $60°$ multiplier turns the hexagon's six sides into $360°$.

Note $C$ in the denominator: as a pixel approaches gray, $C \to 0$ and $H$ becomes numerically explosive then undefined — which is geometrically obvious once you have the picture, because a point *on* the grey axis has no direction to point in. Any hue-based mask must be gated on a minimum saturation.

**RGB → LAB** goes through CIE XYZ:

$$
\begin{bmatrix}X\\Y\\Z\end{bmatrix} = M_{\text{sRGB}\to\text{XYZ}} \begin{bmatrix}R_{\text{lin}}\\G_{\text{lin}}\\B_{\text{lin}}\end{bmatrix},\qquad
L^* = 116\,f(Y/Y_n) - 16,\quad a^* = 500\,[f(X/X_n) - f(Y/Y_n)]
$$

with $f(t) = t^{1/3}$ for $t > (6/29)^3$ and linear below. **You must linearise (undo gamma) before the matrix multiply** — this is the single most common bug in colour-space code. The reason is the same one as above: the sRGB→XYZ matrix is a *linear* map between physical light quantities, so feeding it gamma-encoded numbers means applying a linear operator to a non-linear function of the thing it was defined on.

The cube root is the point: it makes Euclidean distance in LAB approximate *perceived* colour difference ($\Delta E_{76} = \|\Delta \mathbf{Lab}\|_2$, with $\Delta E \approx 2.3$ being the just-noticeable difference). That is why k-means colour quantisation, colour transfer, and superpixel algorithms (SLIC) all operate in LAB rather than RGB.

### Sensor-level facts worth knowing

- **Bayer mosaic (RGGB)**: 50% green, 25% each R and B. Demosaicing is interpolation → introduces correlated noise and colour fringing at edges. Raw-domain CV (astronomy, scientific imaging) skips it.
- **Noise model**: photon shot noise is Poisson (variance ∝ signal) plus Gaussian read noise. The Poisson part is not an empirical curiosity — photon arrivals are a counting process, and a Poisson count with mean $\mu$ has variance $\mu$, so bright pixels are *intrinsically* noisier in absolute terms than dark ones. Add the sensor's read noise, which is signal-independent, and you get $\sigma^2 = a\mu + b$. This is why denoising strength should be *signal-dependent* and why a flat Gaussian-noise assumption underperforms — and it's the physical justification for variance-stabilising transforms (Anscombe).
- **Rolling shutter**: CMOS sensors expose rows sequentially → fast motion skews. Breaks the pinhole assumption of Module 5 and is a real failure source in drone/phone SfM.

### Connect it

Everything downstream assumes $I(x,y)$ is a well-behaved scalar field you can differentiate (1.2, 1.3). Gamma and colour space determine whether "intensity difference" means anything physical. In Module 2, this returns as *normalisation choice* for CNN inputs (ImageNet mean/std are computed on gamma-encoded sRGB — the network learns around the non-linearity). In Module 5, the pinhole model picks up exactly where this section ends.

**In your own words:** why is averaging two pixels not the same as averaging the light they represent?

### 🎯 Top-1% distinction

A surface answer lists "RGB, HSV, LAB." A strong answer says:

1. **"HSV is not illumination-invariant, it is illumination-*decoupled* — and only for intensity scaling, not for coloured illuminants."** A tungsten bulb shifts hue itself; only chromatic adaptation / white balance fixes that.
2. **Hue is an angle**, so mean, variance, Euclidean distance and Gaussian blur are all *wrong* on the H channel. You need circular statistics ($\bar{\theta} = \operatorname{atan2}(\overline{\sin\theta}, \overline{\cos\theta})$) or you must operate on $(\cos H, \sin H)$.
3. **Gamma**: blending/resizing must happen in linear light. Being the candidate who says "I'd linearise before averaging" is a distinct signal.
4. **Metamerism** — colour constancy is ill-posed, which is why white balance is a *prior*, not a computation.

### ✅ Mastery check

You threshold a green ball in HSV and it works indoors but fails under a bright window where part of the ball is in direct sun and part in shadow. Both regions are the same physical green. Explain precisely, in terms of the H/S/V equations above, *which* of the three channels shift in each region and why a pure H-range mask still fails at the specular highlight.

<details><summary>Answer sketch</summary>
Shadow: $V$ drops (scaling all of $R,G,B$ by the same factor leaves $H$ and $S$ approximately invariant — this is the illumination *decoupling* property, and a V-agnostic mask survives). Highlight: specular reflection adds an achromatic (white) component to all channels, which raises $m=\min$, so $C = M-m$ shrinks → **$S$ collapses toward 0 and $H$ becomes numerically unstable/undefined**. So the mask fails at the highlight not because hue changed but because hue stopped being defined. Fix: gate on $S > S_{\min}$ and treat low-S pixels separately, or use the dichromatic reflection model to remove the specular component.
</details>

### 🔨 Build + read

**Build (one sitting):** Write `linear_vs_srgb_blend.py`. Load an image, downsample 4× two ways — (a) directly on uint8 sRGB, (b) linearise → downsample → re-encode. Diff the results and show the error map. Then implement RGB→HSV and RGB→LAB from scratch (no `cv2.cvtColor`) and validate against OpenCV to <1 LSB.

**Read:** Szeliski, *Computer Vision: Algorithms and Applications* 2nd ed., §2.2–2.3 (photometric image formation) and §3.1. Then Shree Nayar, "First Principles of Computer Vision" — the Imaging / Radiometry playlist.

---

## 1.2 Filtering & Convolution

### Intuition

A single pixel value tells you almost nothing — noise alone can move it by several levels. Everything meaningful in an image is a *relationship between a pixel and its neighbours*. So the first question is: **what is the simplest possible way to make a pixel depend on its neighbourhood?**

The answer is a weighted sum. A filter slides a small grid of weights over the image and replaces each pixel with a weighted sum of its neighbours. Blur = "average with your neighbours." Sharpen = "exaggerate how different you are from your neighbours." That is the entire idea; the rest is which weights and why.

You already know this operation intimately from the other direction — a CNN conv layer is exactly this, with the weights learned instead of designed. This section is about what happens when you have to *choose* them, and it will tell you what a network's first layer is spending its capacity rediscovering.

Real-life anchor: the "portrait mode" bokeh on your phone is a spatially varying blur whose kernel radius is driven by an estimated depth map — a classical filter steered by a learned model.

### The math

**Correlation vs convolution.** Correlation:

$$
(w \star I)(x,y) = \sum_{i=-k}^{k}\sum_{j=-k}^{k} w(i,j)\, I(x+i,\, y+j)
$$

Convolution flips the kernel:

$$
(w * I)(x,y) = \sum_{i}\sum_{j} w(i,j)\, I(x-i,\, y-j)
$$

They coincide for symmetric kernels (Gaussian, box, Laplacian) and differ for asymmetric ones (Sobel — sign flips). **What "convolution" layers in CNNs actually compute is correlation.** Nobody cares because the weights are learned, but say it out loud in an interview and you sound like someone who has read the maths.

**Properties that matter:**

- Commutative, associative, distributive over $+$, shift-equivariant. Associativity is what lets you pre-compose two filters into one: $(G_1 * G_2) * I = G_1 * (G_2 * I)$.
- **Linear + shift-invariant (LSI)** ⇒ fully characterised by its impulse response ⇒ multiplication in the Fourier domain: $\mathcal{F}\{w*I\} = \mathcal{F}\{w\}\cdot\mathcal{F}\{I\}$.

**Separability.** Here is the trick that makes large filters affordable, and it is pure linear algebra you already own. A $k\times k$ kernel is a $k\times k$ *matrix*. If that matrix has **rank 1**, then by definition it factors as an outer product of two vectors, $W = \mathbf{u}\mathbf{v}^\top$ — and an outer product means "the 2-D weight at $(i,j)$ is just $u_i$ times $v_j$." Substitute that into the convolution sum and the double sum factors into two single sums, one along rows and one along columns:

$$
W * I = \mathbf{u} * (\mathbf{v}^\top * I)
$$

Cost drops from $O(k^2)$ to $O(2k)$ per pixel — for $k=61$ that is 3721 multiply-adds down to 122, a 30× saving with *no approximation whatsoever*. Test for separability: take the SVD of $W$; rank-1 is exactly the statement that only one singular value is non-zero, so if only $\sigma_1 \neq 0$, it is separable and $\mathbf{u} = \sqrt{\sigma_1}\,u_1$, $\mathbf{v} = \sqrt{\sigma_1}\,v_1$ (splitting the singular value between the two factors keeps their scales balanced). The 2-D Gaussian is separable for the reason you can read straight off the exponent — a sum in the exponent is a product outside it: $e^{-(x^2+y^2)/2\sigma^2} = e^{-x^2/2\sigma^2}\cdot e^{-y^2/2\sigma^2}$. Sobel is separable: $[1,2,1]^\top [-1,0,1]$.

**The Gaussian, precisely.**

$$
G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$

Practical facts you should have memorised:

- Truncate at $\pm 3\sigma$ (99.7% of mass) → kernel size $= 2\lceil 3\sigma \rceil + 1$. OpenCV's default relation is $\sigma \approx 0.3\left((k-1)/2 - 1\right) + 0.8$.
- Always renormalise the discrete kernel to sum to 1, or you shift image brightness.
- **Semigroup / cascade property:**
  $$G_{\sigma_1} * G_{\sigma_2} = G_{\sqrt{\sigma_1^2 + \sigma_2^2}}$$
  **Pause:** blur an image with $\sigma=3$, then blur the result with $\sigma=4$. What single blur would have done the same job — $\sigma=7$?

  No: $\sigma = \sqrt{3^2+4^2} = 5$. **Variances add, standard deviations do not.** And you already know why, from probability rather than from image processing: convolving two densities is the density of the *sum* of two independent random variables, and variances of independent variables add. A Gaussian blur is exactly "displace each photon by an independent Gaussian random offset," so doing it twice displaces by the sum of two independent Gaussians, which is Gaussian with variance $\sigma_1^2+\sigma_2^2$. Blurring twice with $\sigma$ therefore equals blurring once with $\sigma\sqrt{2}$, not $2\sigma$.

  Read the identity backwards and it becomes a construction recipe: if you already have a $\sigma_1$-blurred image and you want $\sigma_2 > \sigma_1$, you need only convolve with $\sqrt{\sigma_2^2 - \sigma_1^2}$ — a much smaller kernel than $\sigma_2$. **This is the single most useful identity in the module** — it is how Gaussian pyramids are built incrementally (1.4) and how SIFT avoids re-blurring from the original image at every level.

**Frequency view.** Convolution in space is multiplication in frequency, so the kernel's transform *is* the filter's gain at each frequency — that is the cleanest way to see what a blur does. $\mathcal{F}\{G_\sigma\} = e^{-2\pi^2\sigma^2 \|f\|^2}$: a Gaussian is its own transform, strictly positive, monotonically decreasing, no side lobes ⇒ a *clean* low-pass that attenuates every frequency and inverts none.

You might expect a box blur to behave the same way, just less smoothly. It does not. A box filter transforms to a **sinc**, and a sinc goes *negative* in its side lobes. A negative gain at some frequency means that frequency comes out of the filter with its sign flipped — the filter can literally **invert contrast** on fine texture, and the oscillating lobes produce visible **ringing** near edges. That is the real answer to "why not just use a box blur": it is not that it looks worse, it is that it manufactures structure the image did not contain. (Hold that thought — in 1.4 the same defect reappears as a violation of the scale-space causality axiom.)

**Non-linear filters** (not expressible as convolution). Linearity is what made all the above tractable, so why give it up? Because linearity is also exactly what forces a blur to smear across an edge: a weighted sum cannot ask "is this neighbour even the same surface as me?" Every filter below buys edge awareness by paying with linearity — and therefore loses the Fourier view, the separability trick and the semigroup property along with it.

- **Median** — optimal for salt-and-pepper (impulse) noise; preserves step edges exactly because a step's median is still a step. A Gaussian smears it.
- **Bilateral filter** — edge-preserving:
  $$
  I'(p) = \frac{1}{W_p}\sum_{q \in \Omega} \underbrace{G_{\sigma_s}(\|p-q\|)}_{\text{spatial}} \cdot \underbrace{G_{\sigma_r}(|I(p)-I(q)|)}_{\text{range}} \cdot I(q)
  $$
  Read the two Gaussians as two separate questions asked of every neighbour $q$: *are you near me in space?* and *are you similar to me in intensity?* A neighbour must answer yes to both to get a vote. The range term therefore kills contributions from pixels across an edge, because those differ in intensity by more than $\sigma_r$ no matter how close they are spatially — which is precisely the question a linear filter cannot ask. $W_p$ is just the sum of the weights, renormalising so brightness is preserved. Cost: $O(k^2)$ non-separable naively; accelerated by the bilateral grid / permutohedral lattice. **Failure mode: gradient reversal and cartoon-like banding** at high $\sigma_r$.
- **Guided filter** — $O(1)$ per pixel, linear model of output w.r.t. a guide image; used for matting, dehazing, and as a fast bilateral substitute.

**Boundary handling** — `zero` (dark halo at borders), `replicate`, `reflect`, `reflect_101` (OpenCV default, does not duplicate the edge pixel), `wrap`. Choosing `zero` and then doing edge detection produces a fake frame around your image. Trivial, and people still ship it.

### Connect it

Convolution is the substrate for edges (1.3) and scale (1.4), and it is literally the operation Module 2 replaces with *learned* weights. The separability trick reappears as $1{\times}k$ / $k{\times}1$ factorised convolutions in Inception-v3 and as depthwise-separable convolutions in MobileNet (2.7) — same rank-1 factorisation argument, applied to the channel dimension instead of the spatial one. The frequency view reappears when we discuss aliasing in downsampling (1.4) and stride in CNNs (2.3).

**In your own words:** why do two blurs of $\sigma=3$ and $\sigma=4$ compose into $\sigma=5$ rather than $\sigma=7$?

### 🎯 Top-1% distinction

- **Why Gaussian rather than box:** not "it looks smoother" — it is (a) separable, (b) strictly positive in the Fourier domain so no ringing, (c) the *unique* kernel satisfying the scale-space axioms (proved in 1.4), and (d) closed under composition (semigroup).
- **Cost accounting:** naive $k\times k$ is $O(k^2)$; separable is $O(2k)$; FFT-based is $O(\log N)$ per pixel amortised and only wins for very large $k$ (roughly $k \gtrsim 30$). Knowing the crossover is a systems-flavoured signal.
- **The bilateral filter is not a projection** — iterating it does not converge to a piecewise-constant image in a well-behaved way, which is why "just apply it 5 times" produces artefacts.

### ✅ Mastery check

You need to blur an image with $\sigma = 10$. Naively that is a $61\times61$ kernel = 3721 multiply-adds per pixel. Give **three** different ways to get the same result substantially cheaper, and state the exact cost of each. Then state which one you would use if you also need $\sigma = 2, 4, 6, 8$ of the same image.

<details><summary>Answer sketch</summary>
(1) <b>Separable</b>: two 1-D passes of length 61 → 122 MACs/pixel, ~30× cheaper, mathematically exact.
(2) <b>Repeated box filter</b>: 3–4 passes of a box filter (each $O(1)$/pixel via running sums / integral image) converges to a Gaussian by the CLT; ~4–8 ops/pixel, approximate. This is what real-time blurs use.
(3) <b>Downsample → blur small → upsample</b>: blur at 1/4 resolution with $\sigma=2.5$, then upsample; cost drops ~16×. Valid because Gaussian blur commutes approximately with scaling.
For the multi-$\sigma$ case: use the <b>semigroup property</b> and build them incrementally — get $\sigma=2$, then convolve with $\sigma=\sqrt{4^2-2^2}=3.46$ to reach 4, etc. Total cost is far below computing each from scratch. This is exactly the Gaussian-pyramid construction in 1.4.
</details>

### 🔨 Build + read

**Build:** Implement `my_conv2d(img, kernel, border)` with all four border modes. Then implement `separable_check(K)` using SVD that returns $(\mathbf{u},\mathbf{v})$ or `None`. Verify empirically that 4 successive box-blur passes approximate a Gaussian (plot the 1-D profiles), and time all three approaches from the mastery check.

**Read:** Szeliski §3.2 (linear filtering) and §3.3.1 (bilateral). Then the "Fast Bilateral Filtering" paper (Paris & Durand, ECCV 2006) for the grid trick.

---

## 1.3 Edge Detection (Gradients, Sobel, Canny)

### Intuition

You want to find object boundaries, but the image contains no objects — only numbers. **What measurable property of the numbers marks a boundary?** Brightness usually changes sharply there: the table is one value, the wall another. So to find it, look at how fast intensity changes as you step sideways: that is the derivative. Places where the derivative is large are edge candidates.

The rest of this section is the consequences of that one decision. Derivatives amplify noise, so we must smooth; smoothing moves edges, so we must choose how much; and the derivative is large over a *band* of pixels, not one, so we must thin. Canny is just those three fixes assembled in order.

Real-life anchor: lane-departure warning in a car was, for a decade, Canny + Hough transform. It works because painted lane markings are high-contrast, roughly straight, and geometrically constrained — the exact regime classical edge detection owns.

### The math

**Image gradient:**

$$
\nabla I = \begin{bmatrix} I_x \\ I_y \end{bmatrix},\qquad
\|\nabla I\| = \sqrt{I_x^2 + I_y^2},\qquad
\theta = \operatorname{atan2}(I_y, I_x)
$$

$\theta$ points **across** the edge (perpendicular to it), toward increasing intensity.

**Discrete derivatives.** Central difference $\tfrac{1}{2}[-1, 0, 1]$ is 2nd-order accurate (the first-order error terms cancel by symmetry, unlike the forward difference $[-1,1]$) but it is noise-amplifying. The Fourier view from 1.2 tells you exactly how badly: differentiating multiplies the spectrum by $j\omega$, so a frequency ten times higher is boosted ten times harder. Noise lives at the highest frequencies in the image, signal usually does not — so a raw derivative preferentially amplifies precisely the component you do not want. Hence you must smooth first, and the smoothing is not a cosmetic step, it is what makes the derivative computable at all.

**Sobel** fuses both into one separable kernel:

$$
S_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} = \begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}-1&0&1\end{bmatrix}
$$

i.e. **smooth vertically with a binomial kernel, differentiate horizontally**. $[1,2,1]$ is the 3-tap binomial (Pascal) approximation to a Gaussian. Sobel is therefore a crude derivative-of-Gaussian. **Scharr** $[3,10,3]$ is the optimised 3-tap variant with better rotational symmetry — use it when you actually care about gradient *orientation* accuracy (e.g. feeding an orientation histogram).

**The identity that saves you a pass.** We now need "smooth, then differentiate," which sounds like two passes over the image. It is one. Convolution and differentiation commute — and the reason is one line: $\frac{\partial}{\partial x}\int G(x-u)I(u)\,du$ differentiates under the integral, and only $G$ depends on $x$, so the derivative lands entirely on the kernel:

$$
\frac{\partial}{\partial x}(G_\sigma * I) = \left(\frac{\partial G_\sigma}{\partial x}\right) * I
$$

So you never blur then differentiate in two steps — you convolve once with the analytic derivative-of-Gaussian:

$$
\frac{\partial G_\sigma}{\partial x} = -\frac{x}{\sigma^2}\,G_\sigma(x,y)
$$

(that is just the chain rule on the exponent $-\tfrac{x^2+y^2}{2\sigma^2}$, whose $x$-derivative is $-x/\sigma^2$).

This matters because it means the *only* free parameter is $\sigma$, and $\sigma$ controls the entire detection/localisation trade-off.

**Canny (1986) — the four stages.** Each stage exists because the previous one left a specific defect; read them that way rather than as a recipe.

1. **Smooth** with $G_\sigma$ (or convolve directly with $\partial G_\sigma$). *Fixes:* noise amplification by differentiation.
2. **Gradient magnitude and orientation** at every pixel. *Leaves:* a thick ridge of large magnitudes, several pixels wide, because a real edge is a ramp and its derivative is non-zero across the whole ramp.
3. **Non-maximum suppression (NMS).** *Fixes stage 2's thick ridge.* Walk along $\theta$; keep pixel $p$ only if $\|\nabla I(p)\|$ exceeds its two neighbours interpolated along $\pm\theta$. This is what thins a fat gradient ridge to a **1-pixel-wide** curve. (Naive implementations quantise $\theta$ to $\{0°,45°,90°,135°\}$; proper ones bilinearly interpolate.)
4. **Hysteresis thresholding.** *Fixes the impossible single threshold.* Any one threshold has to be both high enough to reject noise and low enough to keep a faint real edge, and on a real image no such value exists — a genuine contour dips below the noise level somewhere along its length. Hysteresis escapes the dilemma by using evidence a single threshold cannot see: **connectivity**. Two thresholds $T_{\text{high}} > T_{\text{low}}$. Pixels above $T_{\text{high}}$ are seeds (confident); pixels above $T_{\text{low}}$ are kept **only if connected** to a seed (flood fill / union-find) — i.e. a weak pixel is believed when it is part of a chain that is strong somewhere. A common ratio is $T_{\text{high}} : T_{\text{low}} = 2{:}1$ to $3{:}1$.

**Canny's optimality.** Canny posed edge detection as an explicit optimisation over three criteria for a 1-D step edge in white Gaussian noise:

- **Good detection** — maximise SNR (low miss + low false-alarm rate),
- **Good localisation** — minimise the expected distance between the detected and true edge position,
- **Single response** — one detection per true edge.

Notice the first two criteria are in direct conflict, which is why an optimum exists at all: a wide filter averages more pixels and so has better SNR (detection), but averaging over a wide support smears the edge's position (localisation). Canny made this precise — the SNR term grows like $\sqrt{\text{width}}$ and the localisation term degrades like $1/\sqrt{\text{width}}$, so their *product* is width-independent and the shape of the filter, not its width, is what you optimise over. Maximising that product subject to the third criterion (which penalises multiple responses and pins the width down) yields an optimal filter that is well approximated (within ~20% on the criteria product) by the **first derivative of a Gaussian**.

We are not deriving the variational solution here — it is a calculus-of-variations problem with a fourth-order Euler–Lagrange equation, and §II of the 1986 paper does it properly. What you should carry away is that derivative-of-Gaussian is not a convention or an accident of convenience: it is the near-optimal solution to a stated optimisation problem, and that is the answer to "why derivative-of-Gaussian and not something else."

**Second-derivative view: Laplacian of Gaussian (LoG).** There is a second way to say "peak of the first derivative": a maximum of $|f'|$ is a zero of $f''$. Trading a peak-finding problem for a zero-finding problem is attractive because zeros are easier to locate to sub-pixel precision (interpolate to where a sign change happens) and require no threshold to define.

$$
\nabla^2 G_\sigma(x,y) = \frac{1}{\pi\sigma^4}\left(\frac{x^2+y^2}{2\sigma^2} - 1\right)\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$

Edges are **zero crossings** of $\nabla^2(G*I)$ with sufficient gradient magnitude (Marr–Hildreth). Zero crossings always form closed contours — an aesthetic advantage and an accuracy disadvantage (they hallucinate closure). LoG is the direct bridge into 1.4.

**Trade-off in $\sigma$:** small $\sigma$ → precise localisation, noisy, fragmented. Large $\sigma$ → clean, but edges *move* (curved edges shift toward the centre of curvature, by roughly $\sigma^2/2R$ for radius $R$) and nearby edges merge.

**Pause:** before reading on — you have an image containing both a fine wire mesh and a large low-contrast building outline, and you must detect both. What $\sigma$ do you pick?

There isn't one. The mesh needs a small $\sigma$ or it is blurred away; the building outline needs a large $\sigma$ or it fragments into noise. The two structures have different *sizes*, and $\sigma$ is a size. No single value can serve both, and no amount of threshold tuning helps, because threshold controls contrast and the problem is scale. That deadlock — not any refinement of Canny — is exactly the motivation for scale-space in 1.4: stop choosing, compute all of them.

### Connect it

NMS here is the direct ancestor of NMS in object detection (4.7) — same idea (suppress non-peak responses along a dimension), different dimension: here it is position across the edge, there it is overlapping boxes. The derivative-of-Gaussian is the ancestor of the oriented-gradient histograms in SIFT (1.6) and HOG (4.2). And the fact that a *single* $\sigma$ can never be right is precisely what 1.4 fixes.

**In your own words:** why does Canny need *two* thresholds rather than one carefully chosen one?

### 🎯 Top-1% distinction

- **"Edges are not object boundaries."** Texture (a zebra, a brick wall) produces enormous gradients with no semantic boundary; a real object boundary can have near-zero gradient (a white cup on a white table). This asymmetry — high recall on texture, low recall on low-contrast boundaries — is *why* the field moved to learned boundary detection (HED, 2015) and then to semantic segmentation (4.10). Being able to name that transition is a strong answer.
- **Hysteresis is the interesting part**, not the thresholding. A single threshold either fragments contours (too high) or floods with noise (too low). Hysteresis exploits *spatial coherence*: real edges are connected. It is a mini-CRF.
- **NMS is why Canny output is thin.** Candidates who omit NMS from the pipeline description are visibly reciting.
- **Sobel is a derivative-of-Gaussian in disguise** — $[1,2,1]$ is binomial ≈ Gaussian. State that and the whole family (Prewitt = box smoothing, Scharr = optimised) snaps into one picture.

### ✅ Mastery check

Two parts.

**(a)** You run Canny on an image of a chain-link fence in front of a building. You get thousands of tiny edge fragments and lose the building outline. You increase $\sigma$. Describe *quantitatively* what happens to (i) the fence, (ii) the building corners, (iii) the localisation error of a curved edge of radius $R$.

**(b)** Why can't you fix (a) by just raising $T_{\text{high}}$?

<details><summary>Answer sketch</summary>
(a)(i) The fence is a high-frequency periodic texture; the Gaussian's transfer function $e^{-2\pi^2\sigma^2 f^2}$ attenuates it exponentially in $\sigma^2$, so the fence vanishes rapidly. (ii) Corners are the superposition of two edges; as $\sigma$ grows past the corner scale the two edges merge and the corner is <b>rounded and displaced inward</b> — this is the corner-shrinkage effect. (iii) For a curved edge of radius $R$, Gaussian smoothing displaces it toward the centre of curvature by approximately $\sigma^2/(2R)$ — quadratic in $\sigma$. So doubling $\sigma$ quadruples the localisation error.
(b) Raising $T_{\text{high}}$ is a <b>contrast</b> filter, not a <b>scale</b> filter. A high-contrast fence survives any threshold that a low-contrast building outline survives; in fact raising the threshold kills the building outline first. The two problems live on different axes — contrast vs. spatial frequency — and you need the right tool per axis. This is the argument for multi-scale analysis.
</details>

### 🔨 Build + read

**Build:** Implement Canny end-to-end yourself — DoG convolution, sub-pixel-interpolated NMS (not the 4-direction quantised version), and hysteresis via union-find. Compare against `cv2.Canny` on 5 images. Then sweep $\sigma \in \{1,2,4,8\}$ and produce a figure showing edge count vs. $\sigma$ and the corner displacement you predicted above.

**Read:** Canny, "A Computational Approach to Edge Detection" (PAMI 1986) — read §II (the three criteria and the variational derivation), it is genuinely readable. Then Szeliski §7.2.

---

## 1.4 Scale-Space Theory & Gaussian/DoG Pyramids 🔴

> **This is the load-bearing concept of Module 1.** SIFT, blob detection, image pyramids, FPN in Module 4, and the entire notion of "multi-scale" in vision descend from here. The gap analysis flagged it as critical because without it SIFT's scale invariance is a black box.

### Intuition

1.3 ended in a deadlock: you must choose a $\sigma$, and no choice is right for an image containing structures of different sizes. **What if you refuse to choose?**

Take that seriously and it stops being a dodge. An object in an image has no intrinsic size. A face is 20 pixels wide in a crowd shot and 2000 pixels wide in a portrait. A "corner" viewed up close is a smooth curve; a "blob" viewed from far away is a single pixel. So the question "is there a feature at $(x,y)$?" is malformed — it presumes a size you were never given. The right question is **"is there a feature at $(x, y, \sigma)$?"** — location *and* scale.

Scale-space is the idea that you should not pick one blur level. You build the entire continuum of blurred versions of the image, treat blur level as a third coordinate, and find structures that are extremal in that 3-D volume. The scale at which a structure is extremal *is* that structure's size — the parameter you could not choose becomes an *output* of the detector instead of an input to it. That inversion is the whole idea, and it is worth stating to yourself twice, because everything else in this section is machinery for making it work.

This is the same move as a CNN's feature hierarchy, and it is worth holding the analogy from the start: a deep network handles scale by stacking layers with growing receptive fields, discovering a rough version of this structure from data. Here we are constructing it deliberately, which means we get to say exactly what it guarantees — and later, when FPN (4.4) turns out to be a Laplacian pyramid, you will recognise it rather than memorise it.

Real-life anchor: reverse image search must match a product photo against a thumbnail 8× smaller. It cannot know the zoom factor in advance. Scale-space is how it finds the same keypoints in both.

### The formal object

The **linear scale-space** of $I$ is

$$
L(x, y; t) = G_{\sqrt{t}}(x,y) * I(x,y), \qquad t = \sigma^2 \ge 0
$$

with $L(x,y;0) = I(x,y)$. The parameter $t$ is called *scale* (it has units of area); $\sigma$ is the standard deviation in pixels.

**Why the Gaussian, uniquely (Koenderink 1984, Lindeberg 1994).** We used a Gaussian above without justification. You might reasonably ask why not a box, a triangle, a disc average — anything that blurs. The striking answer is that the Gaussian is not *a* choice, it is *the* choice: write down the properties any sensible notion of "coarser version of this image" must have, and only one kernel survives. Demand these axioms of a scale-space generator:

1. **Linearity** — $\mathcal{T}_t(aI_1 + bI_2) = a\mathcal{T}_t I_1 + b \mathcal{T}_t I_2$.
2. **Shift invariance** — commutes with translation.
3. **Semigroup** — $\mathcal{T}_{t_1} \circ \mathcal{T}_{t_2} = \mathcal{T}_{t_1 + t_2}$ (coarsening is progressive; you can build coarse from fine).
4. **Isotropy / rotational symmetry** — no preferred direction.
5. **Causality / non-enhancement of local extrema** — *no new structure may be created as scale increases*. Formally, at a local maximum $\partial L/\partial t \le 0$ and at a local minimum $\partial L / \partial t \ge 0$ — a peak may only get shorter and a trough only shallower as you coarsen. In 1-D this is equivalent to: the number of zero crossings never increases with $t$.

Axioms 1–4 are the ones you would have guessed. Axiom 5 is the one that does the real work, and it is worth spelling out *why you would demand it*. The entire premise of scale-space is that a bump you find at coarse scale corresponds to something that was actually there in the image. If coarsening could *invent* a local maximum, then a detection at large $\sigma$ would be evidence of nothing — an artefact of your own smoothing, indistinguishable from a real structure. Axiom 5 is the guarantee that every structure you see at a coarse scale can be traced back down to structure that existed at a finer scale.

**The Gaussian is the unique kernel satisfying all five.** This is not a convention; it is a theorem, and we are taking it on trust here — Lindeberg (1994) proves it, and the proof is a functional-equation argument, not something you would reconstruct at a whiteboard. What you *can* reconstruct is why the obvious alternative fails. Recall from 1.2 that a box filter has negative side lobes in the Fourier domain; a negative gain flips the sign of a frequency component, and a flipped component can produce a bump where the image had none. That is precisely a violation of axiom 5, which is why box-filter pyramids hallucinate features — the two facts are the same fact, seen from the frequency and the spatial side.

Equivalently: $L$ satisfies the **heat / diffusion equation**

$$
\frac{\partial L}{\partial t} = \tfrac{1}{2}\nabla^2 L
$$

Blurring an image *is* letting heat diffuse through it, with $t$ playing the role of time. (Some texts drop the $\tfrac12$ by defining $t=\sigma^2/2$; conventions differ, the structure does not.)

### Deriving DoG ≈ scale-normalised Laplacian — the derivation to know cold

**Set up the goal before the algebra.** We want a blob detector. The natural one is the Laplacian: $\nabla^2 L$ is large in magnitude at the centre of a bright-on-dark or dark-on-bright region, and near zero on flat areas and along straight edges. But the LoG kernel is expensive — it is not separable, and we would have to convolve with a fresh one at every scale. Meanwhile, we are *already going to compute* a stack of Gaussian-blurred images, because that is what scale-space is. **Question: can the Laplacian be read off that stack for free?** The derivation below says yes, and it is the reason SIFT is affordable at all.

**Step 1 — differentiate the heat equation, and notice what it is telling you.** $L$ satisfies $\partial L/\partial t = \tfrac12\nabla^2 L$. Because $L = G*I$ and everything is linear in $I$, the same equation holds for the kernel alone:

$$
\frac{\partial G}{\partial t} = \tfrac{1}{2}\nabla^2 G
$$

Stop and read that as an English sentence, because it is the whole trick in one line: **the rate at which the blurred image changes as you coarsen it is (half) its Laplacian.** The Laplacian is not something we have to go and compute separately — it is already sitting there as the *derivative of the blur stack along the scale axis*. Everything that follows is bookkeeping on that observation.

**Step 2 — change variables from $t$ to $\sigma$.** We index our pyramid by $\sigma$ (standard deviation in pixels), not by $t = \sigma^2$, so we need the derivative with respect to the variable we actually step along. Chain rule: $\dfrac{\partial}{\partial \sigma} = \dfrac{dt}{d\sigma}\dfrac{\partial}{\partial t} = 2\sigma \dfrac{\partial}{\partial t}$. Applying it:

$$
\boxed{\ \frac{\partial G}{\partial \sigma} = 2\sigma \cdot \tfrac12 \nabla^2 G = \sigma \nabla^2 G\ }
$$

The factor $\sigma$ on the right is not decoration and it is not an artefact of the change of variables — it will turn into the mandatory scale normalisation in a moment. Notice already that it is there *because* variance, not standard deviation, is the natural diffusion time.

**Step 3 — we cannot take that derivative, so approximate it.** $\partial G/\partial\sigma$ is a derivative along a continuous axis, but our pyramid is a *discrete* stack of blur levels $\sigma, k\sigma, k^2\sigma,\dots$. The only thing you can ever do with a derivative on a discrete grid is replace it with a finite difference between two adjacent samples — the same substitution you make when you replace $\partial I/\partial x$ with $I(x+1)-I(x)$. Here the two adjacent samples are the scales $\sigma$ and $k\sigma$, a step of $k\sigma - \sigma$ apart:

$$
\sigma\nabla^2 G = \frac{\partial G}{\partial \sigma} \approx \frac{G(x,y;k\sigma) - G(x,y;\sigma)}{k\sigma - \sigma}
$$

The numerator is a **difference of two Gaussians** — two images you already have. That is where "DoG" comes from: it is not an invention, it is the finite-difference stencil for $\partial G/\partial\sigma$.

**Step 4 — clear the denominator.** Multiply both sides by $(k-1)\sigma$:

$$
\boxed{\ G(x,y;k\sigma) - G(x,y;\sigma) \;\approx\; (k-1)\,\sigma^2\,\nabla^2 G\ }
$$

**Read what this says, term by term.** On the left: subtract two adjacent blur levels, which costs one subtraction. On the right: $\sigma^2\nabla^2 G$, the **$\sigma^2$-scale-normalised Laplacian of Gaussian** — the thing we actually wanted — multiplied by a leftover constant $(k-1)$.

**Pause:** that stray $(k-1)$ means DoG is not *equal* to the normalised Laplacian, only proportional to it. Doesn't a wrong scale factor ruin the detector?

**No — and understanding why is the crux of the whole derivation.** We are not going to *report* the DoG value; we are going to look for its **extrema in $(x,y,\sigma)$**. Multiplying a function by a positive constant moves no maximum and no minimum: $\arg\max_x c\,f(x) = \arg\max_x f(x)$ for any $c>0$. And $(k-1)$ is a *constant across all scales*, because SIFT fixes a single ratio $k = 2^{1/s}$ for the entire pyramid. It is one global gain on the whole 3-D volume. Had $k$ varied with $\sigma$, the factor would have been a scale-dependent reweighting and would have shifted extrema along the $\sigma$ axis — the detector's scale estimates would be biased. It does not, so it doesn't. Finding extrema of DoG is therefore *equivalent* to finding extrema of $\sigma^2\nabla^2 G * I$: you get the scale-normalised Laplacian detector at the price of a subtraction.

Three payoffs from this one derivation:

1. **DoG is cheap.** You need the blurred images anyway to build the pyramid; DoG is a free subtraction. LoG requires a separate non-separable convolution.
2. **The $\sigma^2$ factor is not decoration — it is mandatory**, and you can see why in one line of dimensional reasoning. Differentiating divides by a length, and the only length in a Gaussian is $\sigma$: each derivative of $G_\sigma$ therefore carries a factor $1/\sigma$, so the $n$-th derivative's amplitude decays as $\sigma^{-n}$. (Concretely, $\partial G/\partial x = -\tfrac{x}{\sigma^2}G$ from 1.3, and $G$ itself has peak height $1/2\pi\sigma^2$ — the more you differentiate, the flatter the response at large $\sigma$.) Now imagine comparing responses *across* scales without correcting for this: the raw Laplacian is systematically larger at small $\sigma$ regardless of what is in the image, so the scale axis has a built-in downhill slope and every extremum slides to the finest scale. Scale selection collapses to "always the smallest," i.e. to no scale selection at all. Multiplying by $\sigma^n$ exactly cancels the decay and puts all scales on a comparable footing. This is Lindeberg's **$\gamma$-normalised derivative**: $\partial_{\xi^n} = \sigma^{n\gamma}\partial_{x^n}$, with $\gamma = 1$ giving $\sigma^2\nabla^2$ for the Laplacian ($n=2$).
3. It explains *why SIFT looks the way it does* — SIFT is "find extrema of the scale-normalised Laplacian, computed via DoG."

### Automatic scale selection and the blob-size relation

**Lindeberg's principle:** the characteristic scale of a structure is the $\sigma$ at which a $\gamma$-normalised differential operator attains a local extremum over scale. This is *covariant*: if you scale the image by $s$, the detected $\sigma$ scales by $s$ too — which is precisely what buys you scale invariance, and note it is the normalisation of the previous paragraph that makes the statement non-vacuous, since without it the extremum is always at the bottom.

**Worked case — the ideal 2-D disk.** Fine, the detector reports a $\sigma$. What does that number *mean* in pixels of actual object? Let us compute it for the one case we can do exactly: a binary disk of radius $R$ on a flat background. The answer will be

$$
\sigma = \frac{R}{\sqrt{2}}
$$

and here is where it comes from, in three steps.

*First, what is the response?* Convolution at the disk's centre is an integral of the kernel against the image, and the image is an indicator function — 1 inside radius $R$, 0 outside. So the response is simply the integral of the LoG kernel over a disk of radius $R$:

$$
F(R) \;=\; \int_0^{R} k(r)\,2\pi r\,dr, \qquad k(r) \;\propto\; \left(\frac{r^2}{2\sigma^2}-1\right)e^{-r^2/2\sigma^2}
$$

(the $2\pi r\,dr$ is just the area element of an annulus — we can integrate radially because both disk and kernel are rotationally symmetric).

*Second, find the extremum.* Differentiate under the integral sign with respect to $R$ — by the fundamental theorem of calculus, $F'(R) = 2\pi R\,k(R)$. Setting this to zero, and discarding the useless root $R=0$, the condition is exactly

$$
k(R) = 0 \quad\Longrightarrow\quad \frac{R^2}{2\sigma^2} - 1 = 0 \quad\Longrightarrow\quad R = \sqrt{2}\,\sigma
$$

So the extremum sits precisely where **the disk's edge coincides with the kernel's zero crossing** — which makes sense once you see it: growing the disk past that radius starts adding kernel weight of the *opposite* sign, so the response stops growing and turns over. Under- or over-shoot the radius and you are either leaving centre-lobe signal on the table or cancelling it against the surround.

*Third, note this is the relation you want.* We derived it by varying $R$ at fixed $\sigma$, whereas the detector varies $\sigma$ at fixed $R$. Those give the same curve: the normalised response depends on $R$ and $\sigma$ only through their ratio (scale both by $s$ and nothing changes — that is the covariance property above), so the stationary condition is a single relation between them either way.

(In $n$ dimensions the relation generalises to $\sigma = R/\sqrt{n}$ — the $n$ appears because the Laplacian sums $n$ second derivatives; the 1-D step/ridge case gives $\sigma = R$.)

**So detected $\sigma$ tells you object size**: a blob detected at $\sigma$ has radius $\approx \sqrt{2}\sigma$. This is how you draw correctly-sized circles in a blob-detection demo, and it is a question interviewers ask to see if you understand or memorised — the tell is whether you can say *why* $\sqrt2$ rather than just quoting it.

### Pyramids

A continuum of blur levels is a lovely idea and an unaffordable one — at full resolution, every level costs a full-image convolution, and the large-$\sigma$ levels are the most expensive. But there is an obvious redundancy to exploit: an image blurred with $\sigma = 8$ has no detail finer than about 8 pixels, so storing it at full resolution is keeping samples that carry no information. Discard them. That is the pyramid.

**Gaussian pyramid.** Repeatedly (a) blur, (b) decimate by 2. The blur is not optional, and the reason is the sampling theorem rather than aesthetics: decimating by 2 halves your sampling rate, so any content above the new Nyquist frequency cannot be represented — and it does not politely disappear, it **aliases**, folding back and masquerading as a *low* frequency. That is the moiré you see on downsampled brick walls, and note it is a violation of the causality axiom by the back door: aliasing creates structure that was not there. Blurring first removes the offending frequencies before they can fold. The correct pre-filter for a factor-2 decimation is $\sigma \approx 0.8$–$1.0$ (the classic Burt–Adelson 5-tap $[1,4,6,4,1]/16$).

**Laplacian pyramid.** A Gaussian pyramid is lossy and redundant at once — each level throws away detail, and neighbouring levels share most of their content. What if instead of storing each level, you store only *what each level added*? That is $L_i = G_i - \text{expand}(G_{i+1})$ — the band-pass residual at each level (level $i$ minus the upsampled coarser level, i.e. exactly the frequencies present at $i$ but not at $i+1$), plus the smallest Gaussian level kept whole so you have somewhere to start rebuilding from. Note this is the same DoG subtraction as above, now used to *represent* the image rather than to detect blobs. It is **invertible** (exact reconstruction by summing back up), which is why it is the classical tool for **multi-band blending** in panoramas (1.9): blend low frequencies over a wide transition band and high frequencies over a narrow one, so seams disappear without ghosting.

**SIFT's exact pyramid bookkeeping** — every number below is forced by a requirement, so derive them rather than memorising them:

- An **octave** is a doubling of $\sigma$, and it is the natural unit because scale is multiplicative, not additive — "twice as big" is the meaningful step, so we sample the scale axis geometrically. You want $s$ scale samples per octave, so $s$ equal *multiplicative* steps must compose to a factor 2: $k^s = 2$, hence $k = 2^{1/s}$. SIFT uses $s = 3$.
- **Why more images than samples.** To call a point a 3-D extremum you must compare it against neighbours in every direction *including along $\sigma$*, so each tested DoG image needs a DoG image above and below it. The top and bottom DoG images of an octave have no such neighbour on one side and therefore cannot be tested. To get $s$ testable images you must build $s+2$ DoG images per octave. And since each DoG is a difference of two adjacent blurred images, $n$ DoG images need $n+1$ blurred ones: $s+3$ blurred images per octave.
- With $s=3$: **6 blurred images, 5 DoG images, 3 of which are tested** for extrema (each compared against 26 neighbours: 8 in-plane + 9 above + 9 below — the full $3\times3\times3$ cube minus the centre).
- The last blurred image of an octave has $\sigma = 2\sigma_0$; **downsample it by 2** and it becomes the first image of the next octave with the correct $\sigma_0$ — no re-blurring from the original. Why that works: halving the resolution halves the pixel-measured $\sigma$, so a $2\sigma_0$ image at half size *is* a $\sigma_0$ image, exactly what the next octave needs to start from. This is the semigroup property (1.2) doing real work: the whole pyramid is built by small incremental blurs, never by a large blur of the original.
- SIFT assumes the input already has $\sigma_{\text{camera}} \approx 0.5$ and pre-blurs to $\sigma_0 = 1.6$. You cannot simply convolve with $1.6$ — the image is *already* blurred by the camera, and blurs compose in quadrature, so you must subtract the existing variance rather than the existing $\sigma$. Hence the actual first convolution uses $\sqrt{1.6^2 - (2\times0.5)^2}$, the $2\times$ accounting for the initial 2× upsampling (which also doubles the camera's effective $\sigma$ in the new pixel units).

### Beyond linear scale-space 🟢

Linear (Gaussian) scale-space blurs *across* edges, so it degrades localisation. **Anisotropic diffusion** (Perona–Malik) replaces $\partial L/\partial t = \nabla^2 L$ with $\partial L / \partial t = \operatorname{div}(c(\|\nabla L\|)\nabla L)$, where $c$ decreases with gradient magnitude — diffuse along edges, not across them. This is what **KAZE / AKAZE** features use instead of a Gaussian pyramid, and it is why AKAZE often beats ORB on blurred or low-texture imagery. Worth one sentence in an interview; it shows you know the Gaussian's cost, not just its benefit.

### Connect it

- 1.3 asked "which $\sigma$?" — this answers "all of them, and pick per-structure."
- 1.5 (Harris) is scale-*variant*; combining Harris (spatial localisation) with LoG (scale selection) yields **Harris–Laplace**.
- 1.6 (SIFT) is DoG extrema + descriptor. You now understand its first stage completely.
- **Module 2/4:** CNNs face the identical problem and solve it with a *learned* pyramid — strided convolutions build a feature hierarchy, and **FPN (4.4)** explicitly reintroduces the Laplacian-pyramid idea (top-down pathway + lateral connections) because plain CNN pyramids lose fine detail at deep levels. When you get to FPN, this section is the thing that makes it obvious rather than arbitrary.
- **Module 6:** ViT patchification throws multi-scale away entirely, which is why hierarchical ViTs (Swin) reintroduce a pyramid. The problem never dies.

**In your own words:** why does subtracting two blurred images give you a Laplacian, and why doesn't the leftover $(k-1)$ matter?

### 🎯 Top-1% distinction

Five things that separate a strong answer:

1. **Derive $\partial G/\partial\sigma = \sigma\nabla^2 G$ and get DoG $\approx (k-1)\sigma^2\nabla^2 G$ on the spot.** Most candidates say "DoG approximates LoG" as a memorised fact. Deriving it — and explaining that $(k-1)$ being scale-constant is *why* it doesn't perturb extrema — is the differentiator.
2. **Explain why $\sigma^2$ normalisation is required**, not optional: derivative magnitudes decay as $\sigma^{-n}$, so without $\gamma$-normalisation every extremum collapses to the finest scale and scale selection is impossible.
3. **State the uniqueness theorem and the causality axiom.** "The Gaussian is the only kernel that doesn't create new extrema as scale increases" is a sentence that immediately marks you.
4. **Know the pyramid arithmetic** ($s+3$ images for $s$ scales, $k=2^{1/s}$, reuse the $2\sigma_0$ image as the next octave's base). This is the difference between having read about SIFT and having implemented it.
5. **Know the blob-size relation $R = \sqrt{2}\sigma$** and be able to explain it from the LoG zero crossing.

**The trap question:** *"Is scale-space scale-invariant?"* No. The scale-space representation itself is scale-*covariant* — scaling the image scales the detected $\sigma$. Invariance is achieved only when you **select** extrema jointly in $(x,y,\sigma)$ and then *normalise* the descriptor patch by the detected $\sigma$. The representation is covariant; the pipeline is invariant. Candidates who conflate the two get caught.

### ✅ Mastery check

You are building a SIFT-like detector with $s = 4$ scales per octave over 5 octaves on a $1024\times1024$ image.

**(a)** How many Gaussian-blurred images and how many DoG images do you compute per octave, and why exactly those numbers?
**(b)** What is $k$?
**(c)** You detect an extremum at octave 2 (i.e. the image has been downsampled twice) at scale index 1 within the octave, with base $\sigma_0 = 1.6$. What is the effective $\sigma$ **in original-image pixels**, and what is the radius of the blob it corresponds to?
**(d)** A colleague removes the $\sigma^2$ factor from the response normalisation to "simplify the code." Predict exactly what breaks and why.

<details><summary>Answer sketch</summary>
(a) $s+3 = 7$ blurred images, $s+2 = 6$ DoG images, of which $s = 4$ are tested for extrema. Reason: an extremum test needs a DoG neighbour above and below, so the top and bottom DoG images can't be tested; and $n$ DoG images need $n+1$ blurred images.
(b) $k = 2^{1/4} \approx 1.189$.
(c) Within-octave scale: $\sigma_{\text{oct}} = \sigma_0 k^{1} = 1.6 \times 1.189 = 1.902$. Octave 2 means the image is downsampled by $2^2 = 4$, so effective $\sigma$ in original pixels $= 1.902 \times 4 = 7.61$. Blob radius $\approx \sqrt{2}\sigma = 10.8$ px.
(d) Every response magnitude now decays like $\sigma^{-2}$ across scale, so the DoG volume is monotonically decreasing in $\sigma$. All scale-space extrema collapse to the finest scale, scale selection returns the same tiny $\sigma$ everywhere, and the detector loses scale invariance entirely — it degenerates into a noisy fine-scale blob detector. Matching across a 2× zoom drops to near chance.
</details>

### 🔨 Build + read

**Build:** Implement a Gaussian pyramid and DoG pyramid from scratch with correct octave bookkeeping ($s+3$ rule, reuse of the $2\sigma_0$ image). Then implement scale-normalised LoG blob detection: find $(x,y,\sigma)$ extrema, and **draw circles of radius $\sqrt{2}\sigma$** over a synthetic image of disks with known radii. Verify your detected radii match ground truth within a few percent — this is a genuinely satisfying test that proves you got the normalisation right. Extend: build a Laplacian pyramid and verify perfect reconstruction (max abs error should be ~0).

**Read:** Lindeberg, "Scale-space theory: A basic tool for analysing structures at different scales" (1994) — read the sections on $\gamma$-normalised derivatives and automatic scale selection. Then Lowe's SIFT paper (IJCV 2004) §3, which is the applied version of exactly this. Optional but excellent: Shree Nayar's First Principles episodes on scale-space.

---

## 1.5 Harris Corners

### Intuition

Scale-space told us *at what size* to look. It did not tell us **which points are worth looking at at all.** What makes a point in an image a good one to remember and later re-find in a different photograph?

The answer is not "high contrast" or "on an edge" — it is **uniqueness under displacement**. Slide a small window around the image. Over flat wall it looks the same wherever you move it, so if you found this patch in another photo you could not say *where* it was. Along an edge it looks the same if you slide *along* the edge, different if you slide across — so you could pin down one coordinate but not the other. At a corner it looks different **in every direction**, so its position is fully determined. A corner is therefore a point of maximum local ambiguity-resolution — the one place you can localise a patch in both $x$ and $y$.

Hold on to that phrasing, because the maths below is nothing more than turning "looks different when I slide it" into a number.

Real-life anchor: this is why a checkerboard is the calibration target of choice (5.3) — its corners are the most precisely localisable structures that exist in an image, down to sub-pixel accuracy.

### The math

**Write down "looks different when I slide it" literally.** Take the window at a point, shift it by $(u,v)$, and measure how much the pixel values disagree — the weighted sum of squared differences:

$$
E(u,v) = \sum_{x,y} w(x,y)\,\bigl[I(x+u, y+v) - I(x,y)\bigr]^2
$$

$w$ is a window function that says which pixels count (we will see shortly that it must be Gaussian). $E(0,0) = 0$ always, and the question "is this a corner?" is now the concrete question **"is $E(u,v)$ large for every direction of $(u,v)$?"** We have converted a vague visual property into a function on the plane. But evaluating $E$ for every possible shift at every pixel is hopeless, so we need a cheap summary of its shape near the origin — and the standard tool for the local shape of a function is a Taylor expansion.

**Step 1 — linearise the image, not the error.** Expand $I$ about $(x,y)$ to first order: $I(x+u,y+v) \approx I(x,y) + u I_x + v I_y$. This is valid for *small* shifts, which is the regime we care about — we are asking about local distinctiveness, not global matching. Substituting, the $I(x,y)$ terms cancel and the bracket collapses to $uI_x + vI_y$:

$$
E(u,v) \approx \sum_{x,y} w(x,y)\,(u I_x + v I_y)^2
$$

**Step 2 — notice the result is exactly second order in $(u,v)$.** We only expanded to *first* order, yet squaring a linear expression gives a quadratic one — that is where the quadratic form comes from, and it is why a "first-order Taylor expansion" produces a second-order model of $E$. Expand the square: $u^2 I_x^2 + 2uv I_xI_y + v^2 I_y^2$. Every term is a product of $u$ or $v$ with $u$ or $v$, so the whole thing is a $\begin{bmatrix}u&v\end{bmatrix}(\cdot)\begin{bmatrix}u\\v\end{bmatrix}$ sandwich waiting to be assembled — the coefficient of $u^2$ goes top-left, of $v^2$ bottom-right, and the $uv$ coefficient splits symmetrically across the off-diagonals:

$$
E(u,v) \approx \begin{bmatrix}u & v\end{bmatrix} M \begin{bmatrix}u\\v\end{bmatrix}
$$

**Step 3 — read off $M$.** Crucially, $(u,v)$ pulls out of the sum entirely — it does not depend on which pixel we are summing over — so all the image-dependent quantities collect into a single $2\times2$ matrix, the **structure tensor / second-moment matrix**:

$$
M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
= \begin{bmatrix} \langle I_x^2\rangle & \langle I_x I_y\rangle \\ \langle I_x I_y\rangle & \langle I_y^2\rangle \end{bmatrix}
$$

That is the payoff: **an entire function of shift direction, compressed into four numbers** (three, by symmetry) per pixel. Note what $M$ *is* — the windowed average of the outer product $\nabla I\,\nabla I^\top$, i.e. an uncentred covariance matrix of the gradient vectors in the window. It is a summary of *how the gradients in this neighbourhood are distributed in direction*, and everything below is reading that distribution.

**Step 4 — the eigenvalues, and why they mean what they mean.** $M$ is symmetric and positive semi-definite (it is a sum of outer products), so it has orthogonal eigenvectors and non-negative eigenvalues. Substituting an eigenvector $\mathbf{e}_i$ as the shift direction gives $E = \lambda_i\|\mathbf{e}_i\|^2$, so **$\lambda_i$ is literally the rate at which the patch changes when you slide it along $\mathbf{e}_i$**. The eigenvectors are the directions of fastest and slowest change; the eigenvalues are how fast. Geometrically, the level sets $E = c$ are ellipses whose axes lie along the eigenvectors with lengths $\propto \lambda_i^{-1/2}$ — a *long* axis means you can slide a long way in that direction before the error reaches $c$, i.e. a small eigenvalue means an uninformative direction.

**Pause:** given that reading, what must the two eigenvalues look like at a point on a straight edge?

One large, one near zero. Along the edge nothing changes (slow direction, tiny $\lambda$); across it everything changes (fast direction, large $\lambda$). All the gradient vectors in the window point the same way, so their outer products sum to a rank-1 matrix. Now the full table falls out with no memorisation:

| Eigenvalues | Structure |
|---|---|
| $\lambda_1 \approx \lambda_2 \approx 0$ | flat region (no gradient in any direction) |
| $\lambda_1 \gg \lambda_2 \approx 0$ | edge (gradient in one direction only) |
| $\lambda_1, \lambda_2$ both large | corner |

**Harris response.** We want "both eigenvalues large," but eigendecomposing a matrix at every pixel of a megapixel image was unthinkable in 1988 and is still wasteful now. So Harris and Stephens asked: **can we test the eigenvalues without computing them?** Yes — because $\det M = \lambda_1\lambda_2$ and $\operatorname{tr} M = \lambda_1 + \lambda_2$ are both readable straight off the matrix entries in a handful of multiplies, and between them they pin down the pair. $\det M$ is already almost the test we want: it is small unless *both* eigenvalues are large. Its problem is that it is also large for a very strong edge with a slightly non-zero $\lambda_2$, so we subtract a penalty that grows when the eigenvalues are lopsided — and $(\operatorname{tr} M)^2$ grows quadratically with the larger eigenvalue while $\det M$ only grows linearly in it. Hence:

$$
R = \det(M) - \kappa\,\bigl(\operatorname{tr} M\bigr)^2 = \lambda_1\lambda_2 - \kappa(\lambda_1+\lambda_2)^2, \qquad \kappa \in [0.04, 0.06]
$$

$R > 0$ and large ⇒ corner; $R < 0$ ⇒ edge (the penalty term wins because $\lambda_1 \gg \lambda_2$); $|R|$ small ⇒ flat (both terms are tiny). Then threshold and apply non-maximum suppression. $\kappa$ has no derivation — it is the knob setting how lopsided is too lopsided, and its arbitrariness is exactly what the variants below remove.

**Variants:**
- **Shi–Tomasi (1994, "Good Features to Track")**: $R = \min(\lambda_1, \lambda_2)$. Requires the eigenvalues but has no magic $\kappa$ and behaves better near the edge/corner boundary. This is what `cv2.goodFeaturesToTrack` implements.
- **Noble / harmonic mean**: $R = \dfrac{\det M}{\operatorname{tr} M + \epsilon}$ — scale-consistent, no $\kappa$.

**$w$ must be Gaussian, not a box.** You might treat the window as a detail — it is only saying which pixels to average over. But think about what a square window *is*: a shape with corners and preferred axes. Rotate the image by 45° and a square window now covers a different set of pixels relative to the underlying structure, so $M$ changes and so does $R$ — a box-windowed Harris response is not rotation-invariant, and the detector fires differently on a rotated copy of the same scene. A Gaussian window is the only weighting with no preferred direction (it is radially symmetric), so it makes the response isotropic and hence rotation-invariant. This is a small detail that separates people who implemented it from people who read about it.

**Invariance audit — memorise this table, it is the template for every detector.** For each transform, the question to ask yourself is mechanical: apply it to $I$, work out what happens to $M$, and see whether the answer changes. Do that for intensity scaling and you get the one genuinely subtle row: $I \to aI$ makes every gradient $a$ times bigger, so every entry of $M$ (a product of two gradients) scales by $a^2$, so $\det M$ scales by $a^4$ and $(\operatorname{tr}M)^2$ likewise — $R$ scales by $a^4$ but the *ranking* of points is untouched. That is why the row says "only up to the threshold": an absolute cut-off is meaningless, a relative one is fine.

| Transform | Harris invariant? | Why |
|---|---|---|
| Rotation | ✅ | Eigenvalues of $M$ are rotation-invariant (the ellipse rotates with the image) |
| Intensity shift $I + b$ | ✅ | Derivatives kill constants |
| Intensity scale $aI$ | ⚠️ only up to the threshold | $M \to a^2 M$, so $R \to a^4 R$ — ordering preserved, absolute threshold not |
| Translation | ✅ | Everything is local |
| **Scale** | ❌ | A corner at scale $s$ looks like a smooth edge at scale $2s$ |
| Affine / viewpoint | ❌ | The gradient ellipse deforms |

**Fixing scale — Harris–Laplace.** Compute Harris responses across a scale-space; keep points that are (a) spatial maxima of the Harris response at some $\sigma$, **and** (b) extrema of the scale-normalised LoG over $\sigma$ at that location. Harris localises in space, LoG selects in scale. This is the clean synthesis of 1.4 and 1.5.

**Fixing viewpoint — Harris-Affine.** Iteratively estimate the second-moment matrix, warp the patch by $M^{-1/2}$ to make the gradient distribution isotropic, recompute, repeat until convergence. This *normalises away* an affine deformation, giving affine-covariant regions (also: MSER, Hessian-Affine). Relevant for wide-baseline matching in 5.x.

**Sub-pixel refinement.** Fit a 2-D quadratic to $R$ in the $3\times3$ neighbourhood of the discrete maximum and solve for the vertex; or use the orthogonality condition $\nabla I^\top (\mathbf{p} - \mathbf{q}) = 0$ (`cv2.cornerSubPix`). Calibration accuracy (5.3) lives or dies on this.

### Connect it

**The structure tensor $M$ is the same matrix that appears in optical flow (5.7).** Lucas–Kanade solves $M \mathbf{v} = -\mathbf{b}$ for the flow vector $\mathbf{v}$; that system is solvable exactly when $M$ is well-conditioned — i.e. exactly at corners. This is why the paper is titled "Good Features to *Track*": Shi & Tomasi derived the corner criterion *from* trackability, not from cornerness. **The aperture problem is the statement that $M$ is rank-1 on an edge.** Being able to say "Harris cornerness and Lucas–Kanade trackability are the same eigenvalue condition" is one of the highest-yield sentences in this entire module.

$M$ also reappears as the "coherency" measure in structure-tensor-based texture analysis and in SIFT's edge-rejection step (1.6), which uses the Hessian's principal-curvature ratio in exactly the same spirit.

**In your own words:** what does each eigenvalue of $M$ physically measure, and why does "both large" mean corner?

### 🎯 Top-1% distinction

1. **Derive $M$ from the Taylor expansion**, don't just present it. The fact that $E(u,v)$ is a quadratic form whose level sets are ellipses is the whole geometric picture.
2. **Say the $\kappa$ trick is about avoiding eigendecomposition** — $\det$ and $\operatorname{tr}$ are the two invariants you can compute with 3 multiplies. Then note Shi–Tomasi's $\min(\lambda_1,\lambda_2)$ is strictly better-behaved but costs the eigenvalues.
3. **Harris is not scale-invariant** and know the fix (Harris–Laplace). Candidates routinely claim SIFT-level invariance for Harris.
4. **The Gaussian-vs-box window rotation-invariance point.**
5. **The Lucas–Kanade connection.** This is the "interviewer sits up" moment.

### ✅ Mastery check

You compute $M$ over a window and find $\lambda_1 = 4000$, $\lambda_2 = 3$.

**(a)** Classify the point and compute $R$ with $\kappa = 0.05$. What sign do you get, and does the sign alone tell you the right answer?
**(b)** Now the image is multiplied by 0.5 (half exposure). What are the new eigenvalues and the new $R$? What breaks?
**(c)** You feed this point to Lucas–Kanade optical flow. What happens to the estimated motion vector, and which component of it is reliable?

<details><summary>Answer sketch</summary>
(a) $\lambda_1 \gg \lambda_2$ ⇒ <b>edge</b>. $R = (4000)(3) - 0.05(4003)^2 = 12000 - 801200 = -789{,}200$. Negative ⇒ edge, so yes the sign is diagnostic here (negative $R$ ⇒ edge; the flat case gives small-magnitude $R$ near zero, so sign alone is insufficient in general — you need the magnitude too).
(b) $M \to 0.25 M$, so $\lambda_i \to 0.25\lambda_i$: $\lambda_1 = 1000$, $\lambda_2 = 0.75$. $R \to 0.25^2 R = -49{,}325$ — i.e. $R$ scales as $a^4$. Relative ordering of points is preserved, but any <b>absolute</b> threshold is now wrong by 16×, so a fixed threshold silently returns far fewer corners on a darker image. Fix: threshold relative to $\max R$ in the image, or normalise the image first.
(c) Rank-deficient $M$ ⇒ the <b>aperture problem</b>. Only the flow component <b>along the gradient direction</b> (perpendicular to the edge, i.e. along $\mathbf{e}_1$) is observable; the component along the edge is in the (near-)null space and is arbitrary. LK will return a vector whose tangential component is dominated by noise, amplified by $1/\lambda_2$. This is exactly why LK is run on Shi–Tomasi corners.
</details>

### 🔨 Build + read

**Build:** Implement Harris from scratch (Sobel gradients → per-pixel outer products → Gaussian-weighted sums → $R$ → threshold → NMS → sub-pixel refine). Then run the **invariance experiment**: rotate an image by 0–90° in 10° steps and measure the *repeatability rate* (fraction of corners re-detected within 2 px of the ground-truth transformed position). Repeat for scaling 1.0–4.0. You should see rotation repeatability stay ~85%+ and scale repeatability collapse past ~1.5×. That plot is worth more than any amount of reading.

**Read:** Harris & Stephens (1988) — 4 pages. Shi & Tomasi, "Good Features to Track" (CVPR 1994) — read the trackability argument. Mikolajczyk & Schmid, "Scale & Affine Invariant Interest Point Detectors" (IJCV 2004) for Harris–Laplace/Harris-Affine.

---

## 1.6 Keypoints & Descriptors: SIFT, ORB 🟡

### Intuition

Harris found you interesting points. Now the harder half: **you have the same physical corner in two photographs taken from different places at different times of day — how do you recognise that they are the same corner?** Comparing raw pixel patches fails immediately, because the two patches differ in scale, rotation and brightness even though the world did not change.

So a detector says *where* to look, and a descriptor says *what it looks like* — a fixed-length vector summarising the patch, built so that the same physical point gives (nearly) the same vector from a different viewpoint, scale, rotation, or lighting. That last clause is a *design specification*, and the way to read SIFT is as a sequence of engineering answers to it: each stage removes one specific nuisance variable. The descriptor is the "embedding" of classical CV; Module 3 is the learned version of this exact idea — and the same specification, met by a loss function instead of by hand.

Real-life anchor: Google Lens, ARKit's plane tracking, and every drone photogrammetry pipeline still run on local descriptors — because you need *correspondence between specific points*, not a global "this image looks like that image."

### SIFT — the full pipeline

David Lowe, IJCV 2004. Five stages — but do not read them as a list of five things SIFT does. Read them as a chain, in which **each stage exists to repair a problem the previous stage created.** Stage 1 gives you candidate points but too many bad ones; stage 2 cleans them up but the patches are still arbitrarily rotated; stage 3 fixes rotation but you still have no vector; stage 4 gives a vector that is still sensitive to lighting; stage 5 fixes lighting. Watch that chain as you go and the design stops looking like folklore.

**The patent expired in March 2020**, so `cv2.SIFT_create()` is now in main OpenCV (it was `xfeatures2d` before 4.4.0). SURF's patent has *not* expired the same way — SURF still lives in `opencv-contrib` behind `OPENCV_ENABLE_NONFREE`, which is the practical install trap the gap analysis flagged.

**Stage 1 — Scale-space extrema detection.** *Problem: where to look, and at what size.* Build the DoG pyramid exactly as in 1.4. A point is a candidate if it is a strict extremum among its **26 neighbours** (8 in-plane, 9 in the DoG above, 9 below). You already own this stage completely — it is 1.4 applied.

*What it leaves broken:* the extremum is located only to the nearest pyramid sample, which in the coarser octaves is several original-image pixels; and comparing against 26 neighbours is a very permissive test, so thousands of the candidates are noise or lie on edges where the position is ill-defined.

**Stage 2 — Keypoint localisation & filtering.** *Fixes both of those.* For the precision problem: the discrete extremum is the largest *sample*, not the true peak, so model the DoG locally as a quadratic — the same second-order Taylor argument as Harris, now in three variables $(x,y,\sigma)$ — and jump to that quadratic's vertex, which is where its own gradient vanishes:

$$
D(\mathbf{x}) \approx D + \frac{\partial D^\top}{\partial \mathbf{x}}\mathbf{x} + \tfrac{1}{2}\mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2}\mathbf{x}, \qquad
\hat{\mathbf{x}} = -\left(\frac{\partial^2 D}{\partial\mathbf{x}^2}\right)^{-1}\frac{\partial D}{\partial \mathbf{x}}
$$

with $\mathbf{x} = (x, y, \sigma)$. (Set the derivative of the quadratic to zero and solve — that is exactly one Newton step, and it is the same formula you would write for any quadratic minimisation.) This gives **sub-pixel and sub-scale** location. If any component of $\hat{\mathbf{x}} > 0.5$ the true peak is closer to a different sample than the one we expanded about, so the quadratic model was fitted in the wrong place: re-centre on the neighbouring sample and repeat.

Then two rejections, one per remaining defect:

- **Low contrast:** discard if $|D(\hat{\mathbf{x}})| < 0.03$ (on $[0,1]$ intensities). A weak extremum is one that noise could plausibly have produced; note we evaluate this at the *refined* location, which is why localisation had to come first. Kills noise-driven extrema.
- **Edge response:** here is the subtle one. DoG responds strongly all along an edge, not just at blobs, and an extremum sitting on an edge is badly localised *in the direction along the edge* — exactly the ambiguity Harris was built to detect, now reappearing in a different detector. So use the same machinery: take the $2\times2$ spatial Hessian $H$ of the DoG, whose eigenvalues $\alpha,\beta$ are the principal curvatures, and note that a blob has comparable curvature in both directions while an edge has one large and one small. We want to test the ratio $r = \alpha/\beta$ **without eigendecomposing** — and we already know the trick from Harris, trace and determinant: 
  $$
  \frac{\operatorname{tr}(H)^2}{\det(H)} = \frac{(\alpha+\beta)^2}{\alpha\beta} = \frac{(r+1)^2}{r}
  $$
  Notice the individual curvatures have vanished — the expression depends only on their *ratio*, which is precisely the scale-free quantity we wanted, and $(r+1)^2/r$ is increasing in $r$ for $r>1$, so thresholding it is the same as thresholding $r$. Discard if this exceeds $(r+1)^2/r$ with $r = 10$. **Same invariant trick as Harris** — trace and determinant instead of eigenvalues. Two detectors, one idea.

*What stage 2 leaves broken:* you now have well-localised, well-behaved keypoints with a scale attached. But rotate the camera and the patch around each keypoint rotates with it, so any vector you compute from raw pixel positions will change.

**Stage 3 — Orientation assignment.** *Fixes rotation, by the standard trick for achieving invariance: find a canonical frame and measure everything relative to it.* If we can attach a repeatable direction to the patch — one determined by the image content, so it rotates with the patch — then describing the patch in that rotated frame makes the description rotation-independent. The natural candidate is "which way do the gradients here mostly point," and the natural estimator is a histogram. In a Gaussian-weighted region (weight $\sigma_w = 1.5\sigma_{\text{kp}}$) around the keypoint, build a **36-bin histogram of gradient orientations**, each vote weighted by gradient magnitude × Gaussian. Take the peak.

**Pause:** what should happen at a patch with two equally strong gradient directions — say a symmetric cross — where the histogram has two nearly-tied peaks?

Picking the taller one is the worst option available, because which of two near-ties wins is decided by noise, so the same physical point gets one orientation in image A and a different one in image B, and the descriptors will not match. Lowe's answer is to refuse to choose: **any other peak within 80% of the maximum spawns a duplicate keypoint** at the same location with that orientation. (This is why SIFT sometimes returns ~1.15× as many keypoints as it detected locations — and it materially improves matching on rotationally ambiguous patches.) Parabola-interpolate the peak bin for sub-bin orientation accuracy.

*What stage 3 leaves broken:* location, scale and orientation are all pinned down, and we still have no vector.

**Stage 4 — Descriptor.** *Finally, the vector.* The obvious option — flatten the raw pixel patch — is hopeless: a one-pixel error in localisation, or a small viewpoint warp, changes every entry. What we want is a summary that is **specific enough to distinguish patches but tolerant of small spatial error**, and the way to get both is to record *what* gradients occur and only *roughly where*. Hence: histogram the orientations (throwing away exact position within a cell), but keep several cells (so coarse position is retained). Take a $16\times16$ grid **in the keypoint's own scale** (i.e. sampled from the pyramid level of the detected $\sigma$), **rotated** so the assigned orientation is at $0°$. Split into $4\times4$ subregions. In each, build an **8-bin orientation histogram**, weighted by gradient magnitude and a Gaussian of $\sigma = \tfrac12 \times$ window width. Concatenate: $4\times4\times8 = \mathbf{128}$ dimensions.

*What stage 4 leaves broken:* the histogram bins are sums of gradient *magnitudes*, and magnitudes move with the lighting. Turn up the lights and every entry grows.

**Stage 5 — Normalisation (the illumination model).** Model the lighting change as $I \to aI + b$ and knock out each term in turn.
1. The offset $b$ is *already* gone, for free — we built the descriptor from gradients, and differentiating annihilates a constant. Nothing to do.
2. The gain $a$ multiplies every gradient, hence every histogram entry, hence the whole vector by $a$. A vector scaled by a positive constant is fixed by **L2-normalising** — so one division removes any affine gain $I \to aI$ (contrast change).
3. That handles *linear* illumination change. Real ones are not linear: a specular highlight or a saturating sensor blows up a few gradients enormously while leaving the rest alone, which no global rescaling can undo. The observation Lowe exploits is that such effects corrupt gradient **magnitudes** far more than **orientations** — the direction of the intensity change survives even when its size does not. So cap magnitude's influence and let orientation carry the information: **clamp every element at 0.2**, then L2-normalise again (the clamp changes the norm, so it must be restored). The clamp is a crude robust estimator — it is the descriptor's version of gradient clipping, and it is there for the same reason: to stop one outlying value dominating the whole vector.

**Interpolation detail:** votes are distributed by **trilinear interpolation** across the two nearest spatial bins in $x$, $y$ and the two nearest orientation bins. Without this, a gradient that shifts slightly across a bin boundary causes a discontinuous descriptor change — the whole point is smooth degradation.

**Why $4\times4\times8$?** There is no derivation here and it would be dishonest to invent one — Lowe swept it empirically. But the *shape* of the trade-off is derivable from what the descriptor is for. More spatial bins ⇒ more discriminative but less tolerant of localisation error and viewpoint warp; fewer ⇒ robust but ambiguous. $4\times4\times8$ was the empirical sweet spot; $2\times2\times8=32$ and $8\times8\times8=512$ both performed worse in his matching experiments. **Being able to state the trade-off (discriminativeness vs. localisation tolerance) rather than the number is the strong answer.**

**Invariance audit:** scale ✅ (from the pyramid), in-plane rotation ✅ (canonical orientation), affine illumination ✅ (normalise + clamp), viewpoint ⚠️ (tolerates roughly ±30° out-of-plane), full projective ❌, non-rigid deformation ❌.

### ORB — Oriented FAST and Rotated BRIEF (Rublee et al., ICCV 2011)

SIFT is excellent and slow. ORB asks a different question: **what is the cheapest thing that still works?** Its answers are worth studying precisely because each one is a deliberate downgrade with a stated cost — and because ORB, not SIFT, is what actually runs on your phone. Designed to be a free, real-time SIFT substitute. Two halves, mirroring SIFT's detector/descriptor split.

**Detector: oriented FAST.** The insight is that SIFT's detector spends most of its cost building a scale-space, but a corner test does not need one: you can decide "is this a corner" by looking at a ring of pixels and asking whether a contiguous arc of them is all brighter or all darker than the centre. That is a handful of comparisons against a handful of *integer* pixel values — no convolutions, no floating point.
- **FAST corner test:** consider the 16-pixel Bresenham circle of radius 3 around $p$. $p$ is a corner if there exist $N$ *contiguous* pixels all brighter than $I_p + t$ or all darker than $I_p - t$. $N = 9$ (FAST-9) is standard. **High-speed rejection:** test pixels 1, 9, then 5, 13 first — if fewer than 3 of those 4 pass, reject immediately. Why those four and why three? They are the compass points, 90° apart; if 9 of 16 contiguous pixels must pass, that arc spans more than half the circle and so must contain at least 3 of any 4 evenly-spaced points. The test is therefore a *sound* early-out — it never rejects a true corner — and since the overwhelming majority of pixels are not corners, almost every pixel in the image exits after four comparisons. This is what makes FAST ~30× faster than Harris.
- FAST has no cornerness measure and no scale, so ORB adds: (a) compute the **Harris response** at each FAST corner and keep the top $N$; (b) run FAST on a **scale pyramid** to get coarse scale.
- **Orientation via intensity centroid** (Rosin): with image moments $m_{pq} = \sum_{x,y} x^p y^q I(x,y)$ over the patch, the centroid is $C = (m_{10}/m_{00},\, m_{01}/m_{00})$ and
  $$\theta = \operatorname{atan2}(m_{01}, m_{10})$$
  the direction from the patch centre to its intensity centroid. Read the moments plainly: $m_{00}$ is total intensity, $m_{10}$ and $m_{01}$ are intensity-weighted sums of $x$ and $y$, so $C$ is the "centre of mass" of brightness and $\theta$ points from the geometric centre towards it. Cheaper than an orientation histogram — one pass of multiply-accumulates instead of a weighted histogram and a peak search — and surprisingly stable. Its weakness is the flip side of its cheapness, and it is the subject of the mastery check below: a patch whose brightness is symmetric has no centre-of-mass offset to point at.

**Descriptor: rotated BRIEF (rBRIEF).**
- **BRIEF** is a binary string: pick $n$ pairs of points $(\mathbf{a}_i, \mathbf{b}_i)$ in the patch; bit $i$ = $\mathbb{1}[I(\mathbf{a}_i) < I(\mathbf{b}_i)]$. ORB uses $n = 256$ ⇒ **32 bytes** (vs SIFT's 128 floats = 512 bytes).
- **Steering:** rotate the whole sampling pattern by $\theta$: $S_\theta = R_\theta S$. Precompute $R_\theta$ for 30 discrete angles (12° increments) into a lookup table.
- **Learning the pattern:** naive steered BRIEF loses variance and gains correlation between bits. Both halves of that sentence are information-theoretic complaints, and they are worth unpacking because they are the same complaints you would make about a bad embedding. A bit whose value is 0.9 of the time carries far less than one bit of information ($H(0.9)\approx0.47$ bits), and a bit that is nearly determined by another bit you already have carries almost none *given* that one. Maximising information per descriptor therefore means driving each bit's mean towards 0.5 and its correlation with the others towards zero. ORB does exactly that: it **greedily selects** 256 tests from ~205k candidates, preferring tests whose mean response over a training set is near 0.5 (max variance ⇒ max information per bit) and whose correlation with already-chosen tests is low. That greedy decorrelation is the "R" in rBRIEF and is the reason ORB is meaningfully better than steered BRIEF.
- **Matching by Hamming distance** — a XOR plus a `popcount` instruction. On modern CPUs this is ~1 cycle per 64 bits, making ORB matching 1–2 orders of magnitude faster than L2 on 128-D floats.

**SIFT vs ORB — the decision table:**

| | SIFT | ORB |
|---|---|---|
| Descriptor | 128-D float (512 B) | 256-bit binary (32 B) |
| Distance | L2 | Hamming (XOR + popcount) |
| Scale invariance | true scale-space | pyramid levels only (coarser) |
| Rotation | orientation histogram | intensity centroid |
| Robustness to viewpoint/blur | higher | lower |
| Speed | ~1× | ~10–40× faster to compute, ~100× faster to match |
| Licence | patent expired 2020, in main OpenCV | free, always was |
| Where used | photogrammetry, COLMAP/SfM, offline matching | ORB-SLAM2/3, real-time AR, embedded |

**Current landscape 🟢 (say this and you sound 2026, not 2014).** Hand-crafted local features have largely been superseded for hard matching problems by **learned** ones: **SuperPoint** (self-supervised detector+descriptor), **DISK**, **ALIKED**, matched with **SuperGlue** → **LightGlue** (attention-based graph matchers), and **detector-free** matchers like **LoFTR**/**RoMa** that skip keypoints entirely and match dense coarse-to-fine. These dominate the IMC (Image Matching Challenge) benchmarks. **But**: SIFT + a good RANSAC is still competitive, still the default in COLMAP, and still the right answer when you need no GPU, no training data, and reproducibility. Knowing both halves of that sentence is the top-1% position.

### Connect it

Descriptor + distance metric + matching is a *retrieval* problem — Module 3 replaces the hand-crafted descriptor with a learned embedding and the linear scan with ANN search (3.7), but the structure is identical, and so are the design pressures: SIFT's normalise-and-clamp is doing by hand what an L2-normalised embedding head does by construction. SIFT's orientation histograms are the direct ancestor of **HOG** (4.2), which is the direct ancestor of the first CNN detection features. The edge-rejection Hessian trick is Harris's trace/det trick again.

**In your own words:** name each of SIFT's five stages by the *problem it solves*, without using the word "histogram."

### 🎯 Top-1% distinction

1. **Explain the 0.2 clamp** — most people know "normalise twice," few can say it models non-linear illumination by capping any single gradient's contribution.
2. **Explain the 80%-of-peak duplicate-orientation rule** and why it improves matching (rotationally ambiguous patches get multiple hypotheses instead of one arbitrary choice).
3. **Explain the edge-rejection ratio $(r+1)^2/r$** and connect it to Harris.
4. **Justify $4\times4\times8$ as a trade-off**, not as trivia.
5. **Know the binary-descriptor economics**: 32 bytes vs 512 bytes is not just memory — it changes what fits in cache and lets you brute-force match 100k descriptors in real time, which is why ORB-SLAM exists and SIFT-SLAM does not.
6. **Name the learned successors** and say when you'd still pick SIFT.

### ✅ Mastery check

You are building a visual-localisation service: a phone uploads a photo, you match it against a 5-million-descriptor map of a building and return a 6-DoF pose. Latency budget: 200 ms end-to-end on a server, no GPU.

**(a)** SIFT or ORB for the map descriptors? Justify with a memory calculation and a matching-cost calculation.
**(b)** Your engineer says "ORB is rotation-invariant because of the steering, so we're fine with any phone orientation." Where is that claim weakest?
**(c)** Photos taken at dusk fail. Which specific stage of each pipeline degrades, and what would you change?

<details><summary>Answer sketch</summary>
(a) Memory: SIFT 5M × 512 B = <b>2.56 GB</b>; ORB 5M × 32 B = <b>160 MB</b> — the ORB map fits in RAM comfortably and largely in L3 for hot shards. Matching: Hamming over 256 bits is a XOR + popcount (~4 ops per 64-bit word × 4 words); L2 over 128 floats is 128 multiply-adds. Roughly 2 orders of magnitude. On a no-GPU 200 ms budget, ORB. The right engineering answer is actually <b>ORB + an ANN index (Module 3.7)</b>, not brute force — but even the index is cheaper on binary codes.
(b) The <b>intensity-centroid orientation</b> is the weak link. It is stable only when the patch has a well-defined intensity asymmetry; on rotationally symmetric or low-contrast patches $m_{10}, m_{01} \to 0$ and $\theta$ becomes noise. A few degrees of orientation error rotates the sampling pattern and corrupts many bits at once (binary descriptors degrade non-gracefully). Also, the steering LUT is quantised to 12°, so there is a built-in ±6° error floor. SIFT's histogram-peak orientation is more robust, and SIFT's descriptor degrades more smoothly under small orientation error thanks to trilinear interpolation.
(c) Dusk ⇒ low SNR and low contrast. SIFT: the <b>low-contrast rejection $|D| < 0.03$</b> throws away most keypoints, and remaining ones are noise-dominated. ORB: the <b>FAST threshold $t$</b> is an absolute intensity difference, so corner count collapses; and BRIEF's binary comparisons flip when $|I(a)-I(b)|$ is within the noise floor — binary descriptors are far more fragile at low contrast for exactly this reason. Fixes: per-image contrast normalisation (CLAHE) before detection; adaptive FAST threshold to hit a target keypoint count; and for the real fix, a learned detector (SuperPoint) trained with photometric augmentation, or switch the map to a learned descriptor. Also worth saying: raise the ratio-test threshold slightly and lean harder on RANSAC, since you'll have fewer, noisier matches.
</details>

### 🔨 Build + read

**Build:** Implement the **SIFT descriptor** (assume keypoints from OpenCV's detector) — the rotated $16\times16$ sampling, trilinear-interpolated $4\times4\times8$ histogram, and the normalise → clamp(0.2) → normalise sequence. Validate by comparing match quality against `cv2.SIFT` descriptors on an image pair. Then implement FAST-9 with the high-speed rejection test and benchmark it against Harris on the same image (you should see roughly an order of magnitude).

**Read:** Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004) — read it end to end, it is the single most important paper in classical CV. Rublee et al., "ORB: an efficient alternative to SIFT or SURF" (ICCV 2011) — short, read §4 on rBRIEF learning. Then skim the **LightGlue** paper (ICCV 2023) to see where the field actually is.

---

## 1.7 Feature Matching & Distance Metrics

### Intuition

You have 2000 descriptors from image A and 2000 from image B. For each descriptor in A, which one in B is the same physical point? Nearest neighbour is the obvious answer and it is wrong roughly half the time.

So the real question of this section is not "how do I find the nearest neighbour" — that is a search-efficiency problem. It is **how do I know whether to believe the nearest neighbour I found?** Every descriptor in A has a nearest neighbour in B whether or not the point is even visible in B; the NN operation always returns something. Deciding what to trust is the interesting part, and the trick that solves it is not the one most people would reach for.

### The math

**Metrics.**
- **L2** for float descriptors (SIFT). $\|a - b\|_2$.
- **Hamming** for binary (ORB/BRIEF/BRISK). `popcount(a XOR b)`.
- **Cosine** $\dfrac{a\cdot b}{\|a\|\|b\|}$ — but note: **for L2-normalised vectors, L2 and cosine are monotonically equivalent.** Expand the squared distance and watch what happens when the norms are both 1:
  $$
  \|a - b\|_2^2 = \|a\|^2 + \|b\|^2 - 2a\cdot b = 2 - 2\cos\theta
  $$
  The first two terms became constants, so the *only* varying quantity left is $\cos\theta$, entering with a negative sign. Squared distance is thus a decreasing affine function of cosine similarity, and an increasing function has no effect on ranking — so ranking by L2 and ranking by cosine give identical orderings. SIFT descriptors are L2-normalised, so this applies. **This identity is asked constantly in ML interviews** and it returns in Module 3 for embedding retrieval.
- **$\chi^2$ / Earth Mover's Distance** for histogram descriptors — better calibrated for histograms than L2, but slower; used in older BoW pipelines.

**Lowe's ratio test.** The obvious way to filter matches is a distance threshold: accept if $d_1 < T$. Try it and it fails badly, and the reason is worth sitting with before seeing the fix — see below. Lowe's alternative uses a quantity you would not think to look at: the *second*-nearest neighbour. For query descriptor $q$, let $d_1, d_2$ be distances to its 1st and 2nd nearest neighbours in the other image. Accept the match only if

$$
\frac{d_1}{d_2} < \tau, \qquad \tau \approx 0.75\text{–}0.8
$$

**Why this works and a global threshold does not.** Descriptors vary enormously in distinctiveness — a corner on a repeated brick pattern has hundreds of near-identical neighbours; a unique logo has one. So $T$ would have to be tight enough to reject the brick corner's spurious best match and loose enough to accept the logo's genuine but noisy one, and no single value does both.

The escape is to stop asking "is $d_1$ small?" (an absolute question, requiring a global calibration you do not have) and instead ask "is $d_1$ *unusually* small for this query?" (a relative question, which the data can answer for itself). The second-nearest neighbour supplies the calibration: it is a **per-query estimate of the local density of the descriptor space** — an estimate of how close things get around here by chance. So $d_2$ tells you what "close" means *for this descriptor*, and $d_1/d_2$ asks whether the best match beats that baseline. A true match should be dramatically closer than the best *wrong* match; a false match is just one of many equally mediocre candidates and its ratio sits near 1.

(This is the same reasoning as a statistical test: $d_1$ is the observation, $d_2$ estimates the null distribution, the ratio is the test statistic.) Lowe's empirical ROC: $\tau = 0.8$ **eliminates ~90% of false matches while discarding only ~5% of correct ones**. Memorise those two numbers — a filter with that asymmetry is rare, and it is why this one line of code matters as much as the descriptor design.

**Mutual / cross-check.** Accept $(a,b)$ only if $b$ is $a$'s NN **and** $a$ is $b$'s NN. Cheap, orthogonal to the ratio test, and typically stacked with it. `cv2.BFMatcher(crossCheck=True)` cannot be combined with `knnMatch`, so in practice you do ratio test → then cross-check manually.

**Search structures.**
- **Brute force**: $O(N_A N_B D)$. Fine up to a few thousand descriptors; embarrassingly parallel; exact.
- **kd-tree**: $O(\log N)$ in low dimensions — but **degenerates to linear scan above roughly 10–20 dimensions**. The mechanism is worth knowing, because it explains why *every* exact partition-based index dies: a kd-tree prunes a branch when the distance to the splitting plane exceeds the best distance found so far. In high dimensions the ratio of nearest to farthest neighbour distance goes to 1 (concentration of measure), so "the best distance so far" is barely smaller than everything else, almost no branch can be pruned, and the search visits nearly every leaf. SIFT is 128-D, so exact kd-tree is useless.
- **FLANN**: randomised kd-tree *forest* (several trees with randomised split dimensions, searched with a shared priority queue and a bounded leaf budget) or hierarchical k-means tree. **Approximate** — you trade a recall percentage for a large speedup. This is the classical ancestor of HNSW/IVF-PQ in 3.7.
- **LSH / multi-probe LSH** for binary descriptors.

**The curse of dimensionality, stated properly.** In high dimensions, for many distributions,
$$
\frac{\mathbb{E}[d_{\max}] - \mathbb{E}[d_{\min}]}{\mathbb{E}[d_{\min}]} \to 0 \quad \text{as } D \to \infty
$$
— all points become roughly equidistant, so "nearest neighbour" loses discriminative meaning and any partition-based index degenerates. Real descriptors live on a much lower-dimensional manifold than 128-D, which is *why* ANN methods work at all in practice.

### Connect it

Matching produces *putative* correspondences that are ~30–60% wrong even after the ratio test. Nothing downstream survives that error rate — and note *why* no better descriptor will fix it: appearance alone cannot distinguish two identical-looking window corners, because they genuinely look identical. The missing information is geometric, not photometric, so the fix must come from a different source of evidence. Hence RANSAC (1.8) as the geometric verifier. And the whole detector→descriptor→ANN→verify chain is structurally identical to a RAG pipeline: chunk → embed → vector index → rerank. Module 3.7 makes that mapping explicit.

**In your own words:** why is the *second*-nearest neighbour more informative than the first?

### 🎯 Top-1% distinction

- **The ratio test's justification** (per-query adaptive density estimate), plus the 90%/5% numbers.
- **The L2 ≡ cosine equivalence for normalised vectors** — and the corollary: if you're going to L2-normalise anyway, don't waste compute on cosine; use L2 and let your ANN index use its faster L2 kernel.
- **Knowing exactly why kd-trees die above ~20-D** and that FLANN is *approximate*, not a faster exact method. Many candidates think FLANN is just "fast kd-tree."
- **Ratio test is not usable when you have multiple valid matches** — e.g. matching against a database with duplicate images, or repeated structure like windows on a facade. There, the second-NN is a *correct* match and the ratio test kills your true positive. Fix: exclude neighbours from the same image / cluster before computing $d_2$. This edge case is a great answer because it shows you've hit it in practice.

### ✅ Mastery check

You match two photos of a glass office tower. After the ratio test at $\tau=0.8$ you retain only 40 of 3000 putative matches, and RANSAC then fails. Diagnose: **(a)** why does the ratio test destroy so many matches here specifically, **(b)** what would you change, and **(c)** what would you change *instead* if the problem were the opposite — 2000 matches surviving but 70% of them wrong?

<details><summary>Answer sketch</summary>
(a) A glass facade is <b>repetitive structure</b>: every window corner has hundreds of near-identical descriptors, so $d_2 \approx d_1$ and the ratio $\to 1$ for almost every true match. The ratio test is designed to reject exactly this ambiguity, and here the ambiguity is real, not spurious — so it rejects almost everything.
(b) Options, best first: (i) relax $\tau$ to ~0.9 and let <b>geometric verification carry the load</b> — RANSAC on a homography can disambiguate repetition because the wrong matches are geometrically inconsistent, whereas the descriptor cannot; (ii) use spatial-consistency filtering / neighbourhood-consensus before RANSAC; (iii) switch to a matcher that reasons globally over the match set (SuperGlue/LightGlue explicitly handle repetitive structure via attention across all keypoints); (iv) use larger-support descriptors so more context disambiguates.
(c) The opposite failure means the descriptors are <b>too permissive</b>: tighten $\tau$ to 0.7, add mutual-NN cross-check, and — most importantly — increase RANSAC iterations, because $N \propto 1/w^s$ and $w = 0.3$ with $s=4$ needs ~560 iterations at $p=0.99$ versus 72 at $w=0.5$. Also consider MSAC/MAGSAC++ which handle low inlier ratios better than vanilla RANSAC.
</details>

### 🔨 Build + read

**Build:** Take an image pair with known ground-truth homography (the **Oxford VGG affine dataset** — graf, boat, bark sequences — ships with ground truth). Sweep $\tau$ from 0.5 to 1.0 and plot precision vs. recall of the matches against ground truth. Reproduce Lowe's ROC curve. Then repeat with SIFT vs ORB on the same pairs and overlay.

**Read:** Lowe 2004 §7.1 (the ratio-test experiment and the figure you're reproducing). Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration" (VISAPP 2009) — the FLANN paper.

---

## 1.8 RANSAC (and MAGSAC++) 🟢

### Intuition

You have 500 candidate matches and maybe 200 are right. **How do you fit a model when you cannot tell in advance which of your data points are lies?**

Your instinct from deep learning is to fit everything and let a robust loss handle it. That instinct is right in spirit and fails here in practice: least squares over all 500 gives garbage, because a squared penalty rewards a single wild outlier for dragging the fit arbitrarily far towards it — least squares has a **breakdown point of 0%**, meaning one bad point out of any number can ruin the estimate. And there is a deeper problem than the loss: with 60% outliers you are not fitting noisy data, you are fitting *two different populations at once*, and no single fit to the union is meaningful.

RANSAC inverts the problem. Instead of fitting the data and hoping the outliers wash out, it **guesses a subset that might be clean, fits that, and then lets the rest of the data vote on whether the guess was good.** Concretely: repeatedly guess a *minimal* set of matches, fit a model to just those, and see how many of the other matches agree. The guess supported by the most agreement wins.

Why *minimal* sets — the smallest number of points that determines the model? Because the whole scheme depends on drawing a sample with no outlier in it, and the probability of that decays exponentially in the sample size. Every extra point you take is exponentially fewer clean samples. Hold that thought; it becomes the derivation below. It's fitting by repeated small democratic elections.

Real-life anchor: this is running inside your phone every time it stitches a panorama, and inside every AR headset's relocalisation.

### The algorithm

```
best_inliers = ∅
for iteration in 1..N:
    S ← random minimal sample of s correspondences
    M ← fit_model(S)                        # exact, minimal solver
    I ← { c : residual(c, M) < t }          # consensus set
    if |I| > |best_inliers|: best_inliers, best_M ← I, M
refit best_M on best_inliers                # least squares / LM on inliers only
```

`s` is the **minimal sample size** for the model: line 2, homography 4, fundamental matrix 7 or 8, essential matrix 5, PnP 3 (P3P).

### Deriving the iteration count $N$ — do this on the whiteboard

The algorithm has an obvious hole: how many times do you loop? Too few and you may never draw a clean sample; too many and you burn time for nothing. The answer is a probability calculation you can do from scratch, and the key move is the standard one for "at least one" events — **compute the probability that it never happens, and subtract.**

Let $w$ = inlier ratio = P(a randomly chosen correspondence is an inlier). Now, one line at a time:

- **P(one sample of $s$ points is all inliers) $= w^s$.** Each of the $s$ draws is an inlier with probability $w$, and we treat the draws as independent, so the probabilities multiply. (Strictly, drawing without replacement makes later draws slightly dependent on earlier ones, but with hundreds of correspondences the correction is negligible — this is the same approximation as sampling with replacement.)
- **P(a given sample is contaminated) $= 1 - w^s$.** Complement of the above: a sample either is all-inlier or it is not.
- **P(all $N$ samples contaminated) $= (1 - w^s)^N$.** The iterations are independent draws, so again the probabilities multiply. This is the quantity we want to be *small*.
- **P(at least one clean sample) $= 1 - (1-w^s)^N$.** Complement again. Call this our success probability and demand it equal $p$ (typically 0.99):

$$
1 - (1 - w^s)^N = p
$$

Now solve for $N$. Rearranging gives $(1-w^s)^N = 1-p$; take logs of both sides, which turns the exponent into a multiplier, $N\log(1-w^s) = \log(1-p)$; divide:

$$
\boxed{\,N = \frac{\log(1-p)}{\log(1 - w^s)}\,}
$$

Sanity-check the signs before trusting it: both $1-p$ and $1-w^s$ are between 0 and 1, so both logs are negative and $N$ comes out positive. And as $w \to 0$ the denominator $\to \log 1 = 0$, so $N \to \infty$ — with no inliers you would search forever, which is correct.

Note carefully what $p$ *is*: not an accuracy, not a confidence in the fitted model, but the probability that you ever drew a clean sample at all. Everything RANSAC promises is conditional on that.

**Worked numbers (know at least two of these):**

| Model | $s$ | $w$ | $N$ at $p = 0.99$ |
|---|---|---|---|
| Line | 2 | 0.5 | 17 |
| **Homography** | **4** | **0.5** | **72** |
| Homography | 4 | 0.3 | 567 |
| **Fundamental (8-pt)** | **8** | **0.5** | **1177** |
| Fundamental (7-pt) | 7 | 0.5 | 587 |
| Essential (5-pt) | 5 | 0.5 | 145 |
| Fundamental (8-pt) | 8 | 0.3 | 70,188 |

**Walk the homography row by hand — you should be able to do this on a whiteboard with no notes.** A homography needs 4 correspondences, so $s=4$, and suppose half your putative matches are good, $w = 0.5$. First, how often does a single draw come up clean? All four points must be inliers, and each is with probability one half, so $w^s = 0.5^4 = 0.0625$ — about one draw in sixteen. So a given draw *fails* 93.75% of the time, and $N$ failures in a row have probability $0.9375^N$. We want that down to 1%, i.e. $0.9375^N = 0.01$. Taking natural logs, $N \times \ln(0.9375) = \ln(0.01)$; numerically $\ln(0.9375) = -0.06454$ and $\ln(0.01) = -4.6052$, so $N = 4.6052/0.06454 = 71.4$, and since you cannot run a fraction of an iteration, round up to **72**.

Sanity-check the magnitude against intuition: one clean draw in sixteen, and we want near-certainty, so a few dozen tries sounds right — you would expect roughly $16 \times \ln(100) \approx 74$, and that is exactly what the formula returned. (The base of the logarithm cancels in the ratio, so use whichever you like.)

**Pause:** predict the 8-point fundamental matrix row at the same $w = 0.5$ before you look back up at the table. Twice the sample size — does $N$ roughly double?

It goes from 72 to **1177**, a factor of 16. Doubling $s$ *squares* the failure-free probability $w^s$, which means $N$ scales like $1/w^s$ — exponentially in $s$, not linearly. That single fact is what the next two paragraphs are about.

**Two things this table screams:**
1. **$N$ grows exponentially in $s$.** This is the entire economic motivation for **minimal solvers** — Nistér's 5-point algorithm for the essential matrix exists because 5 vs 8 is the difference between 145 and 1177 iterations. In Module 5, when someone asks "why do we bother with the 5-point algorithm when the 8-point one is a one-line SVD?", *this* is the answer.
2. **$N$ blows up as $w$ drops.** Going from $w=0.5$ to $w=0.3$ costs 8× for a homography and 60× for an 8-point fundamental matrix. That is why the ratio test (1.7) matters so much — every point of inlier ratio you buy upstream is compounded savings here.

**Adaptive $N$.** There is a circularity in all of the above: the formula needs $w$, and $w$ is the fraction of your matches that are inliers — which is exactly what you are running RANSAC to find out. The way out is to *estimate $w$ from the answer so far*: the best consensus set you have found is itself a lower-bound estimate of the inlier count. So initialise $N = \infty$, and after each iteration set $\hat{w} = |I_{\text{best}}|/n$ and recompute $N$ from the formula. Terminate when the iteration count reaches the current $N$. As better models are found, $\hat w$ rises and the required $N$ falls, so the loop tightens its own budget as it learns — on easy problems it exits in a handful of iterations. Every real implementation (OpenCV included) does this.

### Choosing the threshold $t$

RANSAC has one more free parameter — the residual below which a point counts as agreeing — and unlike $\kappa$ in Harris this one *can* be derived, provided you are willing to state a noise model. Ask the question in the right form: "how large a residual would a genuine inlier plausibly produce, given that keypoint localisation is imperfect?" If the point localisation error is zero-mean Gaussian with std $\sigma$ per coordinate, then the residual is a sum of $m$ squared independent standard normals once you divide by $\sigma^2$ — which is the definition of a $\chi^2$ variable with $m$ degrees of freedom, where $m$ = codimension of the model (dimension of the residual). Choosing $t$ is then just picking a quantile: set it so that a true inlier falls inside with probability $\alpha$.

$$
t^2 = \chi^2_{m, \alpha}\,\sigma^2
$$

- $m = 1$ (point-to-line distance, e.g. epipolar-line residual for $F$): $t^2 = 3.84\sigma^2$ at $\alpha = 95\%$.
- $m = 2$ (point-to-point distance, e.g. reprojection error for $H$): $t^2 = 5.99\sigma^2$.

The reason $m$ differs between the two is worth pausing on: an epipolar residual measures distance to a *line*, and a point can slide freely along that line at no cost, so only one direction is constrained — one degree of freedom. A reprojection residual measures distance to a *point*, constrained in both directions — two. So with 1-pixel keypoint noise, a homography inlier threshold of $\sqrt{5.99} \approx 2.45$ px is the principled choice. **Quoting $\chi^2$ values here is a strong signal** — most candidates say "we tune it."

### Variants worth naming

| Variant | Change | Why |
|---|---|---|
| **MSAC / M-SAC** | cost $= \sum \min(e_i^2, t^2)$ instead of inlier *count* | inliers now vote with quality, not just membership; strictly better at no cost |
| **MLESAC** | maximises a likelihood under an inlier/outlier mixture | principled version of MSAC |
| **LO-RANSAC** | run a local optimisation (inner RANSAC on the current inlier set) whenever a new best is found | dramatically reduces required $N$; near-free accuracy |
| **PROSAC** | sample from matches ordered by descriptor quality first | exploits that good-ratio matches are more likely inliers; often 10–100× fewer iterations |
| **DEGENSAC** | detects degenerate (dominant-plane) samples for $F$ estimation | fixes the single worst failure mode of $F$-RANSAC |
| **GC-RANSAC** | graph-cut local optimisation using spatial coherence | current strong baseline |
| **MAGSAC++** 🟢 | **no threshold** — marginalises the model quality over a range of $\sigma$ ("$\sigma$-consensus"), with an iteratively reweighted least-squares inner loop | removes the most fragile hyperparameter; SOTA on IMC benchmarks; available as `cv2.USAC_MAGSAC` |

Read that table as a single storyline rather than six independent tricks: vanilla RANSAC throws away information at three points — it treats all inliers as equally good (MSAC/MLESAC fix that), it treats all samples as equally likely to be clean (PROSAC fixes that), and it never refines a promising hypothesis before discarding it (LO-RANSAC and GC-RANSAC fix that). MAGSAC++ then removes the last hand-set constant, $t$ itself.

**Practical note:** since OpenCV 4.5.0, `findHomography` / `findFundamentalMat` accept `cv2.USAC_MAGSAC`, `USAC_ACCURATE`, etc. Using `cv2.RANSAC` in 2026 is leaving accuracy on the table for free.

**In your own words:** what exactly is the probability $p$ in the iteration formula the probability *of*?

### 🎯 Top-1% distinction

1. **Derive $N$ live** and immediately draw the two consequences (exponential in $s$ ⇒ minimal solvers; explosion at low $w$ ⇒ upstream match quality matters).
2. **Justify $t$ from a $\chi^2$ noise model** rather than "tuning."
3. **Degeneracy.** RANSAC assumes a *single* dominant model and a *non-degenerate* minimal sample. The classic killer: estimating a **fundamental matrix on a scene where the dominant structure is a plane** — a plane-induced homography explains all those matches perfectly, so RANSAC happily returns an $F$ consistent with a plane and geometrically meaningless. Naming DEGENSAC as the fix is a strong close.
4. **RANSAC is probabilistic, not guaranteed** — it returns a model that is correct with probability $p$, and $p$ is a *design parameter you chose*. The output is a random variable; run it twice, get two answers.
5. **Multiple models:** RANSAC finds one. For multiple planes you need sequential RANSAC (fit, remove inliers, refit — biased) or multi-model methods (J-Linkage, Progressive-X).
6. **Always refit on inliers.** The minimal-sample model is only a hypothesis generator; the final estimate must be a least-squares / LM fit over the consensus set, or you're throwing away most of your data.

### ✅ Mastery check

You are estimating a fundamental matrix from 1000 putative matches with an inlier ratio of 0.35, using the 8-point algorithm.

**(a)** How many iterations for $p = 0.99$? Show the arithmetic.
**(b)** Your budget is 2000 iterations. Give three *distinct* ways to get within budget, and rank them by how much you'd trust the result.
**(c)** The scene is the flat facade of a building. What silently goes wrong, and how would you detect it?

<details><summary>Answer sketch</summary>
(a) $w^s = 0.35^8 = 2.25\times10^{-4}$. $\log(1 - 2.25\times10^{-4}) \approx -2.25\times10^{-4}$. $N = -4.6052 / -2.25\times10^{-4} \approx \mathbf{20{,}450}$ iterations. Ten times over budget.
(b) Ranked:
 1. <b>Use the 7-point algorithm</b> ($s=7$): $w^7 = 6.4\times10^{-4}$, $N \approx 7{,}155$ — still over. Combine with <b>PROSAC</b> (sample good matches first) and/or <b>LO-RANSAC</b> (local optimisation lets you stop far earlier). This is the trustworthy path: same model, better sampling.
 2. <b>Raise the inlier ratio upstream</b> — tighten the ratio test, add mutual-NN, use a better matcher (LightGlue). Going from $w=0.35$ to $w=0.5$ takes 8-point $N$ from 20,470 to 1,177 — <b>in budget</b>, and the estimate is better because the data is better. Arguably this should be #1.
 3. <b>Lower $p$ to 0.95</b>: $N = \log(0.05)/\log(1-2.25\times10^{-4}) \approx 13{,}300$ — still over, and you've bought a 1-in-20 chance of total failure. Weakest option.
 (Also acceptable: if you have calibrated cameras, estimate the <b>essential</b> matrix with the 5-point algorithm — $s=5$ gives $N \approx 875$, comfortably in budget. That's the best answer of all if calibration is available, and it's the Module 5 link.)
(c) A flat facade is a <b>degenerate configuration</b> for $F$: all correspondences are related by a single homography, and any $F$ of the form $F = [\mathbf{e}']_\times H$ fits them perfectly for <i>any</i> epipole $\mathbf{e}'$. So RANSAC reports a huge inlier count and high confidence while the epipolar geometry is unconstrained in one degree of freedom — a silent, confident failure. Detection: fit a homography to the same matches; if the homography inlier count is close to the $F$ inlier count, you are in the degenerate case. Fix: DEGENSAC / QDEGSAC, or use the plane-and-parallax formulation, or move the camera to create real parallax.
</details>

### 🔨 Build + read

**Build:** Implement RANSAC generically (`fit_fn`, `residual_fn`, `s`, `t`, `p`) and use it for both line fitting and homography estimation. Add **adaptive $N$** and verify empirically that the iteration count it converges to matches the closed-form prediction. Then implement MSAC's cost and measure the accuracy difference on synthetic data with known ground truth across inlier ratios 0.2–0.8. Finally, benchmark against `cv2.USAC_MAGSAC`.

**Read:** Fischler & Bolles (1981) — the original, short. Then the MAGSAC++ paper (Barath et al., CVPR 2020) §3 for the $\sigma$-consensus idea. Hartley & Zisserman §4.7 for the $\chi^2$ threshold derivation and the algebraic/geometric error distinction.

---

## 1.9 Homography & Panorama Stitching

### Intuition

RANSAC gave us a way to fit a model to dirty correspondences, but we never said what model. Here is the one that matters most in practice, and the question it answers is: **when are two photographs of the same thing related by a transformation you can actually write down?**

In general they are not — move the camera through a 3-D scene and near things shift more than far things, and no 2-D warp can express that. But in two special cases the relationship collapses to a single $3\times3$ matrix: two views of the **same plane**, or two views from the **same camera centre** (pure rotation), regardless of scene geometry. Both cases kill depth as a variable — in the first because every point has depth determined by the plane, in the second because with no translation depth cannot cause any shift at all. That matrix is the homography. It's why you can rectify a photo of a whiteboard taken at an angle, and why panorama mode tells you to pivot rather than walk sideways.

### The math

In homogeneous coordinates $\tilde{\mathbf{x}} = (x, y, 1)^\top$:

$$
\tilde{\mathbf{x}}' \simeq H \tilde{\mathbf{x}}, \qquad
H = \begin{bmatrix} h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & h_9\end{bmatrix}
$$

$\simeq$ means *equal up to scale*, and that is not a technicality — it is the defining feature of projective geometry. A homogeneous point $(x,y,1)$ and $(2x,2y,2)$ are the *same* image point, because we recover the actual pixel by dividing through by the last coordinate. So if $H$ maps a point correctly, so does $2H$: multiplying $H$ by any non-zero constant produces the same map. Hence $H$ has 9 entries but only **8 degrees of freedom** (scale is free; typically fixed by $\|h\|=1$ or $h_9=1$). Everything awkward and everything elegant about estimating $H$ traces to this one fact, so keep it in view. In inhomogeneous form:

$$
x' = \frac{h_1 x + h_2 y + h_3}{h_7 x + h_8 y + h_9}, \qquad y' = \frac{h_4 x + h_5 y + h_6}{h_7 x + h_8 y + h_9}
$$

The denominator is what makes it *projective* rather than affine — it's what produces perspective foreshortening (parallel lines converging).

**The 2-D transformation hierarchy — know this table:**

| Transform | DoF | Preserves | Matrix form |
|---|---|---|---|
| Translation | 2 | everything but position | $[I \mid \mathbf{t}]$ |
| Euclidean (rigid) | 3 | lengths, angles, areas | $[R \mid \mathbf{t}]$ |
| Similarity | 4 | angles, ratios of lengths | $[sR \mid \mathbf{t}]$ |
| Affine | 6 | parallelism, ratios along a line, centroids | $[A \mid \mathbf{t}]$, last row $(0,0,1)$ |
| **Projective (homography)** | **8** | **straight lines, cross-ratio, incidence** | full $3\times3$ |

**Direct Linear Transform (DLT).** We want to solve for $H$ from correspondences, and we would like the problem to be *linear* in the unknowns so we can use linear algebra rather than an iterative optimiser. The obstacle is the scale ambiguity: we know $\tilde{\mathbf{x}}' \simeq H\tilde{\mathbf{x}}$, which is *not* an equation — it says the two vectors are parallel, not equal, and the unknown proportionality constant is different for every correspondence. Writing $\tilde{\mathbf{x}}' = \lambda_i H\tilde{\mathbf{x}}$ would drag one new unknown $\lambda_i$ into the system per point.

**The trick is to state "parallel" without naming the constant.** Two vectors are parallel exactly when their cross product vanishes:

$$\tilde{\mathbf{x}}' \times H\tilde{\mathbf{x}} = \mathbf{0}$$

This is now a genuine equation, it contains no $\lambda$, and — crucially — it is still *linear in the entries of $H$*, because the cross product is linear in each argument and $H\tilde{\mathbf{x}}$ is linear in $H$. We have removed the scale ambiguity without paying any non-linearity for it. That is the whole idea of the DLT.

The cross product has three components, so each correspondence gives 3 equations — but only **2 are independent**, and you can see why without any algebra: a cross product is always orthogonal to both its arguments, so the three components satisfy one linear relation among themselves; equivalently, "these two vectors are parallel" is a 2-degree-of-freedom statement about directions in 3-space, not a 3-degree-of-freedom one. Writing out two of the three rows:

$$
\begin{bmatrix}
\mathbf{0}^\top & -w'\tilde{\mathbf{x}}^\top & y'\tilde{\mathbf{x}}^\top \\
w'\tilde{\mathbf{x}}^\top & \mathbf{0}^\top & -x'\tilde{\mathbf{x}}^\top
\end{bmatrix}
\begin{bmatrix}\mathbf{h}^1\\ \mathbf{h}^2 \\ \mathbf{h}^3\end{bmatrix} = \mathbf{0}
$$

Here $\mathbf{h}$ is $H$ unrolled into a 9-vector — we are treating the matrix as a vector of unknowns, which is legitimate precisely because the equations are linear in its entries. Stack $n \ge 4$ correspondences → $A \mathbf{h} = \mathbf{0}$ with $A \in \mathbb{R}^{2n \times 9}$. Two equations per point, 8 unknowns' worth of freedom: **4 points in general position** (no 3 collinear) give $8$ equations for 8 DoF — the minimal case, hence $s=4$ in RANSAC.

**Solve by SVD.** Look at what we have: a *homogeneous* system $A\mathbf{h} = \mathbf{0}$, not the familiar $A\mathbf{x} = \mathbf{b}$. You cannot hand this to ordinary least squares, because $\mathbf{h} = \mathbf{0}$ satisfies it perfectly and is useless. But recall the scale ambiguity: any non-zero multiple of the true $\mathbf{h}$ is equally correct, so we are free to impose $\|\mathbf{h}\| = 1$ and rule the zero solution out by fiat. The problem becomes: **minimise $\|A\mathbf{h}\|$ subject to $\|\mathbf{h}\| = 1$** — with noisy data $A\mathbf{h}$ will not be exactly zero, so we ask for the direction that comes closest.

That is a Rayleigh-quotient problem, and the SVD answers it directly. Write $A = U\Sigma V^\top$. Since $U$ is orthogonal it preserves norms, so $\|A\mathbf{h}\| = \|\Sigma V^\top\mathbf{h}\|$; and writing $\mathbf{y} = V^\top\mathbf{h}$ (also unit norm, since $V$ is orthogonal too), the quantity to minimise is $\sqrt{\sum_i \sigma_i^2 y_i^2}$ over unit $\mathbf{y}$. That is a weighted average of the squared singular values with weights summing to 1, so it is minimised by putting *all* the weight on the smallest $\sigma_i$ — i.e. $\mathbf{y} = (0,\dots,0,1)$, which back-substitutes to $\mathbf{h} = $ the **right singular vector of $A$ corresponding to the smallest singular value** (last column of $V$). Reshape to $3\times3$.

So "take the last column of $V$" is not a recipe to memorise: it is the statement that the direction $A$ shrinks the most is the direction closest to $A$'s null space, and with exact noise-free data that smallest singular value would be exactly zero.

**Hartley normalisation — do not skip this.** Before the DLT, transform each image's points so that:
1. the centroid is at the origin, and
2. the RMS distance from the origin is $\sqrt{2}$ (i.e. the average point is at $(1,1)$).

Then $\tilde{H} = T'^{-1} \hat{H} T$ to undo it (apply the normalisation to the points, solve, then conjugate the result back into pixel coordinates).

**Why — and this deserves more than "it conditions the matrix".** Look at what actually sits in a row of $A$. Some entries are products like $x'x \sim 10^3 \times 10^3 = 10^6$; some are single coordinates $\sim 10^3$; some are the homogeneous $1$. So the columns of $A$ have wildly different scales — six orders of magnitude apart.

Why is that fatal? The condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ measures how much a small perturbation of $A$ can rotate its singular vectors, and **the answer we want is precisely a singular vector — the smallest one.** Here is the intuition that makes it concrete. Rounding error perturbs every entry of $A$ by roughly $\varepsilon\,\sigma_{\max}$, where $\varepsilon$ is machine precision — errors scale with the *largest* numbers in the matrix, because that is where the floating-point spacing is coarsest. If $\sigma_{\min}$ is itself smaller than that perturbation, then the direction the matrix "shrinks the most" is being determined by rounding noise rather than by your data, and the vector you extract is arbitrary. Roughly: you lose $\log_{10}\kappa$ significant digits, and you started with about 7 in float32 or 16 in float64.

Run the numbers. Unnormalised, $\kappa(A) \sim 10^6$–$10^8$, so in float32 you have essentially nothing left and even float64 is badly degraded. Normalised, every coordinate is $O(1)$, every entry of $A$ is $O(1)$, and $\kappa(A)$ drops to $O(10)$ — a couple of digits lost instead of all of them. Hartley showed this is the difference between an unusable and a usable 8-point algorithm, and note it is a *purely numerical* fix: the mathematics was correct all along, and the answer was still wrong. **This is a top-tier interview discriminator** — most candidates have never heard of it.

(Why $\sqrt2$ specifically? Because it puts the average point at roughly $(1,1)$, so the coordinate entries and the homogeneous $1$ end up the same size — which is the entire objective.)

**Algebraic vs geometric error.** There is a second, deeper complaint about the DLT, and it is independent of conditioning. DLT minimises $\|A\mathbf{h}\|$ — but what *is* that quantity? It is the residual of the cross-product equations, and nothing more; it has no units of pixels and no geometric meaning. Two correspondences with equal algebraic residual can have very different actual misalignment on screen, because the cross product weights each equation by the point's homogeneous coordinates. So the DLT is minimising the wrong thing — conveniently, and in closed form, but the wrong thing.

**Pause:** if the algebraic error is the wrong objective, why is the DLT used at all?

Because the *right* objective is non-convex and needs an iterative optimiser, and an iterative optimiser needs a starting point — and a bad start on a non-convex problem finds a bad local minimum. DLT is a fast, closed-form, globally-determined initialiser that lands close enough for local refinement to succeed. That division of labour, algebraic initialisation followed by geometric refinement, is one of the most reusable patterns in all of geometric vision. The quantities you actually care about:

- **Symmetric transfer error:** $\sum_i d(\mathbf{x}_i', H\mathbf{x}_i)^2 + d(\mathbf{x}_i, H^{-1}\mathbf{x}_i')^2$
- **Reprojection error (gold standard):** also estimate corrected points $\hat{\mathbf{x}}_i$, minimising $\sum_i d(\mathbf{x}_i, \hat{\mathbf{x}}_i)^2 + d(\mathbf{x}_i', H\hat{\mathbf{x}}_i)^2$

Standard practice: **DLT (+normalisation) for the initial estimate → Levenberg–Marquardt on the geometric error over the inliers.** Exactly the pattern that recurs as bundle adjustment in 5.8.

### The panorama pipeline, end to end

Every stage below is Module 1 cashing out — and each one exists because the previous one leaves a specific visible artefact in the output image. Read the list that way: the last four stages in particular are all answers to "the geometry is right and it still looks wrong."

1. **Detect + describe** (SIFT/ORB) in every image.
2. **Match** pairwise, ratio test. For $n$ images, use a fast global matcher / vocabulary tree rather than $O(n^2)$ exhaustive matching when $n$ is large.
3. **RANSAC a homography** per image pair. Brown & Lowe's probabilistic verification: accept a pair as truly overlapping if $n_i > \alpha + \beta n_f$ (inliers vs features in the overlap), typically $\alpha = 8, \beta = 0.3$.
4. **Bundle adjustment** over all images jointly — optimise rotations and focal lengths to minimise total reprojection error. Pairwise-only chaining accumulates drift and the panorama fails to close.
5. **Choose a warping surface.**
   - **Planar**: fine for < ~60–90° field of view. Beyond that, $\tan$ blows up — a ray at 90° from the optical axis is parallel to the image plane and never meets it, so its projected coordinate is infinite; a 180° panorama would need infinite width. This is a fact about planes, not about the algorithm, which is why the fix is to change the surface you project onto.
   - **Cylindrical**: $(x,y,z) \to (\arctan(x/z),\ y/\sqrt{x^2+z^2})$. Handles 360° horizontally; requires known focal length.
   - **Spherical**: full $360°\times180°$.
6. **Straightening**: the global rotation is under-determined; Brown & Lowe fix it by making the "up" vector the null vector of the covariance of camera $x$-axes (photographers rarely twist the camera).
7. **Exposure / gain compensation**: solve for per-image gains $g_i$ minimising intensity disagreement in overlaps.
8. **Seam finding**: graph cut / dynamic programming to route the seam through low-disagreement regions (around moving people, not through them).
9. **Multi-band blending.** The blending problem contains a genuine contradiction, which is why the naive fix fails. Blend over a *wide* transition and residual exposure differences are hidden — but any small misalignment now appears as a ghost, because both images contribute detail over a wide strip. Blend over a *narrow* transition and detail stays sharp — but the exposure step becomes a visible seam line. One transition width cannot satisfy both.

   The resolution is to notice that the two requirements apply to *different frequency bands*: exposure differences are low-frequency, ghosting is a high-frequency artefact. So decompose each image into frequency bands and give each band its own transition width — build **Laplacian pyramids** (1.4!) of each image and blend band $k$ over a transition width $\propto 2^k$. Low frequencies blend over a wide band (hiding exposure differences), high frequencies over a narrow band (avoiding ghosting/double edges), then sum the blended pyramid back up. This is the single most important quality step and it is a direct application of scale-space — the same decomposition, used to *represent* rather than to detect.

**When the homography model is wrong.** $H$ is exact only if (a) the scene is planar, or (b) the camera undergoes **pure rotation about its optical centre** (zero translation). Any camera translation with non-planar scene ⇒ **parallax** ⇒ no single $H$ fits ⇒ ghosting/misalignment in the overlap. This is why panorama apps show a "pivot in place" guide, and why stitching from a moving car fails. Modern fixes: as-projective-as-possible (APAP) warps, mesh-based / dual-homography warping, or accept it and do full SfM.

### Connect it

$H$ reappears in 5.4 as an element of the projective transformation group and in 5.5 as the plane-induced homography $H = K'(R - \mathbf{t}\mathbf{n}^\top/d)K^{-1}$ relating two views through a plane — which is exactly the degeneracy that breaks fundamental-matrix RANSAC (1.8): if a homography explains all your matches, $F$ is under-determined, and now you can see that as the same statement as "a plane kills depth as a variable." The DLT→SVD→LM pattern is the template for the 8-point algorithm (5.5) and bundle adjustment (5.8).

**In your own words:** why does taking a cross product let you write a linear equation for $H$, and why is the answer a singular vector rather than a least-squares solution?

### 🎯 Top-1% distinction

1. **Hartley normalisation and why** (conditioning). Highest-yield single fact in this section.
2. **8 DoF, not 9** — and be able to say the scale ambiguity is why we solve a homogeneous system with SVD.
3. **The two validity conditions** (planar scene *or* pure rotation) stated crisply, and parallax named as the failure.
4. **Algebraic vs geometric error**, and that DLT is only an initialiser.
5. **Multi-band blending as an application of Laplacian pyramids** — connecting 1.9 back to 1.4 unprompted is exactly the kind of synthesis that reads as mastery.
6. **Degenerate configurations:** any 3 of the 4 points collinear ⇒ $A$ rank-deficient ⇒ no unique $H$. RANSAC should reject such samples cheaply before fitting.

### ✅ Mastery check

**(a)** You have 4 exact correspondences and compute $H$ via DLT without normalisation, in an image of size $4000\times3000$. Estimate the order of magnitude of the condition number of $A$ and explain the mechanism.
**(b)** Your panorama of a room stitches perfectly on the far wall but shows a doubled chair in the foreground. Diagnose precisely and give two fixes at different points in the pipeline.
**(c)** How many correspondences are needed for an affine transform, and why is the linear system easier?

<details><summary>Answer sketch</summary>
(a) Entries of $A$ include terms like $x'\!\cdot\! x \sim 10^3 \times 10^3 = 10^6$, terms like $x \sim 10^3$, and terms equal to $1$. So column norms span roughly $10^6$ to $10^0$ ⇒ condition number on the order of $10^6$–$10^8$. In float32 (≈7 significant digits) the smallest singular vector is dominated by rounding error and $H$ is garbage; even in float64 the accuracy loss is severe. Hartley normalisation makes all coordinates $O(1)$, bringing $\kappa(A)$ to $O(10)$.
(b) The far wall is effectively planar and distant, so a single $H$ fits it. The foreground chair is at very different depth, so if the camera <b>translated at all</b> between shots, the induced parallax means the chair's correct alignment requires a different homography than the wall's. RANSAC picked the dominant (wall) homography; the chair is transferred to the wrong place → ghosting. Fixes: (i) <b>capture</b>: rotate about the camera's optical centre (nodal point), eliminating translation — the physically correct fix; (ii) <b>algorithm</b>: replace the global homography with a spatially-varying warp (APAP / mesh warp) or use seam-finding (graph cut) to route the seam <i>around</i> the chair so only one image contributes there, plus multi-band blending to hide the transition. The deepest answer: a single $H$ is the wrong model class for a scene with parallax; the correct model is full 3D (SfM + depth), and stitching is just accepting the wrong model where the error is small.
(c) Affine has 6 DoF ⇒ <b>3 correspondences</b> (each gives 2 equations). Easier because the last row is fixed at $(0,0,1)$, so the denominator is 1 — the equations are <b>linear and inhomogeneous</b> ($A\mathbf{x} = \mathbf{b}$), solvable by ordinary least squares with no scale ambiguity and no SVD null-space step.
</details>

### 🔨 Build + read

**Build:** Implement `dlt_homography(pts1, pts2, normalize=True)`. Run the ablation: with and without Hartley normalisation, on real pixel coordinates, and report the reprojection error of each — you should see a dramatic, visible difference. Then build a **two-image panorama end to end**: SIFT → ratio test → your RANSAC → your DLT → LM refinement (`scipy.optimize.least_squares` on symmetric transfer error) → warp → **multi-band blend with a 5-level Laplacian pyramid**. Compare against simple alpha blending; the seam difference is the payoff.

**Read:** Hartley & Zisserman, *Multiple View Geometry*, §4.1–4.4 (DLT, normalisation, the gold standard algorithm) — read this properly, it's ~15 pages and it is the foundation for all of Module 5. Then Brown & Lowe, "Automatic Panoramic Image Stitching using Invariant Features" (IJCV 2007) — the full production pipeline in one paper. Burt & Adelson (1983) for multi-band blending.

---

## 1.10 Where Classical CV Works vs. Fails — Synthesis

### The honest verdict

The question this section answers is the one an interviewer will actually ask you, in some disguise: **given that deep learning demonstrably beat hand-crafted features at recognition, why did none of Module 1 go away?**

The answer is that "vision" is two different kinds of problem wearing one name. Some vision tasks are *inference from evidence* — is this a pedestrian? — where there is no formula, only a mapping to be learned from examples, and a network is the right tool by default. Others are *measurement* — where is the camera, how far away is that wall? — where a correct answer exists, defined by projective geometry, and can be computed to arbitrary precision. Learning is how you win the first kind. It is not obviously how you win the second, and empirically it has not been.

So classical CV is not "the old way." It is **the way anything geometric still works**, and it coexists with deep learning in every serious production system. The two tables below are that split made concrete; read each row and check you can attribute it to inference-vs-measurement.

**Classical wins when:**

| Situation | Why |
|---|---|
| **Metric geometry is the output** (pose, depth, calibration, 3D reconstruction) | The problem is a well-posed algebraic/optimisation problem with a correct answer; a network can only approximate it, and with no guarantees. COLMAP is still the default SfM tool in 2026. |
| **Sub-pixel accuracy required** | Checkerboard corner refinement beats any learned detector for calibration. |
| **No training data / no GPU / hard latency budget** | ORB-SLAM3 runs at 30+ FPS on a CPU with zero training. |
| **Controlled environment** (factory inspection, document scanning, fiducial markers) | Variation is bounded, so hand-designed invariances suffice — and they're auditable. |
| **Interpretability / certification matters** | You can prove properties of a RANSAC + LM pipeline. You cannot prove properties of a ResNet. Relevant in medical, aerospace, automotive-safety contexts. |

**Classical fails when:**

| Situation | Why |
|---|---|
| **Semantics** ("is this a pedestrian?") | No hand-crafted feature encodes category. This is the whole reason deep learning took over. |
| **Textureless / repetitive surfaces** | No keypoints to detect (blank wall) or hopelessly ambiguous ones (tiled floor). |
| **Wide baseline / extreme viewpoint** | SIFT tolerates ~30° out-of-plane; beyond that, descriptors stop matching. |
| **Large illumination change** (day↔night, seasonal) | Gradient-based descriptors assume photometric consistency; day/night violates it wholesale. |
| **Deformable / non-rigid objects** | The whole rigid-transform machinery is the wrong model class. |
| **Motion blur, low light, low resolution** | Detectors are contrast-driven and degrade fast. |

### What the hybrid reality looks like in 2026

The modern SfM/SLAM/AR stack is a **layered hybrid**:

```
learned front-end                 classical back-end
─────────────────                 ──────────────────
SuperPoint / DISK / ALIKED   ┐
LightGlue / LoFTR / RoMa     ├──> RANSAC (MAGSAC++) ──> PnP / 5-pt ──> Bundle Adjustment ──> 3D
learned depth priors         ┘         ^                                      ^
                                       │                                      │
                              still pure geometry              still Levenberg–Marquardt
```

- **Front-end (perception, correspondence): learned.** Better repeatability under illumination and viewpoint change than SIFT — because "does this patch depict the same surface as that one" is an inference problem, and the invariances SIFT hand-codes are a strict subset of the ones data can teach.
- **Back-end (geometry, optimisation): classical.** Nobody has replaced bundle adjustment with a network, because BA is a well-conditioned sparse nonlinear least-squares problem with a known correct answer. A network could only approximate that answer, and it would do so without error bars, without a convergence guarantee, and without the ability to exploit sparsity that makes BA tractable on millions of points in the first place.
- Even **3D Gaussian Splatting (5.10)** is typically initialised from a **COLMAP** SfM point cloud — i.e. from SIFT + RANSAC + BA.

**The one-sentence version for an interview:** *"Deep learning replaced the parts of vision that were about recognition and correspondence quality; it has not replaced the parts that are about geometry and optimisation, and current systems are explicitly hybrid."*

### 🎯 Top-1% distinction

The failure mode here is **overclaiming in either direction**. "Classical CV is obsolete" and "you don't need deep learning for real vision" are both wrong and both read as inexperience. The strong position is the layered one above, with a concrete example (ORB-SLAM3 vs. a learned front-end + COLMAP back-end) and an honest statement of where the boundary currently sits.

Second discriminator: **being able to design the right pipeline for a stated constraint.** "Embedded, 5 W power budget, must localise in a warehouse" → ORB + binary descriptors + a small map, not a ViT. "Match tourist photos of a cathedral across seasons" → LoFTR/RoMa, not SIFT. Interviewers test judgement here more than knowledge.

### ✅ Module 1 synthesis check

Design, end to end, a system that takes ~200 handheld phone photos of a building exterior taken over two days (varying sun position, some motion blur) and produces a textured 3D model. Name every stage, the specific algorithm you'd use, and — for each stage — say whether it's classical or learned and **why you chose that**. Then name the three stages most likely to fail on this input and your mitigation for each.

### 🔨 Module 1 capstone build

**Two labs, one sitting each:**

**Lab A — SIFT vs ORB matching benchmark.** On the Oxford VGG affine dataset (graf = viewpoint, boat = rotation+scale, bikes = blur, leuven = illumination), compute for each method and each image pair: number of keypoints, repeatability, precision/recall of matches after ratio test, homography estimation error vs. ground truth, and wall-clock time for detect/describe/match. Produce one figure per dataset. **This single experiment gives you a factual, quotable answer to "when would you use ORB over SIFT?" — which is the most common Module 1 interview question.**

**Lab B — Panorama from scratch.** 5–8 handheld photos, full pipeline: detect → match → RANSAC → DLT+normalisation → LM bundle adjustment over all rotations and a shared focal length → cylindrical warp → gain compensation → graph-cut seam → multi-band blend. Deliberately shoot one set with translation (walk sideways) to *produce* the parallax ghosting failure, and document it. Understanding a failure you caused yourself is worth ten explanations.

---

## Going Deeper — Papers, Sources and Research Scope

*arXiv IDs are given where a paper is on arXiv. Most of Module 1 predates arXiv-by-default, so several entries are journal-only — search the title. If a link 404s, search rather than trusting the ID.*

### A. The canonical papers

Read in this order. The re-examinations matter as much as the originals: knowing that a classic was later shown to be partly wrong is the difference between having read a field and having memorised it.

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Canny, *A Computational Approach to Edge Detection* | 1986, PAMI | Edge detection posed as an **optimisation** with three stated criteria (good detection, good localisation, single response) rather than a recipe. The derivation is why Canny survived and Sobel-plus-threshold didn't. | IEEE PAMI 8(6) |
| Witkin, *Scale-Space Filtering* | 1983, IJCAI | The original argument that scale is a dimension, not a parameter. Short. | Proc. IJCAI |
| Lindeberg, *Feature Detection with Automatic Scale Selection* | 1998, IJCV | Where γ-normalisation comes from and why the LoG response peaks at the blob's true scale. This is the mathematical spine of 1.4. | IJCV 30(2) |
| Harris & Stephens, *A Combined Corner and Edge Detector* | 1988, Alvey Vision Conf. | Six pages. The structure tensor, and the reason you use $\det - k\,\mathrm{tr}^2$ instead of computing eigenvalues. | Alvey |
| Shi & Tomasi, *Good Features to Track* | 1994, CVPR | $\min(\lambda_1,\lambda_2)$ instead of Harris's determinant trick, argued from what tracking actually needs. Pairs with 5.7. | CVPR 1994 |
| **Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*** | 2004, IJCV | The SIFT paper. Read the whole thing — the ablations in §7 are a masterclass in justifying each design choice separately. | IJCV 60(2) |
| Mikolajczyk & Schmid, *A Performance Evaluation of Local Descriptors* | 2005, PAMI | The paper that established *how* to compare descriptors. The methodology is the contribution. | PAMI 27(10) |
| Rublee et al., *ORB: An Efficient Alternative to SIFT or SURF* | 2011, ICCV | rBRIEF and the steered-BRIEF variance/correlation analysis — the interesting part is the learning step that picks decorrelated binary tests. | ICCV 2011 |
| **Fischler & Bolles, *Random Sample Consensus*** | 1981, CACM | Two pages, still exactly correct. Read the original rather than a blog summary; the framing (fit-then-verify, not verify-then-fit) is clearer here than anywhere since. | CACM 24(6) |
| **Hartley, *In Defense of the Eight-Point Algorithm*** | 1997, PAMI | Formally about M5, but read it here: it is the cleanest demonstration in all of CV that a numerically-conditioned bad algorithm beats an unconditioned good one. Normalisation as conditioning. | PAMI 19(6) |
| Barath et al., *MAGSAC++* | 2020, CVPR (1912.05909) | Removes the inlier threshold by marginalising over it. The modern default in OpenCV. | arXiv 1912.05909 |
| Brown & Lowe, *Automatic Panoramic Image Stitching* | 2007, IJCV | The full pipeline end to end — matching, RANSAC, bundle adjustment, blending. This is your 1.9/1.10 practical, written up properly. | IJCV 74(1) |
| **DeTone et al., *SuperPoint*** | 2018, CVPRW (1712.07629) | The learned successor to SIFT, self-supervised via homographic adaptation. Read it to see exactly which parts of the classical pipeline survived. | arXiv 1712.07629 |
| **Sarlin et al., *SuperGlue*** | 2020, CVPR (1911.11763) | Matching as optimal transport with attention, replacing ratio-test + RANSAC. The re-examination that shows how much of Module 1 is now learned. | arXiv 1911.11763 |

### B. The single best source, per hard topic

Not a link dump — one resource per concept, chosen because it explains that concept better than the alternatives.

- **Scale-space and why $\sigma$ is the only kernel (1.4).** Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., §3.5 — free PDF at `szeliski.org/Book`. Read alongside Lindeberg 1998 for the γ-normalisation argument Szeliski compresses.
- **Convolution, filtering and separability (1.2).** Shree Nayar, *First Principles of Computer Vision*, the "Image Filtering" module on YouTube. Nayar draws the signal-processing intuition better than any text.
- **Canny's three criteria (1.3).** The original 1986 paper's §II. It is more readable than its reputation.
- **Harris corners (1.5).** Szeliski §7.1.1 for the derivation; then the Alvey paper for the $k$ trick.
- **SIFT, in implementation detail (1.6).** The `VLFeat` SIFT tutorial (`vlfeat.org/api/sift.html`) — it documents the exact orientation-histogram and descriptor-normalisation details Lowe's paper leaves implicit.
- **RANSAC's iteration count (1.8).** Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., §4.7 (pp. 117–121) — the adaptive-$N$ table on p. 119 is the one to internalise.
- **Homography and the DLT (1.9).** Hartley & Zisserman §4.1 (pp. 88–93). The normalisation discussion in §4.4 is the practically important part.

### C. Reference implementations worth reading

- **`opencv/opencv` → `modules/features2d/src/sift.simd.hpp`.** Look specifically for `calcSIFTDescriptor` — the trilinear interpolation into the 4×4×8 histogram, and the two-stage normalisation (L2 → clamp at 0.2 → L2 again). That clamp is the illumination-robustness hack the paper mentions in one sentence.
- **`opencv/opencv` → `modules/calib3d/src/ptsetreg.cpp`.** `RANSACPointSetRegistrator::run` — see how the iteration count is *updated adaptively* as better models are found, rather than fixed up front.
- **`colmap/colmap` → `src/feature/sift.cc`.** Production-grade SIFT with GPU paths; useful for seeing what matters at scale (guided matching, ratio-test thresholds, cross-checking).
- **`magicleap/SuperGluePretrainedNetwork`.** Read `models/superglue.py` — the Sinkhorn iterations and the dustbin channel that lets a keypoint match *nothing*. That dustbin is the single most elegant idea in learned matching.
- **`rmislam/PythonSIFT`.** A readable, dependency-light SIFT in pure NumPy. Read it *after* you've implemented your own, to check your DoG octave handling.

### D. Open research questions

Honest labelling: most of Module 1 is closed. These are the live threads.

1. **Do learned detectors actually beat SIFT under distribution shift, or only on the benchmarks they were tuned on?** *Why open:* HPatches and the standard benchmarks are largely planar, well-lit and photometrically mild. SuperPoint/SuperGlue win there decisively; the picture under night, weather, and cross-season shift is much less clear, and evaluation papers disagree. *Minimum experiment:* fixed matching protocol, SIFT vs ORB vs SuperPoint vs a modern dense matcher, evaluated on Aachen Day-Night and a season-varying set, reporting pose error rather than match count. Inference-only — laptop GPU is enough. **Feasible.**
2. **How much of RANSAC's practical success is the sampling and how much is the local optimisation?** *Why open:* LO-RANSAC's inner refinement gives large gains, and modern variants bundle several improvements at once, so the ablation is rarely clean. *Minimum experiment:* implement vanilla / LO / MAGSAC++ under one harness, ablate the local-optimisation step alone, on synthetic data with controlled outlier ratio. CPU only. **Very feasible — this is a good weekend project that ends in a real plot.**
3. **Is the ratio test optimal, or merely convenient?** *Why open:* Lowe's 0.8 threshold has been used essentially unchanged for twenty years, and its justification is one histogram in the 2004 paper. *Minimum experiment:* derive the likelihood-ratio test the ratio test approximates under a stated descriptor-distance model, then measure where the approximation breaks. **Feasible and genuinely under-examined.**
4. **Does classical feature matching still win anywhere at inference cost parity?** *Why open:* comparisons usually ignore that SIFT runs on a CPU. *Minimum experiment:* fix a latency budget on a specific device, then ask which method wins *within* the budget. Ties directly to `Supplement-Capstone-Projects.md` #4. **Feasible.**
5. ~~A better corner detector.~~ **Closed.** Do not spend time here; the descriptor and the matcher, not the detector, are where the remaining error lives.

---

## Module 1 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Colour spaces | HSV decouples intensity but Hue is circular and undefined at low saturation; LAB is perceptually uniform so Euclidean distance ≈ perceived difference |
| Gamma | sRGB is non-linear; blending/resizing is only correct in linear light |
| Gaussian | separable, no Fourier side lobes, semigroup $G_{\sigma_1}*G_{\sigma_2} = G_{\sqrt{\sigma_1^2+\sigma_2^2}}$, unique scale-space kernel |
| Canny | DoG → NMS (thins to 1 px) → hysteresis (exploits connectivity); optimal under Canny's 3 criteria |
| Scale-space | $\partial G/\partial\sigma = \sigma\nabla^2 G$ ⇒ DoG $\approx (k-1)\sigma^2\nabla^2 G$; $\sigma^2$ normalisation is mandatory; Gaussian is unique by the causality axiom |
| Blob size | detected $\sigma$ ⇒ radius $\sqrt{2}\sigma$ |
| Harris | structure tensor $M$; $R = \det - \kappa\,\mathrm{tr}^2$; rotation-invariant, **not** scale-invariant; same $M$ governs Lucas–Kanade trackability |
| SIFT | DoG extrema → sub-pixel fit → contrast + edge rejection → orientation histogram → $4{\times}4{\times}8$=128-D → normalise/clamp 0.2/normalise |
| ORB | oriented FAST (Harris-ranked) + rBRIEF (learned decorrelated binary tests), 32 bytes, Hamming distance |
| Matching | ratio test $d_1/d_2 < 0.8$ removes ~90% of false matches, loses ~5% of true; L2 ≡ cosine for normalised vectors |
| RANSAC | $N = \log(1-p)/\log(1-w^s)$; exponential in $s$ ⇒ minimal solvers; $t^2 = \chi^2_{m,\alpha}\sigma^2$; always refit on inliers |
| Homography | 8 DoF; DLT + **Hartley normalisation** + SVD null space; valid only for planar scenes or pure rotation |
| Panorama | RANSAC $H$ → BA → cylindrical warp → gain compensation → graph-cut seam → **multi-band (Laplacian pyramid) blend** |
| Classical vs DL | learned front-end (features/matching), classical back-end (RANSAC, PnP, BA); hybrid is the 2026 answer |

---

*End of Module 1 notes. Drills and self-test are in `Module-01-Drills.md`.*
