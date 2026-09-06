# Module 5 — Geometry & 3D Vision

> **How to use this module.** Your gap analysis called this "the module with the most genuine holes" — the topic names were right but the connective mathematics was missing. So this module is deliberately concentrated: 5.5 (epipolar geometry) and 5.8 (bundle adjustment) are treated at full depth because they are the load-bearing pieces, and the rest is tight but complete.
>
> **Career framing, honestly.** For a GenAI/LLM-engineering track, Module 5 is the *least* directly applicable module. But it is the one that separates people who "know CV" from people who know computer vision — and it is disproportionately used as a filter question precisely because most candidates skip it. Two specific derivations (the essential matrix, and why bundle adjustment's sparsity makes it tractable) are worth more than a broad shallow pass over the whole module.

**Concept map**

```
5.1 Pinhole ──> 5.2 Intrinsics/extrinsics ──> 5.3 Calibration ──> 5.4 Homography revisited
                          │                                              │
                          v                                              v
                  5.5 Epipolar geometry 🔴 ──> 5.6 Stereo/disparity   (projective hierarchy)
                          │                          │
                          v                          v
                  5.7 Optical flow & tracking ──> 5.8 SfM + Bundle Adjustment 🔴 ──> 5.9 SLAM
                                                          │
                                                          v
                                                  5.10 NeRF → 3DGS 🟢 ──> 5.11 Practical
```

---

## 5.1 Pinhole Camera & Perspective Projection

### Intuition

**Start with the question this whole module exists to answer: a photograph is a flat array of numbers, so where did the third dimension go, and can we ever get it back?**

The pinhole model is the answer to the first half. A camera turns a 3-D world into a 2-D image by throwing away depth, and the model says exactly *how* it throws it away: draw a straight line from a 3-D point through a single point (the camera centre) onto the image plane; wherever that line pierces the plane is the pixel. Nothing else happens.

Now notice what that construction does. Every 3-D point on that same line lands on the same pixel — a point 2 m along the ray and a point 200 m along it are indistinguishable in the image. So the mapping is **many-to-one**: an image point doesn't determine a 3-D point, it determines a *ray*. Every difficulty and every technique in this module is a consequence of that one sentence.

Real-life anchor: why two eyes. One eye gives you a ray; two eyes give you two rays, and where they intersect is the point. That sentence is the whole of Module 5 — 5.5 works out when two rays *can* be intersected, 5.6 works out how accurately, and 5.8 works out how to do it with a thousand rays at once.

### The math

**Put the camera at the origin looking down $+Z$, and the image plane at $Z = f$.** A 3-D point $(X,Y,Z)$ and its image $(x,y)$ then sit on the same line through the origin, so the two right-angled triangles they form — one with height $X$ and base $Z$, the other with height $x$ and base $f$ — are similar. Equal ratios of corresponding sides give the whole model:

$$
x = f\frac{X}{Z}, \qquad y = f\frac{Y}{Z}
$$

That division by $Z$ is **the** nonlinearity of vision. It is worth pausing on how much trouble one division causes:
- Distant objects appear smaller — apparent size falls off as $1/Z$, which is a *hyperbola*, not a line, so intuitions built on linear scaling mislead badly at range.
- Parallel lines converge to **vanishing points**. (Push a point out along a fixed direction and its image tends to a finite limit rather than running off the page — a family of parallel 3-D lines with direction $\mathbf{d}$ meets at image point $K\mathbf{d}$, its vanishing point.)
- **Depth is unrecoverable from one view.** Take any point and slide it along its ray: $(X,Y,Z) \to (\lambda X, \lambda Y, \lambda Z)$. The $\lambda$ cancels top and bottom in both $fX/Z$ and $fY/Z$, so $(x,y)$ is *unchanged*. This is the *projective ambiguity*, and it is why monocular metric depth is fundamentally ill-posed without a prior.

**Now the move that makes everything after this page possible — and the one that most often gets filed as "a trick" instead of understood.**

The problem with $x = fX/Z$ is that it is not linear in $(X,Y,Z)$: division by an unknown is not a matrix multiply, so we cannot compose cameras, invert transformations, or stack constraints into a linear system. All of linear algebra is locked out by one division.

**Homogeneous coordinates buy the linearity back by refusing to do the division.** Represent a 3-D point as $\tilde{\mathbf{X}} = (X,Y,Z,1)^\top$ and an image point as $\tilde{\mathbf{x}} = (x,y,1)^\top$, and declare that both are defined **only up to a non-zero scale factor** — $(x,y,1)$, $(2x,2y,2)$ and $(\lambda x, \lambda y, \lambda)$ all name the same image point, and you recover the actual pixel by dividing through by the last coordinate whenever you finally need a number.

Then projection is a matrix:

$$
\lambda\begin{bmatrix}x\\y\\1\end{bmatrix} = \begin{bmatrix}f&0&0&0\\0&f&0&0\\0&0&1&0\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
$$

Multiply out the right-hand side and you get $(fX, fY, Z)^\top$. Divide by the third coordinate — the step the "up to scale" convention lets us postpone — and you get exactly $(fX/Z,\ fY/Z,\ 1)$. **The division didn't disappear; it was moved out of the matrix and into the meaning of the equals sign.** That is the entire trade, and it is why the equation carries a $\lambda$ on the left: the matrix delivers the answer scaled by an unknown factor (here $\lambda = Z$), and we have agreed not to care.

**The thing that trips people up: "equal up to scale" sounds like sloppy bookkeeping. It is not — it is a statement about geometry.** The set of all vectors $\lambda(x,y,1)^\top$ for $\lambda \neq 0$ is a *line through the origin* in $\mathbb{R}^3$. So a homogeneous image point is not a point at all: it is a ray. And that is precisely the object the camera actually gives us, as the Intuition section argued — an image measurement determines a ray, never a point. Homogeneous coordinates are not a hack bolted onto the geometry; they are the notation in which the geometry is honest about what it knows.

Follow the consequence: the scale $\lambda$ we agreed to discard is $Z$ itself. **Throwing away scale and losing depth are the same act.** So the "up to scale" is not a technicality — it *is* the depth ambiguity, encoded algebraically. Homogeneous coordinates make projection linear at the cost of making equality mean "equal up to scale," and that trade is what makes all of multi-view geometry tractable.

One more dividend, which explains the vanishing-point claim above. A homogeneous vector whose last coordinate is $0$ — like $(d_1,d_2,d_3,0)^\top$ — cannot be dehomogenised by dividing, because the division blows up. These are the **points at infinity**, and they encode pure *directions* rather than locations. Projecting the direction $\mathbf{d}$ through the camera gives its vanishing point, which is why the formula is $K\mathbf{d}$ and why it depends on direction alone and not on where the line sits in space. Infinity stops being a limiting process and becomes an ordinary vector you can multiply by a matrix.

**Pause:** before reading on — in homogeneous image coordinates, is $(6, 4, 2)^\top$ the same point as $(3, 2, 1)^\top$? And is $(3,2,0)^\top$ anywhere near either of them?

The first two are the **same** point: $(6,4,2) = 2\cdot(3,2,1)$, and scale doesn't count, so both dehomogenise to the pixel $(3,2)$. The third is *not* a scaled copy of either — no $\lambda$ makes $\lambda\cdot 1 = 0$ — and it isn't a pixel at all. It is the point at infinity in the direction $(3,2)$, i.e. where all image lines of slope $2/3$ meet. Two vectors that look numerically similar can live in completely different parts of the space; what matters is the ratio pattern, never the magnitudes.

**In your own words:** why does adding a coordinate turn a division into a matrix multiply, and what did we give up in exchange?

### Why the perspective camera is not a "camera model choice"

Perspective is not one option on a menu — it is what happens when rays converge to a single centre. But it is worth knowing what you get if that convergence is *negligible*, because the mathematics simplifies dramatically.

If every visible point sits at roughly the same distance $Z \approx Z_0$ — a telephoto lens on a distant subject, say — then $x = fX/Z \approx (f/Z_0)X$ and the troublesome division is now by a *constant*. That is a linear map, and the whole apparatus of homogeneous coordinates is no longer buying you anything. Alternatives worth naming on that spectrum: **orthographic** ($x = X$, no division at all — the limit of an infinitely distant camera), **weak perspective** (scale by the average depth, as just derived), and **affine cameras** generally. These matter because SfM under an affine camera is a *linear* problem — with the division gone, the measurement matrix factorises directly by SVD (Tomasi–Kanade factorisation) — which is why affine SfM was solved decades before projective SfM. The extra difficulty of the projective case is, once again, entirely the fault of dividing by an unknown $Z$.

### 🎯 Top-1% distinction

- **"Depth is lost because projection is defined up to scale"** — state the ambiguity algebraically, not just as "you can't tell depth from one image."
- **Vanishing points are $K\mathbf{d}$** — and they are how you calibrate from a single architectural photo, or estimate horizon and camera rotation from a road scene.
- **Monocular depth networks don't solve the geometry** — they exploit *priors* (object sizes, texture gradients, learned scene statistics), which is exactly why they fail on scenes that violate those priors (a dollhouse, a forced-perspective photo) and why they are metrically ambiguous up to a scale factor unless trained with metric supervision.

### ✅ Mastery check

(a) A 2 m-tall adult stands 10 m from a camera with $f = 800$ px; a 1 m-tall child stands 5 m away. Compute both image heights. What does the result say about estimating physical size from one image?
(b) Railway tracks converge in a photo. Where exactly is the convergence point, in terms of $K$ and the track direction $\mathbf{d}$?
(c) A monocular depth network gives a confident, plausible depth map for a photo of a *dollhouse*. Explain precisely why it is wrong, and what the network actually learned.

<details><summary>Answer sketch</summary>
(a) Adult: $h = fY/Z = 800\times2/10 = \mathbf{160}$ px. Child: $800\times1/5 = \mathbf{160}$ px. <b>Identical.</b> Image size confounds physical size with depth — the projection $y = fY/Z$ has one equation and two unknowns, so <b>size is unrecoverable from one view without a prior</b> (a known object class, a known ground plane, or a second view). This is the scale ambiguity of 5.1 in its most concrete form, and it is why "how far away is that?" is ill-posed monocularly.
(b) At the <b>vanishing point</b> $\mathbf{v} \simeq K\mathbf{d}$ — the image of the direction $\mathbf{d}$, i.e. where a point on the tracks projects as it recedes to infinity. Note it depends only on the <i>direction</i>, so all lines parallel to $\mathbf{d}$ share it, whatever their position.
(c) The network never solved the geometry — it exploited <b>learned priors</b>: apparent object sizes, texture gradients, perspective cues, and scene-layout statistics from its training distribution. A dollhouse violates every one of those priors while preserving the image evidence, so the model confidently reports the depths of a full-sized room. Two corollaries worth stating: monocular depth is <b>relative, not metric</b>, unless trained with metric supervision and known intrinsics; and confident-and-wrong is its characteristic failure, because nothing in the input signals that the priors do not apply. Forced-perspective photos and miniature-model shots break it identically.
</details>


### 🔨 Build + read

**Build:** Take three photos of the same building facade — one wide-angle, one normal, one long-lens (zoom digitally if you must, but a real focal-length change is better) — from progressively further back so the subject stays the same size in frame. Overlay the horizons and measure vanishing points for a set of parallel edges (e.g. window tops) in each. Confirm they move as $K$ changes but the *scene* geometry (parallel lines meeting at one point) doesn't. Then take one photo of railway tracks or a tiled floor receding into the distance and compute the vanishing point pixel coordinates by fitting lines through corresponding edges and intersecting them — compare against $K\mathbf{d}$ if you know your phone's approximate focal length in pixels.

**Read:** Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., §2.1–2.2 and Ch. 6 intro (pp. 25–37, 153–158) for the homogeneous-coordinates and projection formalism. Then Shree Nayar's *First Principles of Computer Vision*, "Image Formation" module, for the physical picture behind the algebra.

---

## 5.2 Intrinsic & Extrinsic Parameters

### The full projection equation

**5.1 cheated twice, and this section pays both debts.** We put the camera at the origin looking down $+Z$ — real cameras are somewhere else, pointing somewhere else. And we measured the image in metres on a plane at $Z=f$ — real images are indexed in pixels from a corner. So the full pipeline is three steps, and each is a matrix precisely because 5.1's homogeneous trick made projection linear:

1. **Move the world into the camera's frame** (a rigid motion — rotation $R$ then translation $\mathbf{t}$): the *extrinsics*.
2. **Project** (the $1/Z$ division of 5.1): the pinhole matrix.
3. **Convert metres on the image plane into pixel indices** (scale by pixel density, shift the origin to the corner): the *intrinsics*.

Chain them and steps 2 and 3 collapse into a single $3\times3$ matrix $K$ sitting in front of the $3\times4$ rigid motion:

$$
\lambda\tilde{\mathbf{x}} = \underbrace{K}_{\text{intrinsics}}\ \underbrace{[\,R \mid \mathbf{t}\,]}_{\text{extrinsics}}\ \tilde{\mathbf{X}}_{\text{world}}, \qquad P = K[R\mid\mathbf{t}]
$$

$P$ is the $3\times4$ **camera matrix**. It has 12 entries, but multiplying the whole thing by any non-zero constant multiplies $\lambda$ by the same constant and leaves the pixel unchanged — the "up to scale" of 5.1 again — so one entry is free. **11 degrees of freedom.** Hold on to that number; we are about to account for all eleven from the physics.

**Intrinsics** — the camera's internal geometry:

$$
K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}
$$

Every entry earns its place by doing one job in step 3:

- $f_x, f_y$: focal length in **pixels**. This is just a unit conversion — the physical focal length divided by the physical width of one pixel, $f_x = f/\text{pixel width}$ — which is why the *same lens* gives different $f_x$ on different sensors. $f_x$ and $f_y$ differ only if pixels are non-square (rare in modern sensors, common in old frame-grabbers and in anamorphic optics).
- $(c_x,c_y)$: **principal point** — where the optical axis pierces the image. This is the origin shift: 5.1 measured from the optical axis, image files index from a corner. Usually near but not exactly at the image centre, because the sensor is never mounted perfectly.
- $s$: **skew** — non-perpendicular pixel axes, i.e. a sensor whose rows and columns are not at right angles. Essentially always 0 for real cameras; keep it in the model only for pathological cases.

That is 5 DoF (or 4 with $s=0$).

**Extrinsics** — where the camera is: $R \in SO(3)$ (3 DoF — a rotation in 3-D needs three numbers, an axis direction and an angle) and $\mathbf{t} \in \mathbb{R}^3$ (3 DoF), for 6 DoF total. Together: $5 + 6 = 11$. ✓ **The eleven from the physics and the eleven from counting matrix entries agree**, which is a small but real sanity check that the model has no slack in it.

**The $\mathbf{t}$ trap — and why the fix is what it is.** $\mathbf{t}$ is **not** the camera's position in world coordinates. It is tempting to read $[\,R\mid\mathbf{t}\,]$ as "the camera is rotated by $R$ and sits at $\mathbf{t}$", but the matrix describes the *world-to-camera* transform, not the camera's pose in the world. Written out:

$$\mathbf{X}_{\text{cam}} = R\mathbf{X}_{\text{world}} + \mathbf{t}$$

**Derive the camera centre rather than memorising it.** Ask what world point $\mathbf{C}$ lands at the camera's own origin — because the camera centre is, by definition, the one point that has coordinates $\mathbf{0}$ in the camera's frame. Set $\mathbf{X}_{\text{cam}} = \mathbf{0}$ and $\mathbf{X}_{\text{world}} = \mathbf{C}$:

$$\mathbf{0} = R\mathbf{C} + \mathbf{t} \quad\Longrightarrow\quad R\mathbf{C} = -\mathbf{t} \quad\Longrightarrow\quad \mathbf{C} = -R^{-1}\mathbf{t}$$

and since $R$ is a rotation, $R^{-1} = R^\top$ — that is the entire content of the orthogonality constraint $R^\top R = I$, and it is why inverting a pose costs a transpose rather than a matrix inversion. Hence

$$
\mathbf{C} = -R^\top \mathbf{t}
$$

Read it physically: $\mathbf{t}$ is the world origin *as seen from the camera*, so to get the camera as seen from the world you must undo the rotation ($R^\top$) and reverse the direction (the minus sign). Three lines of algebra, no memorisation. Getting this backwards is the single most common bug in pose code — it produces a reconstruction that looks *almost* right, with cameras mirrored through the origin — and stating it correctly is a quick credibility signal.

**Normalised coordinates.** If $K$ is the map from rays to pixels, then $K^{-1}$ is the map back. So $\hat{\mathbf{x}} = K^{-1}\tilde{\mathbf{x}}$ takes a pixel measurement and undoes the sensor's idiosyncrasies, leaving **the direction of the ray in the camera's own frame** — what the measurement would have been on an idealised camera with $f=1$ and the principal point at the origin. Two different cameras looking at the same scene produce incomparable pixel coordinates but perfectly comparable normalised ones. This is what "calibrated" means, and it is exactly the difference between the fundamental matrix and the essential matrix (5.5): $E$ operates on the rays, $F$ operates on the pixels and has to carry the unknown $K$ around inside itself.

**In your own words:** why is $\mathbf{t}$ not the camera's position, and what does $R^\top$ undo?

### Rotation representations 🟡

**A rotation has 3 degrees of freedom, but a rotation matrix has 9 numbers. Where do the other 6 go, and does it matter?** They go into constraints: $R^\top R = I$ is 6 equations (3 unit-norm columns, 3 orthogonality conditions between pairs), and $\det R = +1$ picks the rotation branch over the reflection branch. It matters enormously the moment you want to *optimise* over rotations, because a gradient step on 9 free numbers walks straight off the constraint surface and gives you a matrix that is no longer a rotation. Every representation below is a different answer to "how do I carry 3 degrees of freedom without carrying 6 constraints", and each answer costs something:

| Representation | DoF / params | Pros | Cons |
|---|---|---|---|
| **Rotation matrix** $R$ | 3 / 9 | direct, composable | 6 redundant constraints ($R^\top R = I$, $\det = 1$) — hard to optimise |
| **Euler angles** | 3 / 3 | interpretable | **gimbal lock**, order-dependent, non-unique |
| **Axis–angle / rotation vector** $\boldsymbol{\omega}$ | 3 / 3 | minimal, no constraints, the **Lie-algebra $\mathfrak{so}(3)$** parameterisation | singular at $\|\omega\| = 2\pi$; needs exp/log maps |
| **Quaternion** $q$ | 3 / 4 | no gimbal lock, smooth interpolation (SLERP), cheap composition | double cover ($q$ and $-q$ are the same rotation), one norm constraint |
| **6-D continuous** (Zhou et al. 2019) | 3 / 6 | **continuous** — no discontinuity, which matters for neural network regression | not standard in classical geometry |

**For optimisation (bundle adjustment), use the axis–angle / Lie-algebra parameterisation**: it is minimal and unconstrained, so a Gauss–Newton step stays on the manifold via the exponential map. Ceres and g2o both do this. **For neural network regression, use the 6-D representation** — the key insight of Zhou et al. is that quaternions and Euler angles are *discontinuous* as functions on $SO(3)$, so a network must learn a discontinuous map and does so badly. That paper is a great one to name.

### 🎯 Top-1% distinction

- $\mathbf{C} = -R^\top\mathbf{t}$.
- **Focal length is in pixels**, and it is not a property of the lens alone — it depends on the sensor's pixel pitch. Two cameras with the same lens and different sensors have different $f_x$.
- **Explain why $R$ is hard to optimise directly** (over-parameterised with constraints) and name the Lie-algebra fix.
- **The 6-D rotation representation for networks**, with the continuity argument.

### ✅ Mastery check

A camera has $f = 24$ mm, sensor 36×24 mm, image 6000×4000 px. Give $K$ (assume $s=0$, principal point at the centre). Then: the camera is at world position $(1,2,3)$ with orientation $R$. Write $\mathbf{t}$. Finally: you crop the image to the central 3000×2000 px. What is the new $K$?

<details><summary>Answer sketch</summary>
Pixel pitch $= 36\,\text{mm}/6000 = 6\,\mu$m, and $24/4000 = 6\,\mu$m — square pixels. $f_x = f_y = 24\,\text{mm}/0.006\,\text{mm} = \mathbf{4000}$ px. $c_x = 3000$, $c_y = 2000$. So $K = \begin{bmatrix}4000&0&3000\\0&4000&2000\\0&0&1\end{bmatrix}$.
$\mathbf{t} = -R\mathbf{C} = -R(1,2,3)^\top$. (From $\mathbf{C} = -R^\top\mathbf{t}$.)
<b>After a central crop to 3000×2000:</b> focal length is unchanged ($f_x = f_y = 4000$ — cropping doesn't change the optics or the pixel pitch), but the principal point moves because the origin moved: the crop removes 1500 px from the left and 1000 from the top, so $c_x = 3000 - 1500 = 1500$, $c_y = 2000 - 1000 = 1000$. New $K = \begin{bmatrix}4000&0&1500\\0&4000&1000\\0&0&1\end{bmatrix}$. (Contrast with <b>resizing</b> by ½, which scales <i>all four</i> of $f_x,f_y,c_x,c_y$ by ½ — confusing crop with resize is a classic bug.)
</details>


### 🔨 Build + read

**Build:** In Python (NumPy only), construct a synthetic scene: 8–10 known 3D points (e.g. corners of a cube) and a chosen $K$, $R$, $\mathbf{t}$. Project them to pixels via $P = K[R|\mathbf{t}]$. Then invert the problem: given the 3D points and their projected pixels, recover $R$ and $\mathbf{t}$ by hand-rolling a direct linear transform (DLT) for the camera pose, and verify you recover the originals to numerical precision. Perturb $\mathbf{t}$'s recovered value and confirm it is *not* the camera centre — recompute the true centre as $-R^\top\mathbf{t}$ and check it against the ground truth you started with.

**Read:** Hartley & Zisserman §6.1–6.2 (pp. 153–164), specifically the camera-centre derivation on p. 156. Then Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., §2.1 for the intrinsic-matrix parameterisation (skew, principal point, aspect ratio) in the same notation you'll meet in OpenCV.

---

## 5.3 Camera Calibration & Lens Distortion

### Distortion models

The pinhole model is a lie about real lenses. A pinhole is a *point*; a lens is a stack of curved glass with finite aperture, and light passing near its edge bends by slightly the wrong amount. The visible consequence is that straight lines in the world come out curved in the image — the giveaway that the model needs a correction term.

**What kind of correction?** The error is caused by how far off-axis a ray passes, so to first approximation it depends only on the *radius* $r$ from the principal point, not on the angle around it. That immediately tells you the functional form: a radially symmetric function of $r$, and since the distortion must vanish at the centre and be symmetric under $r \to -r$, only even powers survive. Expand as a power series in $r^2$ and truncate — that is where $k_1 r^2 + k_2 r^4 + k_3 r^6$ comes from, and it is why the model has no odd terms and no $\theta$-dependence. **Radial distortion** (barrel/pincushion) dominates, with a smaller *tangential* term for a lens that isn't perfectly parallel to the sensor:

$$
\begin{aligned}
x_{\text{dist}} &= x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \underbrace{2p_1 xy + p_2(r^2 + 2x^2)}_{\text{tangential}}\\
y_{\text{dist}} &= y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \underbrace{p_1(r^2+2y^2) + 2p_2xy}_{\text{tangential}}
\end{aligned}
$$

with $r^2 = x^2+y^2$ in **normalised** coordinates. Note: distortion is applied *in normalised coordinates, before* multiplying by $K$ — getting that order wrong is a common bug. The reason it must be that way round is physical, not conventional: the bending happens in the lens, to the *ray*, before the sensor ever converts anything into pixels. Modelling it after $K$ would be claiming the sensor's pixel grid causes the distortion. It also explains a consequence people get wrong constantly — because $k_1\ldots p_2$ live in normalised coordinates, which are already free of pixel units, **they do not change when you resize an image**, whereas every entry of $K$ does.

- $k_1 < 0$ ⇒ barrel (wide-angle); $k_1 > 0$ ⇒ pincushion (telephoto).
- $k_3$ is only needed for strong wide-angle lenses; fitting it on a normal lens overfits and produces a worse model. **Model selection matters** — more parameters is not better.
- Fisheye lenses need a different model entirely (equidistant/Kannala-Brandt: $r_d = f\theta$ rather than $r_d = f\tan\theta$), because $\tan\theta \to \infty$ at 90° so a perspective model cannot represent a ≥180° field of view.

### Zhang's method (2000) — the standard

**The problem to solve: we need $K$, but every image we take mixes $K$ with an unknown $R$ and $\mathbf{t}$. How do you separate them?** Zhang's answer is to use a calibration object whose geometry you *know* — a flat checkerboard — and then exploit the fact that a rotation matrix's columns are orthonormal. The known board pins down the pose enough that what's left over must be attributable to $K$.

Photograph a planar checkerboard in $\ge 3$ orientations.

1. Detect corners with sub-pixel refinement (Module 1.5 — this is where corner localisation accuracy pays off directly).
2. For each view, the plane→image map is a **homography** (1.9), estimated by DLT + Hartley normalisation.
3. Here is the key step. Put the board at $Z=0$ in world coordinates — you are allowed to, since you chose the world frame. Then the third column of $R$ multiplies a coordinate that is always zero and drops out entirely, so $P = K[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{r}_3\ \mathbf{t}]$ collapses to $H = K[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}]$. **A homography of a plane is a camera matrix with one column deleted** — that is the whole reason planar calibration works. Now invert it: $[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}] \simeq K^{-1}H$, so the first two columns of $K^{-1}H$ *must* be the first two columns of a rotation matrix, i.e. unit length and mutually perpendicular. Those are two facts we know for free about a quantity we can measure, and each one is a **constraint** on the unknown $K$:
   $$
   \mathbf{h}_1^\top K^{-\top}K^{-1}\mathbf{h}_2 = 0, \qquad \mathbf{h}_1^\top K^{-\top}K^{-1}\mathbf{h}_1 = \mathbf{h}_2^\top K^{-\top}K^{-1}\mathbf{h}_2
   $$
   Writing $B = K^{-\top}K^{-1}$ (the **image of the absolute conic**, symmetric, 6 unknowns up to scale = 5 DoF) makes these linear in $B$.
4. With $\ge 3$ views you get $\ge 6$ constraints for 5 DoF → solve linearly, then recover $K$ by Cholesky.
5. Recover per-view $R,\mathbf{t}$, initialise distortion to zero, then **refine everything by non-linear least squares (Levenberg–Marquardt) on total reprojection error** — the same DLT-then-LM pattern as 1.9, and the same pattern as bundle adjustment (5.8).

**Why $\ge 3$ views:** each view gives 2 constraints; 5 intrinsic DoF needs $\ge 3$ views (6 constraints). In practice use 15–25 views covering the whole image area and a range of tilts, because distortion parameters are only observable where you have data — **a calibration whose corners never reach the image corners will have garbage $k_1,k_2$ there**.

### Practical calibration failures

| Symptom | Cause |
|---|---|
| High reprojection error (> ~0.5 px) | blurry images, poor corner detection, wrong board dimensions, or a non-planar (warped) board |
| Distortion wrong at image edges | no calibration images with the board near the edges |
| Unstable focal length | all views at similar depth/orientation ⇒ $f$ and $t_z$ are nearly degenerate |
| Works then drifts | autofocus or zoom changed $f$; thermal expansion; someone bumped the lens |
| Good RMS, bad 3D | RMS is an *average* — check the per-corner error map for structure |

**Autofocus is the enemy of calibration.** Lock focus and zoom, or recalibrate. Phone cameras with continuous autofocus have a *variable* $f$, which is why phone-based photogrammetry pipelines usually re-estimate intrinsics per capture.

### 🎯 Top-1% distinction

- **Zhang's method rests on the orthonormality of the rotation columns**, giving 2 constraints per homography — state that, don't just say "wave a checkerboard."
- **Distortion is applied in normalised coordinates before $K$.**
- **More distortion coefficients is not better** — $k_3$ overfits on normal lenses.
- **Fisheye needs a different projection model**, not more radial terms.
- **RMS reprojection error is an average and can hide structured error** — always look at the residual map.
- **Sub-pixel corner refinement is what makes calibration accurate**, connecting directly to Module 1.5.

### ✅ Mastery check

(a) Your calibration reports an RMS reprojection error of 0.2 px, yet the 3-D reconstruction built on it is visibly wrong. Give three causes that a low RMS would not reveal.
(b) You calibrate at $1920\times1080$, then deploy on images resized to $1280\times720$. What changes in $K$, and what must not?
(c) Why does a phone with continuous autofocus break calibration, and what do photogrammetry pipelines do about it?

<details><summary>Answer sketch</summary>
(a) 1. <b>RMS is an average.</b> A structured residual pattern — systematically radial, or one whole bad image — can sit inside a small mean. Always look at the per-corner residual <i>map</i>, not the scalar. 2. <b>Poor observability.</b> If all views are at similar depth and orientation, $f$ and $t_z$ are nearly degenerate: the optimiser fits the calibration images beautifully with a wrong $f$ compensated by a wrong distance. Low RMS, wrong parameters. 3. <b>No coverage at the image edges.</b> Distortion coefficients are only constrained where you put data; a board that never reached the corners leaves $k_1,k_2$ extrapolating wildly there — and the corners are exactly where distortion matters. (Also valid: overfitting with too many coefficients — fit RMS keeps dropping while held-out error rises; and a warped or mismeasured calibration board, which makes the whole estimate self-consistently wrong.)
(b) Resizing by $s = 2/3$ scales <b>all four</b> of $f_x, f_y, c_x, c_y$ by $s$. What must <i>not</i> change: the <b>distortion coefficients</b> $k_1,k_2,k_3,p_1,p_2$ — they are defined in normalised coordinates (5.3), which are already scale-free, so rescaling them is a real and common bug. Contrast with <b>cropping</b>, which shifts only $c_x,c_y$ and leaves $f_x,f_y$ alone.
(c) Autofocus physically moves the lens, changing the focal length — so $f_x, f_y$ (and slightly the principal point) vary <b>per frame</b>, and a single fixed $K$ is simply the wrong model. Fixes: lock focus and exposure for the whole capture (the correct answer when you control the device); or let the pipeline <b>re-estimate intrinsics per capture</b> during bundle adjustment, which is exactly what COLMAP does by default for phone imagery — at the cost of needing more, better-distributed views to keep the extra parameters observable.
</details>


### 🔨 Build + read

**Build:** Print a checkerboard pattern, photograph it from 12–15 different angles and distances with your phone, and run `cv2.calibrateCamera` end to end: detect corners with `findChessboardCorners`, calibrate, then undistort one of your own photos and visually confirm straight real-world lines (a doorframe, a bookshelf edge) that looked curved in the raw photo are now straight. Report your recovered focal length in pixels and compare it against the value implied by your phone's stated sensor size and field of view.

**Read:** Zhang, *A Flexible New Technique for Camera Calibration* (2000, PAMI), §2–3 for the planar-target method your calibration call implements. Then OpenCV's *Camera Calibration and 3D Reconstruction* documentation for the exact distortion model ($k_1,k_2,p_1,p_2,k_3$) `calibrateCamera` fits.

---

## 5.4 Homography Revisited as a Projective Transform

Module 1.9 covered the estimation — how to fit an $H$ to point correspondences. It did not say *what an $H$ is*, geometrically, or *when one is guaranteed to exist*. Now that we have cameras, both questions have crisp answers, and the second one turns out to be the source of the nastiest failure mode in the whole module.

**The 2-D projective hierarchy** (1.9's table) sits inside a bigger idea: a homography is the general element of $PGL(3)$, the group of projective transformations of the plane — and note the "P", which is 5.1's up-to-scale equivalence showing up again as a quotient in the group itself. The useful way to organise the hierarchy is by asking **what each level refuses to destroy**; that invariant is what defines it:

| Transform | Preserves | Invariant |
|---|---|---|
| Euclidean | lengths, angles, areas | length |
| Similarity | angles, length *ratios* | ratio of lengths |
| Affine | parallelism, ratios along a line, the line at infinity | ratio of areas |
| **Projective** | **straight lines, incidence, the cross-ratio** | **cross-ratio** |

Notice the direction of travel: as you go down the table you gain freedom and lose invariants, until at the projective level almost nothing survives. Lengths go, then ratios of lengths, then parallelism itself (parallel lines meet at a vanishing point — 5.1). What is left at the bottom is the **cross-ratio** of four collinear points: the only projective invariant of the line, and therefore the only quantity you can measure in an image and trust to equal its value in the world without knowing anything about the camera. That is what makes single-view metrology possible (measuring a building's height from one photo given a known reference length) — and the fact that *something* survives is not obvious, which is exactly why it is worth knowing.

**The plane-induced homography** — the identity that connects Modules 1 and 5. It answers "when does one $H$ describe a whole pair of images?", and the answer is legible directly from the formula:

$$
H = K'\left(R - \frac{\mathbf{t}\,\mathbf{n}^\top}{d}\right)K^{-1}
$$

for a plane with unit normal $\mathbf{n}$ at distance $d$ from the first camera. Read it:
- If $\mathbf{t} = \mathbf{0}$ (**pure rotation**), $H = K'RK^{-1}$ — independent of the scene, valid for *every* point. That is why panorama stitching works for arbitrary scenes under pure rotation.
- If $\mathbf{t} \ne \mathbf{0}$, $H$ depends on $\mathbf{n}/d$ — so a *different* plane induces a *different* homography, which is exactly why a scene with depth variation cannot be stitched with one $H$ (parallax, 1.9).

Those two bullets are the two — and only two — situations in which stitching is exact, and they are exact for completely different reasons: one restricts the *scene*, the other restricts the *motion*.

**And this is the degeneracy that breaks fundamental-matrix estimation** (1.8). Here is the worry: if a plane already explains every correspondence perfectly, then the epipolar geometry has nothing left to explain — and an estimator with nothing to constrain it does not fail loudly, it returns garbage confidently. Concretely: if all correspondences lie on one plane, they satisfy a homography $\mathbf{x}' \simeq H\mathbf{x}$, and then $F = [\mathbf{e}']_\times H$ satisfies the epipolar constraint for *any* choice of $\mathbf{e}'$ whatsoever. (The notation $[\,\cdot\,]_\times$ is the cross-product matrix built in 5.5; take it on trust for one paragraph.)

**One line of algebra shows it.** Substitute $H\mathbf{x} \simeq \mathbf{x}'$:

$$
\mathbf{x}'^\top F \mathbf{x} = \mathbf{x}'^\top [\mathbf{e}']_\times H\mathbf{x} \;\simeq\; \mathbf{x}'^\top [\mathbf{e}']_\times \mathbf{x}' = \mathbf{x}'\cdot(\mathbf{e}' \times \mathbf{x}') = 0
$$

— identically zero, because $\mathbf{e}'\times\mathbf{x}'$ is orthogonal to $\mathbf{x}'$ by construction. The residual vanishes for *every* correspondence regardless of $\mathbf{e}'$.

Since $\mathbf{e}'$ is a homogeneous 3-vector defined up to scale, this is a **two-parameter family** of fundamental matrices all fitting the data perfectly — the epipole is completely unconstrained. Hence a huge inlier count and meaningless geometry. Being able to write that derivation, and to say *how many* degrees of freedom are unconstrained, is a strong and specific answer.

### ✅ Mastery check

(a) List the DoF consumed by each level of the 2-D transformation hierarchy, and say which levels you can estimate from 3 point correspondences.
(b) Two photos of a whiteboard taken from two different positions in the room — is a homography exact? Two photos of the whole room taken from a tripod that only rotated — is a homography exact? Explain both from the plane-induced formula.
(c) You measure a building's height from a single photo using a person of known height standing at its base. Which projective invariant makes this possible?

<details><summary>Answer sketch</summary>
(a) Euclidean 3, similarity 4, affine 6, projective 8. Each correspondence gives 2 equations, so 3 points give 6 — enough for <b>affine and below</b> (affine needs exactly 3), not enough for a homography, which needs 4.
(b) <b>Whiteboard from two positions: yes, exact.</b> The scene is planar, so $H = K'(R - \mathbf{t}\mathbf{n}^\top/d)K^{-1}$ applies with that plane's $(\mathbf{n}, d)$ — translation is fine as long as everything you are mapping lies on the one plane. <b>Whole room from a rotating tripod: yes, exact, and for the other reason.</b> With $\mathbf{t} = \mathbf{0}$ the formula collapses to $H = K'RK^{-1}$, which has no dependence on $\mathbf{n}$ or $d$ — so it is valid for <i>every</i> point regardless of depth. The two conditions are genuinely different: one restricts the <i>scene</i>, the other restricts the <i>motion</i>. Anything else (translation + depth variation) produces parallax and no single $H$ fits.
(c) The <b>cross-ratio</b> of four collinear points — the only projective invariant of a line. In practice: take the vertical line through the person and the building, use the ground point, the person's head, the building's top, and the vertical vanishing point as the fourth (at infinity), and the cross-ratio computed in the image equals the cross-ratio in the world, which you solve for the unknown height. This is single-view metrology, and the fact that it works <i>at all</i> under an unknown projective transform is the practical payoff of knowing what each transformation level preserves.
</details>


### 🔨 Build + read

**Build:** Photograph a planar surface (a book cover, a poster, a floor tile pattern) from two different viewpoints. Mark 4+ corresponding points by hand in both images, estimate the homography with `cv2.findHomography`, and warp the second image into the first's frame with `cv2.warpPerspective`. Confirm the warped image aligns with the first. Then repeat with a *non-planar* scene (a room with visible depth variation) using the same pipeline, and observe exactly how and where the alignment breaks — this is the planar-degeneracy failure made visible rather than asserted.

**Read:** Hartley & Zisserman §4.1 and §4.4 (pp. 88–93, normalisation discussion) for the DLT and its conditioning. Then Brown & Lowe, *Automatic Panoramic Image Stitching* (2007, IJCV), for how this homography estimate becomes a full stitching pipeline.

---

## 5.5 Epipolar Geometry: Fundamental & Essential Matrices 🔴

> The bridge between one-camera geometry and stereo. Without it, "depth from disparity" is an assertion. This is the section to know cold.

### Intuition

**You see a point in the left image. Where can it be in the right image?** Sit with that question for a moment, because the naive answer — "anywhere, we'll have to search the whole image" — is wrong in a way that changes everything.

Here is the reasoning, and it uses only 5.1. The left camera didn't tell you *where* the 3-D point is; it told you it lies somewhere on a **ray** through the left camera centre. Fine. Now take that whole ray — a physical line in space — and photograph it with the right camera. A straight line in space, photographed, is a straight line in the image (projection preserves straightness; that is the top row of 5.4's invariant table). So the right image of the ray is a **line**, and the point you're looking for is guaranteed to be on it.

**The search for a correspondence is therefore 1-D, not 2-D.** That is the epipolar constraint, and it is what makes stereo matching tractable at all — it converts an $O(W \cdot H)$ search into an $O(W)$ one, and, just as importantly, it lets you *reject* a match that lands off the line as geometrically impossible rather than merely improbable.

Real-life anchor: this is why stereo cameras are mounted on a rigid horizontal bar. Rectify the images so epipolar lines are horizontal rows, and matching becomes a scan along a single row.

Everything in this section is an answer to one follow-up question: **given the two cameras' relative pose, what is the equation of that line?** The answer will be $\mathbf{l}' = F\mathbf{x}$ — a matrix that eats a point in one image and produces a line in the other.

### The geometry

Two cameras with centres $\mathbf{C}, \mathbf{C}'$. Four definitions, each of which is a thing you could point at in a physical setup:
- **Baseline**: the line $\mathbf{C}\mathbf{C}'$.
- **Epipole** $\mathbf{e}$: the image of $\mathbf{C}'$ in the first camera (and $\mathbf{e}'$ = image of $\mathbf{C}$ in the second). **All epipolar lines pass through the epipole.** If the cameras are parallel (pure sideways translation), the epipoles are at infinity and the epipolar lines are parallel horizontal lines — this is exactly what rectification arranges.
- **Epipolar plane**: the plane through $\mathbf{C}$, $\mathbf{C}'$ and the 3-D point $\mathbf{X}$. Three points determine a plane, and this one contains both camera centres — so as $\mathbf{X}$ moves, the plane *pivots about the baseline* like a door on a hinge. There is one epipolar plane per 3-D point, and they all share the same hinge.
- **Epipolar line** $\mathbf{l}' = F\mathbf{x}$: where that plane cuts the second image. The hinge picture immediately explains the epipole: every plane in the pencil contains the baseline, so every one of their image lines contains the baseline's image, which is the epipole. That is *why* all epipolar lines concur, and we will shortly see the same fact appear algebraically as a rank deficiency.

### Deriving the essential matrix — do this on the whiteboard

**Build the picture before touching algebra.** Stand at the second camera and look at three arrows. The first goes from your camera centre $\mathbf{C}'$ out to the 3-D point: call it $\mathbf{X}'$. The second goes from your centre to the *other* camera's centre — that is the baseline, and in your coordinate frame it is $\mathbf{t}$. The third is the other camera's ray to the same point, rotated into your frame so you can compare it: $R\mathbf{X}$.

Now: those three arrows all lie in the epipolar plane. Of course they do — the plane was *defined* as the one containing both centres and the point, and each arrow runs between two of those three things. So we have three vectors that are forced to be coplanar, and coplanarity is a condition we can write down.

**The bridge from geometry to algebra is volume.** Three vectors span a parallelepiped whose signed volume is the scalar triple product $\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})$. If the three vectors lie in a common plane, that box is flat — zero height, zero volume. So:

> *coplanar* $\iff$ *the box they span has no volume* $\iff$ *the triple product vanishes.*

Applied to our three arrows, that single sentence is the entire content of epipolar geometry:

$$
\mathbf{X}'^\top\,(\mathbf{t}\times R\mathbf{X}) = 0
$$

**Check it algebraically, so you are not taking the picture on faith.** We know $\mathbf{X}' = R\mathbf{X} + \mathbf{t}$ from the rigid motion between the cameras. Substitute and expand into two terms. The first is $(R\mathbf{X})^\top(\mathbf{t}\times R\mathbf{X}) = 0$, because a cross product is orthogonal to both its arguments and $R\mathbf{X}$ is one of them. The second is $\mathbf{t}^\top(\mathbf{t}\times R\mathbf{X}) = 0$, for the same reason with the other argument. Both terms die, the sum is zero, and the coplanarity condition holds identically. **Nothing was assumed except that the two cameras are looking at the same point.**

The remaining work is cosmetic: we want this expressed as *a matrix sandwiched between two image measurements*, because that is a form we can estimate from data — a bilinear form $\mathbf{a}^\top M \mathbf{b}$ is linear in the entries of $M$, so a set of correspondences becomes a linear system in the unknown matrix, and linear systems we can solve. The cross product is in the way, so absorb it into a matrix.

**Where the cross-product matrix comes from — write out one component and read it off.** The cross product $\mathbf{t}\times\mathbf{v}$ has first component $t_2v_3 - t_3v_2$. That is linear in $\mathbf{v}$, so it must be *some* row vector times $\mathbf{v}$ — and inspection says the row is $(0, -t_3, t_2)$. Do the same for the other two components and you have stacked the whole matrix, no memorisation required: $\mathbf{t}\times \mathbf{v} = [\mathbf{t}]_\times \mathbf{v}$ with

$$
[\mathbf{t}]_\times = \begin{bmatrix}0 & -t_3 & t_2\\ t_3 & 0 & -t_1\\ -t_2 & t_1 & 0\end{bmatrix}
$$

It is skew-symmetric ($[\mathbf{t}]_\times^\top = -[\mathbf{t}]_\times$), which you can read straight off the sign pattern, and it is **rank 2** — note that now, because it is the whole explanation of $E$'s rank a page from here. Why rank 2 rather than 3: $[\mathbf{t}]_\times\mathbf{t} = \mathbf{t}\times\mathbf{t} = \mathbf{0}$, so $\mathbf{t}$ is in the null space, and the null space is exactly one-dimensional because $\mathbf{t}\times\mathbf{v}$ vanishes only for $\mathbf{v}$ parallel to $\mathbf{t}$. Then

$$
\mathbf{X}'^\top [\mathbf{t}]_\times R\, \mathbf{X} = 0
$$

**Pause:** we now have a relation between the two *3-D* points $\mathbf{X}$ and $\mathbf{X}'$ — but we cannot measure those, only their images, and projection destroyed the depths. Before reading on: why does that not kill the whole construction?

The reason is that the relation is **homogeneous in each argument separately**. Scale $\mathbf{X}$ by any $\lambda$ and the left side just picks up a factor of $\lambda$; scale $\mathbf{X}'$ by $\lambda'$ and it picks up $\lambda'$. But the right side is zero, and $\lambda\lambda'\cdot 0$ is still $0$. So the equation cannot tell the difference between a point and any other point along the same ray — which is precisely the information the camera threw away in 5.1. The constraint was, by luck or by design, written in exactly the currency we still hold.

**The last step, explicitly.** In normalised coordinates the image point *is* the ray direction up to an unknown positive depth: $\mathbf{X} = \lambda\hat{\mathbf{x}}$ and $\mathbf{X}' = \lambda'\hat{\mathbf{x}}'$ with $\lambda,\lambda' > 0$. Substituting into $\mathbf{X}'^\top [\mathbf{t}]_\times R\,\mathbf{X} = 0$ gives $\lambda\lambda'\,\hat{\mathbf{x}}'^\top [\mathbf{t}]_\times R\,\hat{\mathbf{x}} = 0$, and since $\lambda\lambda' > 0$ we may divide it out. **The bilinear form is scale-invariant in each argument, which is exactly why the relation survives the loss of depth:**

$$
\boxed{\ \hat{\mathbf{x}}'^\top E\, \hat{\mathbf{x}} = 0, \qquad E = [\mathbf{t}]_\times R\ }
$$

Notice what that box is and is not. It is **not** the equation of the epipolar line yet — it is a *test*: hand it two points and it tells you whether they could possibly be images of the same 3-D point. But read it as $(\hat{\mathbf{x}}')^\top(E\hat{\mathbf{x}}) = 0$ and the line falls out for free: $E\hat{\mathbf{x}}$ is some fixed 3-vector $\mathbf{l}'$, and "$\hat{\mathbf{x}}'^\top\mathbf{l}' = 0$" is exactly the homogeneous equation of a line in the second image. **The matrix that tests correspondences and the matrix that generates epipolar lines are the same matrix** — that is the promise made at the end of the Intuition section, now paid.

**The fundamental matrix** is the uncalibrated version, and the derivation of it is a substitution, not a new idea. $E$ eats *normalised* coordinates (5.2: rays in the camera's own frame), but what a camera actually hands you is pixels. The dictionary between them is $\hat{\mathbf{x}} = K^{-1}\tilde{\mathbf{x}}$, so substitute that in for both images and see where the $K$'s land:

$$
\tilde{\mathbf{x}}'^\top \underbrace{K'^{-\top} E K^{-1}}_{F} \tilde{\mathbf{x}} = 0
\qquad\Longrightarrow\qquad
\boxed{\ F = K'^{-\top}[\mathbf{t}]_\times R\, K^{-1}\ }
$$

(The $K'^{-\top}$ rather than $K'^{-1}$ on the left is only because $\hat{\mathbf{x}}'$ appears transposed: $(K'^{-1}\tilde{\mathbf{x}}')^\top = \tilde{\mathbf{x}}'^\top K'^{-\top}$.) **$F$ is not a different concept from $E$ — it is $E$ with the two unknown calibrations swallowed into it.** That single sentence explains every entry of the comparison table below: anything $F$ can do less of, it is because it is carrying two unknown matrices around as passengers.

**In your own words:** why does "three vectors are coplanar" turn into a matrix sandwiched between two image points — and what is the one property of that relation that lets it survive the loss of depth?

### Properties — memorise this table

The table compares them; the paragraphs after it derive every row, so read the table as a summary of things you can reconstruct rather than as a list to memorise cold.

| | $F$ (fundamental) | $E$ (essential) |
|---|---|---|
| Operates on | raw pixel coordinates | normalised (calibrated) coordinates |
| Requires | nothing | known $K, K'$ |
| Size / rank | $3\times3$, **rank 2** | $3\times3$, **rank 2** |
| DoF | **7** (9 entries − 1 scale − 1 rank constraint) | **5** (3 rotation + 3 translation − 1 scale) |
| Extra constraint | $\det F = 0$ | $\det E = 0$ **and two equal non-zero singular values** |
| Minimal solver | 7-point (or 8-point linear) | **5-point (Nistér)** |
| Recovers | projective geometry only | $R$ and $\mathbf{t}$ **up to scale**, 4-fold ambiguity |

**Why rank 2:** $[\mathbf{t}]_\times$ is rank 2 (its null space is $\mathbf{t}$, as we just showed), and $R$ is full rank, so multiplying by it cannot restore the lost dimension — $E = [\mathbf{t}]_\times R$ is rank 2. **The satisfying part is that this algebraic fact and the geometric hinge picture from two pages ago are the same statement.** Rank deficiency means there is a non-zero vector $\mathbf{e}$ with $F\mathbf{e} = \mathbf{0}$. But $F\mathbf{e}$ is "the epipolar line of the point $\mathbf{e}$", and the zero vector is the degenerate line — the one that contains everything. So $\mathbf{e}$ is the one image point whose epipolar line is not a line, which is exactly the epipole, the image of the other camera centre. Likewise $F^\top\mathbf{e}' = \mathbf{0}$ on the other side. **All epipolar lines meeting at a point and $\det F = 0$ are one fact wearing two costumes**, and being able to move between the two readings on demand is what separates a memorised answer from an understood one.

**Why $E$ has two equal singular values:** $E = [\mathbf{t}]_\times R = U\,\text{diag}(\sigma,\sigma,0)\,V^\top$. Here is the reasoning rather than the assertion. $[\mathbf{t}]_\times$ on its own has singular values $\|\mathbf{t}\|, \|\mathbf{t}\|, 0$ — the two non-zero ones are equal because a cross product with $\mathbf{t}$ acts on the plane perpendicular to $\mathbf{t}$ by rotating it a quarter turn and scaling *isotropically* by $\|\mathbf{t}\|$; there is no preferred direction in that plane for it to stretch more. Then right-multiplying by $R$ is an orthogonal transformation, and orthogonal transformations do not change singular values at all — they only re-label which input directions map where. So $E$ inherits $\|\mathbf{t}\|, \|\mathbf{t}\|, 0$ intact. This is a *stronger* condition than $\det E = 0$ (which only says the third one vanishes), and it is exactly the extra structure that drops $E$ from 7 DoF to 5. Enforcing it after estimation ("projecting onto the essential manifold" — set $\sigma_1 = \sigma_2 = (\sigma_1+\sigma_2)/2$, $\sigma_3 = 0$) is a required cleanup step after any linear estimate, for the same reason the rank-2 projection is required for $F$: a linear solver knows nothing about manifolds and will hand you a matrix that is not a legal $E$.

**Why 5 DoF for $E$ but 7 for $F$:** $E$ encodes only relative pose ($R$: 3, $\mathbf{t}$: 3, minus the global scale that is unobservable from images: 1) = 5. $F$ additionally absorbs the two unknown calibrations, and 7 is what remains after removing scale and the rank constraint from 9. **This DoF accounting is a favourite interview question** and the answer explains why the 5-point algorithm is worth its complexity (1.8: $N \propto 1/w^s$, so $s=5$ vs $s=8$ is 145 vs 1177 RANSAC iterations at $w=0.5$).

### Estimating $F$: the normalised 8-point algorithm

**We have the constraint; now we run it backwards.** So far $F$ was something we built from a known $R,\mathbf{t},K$. In practice we know none of those — we have a pile of matched points and want $F$ from them. The move is the one you already use constantly in deep learning under a different name: the constraint is *linear in the unknown*, so a batch of observations becomes a linear system.

Write out $\tilde{\mathbf{x}}'^\top F\tilde{\mathbf{x}} = 0$ term by term with $\tilde{\mathbf{x}} = (x,y,1)^\top$ and $\tilde{\mathbf{x}}' = (x',y',1)^\top$. Each entry $f_{rc}$ of $F$ gets multiplied by the $r$-th component of $\tilde{\mathbf{x}}'$ and the $c$-th of $\tilde{\mathbf{x}}$ — that is all a bilinear form does — so each correspondence gives one scalar equation whose *coefficients* are known products of measured pixel coordinates and whose *unknowns* are the nine entries of $F$:

$$
x'x f_{11} + x'y f_{12} + x' f_{13} + y'x f_{21} + y'y f_{22} + y' f_{23} + x f_{31} + y f_{32} + f_{33} = 0
$$

**Why eight and not nine.** Nine unknowns would normally need nine equations, but $F$ is only defined up to scale (if $F$ satisfies the constraint so does $2F$), so one degree of freedom is unconstrained by construction and eight correspondences suffice. Stack them: $A\mathbf{f} = \mathbf{0}$ with $A$ of size $8\times9$ (or $n\times 9$). A homogeneous system, so we do not want the trivial solution $\mathbf{f} = \mathbf{0}$ — we want the unit-norm vector that comes closest to satisfying it, i.e. $\arg\min_{\|\mathbf{f}\|=1}\|A\mathbf{f}\|$. That minimiser is the **right singular vector of $A$ belonging to the smallest singular value**, which is the standard SVD fact you will now meet for the third time in this course (DLT for homographies in 1.9, Zhang's $B$ in 5.3, here).

**Then enforce rank 2**, which the linear solution does not satisfy — and it is worth being clear about *why* it doesn't. The eight equations know nothing about $\det F = 0$; that constraint is non-linear, so we simply left it out to keep the problem linear, and noise guarantees the returned $F$ will have a small-but-non-zero third singular value. We repair it afterwards: take $F = U\Sigma V^\top$, set $\sigma_3 = 0$, and reconstitute $F' = U\,\text{diag}(\sigma_1,\sigma_2,0)\,V^\top$. This is the closest rank-2 matrix in Frobenius norm (Eckart–Young). **Skipping this step gives epipolar "lines" that don't intersect at a common epipole and a visibly broken result** — which, given what we established above, is not a coincidence but the direct visual read-out of $\det F \ne 0$.

**Pause:** the whole algorithm is "build $A$ from pixel coordinates, take an SVD." Before reading on — what could possibly go wrong numerically, given that $x, y$ are pixel indices in the hundreds or thousands?

**Hartley normalisation is not optional here — it is more critical than for homographies** (1.9). Look at one row of $A$: its entries are $x'x,\ x'y,\ x',\ y'x,\ y'y,\ y',\ x,\ y,\ 1$. With pixel coordinates around $10^3$, the quadratic terms are around $10^6$ while the last entry is exactly $1$ — **six orders of magnitude of spread inside a single row.** The condition number of $A$ is correspondingly enormous, and the quantity we are extracting is the *smallest* singular vector, i.e. the most fragile thing the SVD produces. Rounding error in the large entries swamps the information carried by the small ones. The fix is a similarity transform per image — translate the centroid of the points to the origin and scale so their mean distance from it is $\sqrt2$ — solve for $F$ in those coordinates, then undo the transforms ($F = T'^\top F_{\text{norm}} T$). Hartley's 1997 paper is literally titled "In Defense of the Eight-Point Algorithm" and its entire content is: the algorithm was thought to be hopelessly noise-sensitive, and it isn't — you just have to normalise. **This is one of the best "specific paper, specific fix" stories in classical CV**, and note that the fix is not a better estimator, it is better *coordinates* for the same estimator.

**The 7-point algorithm** is what you get by refusing to throw the non-linear constraint away. Use only 7 correspondences and $A$ is $7\times9$, so its null space is *two*-dimensional — every matrix in the family $\{\alpha F_1 + (1-\alpha)F_2\}$ fits the data equally well, and we need one more equation to pick among them. That equation is the one we discarded before: impose $\det = 0$ on the family. The determinant of a $3\times3$ matrix is cubic in its entries, and the entries here are linear in $\alpha$, so we get a **cubic** in $\alpha$ with 1 or 3 real roots. Fewer RANSAC iterations (7 points instead of 8 — and by 1.8's $N \propto 1/w^s$ that compounds), at the cost of possibly returning three candidate solutions that must each be scored.

### Recovering pose from $E$ — the four-fold ambiguity

**We built $E$ out of $R$ and $\mathbf{t}$; now we want them back out.** This is the step that turns a matrix estimated from pixel matches into something physical — the motion of the camera between two shots — and it is the beginning of every visual odometry and SfM pipeline.

**Where the ambiguity comes from, before the algebra.** Ask what information $E$ can possibly contain. $E$ says, for each pair of rays, that they are coplanar with the baseline. But coplanarity is a statement about *lines*, and a line has no forward direction. So flipping the baseline ($\mathbf{t}\to-\mathbf{t}$) leaves every epipolar plane exactly where it was, and there is a second, less obvious flip: rotating the second camera by 180° about the baseline also preserves the pencil of planes, because the planes all contain the axis you rotated about. Two independent binary flips, and $2\times2 = 4$. **The four-fold ambiguity is not a quirk of the SVD — it is a genuine statement that the epipolar constraint alone cannot distinguish four different physical configurations**, and we will need information from outside the constraint (namely, that real points sit in front of cameras) to break the tie.

Decompose $E = U\,\text{diag}(1,1,0)\,V^\top$ and let $W = \begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$ (which is a 90° rotation about the $z$ axis — $W$ and $W^\top$ are the two directions of that quarter turn, and they are what produce the twisted pair). The four candidate solutions are

$$
(R,\mathbf{t}) \in \{(UWV^\top, +\mathbf{u}_3),\ (UWV^\top, -\mathbf{u}_3),\ (UW^\top V^\top, +\mathbf{u}_3),\ (UW^\top V^\top, -\mathbf{u}_3)\}
$$

where $\mathbf{u}_3$ is the third column of $U$. (Check $\det R = +1$; negate $R$ if not.)

Geometrically these are: the correct solution, the same rotation with the baseline reversed, and two "twisted pair" solutions where the second camera is rotated 180° about the baseline.

**Disambiguate by cheirality**: triangulate one point and keep the solution for which it lies **in front of both cameras** (positive depth). The word means "handedness", and the idea is the only piece of physics we have that the algebra does not: a camera cannot photograph what is behind it. Of the four candidates, exactly one puts the reconstructed points in front of both cameras — in the other three, the points land behind one camera, behind the other, or behind both. One point suffices in the noise-free case; in practice, triangulate all inliers and pick the solution with the most positive-depth points, because a single point near the epipolar-degenerate configuration can vote wrongly under noise.

**The thing that trips people up here:** cheirality is often filed as "a check you run at the end", which makes it sound like validation. It is not validation — it is *the fourth constraint*, and without it the problem is genuinely under-determined. The epipolar constraint gave us everything it had and stopped two bits short.

**And $\mathbf{t}$ is only known up to scale** — $E$ is defined up to scale, so you recover the *direction* of translation, never its magnitude. **Monocular SfM and monocular SLAM therefore reconstruct the world up to an unknown global scale.** That is why monocular visual odometry drifts in scale, and why real systems fix it with a known baseline (stereo), an IMU (visual-inertial), a known object size, or wheel odometry. **This is one of the most practically important facts in the whole module.** And notice it is 5.1's ambiguity resurfacing one level up: there, a single image could not tell a point from a scaled point; here, a pair of images cannot tell a scene from a scaled scene photographed from proportionally scaled positions. Every image in the pair is pixel-for-pixel identical under that scaling, so no algorithm operating on images can possibly recover it.

**In your own words:** why are there exactly four candidate poses, and what kind of information — not what procedure — picks the right one?

### 🎯 Top-1% distinction

1. **Derive $E = [\mathbf{t}]_\times R$ from the coplanarity/triple-product argument** — live, in under three minutes.
2. **Explain rank 2 from $[\mathbf{t}]_\times$**, and connect it to "all epipolar lines meet at the epipole, which is $F$'s null vector."
3. **The DoF accounting**: $F$ 7, $E$ 5 — and why that makes the 5-point algorithm worth it (RANSAC iteration count).
4. **Enforce the rank-2 constraint after the linear solve**, and know it's the Eckart–Young projection.
5. **Hartley normalisation, with the paper title as the punchline.**
6. **The 4-fold ambiguity and cheirality disambiguation.**
7. **Scale is unobservable** — and name the four ways real systems resolve it.
8. **The planar degeneracy** ($F = [\mathbf{e}']_\times H$) from 5.4/1.8, and DEGENSAC as the fix.

### ✅ Mastery check

(a) Derive $E = [\mathbf{t}]_\times R$ from coplanarity. State where each rank/DoF property comes from.
(b) You estimate $F$ with the 8-point algorithm without normalisation and without enforcing rank 2. Describe the two distinct visual symptoms.
(c) You have a calibrated stereo rig and estimate $E$ from 100 matches. You get a valid $R$ and $\mathbf{t}$, but the reconstructed scene is 3× too small. What went wrong — and what if it were mirrored instead?
(d) Why does a drone flying over flat farmland break monocular SLAM?

<details><summary>Answer sketch</summary>
(a) See the derivation above. Rank 2 comes from $[\mathbf{t}]_\times$ being skew-symmetric with null space $\mathbf{t}$. $E$'s 5 DoF = 3 ($R$) + 3 ($\mathbf{t}$) − 1 (global scale unobservable). $F$'s 7 DoF = 9 − 1 (scale) − 1 ($\det F = 0$).
(b) <b>Without normalisation:</b> the estimate is dominated by numerical error — epipolar lines are wildly wrong, often nearly parallel or clustered, and the residuals are large even on the inliers you fitted. <b>Without enforcing rank 2:</b> a subtler and more diagnostic symptom — the epipolar lines <b>do not all pass through a single epipole</b>. Drawing the epipolar lines for a set of points and seeing them fail to concur is the classic visual signature of a missing rank-2 projection.
(c) <b>Nothing went wrong.</b> $E$ is defined up to scale, so $\mathbf{t}$ is recovered only as a <i>direction</i>; the reconstruction is correct up to an arbitrary global scale factor. Fix it with the known stereo baseline: scale the reconstruction so $\|\mathbf{t}\|$ equals the measured baseline in metres. (For a calibrated rig you'd normally not estimate $E$ at all — you know the extrinsics.) <b>If it were mirrored</b>, that's a different bug: you selected the wrong one of the four $E$ decompositions — likely a $\det R = -1$ solution that wasn't corrected, or a cheirality check that passed on a degenerate/noisy point. Re-run the cheirality test over all inliers and enforce $\det R = +1$.
(d) Flat farmland is a <b>plane</b>, which is the classic degenerate configuration for $F$/$E$ estimation (5.4): all correspondences are explained by a plane-induced homography $H = K(R - \mathbf{t}\mathbf{n}^\top/d)K^{-1}$, and $F = [\mathbf{e}']_\times H$ fits them for <i>any</i> epipole. RANSAC reports a huge inlier count with a geometrically meaningless epipolar geometry — a confident, silent failure. Compounding it: with little depth variation there is minimal parallax, so triangulation is ill-conditioned and scale is even less observable than usual. <b>Fixes:</b> detect the degeneracy by comparing the homography inlier count to the $F$ inlier count (if they're close, you're on a plane); use a homography-based initialisation and decomposition instead (ORB-SLAM does exactly this — it computes both $H$ and $F$ every frame, scores them, and picks the model that fits); or fuse an IMU / GPS / barometer to supply the missing constraints and the metric scale.
</details>

### 🔨 Build + read

**Build:** Implement the normalised 8-point algorithm end to end: Hartley normalisation → linear solve via SVD → rank-2 projection → denormalisation. Then **draw the epipolar lines** for a set of points in both images of a real pair — and run the ablation, once without normalisation and once without the rank-2 step, so you see both failure signatures with your own eyes. Then implement $E$ decomposition with the cheirality check and verify against `cv2.recoverPose`.

**Read:** Hartley & Zisserman, *Multiple View Geometry*, Ch. 9 (epipolar geometry) and §11.1–11.2 (the normalised 8-point algorithm) — this is the canonical treatment and it is worth the effort. Hartley, "In Defense of the Eight-Point Algorithm" (PAMI 1997). Nistér, "An Efficient Solution to the Five-Point Relative Pose Problem" (PAMI 2004) — skim for the structure of the argument.

---

## 5.6 Stereo Vision & Depth from Disparity

### The rectified geometry

**5.5 told us a matching point lies on a line. The obvious next question is: fine — but how do I turn "it was here in the left image and there in the right" into a number of metres?**

Two things stand between us and that number, and rectification removes the first. In general the epipolar lines are slanted and different for every pixel, so scanning along them means resampling the image along oblique paths — correct, but awkward and cache-hostile. **Rectification** warps both images so that the epipolar lines become corresponding horizontal scanlines — equivalent to synthesising two cameras with identical intrinsics, no relative rotation, and translation purely along $x$. Why that particular configuration produces horizontal lines is worth checking against 5.5 rather than accepting: with $\mathbf{t}$ along $x$ and $R = I$, the epipole (the image of the other camera centre, which now lies in the direction of the $x$ axis) sits at infinity in the horizontal direction, and every epipolar line must pass through it — so every epipolar line is horizontal. Then correspondence search is 1-D, along a row, with the same row index in both images.

Now the second thing: the number. **Derive it rather than reading it off.** Put the world origin at the left camera and let the right camera sit at $(B,0,0)$ — that is what "baseline $B$, translation purely along $x$" means. A 3-D point $(X,Y,Z)$ then projects by 5.1's rule into the left camera at $x_L = f\frac{X}{Z}$. For the right camera, the *same* point has $x$-coordinate $X - B$ in that camera's own frame, so $x_R = f\frac{X - B}{Z}$. Subtract, and the unknown $X$ — which we never wanted — cancels:

$$d = x_L - x_R = f\frac{X}{Z} - f\frac{X-B}{Z} = \frac{fB}{Z}$$

Rearranged, that is the equation the whole of stereo rests on. For a rectified pair with baseline $B$ and focal length $f$ (pixels), disparity $d = x_L - x_R$:

$$
\boxed{\ Z = \frac{fB}{d}\ }
$$

**The thing that trips people up here is the direction of the relationship.** People expect depth and disparity to be proportional, because "further away, more shift" sounds right. It is exactly backwards: hold your thumb up and blink alternate eyes, and the *near* thumb jumps furthest while the far wall barely moves. Disparity is the *inverse* of depth, and almost every counter-intuitive property of stereo below is a consequence of that one inversion.

**Pause:** given $Z = fB/d$ and nothing else — before reading on, predict how the depth error behaves as an object gets further away. Linear in $Z$? Constant? Something worse?

**Consequences you must be able to state:**

- **Depth error grows quadratically with depth.** Here is why, and the reason is the inversion we just flagged. Your sensor's error is in *disparity* — a matcher localises a match to some fraction of a pixel, and that accuracy $\delta d$ is roughly constant regardless of how far away the object is. To find out what that costs in metres, ask how sensitive $Z$ is to $d$, i.e. differentiate $Z = fB/d$:
  $$\left|\frac{\partial Z}{\partial d}\right| = \frac{fB}{d^2}$$
  which is already the answer, but it is expressed in $d$ and we want it in $Z$. Substitute $d = fB/Z$: the $d^2$ in the denominator becomes $(fB)^2/Z^2$, and the $fB$ on top cancels one factor, leaving $Z^2/(fB)$. So a disparity error $\delta d$ gives
  $$
  \delta Z \approx \frac{Z^2}{fB}\,\delta d
  $$
  **This single formula is the most useful thing in stereo.** Doubling the distance quadruples the depth error — and note that the quadratic came from nowhere exotic: it is purely the chain rule applied to a reciprocal, the same reason $1/x$ gets flat and useless for large $x$. It is why stereo rigs on cars have long baselines, why depth cameras have a spec'd working range, and why long-range perception uses lidar or radar rather than stereo.
- **A larger baseline $B$ improves depth accuracy but shrinks the overlap and makes matching harder** (larger viewpoint change ⇒ more occlusion and appearance change). Classic engineering trade-off.
- **Disparity is inversely proportional to depth**, so disparity maps look like "closer = brighter."
- **$d = 0$ means infinity**; negative disparity means a rectification or ordering error.

**In your own words:** why does a constant sub-pixel matching accuracy turn into a depth error that gets *worse the further away you look*?

### Matching

**We have the formula; we still have to find $d$.** And that is the hard part — the geometry took three lines, the matching is where sixty years of research went. The question every method below answers differently is: *for this pixel in the left image, which pixel on the corresponding right-image row is the same physical point?*

**Classical block matching**: for each pixel, slide a window along the scanline and minimise SSD / SAD / **NCC** (normalised cross-correlation — invariant to affine brightness change, which is why it beats SSD across cameras with different gains) or **Census transform / Hamming** (robust to non-linear illumination differences; the standard in hardware stereo).

**Semi-Global Matching (SGM, Hirschmüller 2005)** — the classical workhorse, still used in production and in hardware. Block matching decides every pixel independently, which is why its output is speckled: a textureless patch has no reason to prefer one disparity over another, so noise decides. The fix is to stop treating pixels as independent and add a prior that neighbouring pixels usually have similar depth — i.e. turn matching into an energy minimisation with a data term and a smoothness term. This is the same structure as a CRF, and the same structure as the regularised objectives you know from deep learning: fit the evidence, but pay for roughness. Minimise a global energy

$$
E(D) = \sum_p C(p, D_p) + \sum_{q\in N_p}\Bigl[P_1\,\mathbb{1}[|D_p - D_q| = 1] + P_2\,\mathbb{1}[|D_p-D_q| > 1]\Bigr]
$$

— a data term plus a smoothness term with a small penalty $P_1$ for a 1-pixel disparity change and a large penalty $P_2$ for a jump. **Read the two penalties as encoding two different beliefs about the world.** A single-pixel disparity step is what a *slanted* surface looks like — a table top receding from you changes depth gradually — and slanted surfaces are everywhere, so penalise that lightly. A multi-pixel jump is what a genuine object boundary looks like, and boundaries are rare but real, so penalise it heavily but not infinitely. A single quadratic smoothness penalty cannot express both, because it grows with the size of the step and therefore fights hardest exactly at the depth discontinuities you most want to keep — that is Horn–Schunck's failure mode in 5.7, arriving early.

True 2-D optimisation of this energy is NP-hard, so SGM approximates it by aggregating exact 1-D dynamic-programming solutions along 8 or 16 directions. **The trick is worth stating precisely:** along a single scanline the problem is a chain, and a chain is exactly what dynamic programming solves optimally in one pass; the 2-D problem is hard only because of the loops in the pixel grid's neighbourhood graph. So SGM solves the easy version many times, along many directions through each pixel, and sums the costs — an approximation with no optimality guarantee that works extremely well in practice. **The $P_1/P_2$ split is the clever part** — it's a discontinuity-preserving smoothness prior, the same idea as the bilateral filter's range kernel (1.2).

**Post-processing that matters**: left–right consistency check (match $L\to R$ and $R\to L$; disagreement ⇒ occlusion, mark invalid), sub-pixel refinement by parabola fitting on the cost curve, speckle filtering, and hole filling. The first two deserve a sentence each because both are more principled than they look. **Left–right consistency is the only occlusion detector you get for free:** a pixel visible in the left view but hidden in the right has *no* correct match, and a matcher will return one anyway — but the reverse match from that returned pixel will point somewhere else, and the disagreement is the tell. **Sub-pixel refinement is where the $\delta d$ in the error formula actually comes from:** the cost function is evaluated only at integer disparities, so fitting a parabola through the minimum and its two neighbours and taking the vertex recovers a fractional disparity — and going from 1 px to ¼ px accuracy is, by $\delta Z \approx Z^2\delta d/(fB)$, a fourfold depth improvement for a few flops.

**Learned stereo**: GC-Net, PSMNet, GA-Net (**cost-volume** networks — build a $H\times W\times D_{\max}$ volume of matching costs and 3-D-convolve it), RAFT-Stereo (iterative refinement with a GRU, the current strong baseline), and foundation-model-flavoured approaches (FoundationStereo, DepthAnything-derived stereo). **The cost volume is the key structural idea** — it's the learned version of the classical disparity search, and networks that build one generalise far better than those that regress depth directly.

### Where stereo fails

**Every failure below is the same failure**, and it is worth naming once rather than five times: stereo assumes that a physical point looks the same from both cameras and looks different from its neighbours. The first half is the photometric assumption; the second is what makes a match *unique*. Break either and matching has nothing to work with.

Textureless regions (a white wall — no matching signal), repetitive texture (a brick wall — ambiguous matches), occlusions (visible in one view only), specular/transparent surfaces (the "same" point looks different from each view, violating the photometric assumption), and thin structures. **Active stereo** (projecting a texture pattern — Kinect v1, Intel RealSense) solves the textureless case directly and is the standard industrial answer.

### 🎯 Top-1% distinction

1. **Derive $Z = fB/d$** and immediately give **$\delta Z \approx \frac{Z^2}{fB}\delta d$** with its consequences.
2. **The baseline trade-off** (accuracy vs matching difficulty vs occlusion).
3. **SGM's $P_1/P_2$ smoothness design** and why 1-D aggregation along many directions approximates the intractable 2-D problem.
4. **Left–right consistency as the principled occlusion detector.**
5. **NCC/Census over SSD**, with the illumination-invariance reason.
6. **The cost volume as the structural bridge** from classical to learned stereo.

### ✅ Mastery check

A stereo rig has $B = 12$ cm, $f = 700$ px, and achieves ¼-pixel disparity accuracy.
(a) Depth error at 2 m, 10 m, and 50 m.
(b) You need ±10 cm accuracy at 30 m. What baseline?
(c) Is that baseline practical? What would you actually do?

<details><summary>Answer sketch</summary>
(a) $fB = 700 \times 0.12 = 84$ px·m. $\delta Z = Z^2 \delta d/(fB) = Z^2 \times 0.25/84$.
 At 2 m: $4 \times 0.25/84 = \mathbf{1.2}$ cm. At 10 m: $100\times0.25/84 = \mathbf{30}$ cm. At 50 m: $2500\times0.25/84 = \mathbf{7.4}$ m.
 Note the 2 m → 50 m degradation: 1.2 cm → 7.4 m, a factor of ~620 for a 25× distance increase (25² = 625 ✓).
(b) Need $\delta Z = 0.1$ m at $Z = 30$ m: $0.1 = 900 \times 0.25/(700B) \Rightarrow 700B = 2250 \Rightarrow B = \mathbf{3.2}$ m.
(c) A 3.2 m baseline is impractical on a vehicle (it exceeds the width of a car) and would create severe occlusion and a large viewpoint difference that makes matching much harder — the two cameras would see substantially different scenes. What you'd actually do, in order: (i) <b>increase $f$ instead of $B$</b> — a longer lens or a higher-resolution sensor raises $fB$ without a physical baseline change (at the cost of field of view, so you might use a narrow-FoV "far" camera alongside a wide "near" one, which is exactly what production autonomous-driving stacks do); (ii) <b>improve sub-pixel disparity accuracy</b> — going from ¼ px to 1/10 px is a 2.5× improvement for free, achievable with better cost interpolation or a learned stereo network; (iii) <b>use a different sensing modality at range</b> — lidar or radar, where accuracy doesn't degrade quadratically; (iv) <b>fuse over time</b> — structure from motion over a moving platform gives a much larger <i>effective</i> baseline than the physical rig. The honest headline: <b>stereo is a short-to-medium-range sensor by physics, not by engineering quality.</b>
</details>


### 🔨 Build + read

**Build:** Using a stereo pair (the Middlebury stereo dataset has ground-truth disparity, or rig two webcams a fixed baseline apart yourself), implement block-matching stereo from scratch: for each pixel in the left image, search a horizontal window in the right image for the best SSD/SAD match, and produce a disparity map. Convert to depth via $Z = fB/d$ using the dataset's stated $f$ and $B$. Compare your naive block-matcher's disparity map against OpenCV's `StereoSGBM`, and identify where yours fails (textureless regions, repeated patterns, occlusion boundaries) that SGM's smoothness term handles.

**Read:** Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Ch. 12 (stereo). Then Hirschmüller, *Stereo Processing by Semi-Global Matching* (2008, PAMI) for the energy that `StereoSGBM` actually optimises.

---

## 5.7 Optical Flow & Tracking

### Intuition

**Two consecutive video frames, and no labels, no detector, no idea what any object is. What can you still say about what moved where?**

Optical flow is the answer: the per-pixel motion field between two frames — for every pixel, an arrow saying where it went. It is what lets a system reason about motion without knowing what anything *is*, which is why it sits underneath video compression, frame interpolation, visual odometry front-ends, and action recognition alike. The catch is that a pixel is a number, not a name, so "where did *this* pixel go" is not a well-posed question until we assume something. The rest of this section is about what we have to assume, and what that assumption costs.

### The brightness constancy assumption

**The assumption is the smallest one that could possibly work:** when a bit of surface moves, its brightness comes with it. The patch of world that was at $(x,y)$ in frame $t$ is at $(x+u, y+v)$ in frame $t+1$, and it has the same intensity there.

$$
I(x, y, t) = I(x + u, y + v, t + 1)
$$

This is one equation per pixel and it is useless as it stands, because $I$ is an arbitrary function of its arguments and we cannot invert it. **The move that makes it usable is linearisation** — the same move as everywhere else in optimisation: assume the displacement is small and replace the function by its first-order Taylor expansion around $(x,y,t)$:

$$I(x+u,\, y+v,\, t+1) \approx I(x,y,t) + \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t}\cdot 1$$

Set that equal to $I(x,y,t)$ as the assumption demands, and the $I(x,y,t)$ terms cancel from both sides, leaving only the derivative terms:

$$
I_x u + I_y v + I_t = 0
$$

— **the optical flow constraint equation**. Note how cheap the ingredients are: $I_x, I_y$ are the spatial gradients you already compute with a Sobel filter (1.3), and $I_t$ is just the difference between the two frames. **Everything on the left except $u$ and $v$ is measurable from two images with three convolutions.**

**Pause:** we have one linear equation per pixel and two unknowns per pixel, so the system is under-determined by exactly one degree of freedom everywhere. Before reading on — geometrically, *which* direction of motion is the one we cannot see?

**One equation, two unknowns per pixel: this is the aperture problem**, and the equation tells you precisely which component survives. Write it as a dot product: $\nabla I \cdot (u,v) = -I_t$. Any $(u,v)$ that differs from the true flow by a vector *perpendicular to the gradient* satisfies the equation equally well, because that perpendicular part contributes nothing to the dot product. So only the flow component **along** the image gradient — the normal flow — is observable; the tangential component is completely unconstrained by this pixel. And the image gradient at an edge points across the edge, so the unobservable direction is along the edge. (Look at a moving edge through a small hole and you cannot tell if it's sliding along itself — the classic barber-pole illusion is your own visual system falling for exactly this.)

**The thing that trips people up here is calling the aperture problem a limitation of the *algorithm*.** It is not. The information is absent from the images. No estimator, learned or classical, can recover the tangential motion of a featureless straight edge from local evidence alone — it can only *guess* it from context, which is exactly what a global smoothness term or a trained network does.

**In your own words:** why does one brightness-constancy equation per pixel leave one direction of motion invisible, and which direction is it?

### Lucas–Kanade: the local solution

**If one equation is not enough, get more equations.** The cheapest extra assumption available is that motion is locally coherent: neighbouring pixels belong to the same surface and therefore move together. Assume flow is constant in a small window $W$ — say $5\times5$ — and suddenly you have 25 equations in the same two unknowns. Over-determined instead of under-determined, so solve it in the least-squares sense. Stack the constraint over all pixels in $W$ and form the normal equations $A^\top A\,\mathbf{u} = A^\top\mathbf{b}$, and what comes out is a $2\times2$ system you have seen before:

$$
\underbrace{\begin{bmatrix}\sum I_x^2 & \sum I_xI_y\\ \sum I_xI_y & \sum I_y^2\end{bmatrix}}_{M \text{ — the structure tensor!}}\begin{bmatrix}u\\v\end{bmatrix} = -\begin{bmatrix}\sum I_xI_t\\ \sum I_yI_t\end{bmatrix}
$$

**That $M$ is exactly the Harris structure tensor from Module 1.5** — the same sums of products of gradients over a window, arrived at from a completely different starting point. That coincidence is not a coincidence, and following it through is one of the most satisfying connections in classical vision.

Trace the logic. The system $M\mathbf{u} = \mathbf{b}$ has a unique, stable solution exactly when $M$ is well-conditioned — both eigenvalues large. Now recall what Harris said the eigenvalues of $M$ mean: two small eigenvalues is a flat region, one large and one small is an edge, two large is a corner. **So "LK is solvable here" and "this is a corner" are the same test.** And the middle case is the aperture problem in matrix form: on an edge, every gradient in the window points the same way, so all rows of $A$ are parallel, $M$ is **rank-1**, and it has a null direction — along the edge — in which we cannot solve. The geometric argument from the previous section and the linear-algebra argument here are the same argument.

This is why Shi & Tomasi titled their corner paper "Good Features to **Track**": they derived the corner criterion from trackability, not from an aesthetic notion of cornerness. Their refinement over Harris follows immediately from the above — since it is the *smaller* eigenvalue that limits solvability, score a candidate by $\min(\lambda_1,\lambda_2)$ directly rather than by Harris's determinant-minus-trace surrogate.

**In your own words:** why is "a good corner" and "a pixel whose motion is recoverable" the same condition?

**Coarse-to-fine LK** (pyramidal LK, `cv2.calcOpticalFlowPyrLK`). Go back to the Taylor expansion: we truncated after the first-order term, which is only accurate while $(u,v)$ is small enough that $I$ is approximately linear over the displacement — in practice about a pixel, since that is the scale on which image intensity varies. A car crossing the frame moves thirty pixels between frames and the linearisation is simply wrong. **The fix is to change the units of "one pixel" rather than to change the algorithm:** downsample the image by 16× and a 30-pixel motion becomes a 2-pixel motion, which the linearisation handles. So run LK on a Gaussian pyramid (1.4!) from coarse to fine, and at each level warp the second image by the current estimate before re-running LK, so each level only has to explain the small *residual* motion its parent could not. This handles motions of tens of pixels. **Yet another use of the image pyramid** — and note the shape of the argument, which recurs: when a linearisation's validity depends on a scale, build a representation in which you can choose the scale.

### Dense flow

**LK gives you flow at corners. Dense flow wants it everywhere — including at the edges and flat regions where, as we just established, the information genuinely is not there.** So every dense method must supply the missing information from an assumption, and the methods differ mainly in which assumption they make and how gracefully it fails.

- **Horn–Schunck (1981)**: global variational method, $\min \iint (I_xu+I_yv+I_t)^2 + \lambda(\|\nabla u\|^2 + \|\nabla v\|^2)$ — the data term plus a smoothness regulariser. Read it as the direct answer to the aperture problem: where the data term constrains only one direction, the smoothness term supplies the other by copying it from the neighbours, and the two terms trade off through $\lambda$. Its weakness follows from the same reading — the penalty is *quadratic* in $\|\nabla u\|$, so it grows fastest exactly where the flow changes fastest, which is at motion boundaries. It therefore smooths hardest precisely where you least want it to, blurring a moving car into the background it occludes. (This is the same pathology the $P_1/P_2$ split in SGM was designed to avoid, 5.6, and the modern fix is the same: a robust, sub-quadratic penalty that saturates instead of growing.)
- **Farnebäck**: polynomial expansion, OpenCV's default dense method.
- **FlowNet / FlowNet2** (2015–17): the first learned flow, trained on synthetic data (FlyingChairs) — the paper that established synthetic-to-real transfer for flow.
- **PWC-Net** (2018): **P**yramid, **W**arping, **C**ost volume — encodes the classical recipe as network structure, which is why it was both accurate and small.
- **RAFT** (ECCV 2020, best paper): builds an **all-pairs 4-D correlation volume** and iteratively refines flow with a shared GRU update operator, at a single high resolution. Currently the reference design; its descendants (RAFT-Stereo, CRAFT, FlowFormer, SEA-RAFT) dominate.

**The pattern to notice:** the winning learned architectures (PWC-Net, RAFT) *encode classical structure* — pyramids, warping, cost volumes — rather than replacing it with a generic network. Same story as FPN encoding the Laplacian pyramid.

### Tracking

**Flow answers "where did this pixel go, once". Tracking answers "which observations across a hundred frames are the same object", which is a different problem** — it is an *association* problem, not an estimation one, and the table below is really a table of answers to "what do you match on": image gradients (KLT), a learned appearance embedding (DeepSORT, Siamese), or predicted position (SORT's Kalman filter). Each row buys robustness to a different thing.

| Paradigm | Method | Note |
|---|---|---|
| Feature tracking | KLT (LK + Shi-Tomasi corners) | the classical workhorse; still used in VO front-ends |
| Correlation filters | MOSSE, KCF, CSRT | very fast, single-object, no learning |
| **Tracking-by-detection** | **SORT** (Kalman + Hungarian on IoU), **DeepSORT** (+ appearance embedding), **ByteTrack** (associate *low*-confidence detections too), **BoT-SORT** | the dominant multi-object paradigm |
| Siamese trackers | SiamFC, SiamRPN++ | template matching in embedding space — **Module 3.4 again** |
| Transformer trackers | TransT, MixFormer, OSTrack | current SOTA for single-object tracking |
| Any-point tracking | TAP-Vid, CoTracker, TAPIR | track arbitrary points through occlusion — the modern frontier |

**Two things worth knowing precisely about MOT:**
- **SORT = Kalman filter (constant velocity) + Hungarian assignment on IoU.** The Hungarian algorithm again (4.8) — same tool, different problem.
- **ByteTrack's insight is embarrassingly simple and very effective**: don't throw away low-confidence detections. Associate high-confidence detections first, then try to match the *remaining* tracks against the low-confidence detections, which are usually occluded objects rather than noise. It topped MOT benchmarks with no new network.
- **Metrics**: MOTA (dominated by detection quality), IDF1 (measures identity preservation), and **HOTA** (Higher Order Tracking Accuracy — explicitly decomposes into detection accuracy × association accuracy, and is now the preferred metric because MOTA and IDF1 each hide half the story). Naming HOTA and its factorisation is a good currency signal, and it rhymes with PQ = SQ × RQ from 4.12.

### 🎯 Top-1% distinction

1. **Derive the flow constraint and state the aperture problem as "$M$ is rank-1 on an edge"** — explicitly linking to Harris.
2. **Coarse-to-fine is required because the linearisation assumes sub-pixel motion.**
3. **PWC-Net and RAFT encode classical structure** (pyramid/warp/cost-volume) rather than discarding it.
4. **ByteTrack's low-confidence association trick.**
5. **HOTA = detection × association**, and why MOTA alone is misleading.
6. **Brightness constancy fails** under illumination change, specularity, transparency, and motion blur — which is why photometric methods lose to feature/learned methods in the wild.

### ✅ Mastery check

(a) You run Lucas–Kanade on a rotating fan blade. Describe what happens at a point in the middle of a blade's straight edge versus at the hub where blades meet.
(b) Brightness constancy is the founding assumption. Name four physical situations that violate it, and say which one is worst for a road-scene system.
(c) A multi-object tracker reports MOTA 0.72 and IDF1 0.41. Diagnose what is happening, and say which metric would have told you directly.

<details><summary>Answer sketch</summary>
(a) <b>On the blade edge:</b> the structure tensor $M$ is rank-1 — one large eigenvalue along the gradient (across the edge), one near zero along it. This is the <b>aperture problem</b>: only the normal component of motion is observable, so LK returns a vector whose tangential part is noise amplified by $1/\lambda_2$, and the blade appears to move perpendicular to itself regardless of its true motion. <b>At the hub:</b> gradients in many directions ⇒ both eigenvalues large ⇒ $M$ well-conditioned ⇒ the full 2-D motion is recoverable. This is exactly Shi–Tomasi's "good features to track" criterion, and it is why trackers seed on corners. (Bonus: a fast fan also breaks the small-motion linearisation entirely — coarse-to-fine helps, motion blur does not.)
(b) <b>Illumination change</b> (a cloud, a streetlight, auto-exposure adjusting between frames); <b>specular highlights</b> (a highlight is fixed to the viewing geometry, not to the surface, so it slides across a moving object); <b>transparency and reflections</b> (a windscreen shows two motions at one pixel); <b>occlusion boundaries</b> (a pixel's content is not a moved version of anything — there is no correct flow). For road scenes the worst is <b>auto-exposure / illumination</b>, because it is global, frequent (entering a tunnel, sun through trees) and corrupts <i>every</i> pixel simultaneously rather than a region — which is precisely why direct/photometric methods lose to feature-based and learned methods outdoors.
(c) High MOTA with low IDF1 means <b>detection is good but association is bad</b>: objects are found in nearly every frame (MOTA is dominated by FP/FN detection counts) but identities are swapped constantly, so each ground-truth track is covered by many fragmented predicted IDs. Typical causes: occlusion crossings, a weak or absent appearance embedding, or a motion model that fails on non-linear paths. <b>HOTA</b> would have told you directly, because it factorises into detection accuracy × association accuracy — you would have seen DetA high and AssA low in one number. This is the same "factorise the metric until it maps onto your failure modes" move as PQ = SQ × RQ in 4.12.
</details>


### 🔨 Build + read

**Build:** Implement Lucas–Kanade optical flow from scratch (no `cv2.calcOpticalFlowPyrLK`) on a short webcam clip or a public video: for a set of Harris/Shi–Tomasi corners, solve the $2\times2$ least-squares system per point per frame pair, and draw the flow vectors. Then run `cv2.calcOpticalFlowPyrLK` on the same clip and compare — yours should agree closely on well-textured points and diverge exactly at the aperture-problem cases (edges, textureless patches) you'd predict from 5.7's derivation.

**Read:** Baker & Matthews, *Lucas-Kanade 20 Years On: A Unifying Framework* (2004, IJCV), §2 for the inverse-compositional formulation modern implementations use. Then Shi & Tomasi, *Good Features to Track* (1994, CVPR) for why $\min(\lambda_1,\lambda_2)$, not Harris's determinant trick, is the right point-selection criterion for tracking specifically.

---

## 5.8 Structure from Motion & Bundle Adjustment 🔴

> The gap analysis put it perfectly: listing SfM without bundle adjustment is like teaching neural networks without mentioning backpropagation. BA is the optimisation engine underneath every 3-D reconstruction and every SLAM system.

### Intuition

**5.5 gave us the pose between two cameras. So why not just do that a thousand times and stitch the answers together?**

That question is worth taking seriously, because the naive answer — chain the pairwise estimates — is exactly what people try first and exactly why SfM needed thirty years of work. Each pairwise estimate carries error. Chaining composes those errors multiplicatively, so by camera 50 the pose is nonsense, and worse, the *scale* of each pair is independent (5.5: $\mathbf{t}$ is only a direction), so even the units drift. Meanwhile the constraints you are throwing away are enormous: a point seen in twenty images gives twenty constraints, and a pairwise pipeline uses them two at a time.

So state the problem properly instead. Given many photos of a scene from unknown positions, recover both **where every camera was** and **where every 3-D point is**, simultaneously. Bundle adjustment is that simultaneous solve: adjust all cameras and all points at once so that every 3-D point, when re-projected into every camera that saw it, lands as close as possible to where it was actually observed. **Nothing is chained; every constraint votes on every parameter at once.**

The name is literal: it adjusts the *bundles* of light rays converging at each camera centre.

If you want an anchor from what you already know: this is a loss function over hundreds of thousands of parameters, minimised by a second-order method with an explicitly constructed and factorised Hessian. The unfamiliar part is not the optimisation — you know optimisation — it is that here the parameters have geometric meaning, the Hessian is sparse in a *structured* way you can exploit, and the problem is small enough to afford second-order steps that a neural network never could.

### The incremental SfM pipeline (COLMAP)

**Why a pipeline at all, if BA solves everything?** Because BA is non-convex and only refines: it descends from wherever you start it, and reprojection error has plenty of local minima to fall into. Everything below exists to hand BA a starting point close enough to the truth that descending from it works. Read the pipeline as "construct an initial guess, incrementally, never letting the error grow far enough to matter."

```
1. Feature extraction              SIFT (or SuperPoint) on every image
2. Feature matching                exhaustive / vocabulary-tree / sequential
3. Geometric verification          RANSAC on F or H per pair → the "scene graph"
4. Initialisation                  pick a good pair (many inliers, WIDE baseline,
                                   not planar) → E → R,t → triangulate
5. Incremental loop:
     a. Choose the next image      most 2D-3D correspondences with the current model
     b. Register it                PnP + RANSAC → its pose
     c. Triangulate new points     from the newly available views
     d. LOCAL bundle adjustment    over the recent cameras and their points
     e. Filter outliers            reprojection error, triangulation angle, cheirality
6. Periodic GLOBAL bundle adjustment
```

**PnP (Perspective-n-Point)** — given $n$ known 3-D points and their 2-D observations in a *new* image, find that camera's pose. **Note that this is a strictly easier problem than 5.5's**, and that is the point of the incremental structure: once you have a partial reconstruction, a new image is no longer being matched against another *unknown* camera, it is being fitted to *known* 3-D geometry, so there is no scale ambiguity and no four-fold decomposition. A pose has 6 DoF and each 2-D observation of a known 3-D point gives 2 equations, so 3 points give 6 equations for 6 unknowns — hence the minimal case is **P3P** ($n=3$, and the non-linearity leaves up to 4 solutions, disambiguated by a 4th point). EPnP is the standard $O(n)$ solver for larger $n$. Always wrapped in RANSAC, because the 2D–3D correspondences come from feature matching and are therefore contaminated. PnP is what "registers" each new image.

**Choosing the initial pair matters enormously**, and both failure modes are ones you can now derive rather than accept. A pair with a **short baseline** gives badly-conditioned triangulation: the two rays to a point are nearly parallel, so a sub-pixel error in either one slides the intersection a long way along the line of sight — which is precisely the $\delta Z \approx Z^2\delta d/(fB)$ law of 5.6 with a tiny $fB$. A **planar** pair is degenerate for a different reason: $F$ is unidentifiable up to a two-parameter family (5.4), so RANSAC returns a confident, meaningless epipolar geometry. COLMAP explicitly scores candidate pairs on inlier count *and* triangulation angle, which is the operational form of "enough matches, and enough parallax."

**Pause:** the initialisation picks a *wide*-baseline pair for good triangulation. But 5.6 pointed out that a wide baseline makes matching harder. Before reading on — why is that trade-off not fatal here, when it is a real constraint for a stereo rig?

Because SfM does not need dense correspondence. A stereo rig wants a disparity at *every* pixel, so it needs the two views to look alike everywhere; SfM needs a few hundred well-localised, viewpoint-robust keypoints, and SIFT was built precisely to survive large viewpoint change (1.6). **The trade-off is real in both cases; SfM can afford to sit further along it because sparse matching degrades far more gracefully than dense matching.**

### Bundle adjustment, formally

**Write down what you actually want and let the objective fall out of it.** What we want is: every 3-D point, pushed through every camera that saw it, should land where the feature detector said it did. So take the discrepancy for one (point, camera) pair — the **reprojection error** — square it, and sum over every observation in the dataset. Minimise total reprojection error over all camera parameters $\{\mathbf{c}_j\}$ and all 3-D points $\{\mathbf{X}_i\}$:

$$
\boxed{\ \min_{\{\mathbf{c}_j\},\{\mathbf{X}_i\}} \sum_{i}\sum_{j} v_{ij}\ \rho\Bigl(\bigl\| \pi(\mathbf{c}_j, \mathbf{X}_i) - \mathbf{x}_{ij}\bigr\|^2\Bigr)\ }
$$

- $v_{ij} = 1$ if point $i$ is visible in camera $j$, else 0.
- $\pi$ is the projection function (including distortion).
- $\rho$ is a **robust loss** — Huber or Cauchy — because a few mismatched features would otherwise dominate a pure least-squares objective. (Same reasoning as smooth L1 in 4.3.)

**This is the maximum-likelihood estimate** under the assumption of independent zero-mean Gaussian noise on the image measurements — and the reason is the one you already know from deep learning, imported wholesale. If each observed feature location is the true projection plus Gaussian noise, then the log-likelihood of the whole dataset is a constant minus the sum of squared reprojection residuals, so maximising likelihood *is* minimising this objective. Nothing special about vision here; it is the same derivation that makes MSE the right loss for Gaussian regression.

**Why that matters, and it is the reason the phrase "gold standard" gets used:** the noise is genuinely in the image, in pixels, because that is where the measurement was made. Any objective that measures error somewhere else — algebraic error in $F$'s entries, error in 3-D, error in a homography's coefficients — is minimising a quantity that is *not* the one the noise model describes, and so is not the ML estimate however elegant it looks. The 8-point algorithm of 5.5 minimises an algebraic error precisely because it buys a linear solve, and that is exactly why its output is an initialisation to be refined rather than an answer.

**In your own words:** why is reprojection error the "right" thing to minimise, rather than error measured in 3-D?

### Why it is tractable: sparsity and the Schur complement 🔴

**Do the arithmetic before believing the problem is easy.** 1000 cameras × 9 parameters + 100,000 points × 3 = **309,000 unknowns**. Gauss–Newton says: build the approximate Hessian $J^\top J$, solve one linear system per iteration, repeat. But that system is $309{,}000 \times 309{,}000$ — about $10^{11}$ entries, roughly 760 GB in double precision, and a dense factorisation costs $O(n^3) \approx 3\times10^{16}$ flops. Not "slow": impossible. **And yet COLMAP does exactly this on a laptop.** The rest of this subsection is how.

**Start from the residuals and ask what each one touches.** A residual $\mathbf{r}_{ij}$ is "where point $i$ landed in camera $j$, minus where it was observed." Differentiate it with respect to every parameter in the problem: it depends on the 9 parameters of camera $j$, and the 3 coordinates of point $i$, and **nothing else** — not on any other camera, not on any other point. So its row of the Jacobian $J$ has 12 non-zeros out of 309,000. That is the structural fact everything below rests on, and it is a direct consequence of how the objective was written: each term couples exactly one camera to exactly one point.

**Now push that through to $\mathcal{H} = J^\top J$.** An off-diagonal entry of $J^\top J$ between parameter $a$ and parameter $b$ is non-zero only if some single residual depends on both. Work through the three cases:
- **Two cameras.** No residual depends on two cameras at once, so the camera–camera block has entries only *within* each camera's own 9 parameters. Block-diagonal.
- **Two points.** Same argument. Block-diagonal, with $3\times3$ blocks.
- **A camera and a point.** Non-zero exactly when that camera observed that point — which is the visibility pattern $v_{ij}$, and is itself very sparse, since a given camera sees a small fraction of the scene.

So $J^\top J$ (the approximate Hessian, written $\mathcal{H}$ to avoid collision with the homography $H$ of 5.4) has a characteristic **arrowhead / bordered block-diagonal** structure — and it was not designed to; it is what the visibility structure of a photo collection looks like when you write it as a matrix:

$$
\mathcal{H} = \begin{bmatrix} B & E \\ E^\top & C \end{bmatrix}
\qquad
\begin{aligned}
B &: \text{block-diagonal, } 9\times9 \text{ per camera}\\
C &: \text{block-diagonal, } 3\times3 \text{ per point}\\
E &: \text{sparse camera–point coupling}
\end{aligned}
$$

**Pause:** 97% of the unknowns are points, and $C$ — the entire point-to-point block — is block-diagonal with $3\times3$ blocks. Before reading on: what does block-diagonality let you do that a general matrix does not, and why is that the whole solution?

**$C$ is block-diagonal with $3\times3$ blocks, so $C^{-1}$ is trivial.** A block-diagonal matrix inverts blockwise — the inverse of $\text{diag}(C_1,\dots,C_n)$ is $\text{diag}(C_1^{-1},\dots,C_n^{-1})$ — so inverting $C$ means inverting 100,000 independent $3\times3$ matrices, which is $O(N_{\text{pts}})$ work, embarrassingly parallel, and cheap enough to be a rounding error in the total runtime. **The expensive 97% of the problem is the part we can invert for free.** So: eliminate it.

That is the **Schur complement**. It is nothing more exotic than block Gaussian elimination — the same "solve one variable, substitute it into the rest" you did with $2\times2$ systems in school, with scalars promoted to matrix blocks. Written out on $\mathcal{H}\Delta = -g$ split as $B\Delta\mathbf{c} + E\Delta\mathbf{X} = \mathbf{v}$ and $E^\top\Delta\mathbf{c} + C\Delta\mathbf{X} = \mathbf{w}$: take the second equation, solve it for $\Delta\mathbf{X} = C^{-1}(\mathbf{w} - E^\top\Delta\mathbf{c})$ — legal precisely because $C^{-1}$ is cheap — and substitute into the first. The $\Delta\mathbf{X}$ disappears, and what is left is a system in the cameras alone:

$$
\underbrace{(B - EC^{-1}E^\top)}_{\text{reduced camera system, } S}\ \Delta\mathbf{c} = \mathbf{v} - EC^{-1}\mathbf{w}
$$

then back-substitute for the points:

$$
\Delta\mathbf{X} = C^{-1}\bigl(\mathbf{w} - E^\top\Delta\mathbf{c}\bigr)
$$

**Where $\mathbf{v}$ and $\mathbf{w}$ come from:** the normal equations are $\mathcal{H}\,\Delta = -g$ with the unknown split to match the blocks, $\Delta = [\Delta\mathbf{c};\ \Delta\mathbf{X}]$. Split the right-hand side the same way, $-g = [\mathbf{v};\ \mathbf{w}]$, so $\mathbf{v} \in \mathbb{R}^{9N_{\text{cam}}}$ is the camera block of the negative gradient and $\mathbf{w} \in \mathbb{R}^{3N_{\text{pts}}}$ the point block. That is all they are — **the Schur complement is pure block elimination on $\mathcal{H}\Delta = -g$.** No approximation has been made anywhere: $\Delta\mathbf{c}$ from the reduced system and $\Delta\mathbf{X}$ from the back-substitution are the *exact* solution of the full system.

**The reduced system $S$ is $9N_{\text{cam}} \times 9N_{\text{cam}}$** — for 1000 cameras that's $9000\times9000$, entirely tractable. The 100,000 points, which are 97% of the unknowns, were eliminated almost for free. Cubic cost went from $(309{,}000)^3$ to $(9{,}000)^3$: a factor of about $4\times10^{4}$.

**You might expect eliminating the points to lose information about them.** It doesn't, and it is worth seeing why not, because this is the conceptual heart of the trick. Look at what $S = B - EC^{-1}E^\top$ actually is: the original camera block $B$, *corrected* by a term built from the point block. That correction term is the points' influence, folded into the camera equations before the points are removed. So the reduced system already knows everything the points had to say about the cameras — and the back-substitution then recovers the points exactly. **Nothing is discarded; the points are summarised, solved around, and reconstructed.**

**This is the single most important algorithmic fact in 3-D reconstruction**, and being able to explain it — sparsity from the visibility structure, $C$ block-diagonal because each point touches only itself, Schur complement to marginalise the points — is a strong, specific, senior-sounding answer. And the word *marginalise* is not loose analogy: for a Gaussian, the Schur complement of a joint precision matrix **is** the precision of the marginal. So "eliminate the 3-D points from the linear system" and "integrate the latent 3-D points out of the joint posterior" are literally the same operation, which is precisely analogous to marginalising latent variables in a graphical model — the right way to frame it if your interviewer is ML-flavoured rather than robotics-flavoured. The visibility graph is the graphical model; the cameras are what remains after eliminating the point nodes.

Further scaling: $S$ is itself sparse, and the reason is legible in the same terms — an off-diagonal block of $S$ between cameras $j$ and $k$ comes from $EC^{-1}E^\top$, which is non-zero only if some point is seen by *both*. **Two cameras couple only if they share a point**, so a sequential capture with limited overlap gives a banded $S$. Hence sparse Cholesky, or **preconditioned conjugate gradient** (Ceres' `ITERATIVE_SCHUR`) for very large problems, where you never form $S$ at all and only ever apply it to vectors.

**In your own words:** why is the Hessian sparse, and what makes the *points* the right block to eliminate rather than the cameras?

**Levenberg–Marquardt** wraps the whole thing: $(\mathcal{H} + \lambda\,\text{diag}(\mathcal{H}))\Delta = -g$, interpolating between Gauss–Newton ($\lambda\to0$, fast near the optimum) and gradient descent ($\lambda\to\infty$, robust when far away), with $\lambda$ adapted based on whether each step reduced the cost.

### Gauge freedom

**One loose end: is the solution even unique?** Ask what happens if you pick up the entire reconstruction — every camera and every point together — and move it. Every camera's view of every point is unchanged, because you moved the observer along with the observed. So the reprojection error is *identical*, and the optimiser cannot tell the two configurations apart.

The objective is **invariant to a global similarity transform** — translate, rotate, and scale the entire reconstruction (cameras and points together) and every reprojection is identical. **Count the directions of that invariance and you have counted the rank deficiency**, because a direction in parameter space along which the cost is exactly flat is a direction in which the second derivative is zero — i.e. a null vector of $\mathcal{H}$. Translation contributes 3, rotation 3, uniform scale 1. So $\mathcal{H}$ is **rank-deficient by 7**. (Scale is in the list for a reason worth pausing on: doubling the scene and doubling every camera's distance from it leaves every projection identical, since $x = fX/Z$ depends only on the *ratio*. That is 5.1's ambiguity, one last time.)

Handle it by fixing one camera's pose and one distance (gauge fixing), or by adding a damping term (LM's $\lambda$ does this implicitly — damping adds $\lambda\,\text{diag}(\mathcal{H})$, which lifts the zero eigenvalues off zero and makes the system solvable, at the cost of an arbitrary but harmless choice of gauge), or by using a pseudo-inverse. **Knowing why $\mathcal{H}$ is singular, and by exactly 7, is a precise and impressive detail.** It is also the algebraic statement of "monocular reconstruction has no absolute scale" from 5.5 — the same fact, first met as a property of $E$, then as a property of the Hessian's null space.

### Practical notes

- **Ceres Solver** (Google) and **g2o** are the standard implementations; **GTSAM** for the factor-graph formulation.
- **Local BA** (recent keyframes + their points) runs continuously in SLAM; **global BA** runs after loop closure.
- **Initialisation matters**: BA is a non-convex problem and LM finds a local minimum. It cleans up a good initial estimate; it does not rescue a bad one.
- **Analytic Jacobians** (or automatic differentiation) — numeric differentiation is far too slow and less accurate.
- **Robust loss is not optional** at real outlier rates.

### 🎯 Top-1% distinction

1. **Write the BA objective and call it the maximum-likelihood estimate** under Gaussian image noise.
2. **Explain the sparsity structure and the Schur complement** — this is the differentiator.
3. **Gauge freedom: $\mathcal{H}$ is rank-deficient by exactly 7.**
4. **Robust loss (Huber/Cauchy) is mandatory**, for the same reason as smooth L1.
5. **BA refines, it does not initialise** — non-convexity means the incremental pipeline's job is to hand BA a good starting point.
6. **Local vs global BA** and why SLAM needs both.
7. **Frame it for an ML interviewer**: "it's a large sparse non-linear least-squares problem, and the Schur complement is marginalising out the 3-D points, the same move as eliminating latent variables in a graphical model."

### ✅ Mastery check

500 cameras (6 pose + 3 intrinsic params each), 80,000 points, ~200 observations per camera.
(a) Total unknowns. Size of the full Hessian $\mathcal{H}$.
(b) Size of the reduced camera system after the Schur complement. Speed-up factor in the dominant cost?
(c) Why is $\mathcal{H}$ singular, and by how much?
(d) Your BA converges but the reconstruction has a systematic "bowl" curvature. Diagnose.

<details><summary>Answer sketch</summary>
(a) Cameras: $500\times9 = 4500$. Points: $80{,}000\times3 = 240{,}000$. Total $= \mathbf{244{,}500}$ unknowns. Full $H$ is $244{,}500^2 \approx 6\times10^{10}$ entries ≈ <b>478 GB in float64</b> if dense. Direct dense factorisation is $O(n^3) \approx 1.5\times10^{16}$ flops — completely infeasible.
(b) Reduced system $S$ is $4500 \times 4500$ ≈ $2\times10^7$ entries ≈ 162 MB dense, factorised in $O(4500^3) \approx 9\times10^{10}$ flops — seconds. The dominant-cost ratio is roughly $(244{,}500/4500)^3 \approx \mathbf{1.6\times10^5}$, i.e. about <b>five orders of magnitude</b>, plus the $C^{-1}$ step is linear in the number of points because each block is $3\times3$.
(c) <b>Gauge freedom.</b> The reprojection error is invariant under a global similarity transform of the whole reconstruction: 3 translation + 3 rotation + 1 scale = <b>7 degrees of freedom</b> along which the cost is exactly flat. So $\mathcal{H}$ is rank-deficient by 7. Fix by constraining one camera's pose (6) and one baseline length or point distance (1), or rely on LM's damping.
(d) A systematic bowl/dome curvature in a reconstruction is the classic signature of <b>unmodelled or badly-estimated radial distortion</b> — "doming" is a well-known failure in drone photogrammetry. Mechanism: a small residual radial error is consistent across every image, and BA can partially absorb it by bending the whole scene rather than by correcting the distortion parameters, especially when the imagery lacks the variety needed to make distortion observable (e.g. all nadir images at constant altitude, all with the board/features in the same image region). <b>Fixes:</b> include oblique images and varied altitude in the capture; calibrate intrinsics separately with a proper checkerboard and hold them fixed (or partially fixed) during BA; add ground control points or GPS priors to pin the geometry; and check whether you're over-parameterising distortion ($k_3$ on a lens that doesn't need it). Worth also ruling out: a poor initialisation that LM settled into a local minimum, and drift from a purely sequential matching graph with no loop closures.
</details>

### 🔨 Build + read

**Build:** Implement bundle adjustment for a small synthetic problem (10 cameras, 200 points) using `scipy.optimize.least_squares` with a **sparse Jacobian** (`jac_sparsity`) — SciPy's own BA tutorial is a good scaffold. Then implement the Schur complement yourself and compare wall-clock against the naive dense solve as you scale the problem up; plotting time vs. number of points for both is the experiment that makes the concept permanent. Then run **COLMAP** on 40 of your own photos of a building and inspect the reconstruction, the per-image reprojection errors, and the effect of enabling/disabling refinement of the intrinsics.

**Read:** Triggs et al., "Bundle Adjustment — A Modern Synthesis" (2000) — the canonical reference; read §3 (parameterisation) and §6 (network structure and the Schur complement). Hartley & Zisserman, Appendix 6 (sparse LM). Schönberger & Frahm, "Structure-from-Motion Revisited" (COLMAP, CVPR 2016) — read §4 for the practical pipeline decisions.

---

## 5.9 SLAM — Conceptual Overview

### SLAM vs SfM

**5.8 solved reconstruction offline, with all the photos in hand and as much time as you like. Now put the same problem on a robot: the frames arrive one at a time, you must answer before the next one lands, and you can never go back and re-plan the capture. What has to change?**

Almost everything about the *engineering*, and almost nothing about the *mathematics* — which is the single most useful thing to notice about SLAM. The objective is still reprojection error; the solver is still sparse non-linear least squares; the Schur complement still does the work. What changes is that you cannot afford to solve the whole problem every frame, and that errors you make now cannot be fixed by a better capture later. Same underlying problem — estimate camera poses and scene structure — with different constraints:

| | **SfM** | **SLAM** |
|---|---|---|
| Data | an unordered photo collection | a sequential stream |
| Timing | offline, batch | **real-time**, causal |
| Optimisation | global BA over everything | local BA + a sliding window, global only at loop closure |
| Priority | accuracy | latency and consistency |
| Extra machinery | — | loop closure, relocalisation, map management, IMU fusion |

### The architecture

```
                    ┌─────────── FRONT-END (fast, per frame) ───────────┐
sensor stream ────> │ feature extraction / photometric alignment        │
                    │ data association, pose estimate (PnP or direct)   │
                    │ keyframe decision                                 │
                    └──────────────────────┬───────────────────────────┘
                                           v
                    ┌─────────── BACK-END (slower, per keyframe) ───────┐
                    │ local bundle adjustment / pose-graph optimisation │
                    │ map point culling and creation                   │
                    └──────────────────────┬───────────────────────────┘
                                           v
                    ┌─────────── LOOP CLOSURE ──────────────────────────┐
                    │ place recognition (DBoW2 / NetVLAD — Module 3!)   │
                    │ geometric verification → add a loop constraint    │
                    │ pose-graph optimisation, then global BA           │
                    └──────────────────────────────────────────────────┘
```

**Pause:** a robot drives a perfect circle around a building and returns to its starting point. Its pose estimate says it has ended up 30 m from where it started. Nothing is broken — every individual estimate was as good as it could be. Before reading on: what information does the system have available that would let it fix this, and where does that information come from?

**Loop closure is the defining feature of SLAM.** Without it, pose estimates drift monotonically — every frame-to-frame estimate has error and the errors integrate, so the uncertainty grows without bound no matter how good each individual step is. **The information that fixes it is not in any single frame; it is in the fact that two frames far apart in time look at the same place.** Recognising "I have been here before" adds a constraint between two nodes that are distant in the trajectory but adjacent in space, and that constraint turns an open chain into a loop — which is exactly what lets the optimiser redistribute the accumulated error around the whole circuit instead of dumping it all at the end. **And place recognition is exactly image retrieval (Module 3)**: DBoW2 is a bag-of-visual-words vocabulary tree (3.2!), and modern systems use NetVLAD or learned global descriptors. **That cross-module connection is worth volunteering.**

**Pose-graph optimisation vs BA:** after loop closure, running full BA is expensive, so systems first optimise a *pose graph* — nodes are keyframe poses, edges are relative-pose constraints, and 3-D points are marginalised out. It's much smaller and fixes the gross drift; full BA then polishes. **And "marginalised out" is the exact same word as in 5.8, doing the exact same job** — you have already seen this operation: eliminating the point block from the system, leaving a problem in the cameras alone. Pose-graph optimisation is what you get when you commit to that elimination permanently rather than back-substituting the points afterwards. That is why it is cheap, and also why it is only an approximation: once the points are gone for good, the optimiser can no longer trade a point's position against a camera's.

**In your own words:** why does a loop closure fix drift that no amount of improving the per-frame estimate ever could?

### The families

**The one axis that organises the whole table is: what do you measure the error in?** Indirect methods extract keypoints first and then minimise *reprojection* error in pixels of feature location — throwing away 99% of the image but gaining invariance to how bright it is. Direct methods skip the abstraction and minimise *photometric* error over raw intensities — using all the information but inheriting the brightness-constancy assumption of 5.7, with all its failure modes. Everything else in the table is a way of getting some of both.

| Approach | Idea | Examples |
|---|---|---|
| **Feature-based (indirect)** | extract keypoints, minimise **reprojection** error | **ORB-SLAM2/3** (the reference system) |
| **Direct** | skip features, minimise **photometric** error over pixels | LSD-SLAM, **DSO**, DTAM |
| **Semi-direct** | features for tracking, direct for mapping | SVO |
| **Visual-inertial (VIO)** | fuse an IMU | ORB-SLAM3, VINS-Mono, OKVIS — **this is what phones and headsets actually run** |
| **Learned / neural** | learned front-ends or fully differentiable | DROID-SLAM, and neural-map systems (NICE-SLAM, Gaussian-Splatting SLAM) |

**Why VIO dominates in products:** an IMU gives metric scale (resolving 5.5's scale ambiguity), gravity direction, and high-rate motion estimates that survive motion blur and textureless scenes. Every AR headset and phone AR framework (ARKit, ARCore) is visual-inertial. **Saying "monocular SLAM is scale-ambiguous, which is why real systems are visual-inertial" ties 5.5 to 5.9 in one sentence.**

### 🎯 Top-1% distinction

1. **Loop closure is the difference between SLAM and visual odometry** — VO has no map-level loop closure and therefore drifts without bound.
2. **Place recognition = image retrieval** (DBoW2 ≈ BoVW, NetVLAD) — Module 3 connection.
3. **Pose-graph optimisation as marginalised BA.**
4. **Direct vs indirect trade-off**: direct methods use all pixels (better in low-texture scenes) but assume brightness constancy (fragile to exposure/illumination change, and to rolling shutter); indirect methods are robust to photometric change but throw away most of the image.
5. **VIO for metric scale** — and note the IMU's own problem (bias drift), which is why it must be *jointly* estimated, not just integrated.
6. **Rolling shutter** breaks the single-projection-centre assumption on fast motion; serious systems model it explicitly.

### ✅ Mastery check

(a) Your visual odometry drifts by about 3% of distance travelled. The robot completes a 1 km loop and returns to its start. What is the position error at closure, and what single mechanism removes most of it?
(b) Why would a direct method outperform a feature-based one in a bare white corridor, and underperform it under flickering fluorescent lighting?
(c) An AR headset places a virtual object on a real table and it stays put at the right size. Where did the metric scale come from, given 5.5 says monocular reconstruction has none?

<details><summary>Answer sketch</summary>
(a) Roughly <b>30 m</b> — and the qualitative point matters more than the number: the error is <i>unbounded and accumulating</i>, because every frame-to-frame estimate contributes and nothing ever corrects the accumulated total. <b>Loop closure</b> removes most of it: recognising "I have been here before" adds a constraint between the current pose and the start pose, and pose-graph optimisation redistributes the accumulated error around the whole loop rather than leaving it at the end. This is precisely the difference between visual odometry and SLAM.
(b) <b>White corridor:</b> almost no keypoints — a feature-based front end finds nothing to track and fails outright. A direct method uses <i>every</i> pixel with any gradient at all, including the faint intensity variations a detector would reject, so it survives. <b>Flickering fluorescents:</b> direct methods minimise <b>photometric</b> error and therefore assume brightness constancy, which flicker violates globally every frame; a feature-based method using gradient-orientation descriptors is largely invariant to that. The general trade: direct methods use more of the image (better in low texture) but assume more about it (worse under photometric change) — and this is why real systems either fuse both (semi-direct, SVO) or add photometric calibration and exposure modelling (DSO).
(c) From the <b>IMU</b> — the headset runs <b>visual-inertial</b> odometry, not monocular. The accelerometer measures acceleration in physical units, so double-integrating it over a short window supplies the metric baseline that vision alone cannot observe, and gravity fixes two rotational degrees of freedom besides. The scale is estimated <i>jointly</i> with pose and IMU biases rather than integrated open-loop, because raw IMU integration drifts quadratically. This is the practical resolution of 5.5's scale ambiguity and the reason essentially every shipped AR system is visual-inertial rather than purely visual.
</details>

---

## 5.10 Modern 3D Representations: NeRF → 3D Gaussian Splatting 🟢

### The shift

**Everything so far answered "where is the geometry?". Ask a different question — "what would this scene look like from a camera position I never photographed?" — and the whole design changes.**

That is not a cosmetic difference. Classical reconstruction produces **explicit geometry** (a point cloud or mesh), and if you want a new view you render that geometry, which means every hole, every missing surface, every mis-triangulated point shows up as a visible artefact. The modern line refuses to build geometry as an intermediate product at all: it builds a representation optimised *directly* for rendering new views, and trains it by differentiable rendering against the input photos.

**And here is why you, specifically, already have the right intuition for this.** The recipe is one you use every day: define a parameterised function, define a differentiable forward process that turns it into a prediction, compare that prediction to observed data, backpropagate. The only unfamiliar piece is that the forward process is a *renderer* rather than a network layer stack — so the whole problem becomes "make rendering differentiable, then let gradient descent do inverse graphics." Everything in 5.10 is a choice about what the parameterised function is and how the renderer works.

### NeRF (Mildenhall et al., ECCV 2020)

Represent the scene as a **continuous function** — an MLP:

$$
F_\Theta: (\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)
$$

mapping a 3-D position and viewing direction to colour and volume density. Render a pixel by **volume rendering** along its ray:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt,
\qquad T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)
$$

**Read the integral rather than skipping it — every symbol is doing an obvious job.** March along the ray. At each depth $t$, the network reports a density $\sigma$ (how much stuff is there) and a colour $\mathbf{c}$ (what colour that stuff is, seen from direction $\mathbf{d}$). The contribution of that slice to the final pixel is "how much stuff × what colour", but weighted by whether light from there can actually reach the camera — and that is $T$, the **transmittance**, the probability the ray gets from the near plane to $t$ without being absorbed. Its exponential-of-negative-integral form is just the standard survival probability of something dying at rate $\sigma$: accumulate the density you have passed through, and exponentiate the negative. **So the integral says: sum up the colour of everything along the ray, discounted by how occluded it is.** Nothing about it is specific to neural networks; it is the classical volume-rendering equation from the 1980s, which is exactly why it was available to be borrowed.

The one property that matters for us is that every operation in it — the network evaluation, the products, the exponential, the discretised sum — is differentiable. Discretised with stratified sampling, this gives a fully differentiable path from $\Theta$ to a rendered pixel, so you can train $\Theta$ by comparing rendered pixels to the input photos. **The loss is just photometric MSE against the training images** — there is no 3-D supervision at all, which is the beautiful part: geometry emerges because it is the only explanation consistent with all the photos at once.

**Pause:** the MLP takes a raw 3-D coordinate as input and must output sharp, high-frequency detail. Given what you know about how networks fit functions, predict what goes wrong.

Two essential details fix exactly that. **Positional encoding** ($\gamma(p) = (\sin 2^0\pi p, \cos 2^0 \pi p, \ldots)$) — without it the MLP's **spectral bias** (its strong preference for fitting low-frequency functions first and high-frequency ones barely at all) makes it learn only smooth, blurry geometry; the sinusoidal lift makes high-frequency variation linearly accessible so the same network can express sharp edges. And **hierarchical sampling** (a coarse network guides where the fine network samples), because a scene is mostly empty and uniformly spaced samples spend nearly all their compute on nothing.

**Costs:** days to train, seconds per frame to render, and it needs known camera poses — **which come from COLMAP**, i.e. from SIFT + RANSAC + bundle adjustment. Instant-NGP (2022) cut training to seconds with a multi-resolution hash grid.

### 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)

**Ask where NeRF's cost actually goes and the redesign writes itself.** Rendering one pixel means hundreds of MLP evaluations along one ray, and there are a million pixels. The MLP is queried *per sample point*, and sample points are what you need because the scene is stored as a function you can only probe. So: what if the scene were stored as a list of things instead of a function to probe, so that each thing could be drawn once and splashed onto all the pixels it covers at the same time?

Replace the implicit MLP with an **explicit** set of millions of 3-D Gaussians, each with a position $\boldsymbol{\mu}$, covariance $\Sigma$ (stored as scale + rotation quaternion), opacity $\alpha$, and view-dependent colour as spherical-harmonic coefficients.

Render by **splatting**: project each Gaussian to the image plane (an affine approximation gives a 2-D Gaussian), sort by depth per tile, and alpha-composite front to back. This is **rasterisation, not ray marching** — so it runs on the GPU's native strengths.

**Why a Gaussian, of all shapes?** Because a Gaussian is the one blob whose *projection is also a Gaussian* — an affine map of a Gaussian is a Gaussian, so projecting a 3-D one to the image plane costs a covariance transform rather than a rendering algorithm. It also has no boundary, which means the alpha it contributes to each pixel is a smooth function of its parameters, which means gradients flow. **A shape with a hard edge would have zero gradient almost everywhere and a discontinuity at the edge, and could not be optimised at all** — the same reason you use soft attention rather than hard selection.

Optimise the Gaussians' parameters by the same photometric loss, plus **adaptive density control**: clone Gaussians in under-reconstructed regions, split large ones in over-reconstructed regions, and prune near-transparent ones. That last mechanism has no analogue in NeRF and is easy to underrate: **the model's own capacity is part of what is being optimised**, so the representation grows where the photos disagree with it and shrinks where they don't — closer to architecture search interleaved with training than to ordinary gradient descent.

**In your own words:** both methods minimise the same photometric loss against the same photos. What is the one design choice that separates them, and what follows from it?

| | NeRF | 3DGS |
|---|---|---|
| Representation | implicit (MLP) | explicit (millions of Gaussians) |
| Rendering | ray marching (many MLP evals/ray) | rasterisation + alpha blending |
| Training | hours–days (Instant-NGP: minutes) | **minutes** |
| Rendering speed | seconds/frame (Instant-NGP: interactive) | **>100 FPS** |
| Memory | small model (MB) | large (hundreds of MB–GB) |
| Editing | hard (weights are opaque) | **easy** (move/delete Gaussians directly) |
| Both need | **COLMAP poses + sparse point cloud for init** | same |

**As of 2026, 3DGS has largely displaced NeRF for practical novel-view synthesis** — real-time rendering, fast training, editable, and mesh/point-cloud-friendly. NeRF's ideas (volume rendering, differentiable inverse rendering) remain foundational, and the research frontier has moved to dynamic scenes (4D-GS), large-scale/city-scale reconstruction, generative 3D, SLAM integration, and compression of the Gaussian set.

### 🎯 Top-1% distinction

1. **"Both are initialised from a classical SfM reconstruction."** 3DGS is typically seeded from COLMAP's sparse point cloud, and both need COLMAP poses. **So the most modern 3-D method in the field still runs on SIFT + RANSAC + bundle adjustment.** That is the perfect closing sentence for Module 5 — and a genuinely good answer to "is classical CV obsolete?"
2. **The core distinction is implicit-vs-explicit and ray-marching-vs-rasterisation**, and rasterisation is why 3DGS is fast — it maps onto what GPUs already do.
3. **Positional encoding fixes the MLP's spectral bias** — a specific, checkable detail.
4. **These are novel-view-synthesis methods, not geometry methods.** They optimise rendering quality, and the extracted geometry can be poor (floaters, wrong surfaces in unobserved regions) — a distinction that matters if you need a mesh for physics or measurement rather than pictures.

### ✅ Mastery check

(a) NeRF without positional encoding produces blurry, over-smooth geometry. Name the mechanism and why the encoding fixes it.
(b) 3DGS renders at >100 FPS where NeRF renders at ~1 FPS. Give the single architectural reason, not a list.
(c) You need a watertight mesh for 3D printing from 40 phone photos of a sculpture. NeRF, 3D Gaussian Splatting, or COLMAP + MVS? Defend the choice.

<details><summary>Answer sketch</summary>
(a) <b>Spectral bias.</b> A coordinate MLP with standard activations preferentially learns low-frequency functions — it fits the smooth part of the signal fast and the high-frequency part barely at all. Positional encoding $\gamma(p) = (\sin 2^0\pi p, \cos 2^0\pi p, \dots, \sin 2^{L-1}\pi p, \cos 2^{L-1}\pi p)$ lifts the 3-D input into a high-dimensional space where high-frequency variation is <i>linearly</i> accessible, so the same MLP can represent sharp geometry and texture. (This is the same insight as Fourier features / random features, and the same reason ViTs need positional information injected rather than learned from raw indices.)
(b) <b>Rasterisation instead of ray marching.</b> NeRF must evaluate the MLP at many sample points along <i>every</i> ray — hundreds of network evaluations per pixel. 3DGS projects each Gaussian to the image plane once and alpha-composites the sorted result, which is the operation GPU hardware was built for. Everything else (explicit vs implicit, SH colour, adaptive density) follows from or supports that choice.
(c) <b>COLMAP + MVS</b>, and it is not close. NeRF and 3DGS optimise <b>novel-view rendering quality</b>, not surface accuracy — their extracted geometry has floaters, hallucinated surfaces in unobserved regions, and no watertightness guarantee, because nothing in their objective penalises being wrong where no camera looked. A classical SfM + multi-view-stereo + Poisson-reconstruction pipeline optimises geometric consistency directly and yields a watertight mesh with a defensible error characterisation. The general principle worth stating: <b>match the method's objective to your deliverable</b> — if you need pictures, use a rendering method; if you need a surface, use a geometry method. (Reasonable refinement: run COLMAP anyway, since 3DGS needs its poses regardless, then compare — and note that SuGaR / 2DGS-style variants add explicit surface regularisation precisely to close this gap.)
</details>

---

## 5.11 Practical: Camera Calibration + Stereo Depth Lab

**Why do these labs rather than read another chapter?** Because every claim in this module is falsifiable with hardware you already own, and a geometric law you have *measured* is held differently from one you have read. The three labs are each built around a single prediction the module made — that distortion coefficients overfit, that depth error grows as $Z^2$, that planar scenes silently break $F$ estimation — and each is designed so that you find out whether the notes were telling the truth.

**In your own words:** before starting, write one sentence predicting what the depth-error-vs-distance plot in Lab B will look like. Then check it against what you measure — that comparison is the whole point of the lab.

**Lab A — Calibration.** Print a checkerboard, mount it flat, capture 20+ images covering the whole frame including corners and a range of tilts. Then:
1. Detect corners with sub-pixel refinement; calibrate with `cv2.calibrateCamera`.
2. Report RMS reprojection error and, more importantly, **plot the per-corner residual vectors over the image** — look for structure (systematic radial pattern ⇒ under-modelled distortion; one bad image ⇒ blur or a mis-detected board).
3. Ablate the distortion model: fit with $k_1$; $k_1,k_2$; $k_1,k_2,k_3$; $+p_1,p_2$. Report RMS **and** cross-validated error on held-out images. You should see RMS keep dropping while held-out error stops improving — **overfitting, demonstrated on your own calibration.**
4. Undistort an image of a straight-edged object and verify the edges are straight.

**Lab B — Stereo depth.** With two calibrated cameras on a rigid bar:
1. Stereo-calibrate for $R,\mathbf{t}$; rectify; verify by drawing horizontal lines across both images and checking that features fall on the same rows.
2. Compute disparity with block matching and with SGBM; compare.
3. Convert to depth and compare against tape-measured ground truth at 5 known distances. **Plot measured depth error vs. distance and fit the predicted $Z^2/(fB)$ curve.** Confirming the quadratic law with your own tape measure is the single most valuable hour in this module.
4. Add the left–right consistency check and see the occlusion regions light up.

**Lab C — SfM.** Run COLMAP on 40 photos of a building. Then rerun with (i) a deliberately short baseline set, (ii) photos of a flat wall only. Document both failures — you will have reproduced the initialisation and planar-degeneracy problems yourself.

**Pause:** before you run case (ii), predict what COLMAP will do with photos of a flat wall. Will it error out, produce a visibly broken reconstruction, or report success? Commit to an answer, then run it — the fact that the planar case usually fails *quietly*, with a healthy-looking inlier count (5.4), is exactly the lesson the lab exists to burn in, and it only lands if you predicted otherwise first.

---

## Going Deeper — Papers, Sources and Research Scope

*Geometry is the oldest and most settled part of the curriculum — depth and geometry fell to ~1.2% of CVPR 2026 highlights. Two things follow. First: the classical material below is **reference-grade and permanent**; Hartley & Zisserman will still be correct in 2040, which is not true of most of Module 6. Second: the live research in 3D has moved almost entirely to learned, feed-forward methods, and that is where §D points.*

### A. The canonical papers

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Longuet-Higgins, *A computer algorithm for reconstructing a scene from two projections* | 1981, Nature | The essential matrix and the eight-point algorithm, in one page. Read the original — it is startlingly direct. | Nature 293 |
| **Hartley, *In Defense of the Eight-Point Algorithm*** | 1997, PAMI | Normalisation turns a famously unstable algorithm into a good one. The condition-number argument is the lesson. | PAMI 19(6) |
| Nistér, *An Efficient Solution to the Five-Point Relative Pose Problem* | 2004, PAMI | Uses the essential matrix's internal constraints instead of discarding them; five points instead of eight, which transforms RANSAC's iteration count. | PAMI 26(6) |
| Zhang, *A Flexible New Technique for Camera Calibration* | 2000, PAMI | The planar-target calibration everyone uses. OpenCV's `calibrateCamera` is this paper. | PAMI 22(11) |
| Lucas & Kanade, *An Iterative Image Registration Technique* | 1981, IJCAI | Brightness constancy, the linearisation, and the least-squares solve. Two pages. | IJCAI 1981 |
| Horn & Schunck, *Determining Optical Flow* | 1981, Artificial Intelligence | The global/variational alternative to LK, with the smoothness prior. Read the two together — they define the whole design space. | AI 17 |
| **Triggs et al., *Bundle Adjustment — A Modern Synthesis*** 🔴 | 1999, Vision Algorithms workshop | The definitive treatment of the sparse-Hessian structure and the Schur complement. 70 pages; skim, then use as reference. §6 and §8 are the load-bearing parts. | Springer LNCS 1883 |
| Mur-Artal et al., *ORB-SLAM* | 2015, T-RO (1502.00956) | The reference monocular SLAM system — tracking, local mapping, loop closing as three threads. | arXiv 1502.00956 |
| Schönberger & Frahm, *Structure-from-Motion Revisited* (COLMAP) | 2016, CVPR | The incremental-SfM pipeline that is still the field's ground-truth generator. | CVPR 2016 |
| Engel et al., *Direct Sparse Odometry* | 2016, PAMI (1607.02565) | The direct (photometric) alternative to feature-based SLAM — worth reading precisely because it rejects Module 1's whole approach. | arXiv 1607.02565 |
| **Mildenhall et al., *NeRF*** | 2020, ECCV (2003.08934) | Volume rendering made differentiable; positional encoding as a fix for MLP spectral bias. | arXiv 2003.08934 |
| Müller et al., *Instant-NGP* | 2022, SIGGRAPH (2201.05989) | Multiresolution hash encoding — NeRF from hours to seconds. The encoding, not the renderer, was the bottleneck. | arXiv 2201.05989 |
| **Kerbl et al., *3D Gaussian Splatting*** | 2023, SIGGRAPH (2308.04079) | Explicit primitives + tile-based differentiable rasterisation. Real-time, and it displaced NeRF for most reconstruction use. | arXiv 2308.04079 |
| **Wang et al., *DUSt3R*** | 2024, CVPR (2312.14132) | **The paradigm shift.** Two uncalibrated images in, pointmaps out, no camera parameters, no SfM. Read it and ask honestly which parts of 5.1–5.8 it makes optional. | arXiv 2312.14132 |
| Leroy et al., *MASt3R* | 2024, ECCV (2406.09756) | Adds matching to DUSt3R's grounding; the practical version. | arXiv 2406.09756 |
| Wang et al., *VGGT: Visual Geometry Grounded Transformer* | 2025, CVPR (2503.11651) | Feed-forward camera poses, depth and point tracks from many views in one pass. CVPR 2025 Best Paper. | arXiv 2503.11651 |
| Yang et al., *Depth Anything V2* | 2024, NeurIPS (2406.09414) | Monocular relative depth as a solved commodity. Metric depth is still not. | arXiv 2406.09414 |
| Sarlin et al., *SuperGlue* | 2020, CVPR (1911.11763) · Lindenberger et al., *LightGlue*, 2023, ICCV (2306.13643) | What replaced ratio-test matching in modern geometry pipelines. | arXiv |

### B. The single best source, per hard topic

Hartley & Zisserman (*Multiple View Geometry in Computer Vision*, 2nd ed., 2004 — "H&Z") is the reference for almost everything here; page numbers below are from the 2nd edition.

- **Homogeneous coordinates and projective geometry (5.1).** H&Z **Ch. 2, §2.1–2.2 (pp. 25–37)**. Then Shree Nayar's *First Principles* "Image Formation" module for the physical picture.
- **Intrinsics, extrinsics, and the P = K[R|t] decomposition (5.2).** H&Z **§6.1–6.2 (pp. 153–164)**. The gotcha that $\mathbf{t}$ is not the camera centre is on p. 156.
- **Calibration and distortion (5.3).** Zhang 2000 §2–3, then OpenCV's *Camera Calibration and 3D Reconstruction* docs for the exact distortion model implemented.
- **Epipolar geometry (5.5) 🔴.** H&Z **Ch. 9, §9.1–9.3 (pp. 239–254)** for F, and **§9.6 (pp. 257–260)** for E and the four-fold pose ambiguity. This is the single most valuable 20 pages in the module.
- **The eight-point algorithm and normalisation (5.5).** H&Z **§11.1–11.2 (pp. 279–285)**, then Hartley 1997 itself.
- **Triangulation (5.6).** H&Z **Ch. 12 (pp. 310–324)** — in particular why the naive DLT triangulation is not optimal and what the optimal method costs.
- **Stereo and disparity (5.6).** Szeliski **Ch. 12**, plus Hirschmüller's SGM paper (2008, PAMI) for the semi-global energy that everything practical still uses.
- **Optical flow (5.7).** Baker & Matthews, *Lucas-Kanade 20 Years On* (IJCV 2004) — the definitive unification of the LK variants, and it explains the inverse-compositional trick properly.
- **Bundle adjustment (5.8) 🔴.** Triggs et al. §6 (the sparse structure) and §8 (implementation). Then read Ceres Solver's *Bundle Adjustment* tutorial, which is the same maths in code you can run.
- **SLAM overview (5.9).** Cadena et al., *Past, Present, and Future of SLAM* (2016, T-RO, arXiv 1606.05830) — the best single survey, and its open-problems section still largely holds.

### C. Reference implementations worth reading

- **`colmap/colmap` → `src/colmap/estimators/bundle_adjustment.cc`.** See how the Schur complement is invoked via Ceres (`SPARSE_SCHUR`), and — more instructively — how the gauge freedom is fixed (which parameters are held constant, and why it must be seven of them).
- **`ceres-solver/ceres-solver` → `examples/bundle_adjuster.cc`.** The canonical 300-line BA program. Read the `ReprojectionError` functor and the ordering setup; the ordering *is* the Schur trick.
- **`opencv/opencv` → `modules/calib3d/src/five-point.cpp`.** Nistér's 5-point solver. You will not follow every line of the polynomial elimination, and that's fine — read it to appreciate what "use the internal constraints" costs in practice.
- **`graphdeco-inria/gaussian-splatting` → `submodules/diff-gaussian-rasterization`.** The CUDA rasteriser. Look for the tile binning and the front-to-back alpha compositing — the differentiability of *sorting* is the trick that makes the whole thing trainable.
- **`naver/dust3r` → `dust3r/model.py` and the pointmap head.** Then ask: where did the camera intrinsics go? (Answer: implicit in the pointmap. That's the whole idea.)
- **`UZ-SLAMLab/ORB_SLAM3`.** Read `src/Optimizer.cc` for the pose-graph and local-BA formulations. Heavy going, but it is what a real SLAM system looks like.

### D. Open research questions

1. **Where does feed-forward 3D break, and does it degrade gracefully?** *Why open:* DUSt3R/VGGT are evaluated on standard benchmarks; the failure boundary (textureless surfaces, repetitive structure, very wide baselines, non-Lambertian materials) is anecdotal. *Minimum experiment:* generate controlled Blender scenes sweeping texture density, baseline angle and repetition; run VGGT, DUSt3R and COLMAP; map where each fails. **Inference-only — very feasible, and genuinely useful. My pick from this module.**
2. **Does classical bundle adjustment still improve a feed-forward reconstruction, and by how much?** *Why open:* the natural hybrid (feed-forward init → BA refine) is obvious and under-quantified. *Minimum experiment:* VGGT output as BA initialisation vs COLMAP's own init, compare final reprojection error and convergence iterations. **Feasible with Ceres/pycolmap.**
3. **Is monocular *metric* depth achievable without intrinsics, or is the ambiguity fundamental?** *Why open:* focal length and depth are entangled in a single view; models that claim metric depth are implicitly predicting intrinsics. Nobody has cleanly separated how much is prediction vs prior. *Minimum experiment:* feed the same image content at varying synthetic focal lengths and measure whether predicted metric depth tracks the true scale. **Very feasible and a clean falsifiable test.**
4. **How much of SLAM's drift is front-end vs back-end?** *Why open:* end-to-end system numbers dominate; the decomposition is rarely reported. *Minimum experiment:* swap ORB features for SuperPoint/LightGlue in ORB-SLAM3's front end, hold the back end fixed, measure ATE on TUM/EuRoC. **Moderate — the engineering is the hard part, not the compute.**
5. **Do 3DGS reconstructions preserve *metric* geometry, or only appearance?** *Why open:* 3DGS is evaluated with PSNR/SSIM — photometric metrics — and its geometric accuracy is much less examined. *Minimum experiment:* reconstruct scenes with known ground-truth geometry, measure point-to-surface error rather than rendered-image quality. **Feasible, and the answer matters for anyone using 3DGS for measurement rather than viewing.**
6. ~~A better SfM pipeline.~~ **Closed.** COLMAP is a decade of engineering. Compete with it and you will lose; build on it instead.

---

## Module 5 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Projection | $x = fX/Z$; homogeneous coords make it linear at the cost of "equality up to scale" — **which is the depth ambiguity, algebraically** |
| Camera matrix | $P = K[R\mid\mathbf{t}]$, 11 DoF (5 intrinsic + 6 extrinsic) |
| Camera centre | $\mathbf{C} = -R^\top\mathbf{t}$ — $\mathbf{t}$ is *not* the position |
| Rotation params | axis–angle/Lie algebra for optimisation (minimal, unconstrained); **6-D continuous** for network regression (quaternions are discontinuous on $SO(3)$) |
| Distortion | applied in **normalised** coords before $K$; $k_3$ overfits on normal lenses; fisheye needs a different projection model |
| Zhang calibration | plane→image homographies; orthonormality of $\mathbf{r}_1,\mathbf{r}_2$ gives **2 constraints per view**; $\ge 3$ views for 5 DoF; then LM on reprojection error |
| Plane-induced $H$ | $H = K'(R - \mathbf{t}\mathbf{n}^\top/d)K^{-1}$; $\mathbf{t}=0$ ⇒ scene-independent (panoramas); explains the $F$ planar degeneracy |
| **Epipolar** | coplanarity ⇒ $\hat{\mathbf{x}}'^\top[\mathbf{t}]_\times R\,\hat{\mathbf{x}} = 0$; $E = [\mathbf{t}]_\times R$, $F = K'^{-\top}EK^{-1}$ |
| $F$ vs $E$ | $F$: pixels, 7 DoF, $\det F{=}0$. $E$: normalised, **5 DoF**, two equal singular values. Both rank 2 (from $[\mathbf{t}]_\times$) |
| 8-point | **Hartley-normalise**, SVD, then **project to rank 2** (Eckart–Young). Skipping either has distinct visual symptoms |
| Pose from $E$ | 4-fold ambiguity → **cheirality** (positive depth in both cameras); $\mathbf{t}$ only up to **scale** ⇒ monocular SfM/SLAM has no metric scale |
| Stereo | $Z = fB/d$; **$\delta Z \approx \frac{Z^2}{fB}\delta d$** — depth error grows quadratically; larger $B$ = better depth, worse matching |
| SGM | data term + $P_1$ (small, allows slant) / $P_2$ (large, preserves discontinuities); 1-D DP aggregated over 8–16 directions |
| Optical flow | $I_xu + I_yv + I_t = 0$; aperture problem = **the structure tensor $M$ is rank-1 on an edge** (Harris again); coarse-to-fine because linearisation assumes sub-pixel motion |
| Modern flow | PWC-Net and RAFT **encode classical structure** (pyramid, warp, cost volume) |
| Tracking | SORT = Kalman + Hungarian on IoU; ByteTrack associates **low-confidence** detections too; **HOTA = detection × association** |
| **Bundle adjustment** | minimise reprojection error over all cameras and points with a robust loss — the **MLE** under Gaussian image noise |
| **BA tractability** | Jacobian sparsity ⇒ arrowhead Hessian $\mathcal{H}$; $C$ (points) is block-diagonal $3\times3$ ⇒ **Schur complement** marginalises the points, leaving a $9N_{\text{cam}}$ system |
| Gauge freedom | $\mathcal{H}$ is rank-deficient by exactly **7** (3 translation + 3 rotation + 1 scale) |
| SfM pipeline | match → verify → good initial pair (wide baseline, non-planar) → incremental PnP + triangulate + local BA → global BA |
| SLAM | **loop closure** is what distinguishes it from VO; place recognition = image retrieval (DBoW2/NetVLAD); pose-graph = BA with points marginalised; VIO gives metric scale |
| NeRF | MLP $(\mathbf{x},\mathbf{d})\to(\mathbf{c},\sigma)$ + volume rendering; positional encoding defeats spectral bias; trained on photometric loss alone |
| 3DGS | explicit Gaussians + **rasterisation** ⇒ real-time; adaptive density control; has largely displaced NeRF in practice |
| The closing point | **both NeRF and 3DGS are initialised by COLMAP** — SIFT + RANSAC + bundle adjustment. Classical geometry is the foundation of the most modern 3-D methods |

---

*End of Module 5 notes. Drills in `Module-05-Drills.md`.*
