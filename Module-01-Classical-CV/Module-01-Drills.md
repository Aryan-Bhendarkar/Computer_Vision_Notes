# Module 1 — Drills, Interview Bank & Spaced Repetition

> Use this file **after** you've read `Module-01-Notes.md` once. Do not read the answers first. Time yourself: an interview answer should take 60–120 seconds spoken.

---

## Part A — Rapid-fire (answer in one or two sentences, out loud)

1. Why is the Gaussian the *unique* smoothing kernel for scale-space?
2. Write the semigroup property of Gaussians. Why does SIFT care?
3. Derive DoG ≈ scale-normalised LoG. What is the constant factor and why doesn't it matter?
4. Why must the Laplacian be multiplied by $\sigma^2$ before comparing across scales?
5. A blob is detected at $\sigma = 4$. What is its radius in pixels?
6. What is the structure tensor and what do its eigenvalues mean?
7. Why does Harris use $\det - \kappa\,\mathrm{tr}^2$ instead of the eigenvalues directly?
8. Is Harris scale-invariant? What fixes it?
9. State the aperture problem in terms of the structure tensor.
10. Why is Sobel really a derivative-of-Gaussian?
11. What does non-maximum suppression do in Canny, and what breaks without it?
12. Why two thresholds in Canny instead of one?
13. Name Canny's three optimality criteria.
14. Why is SIFT's descriptor $4\times4\times8$ and not $8\times8\times8$?
15. What does the 0.2 clamping in SIFT normalisation model?
16. Why does SIFT sometimes emit two keypoints at the same location?
17. What is SIFT's edge-rejection test and which earlier concept does it echo?
18. Give the ORB descriptor size in bytes and the matching operation.
19. How does ORB assign orientation, and when does that fail?
20. What is the "R" in rBRIEF actually doing?
21. State Lowe's ratio test and the two numbers from his ROC curve.
22. Why does a global distance threshold fail where the ratio test works?
23. Prove L2 and cosine give the same ranking for L2-normalised vectors.
24. Why do exact kd-trees fail on 128-D SIFT descriptors?
25. Derive $N = \log(1-p)/\log(1-w^s)$.
26. Homography with $w=0.5$, $p=0.99$: how many RANSAC iterations?
27. Why does the 5-point algorithm exist when the 8-point one is simpler?
28. How do you principledly choose RANSAC's inlier threshold?
29. What is MSAC's cost function and why is it better than counting?
30. What does MAGSAC++ remove from the pipeline?
31. Name the degenerate configuration that silently breaks fundamental-matrix RANSAC.
32. How many DoF does a homography have and why not 9?
33. Why solve $A\mathbf{h}=\mathbf{0}$ with SVD instead of least squares?
34. What is Hartley normalisation and what happens without it?
35. Name the two — and only two — conditions under which a homography exactly relates two views.
36. What causes ghosting in a panorama, and what is the *capture-side* fix?
37. Why multi-band blending instead of alpha blending?
38. Why cylindrical rather than planar warping past ~90° FoV?
39. Give three situations in 2026 where you'd still choose classical CV over a network.
40. What is the modern hybrid SfM stack, layer by layer?

---

## Part B — Whiteboard problems (work these with pen and paper)

**B1 — Scale-space arithmetic.**
You build a SIFT pyramid with $s = 3$, base $\sigma_0 = 1.6$, 4 octaves.
(a) List every $\sigma$ value computed in octave 0.
(b) You already have the image blurred to $\sigma = 1.6$ and want $\sigma = 2.016$. What single Gaussian do you convolve with? Give the number.
(c) Which image in octave 0 becomes the base of octave 1, and what operation is applied?
(d) A keypoint is found in octave 3 at scale index 2. Give its $\sigma$ in original-image pixels.

**B2 — Harris under transformation.**
An image patch has $M = \begin{bmatrix}900 & 300\\300 & 400\end{bmatrix}$.
(a) Eigenvalues. Classify the point.
(b) Harris response with $\kappa=0.04$; Shi–Tomasi response.
(c) The image is rotated 45°. What is the new $M$ and the new $R$? Show that $R$ is unchanged.
(d) The image is scaled 2×. Qualitatively, what happens to $\lambda_2/\lambda_1$ and why?

**B3 — RANSAC budget planning.**
You must estimate a homography in 20 ms. Each iteration (sample + 4-point DLT + residual over 1000 points) costs 15 µs.
(a) How many iterations fit in budget?
(b) With $p=0.99$, what is the *lowest* inlier ratio $w$ you can tolerate? Solve for $w$.
(c) Your measured $w$ is 0.25. Give three concrete interventions and the new $N$ for each.
(d) You switch to PROSAC. Explain in one sentence why the effective iteration count drops without changing $w$.

**B4 — Homography derivation.**
(a) Starting from $\tilde{\mathbf{x}}' \simeq H\tilde{\mathbf{x}}$, derive the two linear equations per correspondence via the cross product.
(b) Show why the third equation is linearly dependent on the first two.
(c) Explain why 3 collinear points make $A$ rank-deficient.
(d) Write the full estimation recipe you'd actually ship, from raw matches to final $H$.

**B5 — Full-system design.**
An AR app must anchor a virtual object to a printed poster on a wall, on a mid-range Android phone, at 30 FPS.
(a) Detector + descriptor choice, with justification tied to a compute budget.
(b) How do you get 6-DoF pose from a poster? (Hint: the poster is planar — what does that let you do?)
(c) How do you handle the phone being moved so the poster is 70% occluded?
(d) What breaks when the user turns off the room light and uses a phone torch?

---

## Part C — The traps (questions designed to catch memorisers)

| Trap question | The wrong-but-common answer | What you should say |
|---|---|---|
| "Is scale-space scale-invariant?" | "Yes, that's the point." | It's scale-**covariant**. Invariance comes from selecting extrema in $(x,y,\sigma)$ *and* normalising the descriptor patch by the detected $\sigma$. |
| "SIFT is rotation invariant, so it handles any camera orientation?" | "Yes." | In-plane rotation, yes. Out-of-plane (viewpoint) rotation only to ~30°; it's not affine- or projective-invariant. |
| "Just use FLANN, it's a fast kd-tree." | agreeing | FLANN is **approximate**. Exact kd-tree degenerates to linear scan above ~20-D. FLANN trades recall for speed. |
| "Raise the Canny threshold to remove texture." | "Sure." | Threshold filters by **contrast**, texture is a **frequency** problem. You need $\sigma$, not $T$. |
| "HSV is illumination invariant." | "Yes." | Decoupled from intensity *scaling* only. Coloured illuminants shift Hue; specular highlights collapse Saturation and make Hue undefined. |
| "More RANSAC iterations always help." | "Yes." | Not under degeneracy — a plane-dominated scene gives $F$ estimates with high inlier counts and no valid epipolar geometry, no matter how long you run. |
| "Bigger descriptor = better." | "Yes." | More spatial bins ⇒ more discriminative but less tolerant of localisation error and viewpoint warp. $4{\times}4{\times}8$ was empirically optimal. |
| "Blur then downsample, or downsample then blur — same thing." | "Same." | Blur **must** precede decimation or you alias. Order matters and it's Nyquist, not preference. |
| "We use cosine similarity because it's better than L2 for embeddings." | "Right." | For L2-normalised vectors they are monotonically equivalent — $\|a-b\|^2 = 2-2\cos\theta$. The choice only matters for unnormalised vectors. |
| "DLT gives you the homography." | "Yes." | DLT gives an **algebraic** solution that must be (a) Hartley-normalised and (b) refined by LM on geometric error. |

---

## Part D — Spaced-repetition cards

Copy into Anki. Front → Back.

```
Q: Semigroup property of Gaussians
A: G_σ1 * G_σ2 = G_√(σ1²+σ2²) — blurring twice with σ equals blurring once with σ√2. Enables incremental pyramid construction.

Q: Relation between ∂G/∂σ and the Laplacian
A: ∂G/∂σ = σ ∇²G  (from the heat equation ∂L/∂t = ½∇²L with t = σ²)

Q: DoG ≈ ?
A: G(kσ) − G(σ) ≈ (k−1) σ² ∇²G — the σ²-scale-normalised LoG. (k−1) is scale-constant so extrema locations are unaffected.

Q: Why σ² normalisation in scale selection?
A: Gaussian n-th derivative magnitudes decay as σ^(−n). Without γ-normalisation every extremum collapses to the finest scale.

Q: Blob radius from detected σ (2D)
A: R = √2 · σ  (from the LoG zero crossing at r = √2σ). In n-D: R = √n · σ.

Q: SIFT pyramid image counts for s scales/octave
A: s+3 Gaussian-blurred, s+2 DoG, s tested for extrema. k = 2^(1/s). Reuse the 2σ₀ image (downsampled 2×) as the next octave's base.

Q: Structure tensor M
A: M = Σ w(x,y) [[Ix², IxIy],[IxIy, Iy²]] — Gaussian-weighted covariance of gradients. Eigenvalues: both small=flat, one large=edge, both large=corner.

Q: Harris response
A: R = det(M) − κ·tr(M)², κ ∈ [0.04, 0.06]. Avoids eigendecomposition. Shi–Tomasi: R = min(λ1, λ2).

Q: Harris ↔ Lucas–Kanade
A: Same matrix M. LK solves Mv = −b; solvable iff M is well-conditioned, i.e. at corners. The aperture problem = M is rank-1 on an edge.

Q: Canny's four stages
A: (1) Gaussian/DoG smoothing (2) gradient magnitude + orientation (3) non-max suppression along ∇ direction → 1px wide (4) hysteresis thresholding using connectivity.

Q: Canny's three optimality criteria
A: Good detection (max SNR), good localisation (min displacement), single response per edge. Optimum ≈ first derivative of a Gaussian.

Q: SIFT descriptor construction
A: 16×16 patch at keypoint scale, rotated to canonical orientation → 4×4 subregions × 8 orientation bins (trilinear interpolation) = 128-D → L2 normalise → clamp at 0.2 → L2 normalise.

Q: Why clamp SIFT descriptor at 0.2?
A: Non-linear illumination (saturation, specularity) distorts gradient magnitudes far more than orientations. Capping limits any single gradient's influence.

Q: SIFT edge rejection test
A: tr(H)²/det(H) < (r+1)²/r with r = 10, on the 2×2 spatial Hessian. Same trace/determinant invariant trick as Harris.

Q: ORB = ?
A: Oriented FAST (FAST-9 corners, Harris-ranked, on a pyramid, orientation from intensity centroid) + rotated BRIEF (256 learned decorrelated binary tests, steered by θ). 32 bytes, Hamming distance.

Q: ORB orientation
A: θ = atan2(m01, m10) — direction from patch centre to intensity centroid. Fails on rotationally symmetric or low-contrast patches.

Q: Lowe's ratio test
A: Accept if d1/d2 < 0.8. Removes ~90% of false matches, loses ~5% of true ones. The 2nd-NN is a per-query estimate of local descriptor density.

Q: L2 vs cosine for normalised vectors
A: ||a−b||² = 2 − 2·cos θ. Monotonically equivalent → identical ranking.

Q: RANSAC iteration count
A: N = log(1−p) / log(1 − w^s). Homography (s=4, w=0.5, p=0.99) → 72. 8-pt F (s=8, w=0.5) → 1177.

Q: RANSAC inlier threshold, principled
A: t² = χ²_{m,α} σ². m=1 (point-to-line): 3.84σ². m=2 (point-to-point): 5.99σ².

Q: RANSAC degeneracy for F
A: A dominant plane. Any F = [e']× H fits all plane matches for any epipole → high inlier count, meaningless geometry. Detect by comparing H inlier count to F inlier count. Fix: DEGENSAC.

Q: MAGSAC++ contribution
A: Removes the inlier threshold — marginalises model quality over a range of noise scales (σ-consensus) with IRLS. cv2.USAC_MAGSAC.

Q: Homography DoF and solver
A: 8 DoF (3×3 up to scale). DLT: cross product → 2 eqs/correspondence → A h = 0 → h = right singular vector of smallest singular value. Needs 4 points, no 3 collinear.

Q: Hartley normalisation
A: Translate centroid to origin, scale so RMS distance = √2, per image. Without it κ(A) ~ 1e6–1e8 and the DLT is numerically useless.

Q: When is a homography exact?
A: (1) planar scene, or (2) pure camera rotation about the optical centre. Any translation + depth variation ⇒ parallax ⇒ no single H.

Q: Multi-band blending
A: Blend Laplacian-pyramid band k over a transition width ∝ 2^k. Low frequencies wide (hides exposure), high frequencies narrow (avoids ghosting).

Q: 2026 hybrid vision stack
A: Learned front-end (SuperPoint/DISK + LightGlue/LoFTR) → classical back-end (MAGSAC++ RANSAC → 5-pt/PnP → bundle adjustment). Even 3DGS is initialised from COLMAP.
```

---

## Part E — Self-assessment rubric

Rate yourself 1–5 per row. Anything ≤3 goes back into the notes before you move to Module 2.

| Skill | 1 (surface) | 3 (competent) | 5 (top 1%) |
|---|---|---|---|
| Scale-space | "SIFT is scale-invariant" | knows DoG≈LoG, builds a pyramid | derives ∂G/∂σ=σ∇²G, explains uniqueness/causality, knows the s+3 arithmetic and R=√2σ |
| Harris | knows the eigenvalue table | derives M from Taylor expansion | connects M to Lucas–Kanade trackability and the aperture problem unprompted |
| SIFT | lists the 5 stages | explains orientation + descriptor construction | justifies 4×4×8, the 0.2 clamp, the 80% duplicate rule, and names learned successors |
| Matching | knows the ratio test | explains why ratio beats threshold | knows the 90%/5% numbers, L2≡cosine proof, and when the ratio test *hurts* (repetitive structure) |
| RANSAC | knows the loop | derives N | derives N, chooses t from χ², names degeneracy + DEGENSAC + MAGSAC++, always refits on inliers |
| Homography | knows it's 3×3 | runs DLT | Hartley normalisation + why, validity conditions, algebraic vs geometric error, LM refinement |
| Judgement | "classical is old" | knows both exist | designs a correct hybrid pipeline for a stated compute/accuracy constraint |

---

## Part F — Module 1 exit criteria

You are done with Module 1 when you can, **without notes**:

- [ ] Derive DoG ≈ $(k-1)\sigma^2\nabla^2 G$ on a whiteboard in under 3 minutes.
- [ ] Derive the RANSAC iteration formula and compute two rows of the table from memory.
- [ ] Derive $M$ from the SSD Taylor expansion and state the LK connection.
- [ ] Write the DLT setup for a homography and explain Hartley normalisation.
- [ ] Have working code for: Canny from scratch, a DoG blob detector that reports correct radii, RANSAC + DLT homography, and a multi-band-blended panorama.
- [ ] Have run Lab A and be able to quote *your own numbers* for SIFT vs ORB.
