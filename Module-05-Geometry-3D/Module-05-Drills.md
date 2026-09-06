# Module 5 — Drills, Interview Bank & Spaced Repetition

---

## Part A — Rapid-fire

1. Why is depth unrecoverable from a single image? State it algebraically.
2. What is a vanishing point in terms of $K$ and a 3-D direction?
3. Write $P = K[R|\mathbf{t}]$ and count the degrees of freedom.
4. Where is the camera centre in world coordinates?
5. What is $f_x$ measured in, and what does it depend on besides the lens?
6. Which rotation parameterisation for bundle adjustment, and why? Which for a neural network, and why?
7. In what coordinate frame is lens distortion applied?
8. Why does adding $k_3$ sometimes make calibration worse?
9. Why do fisheye lenses need a different model, not just more radial terms?
10. How many constraints does one checkerboard view give on $K$? Why?
11. Why do you need ≥3 views for Zhang's method?
12. What happens to $K$ when you crop an image? When you resize it?
13. Write the plane-induced homography. What happens when $\mathbf{t} = 0$?
14. Define the epipole, epipolar line, epipolar plane, baseline.
15. Derive $E = [\mathbf{t}]_\times R$ from coplanarity.
16. Why is $E$ rank 2?
17. Why does $E$ have two equal non-zero singular values?
18. $F$ has 7 DoF, $E$ has 5. Account for both.
19. Why does the DoF difference justify the 5-point algorithm? (Give numbers.)
20. What two post-processing steps must follow a linear $F$ estimate?
21. What is the visual symptom of skipping the rank-2 projection?
22. What is the title and content of Hartley's 1997 paper, and why does it matter?
23. How many pose solutions come from decomposing $E$? How do you pick?
24. Why is monocular reconstruction scale-ambiguous? Name four ways real systems fix it.
25. What is the planar degeneracy for $F$, and how do you detect it?
26. Derive $Z = fB/d$.
27. Derive the depth-error formula and state its consequence.
28. What is the baseline trade-off?
29. What does SGM's $P_1$/$P_2$ split accomplish?
30. Why NCC or Census rather than SSD?
31. What does a left–right consistency check detect?
32. Name four situations where stereo fails, and the standard industrial fix for one of them.
33. Write the optical flow constraint equation. Why is it under-determined?
34. State the aperture problem in terms of the structure tensor.
35. Why must Lucas–Kanade be run coarse-to-fine?
36. What classical structures do PWC-Net and RAFT encode?
37. What is SORT? What does ByteTrack add?
38. What does HOTA decompose into, and why does that matter?
39. Write the bundle-adjustment objective. Why is it the MLE?
40. Why is a robust loss mandatory in BA?
41. Describe the Hessian's sparsity structure. Why is $C$ easy to invert?
42. Write the Schur complement step and say what it accomplishes.
43. Why is the BA Hessian $\mathcal{H}$ singular, and by exactly how much?
44. Does BA initialise or refine? What follows from that?
45. Why does the choice of the initial image pair matter in incremental SfM?
46. What is PnP and where does it appear in the pipeline?
47. What distinguishes SLAM from visual odometry?
48. How is loop closure related to Module 3?
49. What is pose-graph optimisation, in one sentence, relative to BA?
50. Why is every production AR system visual-inertial?
51. Direct vs indirect SLAM — trade-off in one sentence each.
52. Write NeRF's volume rendering integral. What is $T(t)$?
53. Why does NeRF need positional encoding?
54. Why is 3DGS fast? What is adaptive density control?
55. What do NeRF and 3DGS both depend on, and why is that ironic?

---

## Part B — Whiteboard problems

**B1 — Intrinsics.** A 1/2.3" sensor (6.17×4.55 mm) at 4032×3024 px with a 4.3 mm lens. Give $K$. Then the image is resized to 1008×756 and centre-cropped to 800×600. Give the final $K$.

**B2 — Epipolar derivation.** Derive $E = [\mathbf{t}]_\times R$ from scratch. Then derive $F$ from $E$. Then show $F\mathbf{e} = 0$ and interpret.

**B3 — Rank-2 projection.** Given a $3\times3$ matrix with singular values $(4.2, 1.7, 0.31)$, write the rank-2 projection and explain why it is optimal in Frobenius norm.

**B4 — Stereo design.** Design a stereo rig for a warehouse robot needing ±2 cm accuracy from 0.5 m to 8 m, on a chassis at most 60 cm wide, with a 90° horizontal field of view.
(a) Pick $f$ (px) and $B$; show the accuracy at both range extremes.
(b) What is the minimum disparity you must be able to measure, and what does that imply about the search range?
(c) The nearest 0.5 m has huge disparity — what practical problem does that create?

**B5 — Bundle adjustment scaling.** 2000 cameras, 500,000 points.
(a) Unknowns; dense Hessian memory in float64.
(b) Reduced camera system size after Schur.
(c) $S$ is itself sparse — under what condition are two cameras coupled? Estimate the fill for a sequential video capture vs. an unordered photo collection.
(d) Which solver would you choose for each case and why?

**B6 — Failure diagnosis.** A monocular SLAM system on a warehouse robot works in the aisles and loses tracking in the open loading bay. Give four candidate causes with a distinguishing test for each.

---

## Part C — Traps

| Trap | Common wrong answer | Correct answer |
|---|---|---|
| "$\mathbf{t}$ is the camera position." | "Yes." | $\mathbf{C} = -R^\top\mathbf{t}$. $\mathbf{t}$ is the translation of the *world-to-camera* transform. |
| "More distortion coefficients = better calibration." | "Yes." | $k_3$ overfits on normal lenses. Validate on held-out images, not on the fitting RMS. |
| "Cropping changes the focal length." | "Yes." | Cropping changes only $c_x, c_y$. **Resizing** scales all four of $f_x,f_y,c_x,c_y$. |
| "The 8-point algorithm is noise-sensitive." | "Yes, that's why we use 7-point." | It is — **without normalisation**. Hartley's whole paper is that normalising fixes it. |
| "You can get metric depth from monocular SfM." | "With a good network, yes." | $E$ is defined up to scale, so $\mathbf{t}$ is a direction only. Scale needs a stereo baseline, an IMU, a known size, or odometry. Learned monocular depth is a *prior*, not geometry, and is itself scale-ambiguous unless metrically supervised. |
| "More RANSAC iterations will fix my fundamental matrix." | "Yes." | Not under planar degeneracy — you get a high inlier count and meaningless geometry no matter how long you run. Compare $H$ and $F$ inlier counts; use DEGENSAC. |
| "Bigger baseline is always better for stereo." | "Yes." | Better depth precision, worse matching (viewpoint change, occlusion, smaller overlap). It's a trade. |
| "Depth error grows linearly with distance." | "Yes." | **Quadratically**: $\delta Z \approx Z^2\delta d/(fB)$. |
| "Optical flow gives you the motion of every pixel." | "Yes." | Only the component along the gradient is observable per pixel (aperture problem). Dense flow relies on a smoothness prior to fill in the rest. |
| "Bundle adjustment finds the global optimum." | "It's least squares, so yes." | It's **non-convex**. LM finds a local minimum. BA refines a good initialisation; it doesn't rescue a bad one. |
| "BA is intractable for large scenes." | "Yes, that's why we use pose graphs." | The Schur complement makes it very tractable — the points are eliminated almost for free because $C$ is block-diagonal. |
| "SLAM is just real-time SfM." | "Yes." | Loop closure, relocalisation, map management, and causality are structurally different problems. And VO ≠ SLAM: no loop closure means unbounded drift. |
| "3D Gaussian Splatting replaced classical 3D reconstruction." | "Yes." | It is initialised from **COLMAP** poses and its sparse point cloud — i.e. from SIFT + RANSAC + bundle adjustment. |
| "NeRF/3DGS give you accurate geometry." | "Yes." | They optimise **novel-view rendering quality**. Extracted geometry can have floaters and wrong surfaces in unobserved regions. Different objective, different failure modes. |

---

## Part D — Spaced-repetition cards

```
Q: Why is depth lost in a single view?
A: Projection is defined up to scale: (X,Y,Z) and (λX,λY,λZ) map to the same (x,y). Homogeneous coordinates encode this as "equality up to scale".

Q: Camera matrix and DoF
A: P = K[R|t], 3×4, 11 DoF = 5 intrinsic (fx, fy, cx, cy, s) + 6 extrinsic.

Q: Camera centre
A: C = −Rᵀt. (From X_cam = R·X_world + t and X_cam = 0.)

Q: Rotation parameterisation
A: Optimisation (BA): axis–angle / Lie algebra so(3) — minimal and unconstrained, step via exp map. Network regression: 6-D continuous representation (Zhou 2019) — quaternions/Euler are DISCONTINUOUS on SO(3).

Q: Distortion model
A: x_d = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + tangential, in NORMALISED coords BEFORE multiplying by K. r² = x²+y². k1<0 barrel, k1>0 pincushion.

Q: Zhang's calibration
A: Board plane → image is a homography per view. Orthonormality of r1, r2 gives 2 constraints on B = K⁻ᵀK⁻¹ per view; ≥3 views for 5 DoF; solve linearly, Cholesky for K, then LM on total reprojection error.

Q: Crop vs resize effect on K
A: Crop: only cx, cy shift. Resize by s: fx, fy, cx, cy all scale by s.

Q: Plane-induced homography
A: H = K'(R − t·nᵀ/d)K⁻¹. If t = 0 (pure rotation) H = K'RK⁻¹, independent of the scene — why panoramas work. Different planes ⇒ different H ⇒ parallax breaks single-H stitching.

Q: Essential matrix derivation
A: X', t, RX are coplanar (all in the epipolar plane) ⇒ triple product X'ᵀ(t × RX) = 0 ⇒ x̂'ᵀ[t]ₓR x̂ = 0 ⇒ E = [t]ₓR.

Q: F from E
A: x̂ = K⁻¹x̃ ⇒ F = K'⁻ᵀ E K⁻¹ = K'⁻ᵀ[t]ₓR K⁻¹.

Q: F vs E properties
A: Both 3×3 rank 2 (from [t]ₓ). F: pixel coords, 7 DoF (9 − scale − det=0), needs nothing. E: normalised coords, 5 DoF (3 R + 3 t − scale), two EQUAL non-zero singular values, needs known K.

Q: Why the 5-point algorithm
A: RANSAC N = log(1−p)/log(1−w^s). At w=0.5, p=0.99: s=5 ⇒ 145 iterations; s=8 ⇒ 1177. Exponential in s.

Q: 8-point algorithm, correctly
A: (1) HARTLEY NORMALISE both point sets (centroid at origin, RMS distance √2). (2) Build A, solve Af=0 by SVD (smallest right singular vector). (3) PROJECT TO RANK 2: SVD, set σ3=0, reconstitute (Eckart–Young optimal). (4) Denormalise.

Q: Symptom of skipping rank-2 projection
A: The epipolar lines do not all pass through a single epipole.

Q: Pose from E
A: E = U diag(1,1,0) Vᵀ, W = [[0,−1,0],[1,0,0],[0,0,1]]. Four solutions: R ∈ {UWVᵀ, UWᵀVᵀ}, t = ±u₃. Enforce det R = +1; disambiguate by CHEIRALITY (positive depth in both cameras). t is direction only — SCALE IS UNOBSERVABLE.

Q: Fixing monocular scale
A: Stereo baseline · IMU (visual-inertial) · known object size · wheel/GPS odometry.

Q: Stereo depth
A: Z = fB/d (from similar triangles: x_L = fX/Z, x_R = f(X−B)/Z).

Q: Stereo depth error
A: δZ ≈ (Z²/fB)·δd — QUADRATIC in depth. Doubling distance quadruples the error. Stereo is a short-to-medium-range sensor by physics.

Q: SGM
A: E(D) = Σ C(p,D_p) + Σ[P1·1(|ΔD|=1) + P2·1(|ΔD|>1)]. Small P1 allows slanted surfaces, large P2 preserves depth discontinuities. 2-D optimisation is NP-hard ⇒ aggregate exact 1-D DP along 8–16 directions.

Q: Optical flow constraint
A: I_x u + I_y v + I_t = 0 (from brightness constancy + first-order Taylor). One equation, two unknowns ⇒ aperture problem.

Q: Lucas–Kanade
A: Assume constant flow in a window ⇒ M v = −b with M = the HARRIS STRUCTURE TENSOR. Solvable iff M is well-conditioned ⇒ at corners. Aperture problem = M is rank-1 on an edge. Run coarse-to-fine on a pyramid because the linearisation assumes sub-pixel motion.

Q: MOT essentials
A: SORT = Kalman (constant velocity) + Hungarian on IoU. DeepSORT adds an appearance embedding. ByteTrack also associates LOW-confidence detections (usually occluded objects, not noise). HOTA = detection accuracy × association accuracy.

Q: Bundle adjustment objective
A: min over cameras and points of Σ_ij v_ij · ρ( ||π(c_j, X_i) − x_ij||² ), ρ = Huber/Cauchy. It is the MLE under independent Gaussian image noise.

Q: BA Hessian structure
A: 𝓗 = [[B, E],[Eᵀ, C]]  (written 𝓗, NOT H — H is the homography) — B block-diagonal 9×9 per camera, C block-diagonal 3×3 per point, E sparse coupling. Sparsity comes from each residual depending on exactly one camera and one point.

Q: Schur complement in BA
A: From 𝓗Δ = −g with −g split into camera and point blocks [v; w]: (B − E C⁻¹ Eᵀ)Δc = v − E C⁻¹ w, then ΔX = C⁻¹(w − EᵀΔc). C⁻¹ is trivial (3×3 blocks), so the points are marginalised almost free, leaving a 9·N_cam system. This is marginalising latent variables in a graphical model.

Q: BA gauge freedom
A: The cost is invariant to a global similarity transform ⇒ 𝓗 is rank-deficient by exactly 7 (3 translation + 3 rotation + 1 scale). Fix a camera and a distance, or rely on LM damping.

Q: Incremental SfM pipeline
A: features → matching → RANSAC geometric verification (scene graph) → initial pair (many inliers, WIDE baseline, non-planar) → E → R,t → triangulate → loop {next image by 2D-3D count, PnP+RANSAC, triangulate, LOCAL BA, filter} → periodic GLOBAL BA.

Q: SLAM vs VO vs SfM
A: SfM = offline, unordered, global BA. VO = sequential, real-time, no loop closure ⇒ unbounded drift. SLAM = VO + map + LOOP CLOSURE + relocalisation. Place recognition = image retrieval (DBoW2 ≈ BoVW, NetVLAD).

Q: Pose-graph optimisation
A: Bundle adjustment with the 3-D points marginalised out: nodes = keyframe poses, edges = relative-pose constraints. Much smaller; fixes gross drift after loop closure, then full BA polishes.

Q: Why VIO
A: An IMU supplies metric scale (resolving E's scale ambiguity), gravity direction, and high-rate motion that survives blur and textureless scenes. IMU bias must be jointly estimated, not just integrated.

Q: NeRF
A: MLP F_Θ(x, d) → (c, σ). C(r) = ∫ T(t)σ(r(t))c(r(t),d)dt with T(t) = exp(−∫σ ds) = transmittance. Trained on photometric loss against input images only. Positional encoding is essential — defeats the MLP's spectral bias.

Q: 3D Gaussian Splatting
A: Explicit millions of 3-D Gaussians (μ, Σ as scale+quaternion, α, SH colour), rendered by projecting to 2-D and alpha-compositing per tile — RASTERISATION, not ray marching, hence >100 FPS. Adaptive density control clones/splits/prunes Gaussians during optimisation.

Q: The Module 5 closing point
A: Both NeRF and 3DGS are initialised from COLMAP poses (and 3DGS from its sparse point cloud) — i.e. from SIFT + RANSAC + bundle adjustment. The most modern 3-D methods run on classical geometry.
```

---

## Part E — Self-assessment rubric

| Skill | 1 | 3 | 5 (top 1%) |
|---|---|---|---|
| Projection | knows $x=fX/Z$ | writes $P=K[R\|t]$ | states the scale ambiguity algebraically, knows $C=-R^\top t$, counts DoF, handles crop vs resize |
| Calibration | "wave a checkerboard" | runs `calibrateCamera` | explains the orthonormality constraints, knows distortion order and overfitting, reads residual maps |
| Epipolar | "there's a line" | writes $x'^\top Fx = 0$ | **derives $E=[t]_\times R$ live**, explains rank 2 and the DoF accounting, knows normalisation + rank-2 projection and their distinct symptoms |
| Pose recovery | knows SVD is involved | knows 4 solutions | cheirality, $\det R=+1$, and the scale ambiguity with its four practical fixes |
| Stereo | knows $Z=fB/d$ | derives it | derives $\delta Z \approx Z^2\delta d/(fB)$ and reasons about rig design from it; explains SGM's $P_1/P_2$ |
| Flow | knows LK exists | writes the constraint | states the aperture problem via the structure tensor, links to Harris, explains coarse-to-fine |
| **BA** | "it optimises everything" | writes the objective | explains sparsity + Schur complement + gauge freedom, knows it refines rather than initialises, frames it as marginalisation for an ML audience |
| SLAM | "real-time SfM" | knows front/back end | loop closure as the defining feature, place recognition = retrieval, pose-graph as marginalised BA, VIO for scale |
| Modern 3D | "NeRF makes 3D" | knows NeRF vs 3DGS | volume rendering integral, positional encoding, rasterisation-vs-raymarching, and **both depend on COLMAP** |

---

## Part F — Module 5 exit criteria

- [ ] Derive $E = [\mathbf{t}]_\times R$ on a whiteboard in under three minutes.
- [ ] State $F$ 7 DoF / $E$ 5 DoF with the full accounting, and the RANSAC consequence.
- [ ] Derive $Z = fB/d$ and $\delta Z \approx Z^2\delta d/(fB)$, and design a rig from a stated accuracy requirement.
- [ ] Explain BA's sparsity and the Schur complement well enough to size the reduced system for arbitrary $N_{\text{cam}}$, $N_{\text{pts}}$.
- [ ] State the gauge-freedom rank deficiency as exactly 7 and explain it.
- [ ] Have implemented: normalised 8-point $F$ with rank-2 projection and drawn epipolar lines; $E$ decomposition with cheirality; a small sparse BA.
- [ ] Have calibrated a real camera, measured the distortion-model overfitting curve, and confirmed the quadratic stereo depth-error law against a tape measure.
- [ ] Have run COLMAP successfully, and have *deliberately broken it* two ways (short baseline, planar scene).
