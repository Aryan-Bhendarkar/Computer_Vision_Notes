# Module 5 — Appendix: 3D Deep Learning, Depth, and Other Sensors

> **Why this appendix exists.** The main Module 5 teaches 3-D *geometry* — cameras, epipolar constraints, stereo, bundle adjustment — and Modules 2–6 teach *deep learning on images*. The bridge between them is missing entirely: **there is no deep learning on 3-D data anywhere in the curriculum.** A coverage audit called this the single largest structural hole, and it is, because it removes robotics, autonomous driving, AR, and industrial 3-D inspection from the set of roles the material prepares you for.
>
> This appendix closes it: how networks consume point clouds, how monocular depth actually works, how you get from a sparse SfM reconstruction to a dense surface, and what changes when the sensor isn't a colour camera.

---

## A5.1 Deep Learning on Point Clouds

### The problem: point clouds break every assumption CNNs rely on

A LiDAR sweep is a set of $(x, y, z, \text{intensity})$ tuples — typically 30k–120k of them per frame. Three properties make a CNN inapplicable:

1. **Unordered.** A point cloud is a *set*. Permute the rows and it is the same cloud, so the network's output must be **permutation-invariant**. A convolution over an array is not.
2. **Irregular.** Points are not on a grid, and density varies enormously with range — a LiDAR returns thousands of points on a nearby car and a dozen on the same car at 60 m.
3. **Sparse in the extreme.** Voxelise a $100\times100\times6$ m scene at 10 cm and >99% of cells are empty. Dense 3-D convolution wastes almost all its compute on air.

**Every architecture below is a different answer to those three facts.** That framing is the thing to lead with.

### PointNet (Qi et al., CVPR 2017) — consume the set directly

The insight is a theorem: **any permutation-invariant function on a set can be approximated as**

$$
f(\{x_1,\dots,x_n\}) \approx \gamma\Bigl(\operatorname*{MAX}_{i=1..n}\ \{\,h(x_i)\,\}\Bigr)
$$

— apply the *same* function $h$ to every point independently, then aggregate with a **symmetric function**. PointNet implements $h$ as a shared MLP (identical weights for every point, so ordering is irrelevant by construction) and the aggregator as **element-wise max-pooling**, giving a single global feature vector; $\gamma$ is another MLP producing the output.

Max-pooling is the load-bearing choice. Sum and mean are also symmetric, but max makes the global feature depend on a small set of points that "win" each dimension — the **critical point set**, which the paper shows is essentially the object's skeleton/outline. This gives PointNet notable robustness: delete 50% of the points at random and accuracy barely moves, because the critical points usually survive.

**T-Net** modules predict a small transformation matrix applied to the input points (and later to features) to canonicalise pose — a learned alignment, regularised toward orthogonality.

**PointNet's weakness is structural: there is no local neighbourhood anywhere.** Every point is processed in isolation and then everything is max-pooled globally, so the model has no notion of local geometry — the exact property that makes CNNs work on images.

### PointNet++ — put the hierarchy back

Recover locality by applying PointNet **recursively on local neighbourhoods**. Each *set abstraction* layer:

1. **Sample** — farthest-point sampling picks a well-spread subset of centroids (better coverage than random sampling, and worth naming).
2. **Group** — ball query gathers the points within radius $r$ of each centroid (preferred over k-NN because it gives a *fixed spatial scale*, which generalises better across varying density).
3. **PointNet** — apply a mini-PointNet to each group to produce a feature for that centroid.

Stack these and you have exactly the CNN hierarchy — growing receptive field, increasing abstraction, decreasing resolution — built on sets instead of grids. **Multi-scale grouping** (concatenating features from several radii) handles the non-uniform-density problem directly.

### The voxel and sparse-convolution line

Alternative answer: **impose a grid**, then use 3-D convolutions.

- **VoxelNet** — voxelise, encode the points in each voxel with a small PointNet, then run 3-D convolutions. Correct but slow.
- **Sparse convolution** (SECOND, MinkowskiNet) — the key efficiency idea. Compute **only at occupied voxels**, maintaining a hash map from coordinates to features. *Submanifold* sparse convolution additionally refuses to dilate the occupancy pattern (an output site is active only if the corresponding input site was), which keeps sparsity from decaying layer by layer — without that, a few layers of ordinary sparse conv fill the volume and you are back to dense. **Naming submanifold sparse convolution and why it exists is a strong, specific signal.**
- **PointPillars** — the pragmatic winner for driving. Voxelise only in $x$–$y$ into vertical **pillars** (no $z$ discretisation), encode each pillar with a PointNet, scatter the results into a 2-D pseudo-image, then run **an ordinary 2-D CNN**. This is the crucial move: it converts a 3-D problem into the 2-D problem the entire detection literature already solved (Module 4), so you inherit FPN, anchors, focal loss and every other advance for free. Fast enough for real-time and still a strong baseline.

### BEV — bird's-eye view as the fusion space

Modern driving perception represents the world in a **top-down bird's-eye-view grid** centred on the vehicle. Why it wins:

- Objects **do not overlap** in BEV (two cars occupy different ground area), unlike in a camera image where occlusion is constant.
- Object size is **scale-invariant** in BEV — a car is ~4.5 m long whether it is 5 m or 50 m away, which removes the multi-scale problem that dominates image-space detection.
- It is the natural space for **fusing sensors**: LiDAR is already 3-D, and camera features can be lifted into BEV (LSS "lift-splat-shoot" predicts a per-pixel depth distribution and splats features into the grid; **BEVFormer** and its successors instead use transformer queries that attend from BEV cells back into multi-camera image features).
- It is the space downstream planning wants anyway.

**Fusion taxonomy** worth being able to recite: **early** (raw sensor data concatenated — maximum information, hardest to align and least robust to a failed sensor), **mid/feature-level** (fuse in BEV — the current standard), **late** (fuse per-modality detections — most robust to sensor failure, loses cross-modal evidence).

### 3-D object detection

Output is a **7-DoF box**: $(x, y, z, l, w, h, \theta)$ — centre, dimensions, and yaw (roll and pitch are usually assumed zero for road vehicles). Two details that matter:

- **Yaw regression is periodic**, so regressing $\theta$ directly with L1 is wrong at the wrap-around. Standard fixes: regress $(\sin\theta, \cos\theta)$, or bin the angle coarsely and regress a residual within the bin.
- **IoU is 3-D (or BEV) IoU**, which is far more expensive to compute and much harsher than 2-D IoU — a small localisation error in depth destroys it. This is why 3-D detection AP numbers look low next to COCO's.

Benchmarks: KITTI (small, classic, saturated), **nuScenes** (multi-sensor, 360°, with the NDS metric that combines mAP with translation/scale/orientation/velocity/attribute errors), Waymo Open.

### 🎯 Top-1% distinction

1. **Lead with the three properties** (unordered, irregular, sparse) and present each architecture as an answer to them.
2. **State PointNet's theorem** — symmetric function of per-point embeddings — and why **max** specifically (critical point set, robustness to dropout).
3. **PointNet has no local structure; PointNet++ adds the hierarchy** via sample → group → PointNet.
4. **Submanifold sparse convolution exists to stop sparsity decaying** through layers.
5. **PointPillars' real contribution is reducing 3-D to 2-D** so the whole 2-D detection stack applies.
6. **BEV wins because objects don't overlap and scale is constant** — say those two, not "it's a top-down view."
7. **Yaw is periodic** — a small, checkable detail that shows you have implemented one.

### ✅ Mastery check

You must detect vehicles and pedestrians from a 64-beam LiDAR at 10 Hz on an automotive compute budget.

(a) PointNet++, sparse convolution, or PointPillars? Justify from the compute characteristics, not from benchmark numbers.
(b) Pedestrians at 50 m return ~15 points. What specifically fails, and what would you change?
(c) You add cameras. Where in the pipeline do you fuse, and what is the failure mode of your choice?

<details><summary>Answer sketch</summary>
(a) <b>PointPillars.</b> PointNet++'s ball query and farthest-point sampling are irregular, memory-bound gather operations that map badly to automotive accelerators and scale poorly to 100k points at 10 Hz. Sparse convolution is efficient but needs hash-map machinery and custom kernels that many embedded runtimes (TensorRT, and most NPUs) do not support well. PointPillars produces a <b>dense 2-D pseudo-image</b> after the pillar encoder, so everything downstream is standard 2-D convolution — fully supported, easily quantised, and directly deployable with the tooling you already have. That deployment argument, not the AP, is the real reason it dominates production.
(b) With ~15 points the object is at the edge of representability: the pillar encoder sees one or two occupied pillars, the BEV feature for that location is nearly empty, and after a stride-2 backbone the pedestrian occupies a sub-cell. It fails as a <b>resolution and evidence</b> problem, not a modelling one. Changes: (i) <b>finer pillar resolution</b> in the near-to-mid range, or a range-adaptive grid; (ii) <b>reduce the backbone stride</b> or attach the head to a higher-resolution feature level — the FPN argument from 4.4, applied in BEV; (iii) <b>fuse camera</b>, which has far more angular resolution at range and is the honest fix — a pedestrian at 50 m is dozens of camera pixels and 15 LiDAR points; (iv) <b>accumulate multiple sweeps</b> with ego-motion compensation, which multiplies point count at the cost of smearing moving objects. And say the safety-relevant part: at that range you should be reporting <i>uncertainty</i>, not a confident box.
(c) <b>Mid-level fusion in BEV</b> is the standard choice: lift camera features into the BEV grid (LSS-style depth splatting or BEVFormer-style attention) and concatenate with the LiDAR BEV features before the detection head. It keeps cross-modal evidence — camera supplies semantics and angular resolution, LiDAR supplies metric geometry — while avoiding early fusion's brittle raw-data alignment. <b>The failure mode is sensor dropout and miscalibration:</b> a single fused backbone trained on both modalities can degrade badly or unpredictably when one sensor fails, is occluded (mud on a lens), or drifts out of extrinsic calibration, because the network has learned to rely on both. Mitigations: train with <b>modality dropout</b> so the model is robust to a missing input, monitor extrinsics online, and keep a <b>late-fusion LiDAR-only path</b> as a safety fallback — which is why safety-critical stacks often run both.
</details>

---

## A5.2 Monocular Depth Estimation

### The task and its fundamental catch

Predict a depth map from a single image. Module 5.1 established this is **geometrically ill-posed** — projection destroys depth, and scaling the whole scene leaves the image unchanged. So a monocular depth network is not solving geometry; it is applying **learned priors** about object sizes, texture gradients, perspective, and scene layout.

**The direct consequence: monocular depth is scale-ambiguous.** A network can produce a depth map that is correct up to an unknown global scale (and often an unknown shift), which is why the field distinguishes:

- **Relative depth** — ordering and proportions are right, absolute metres are not. This is what most models predict.
- **Metric depth** — actual metres. Requires either metrically-supervised training on a fixed camera, or explicit conditioning on the camera intrinsics (the field of view tells you a great deal about scale), or a reference of known size.

### How the strong models are actually trained

**MiDaS** (Ranftl et al.) made the key methodological move: to train on many datasets with incompatible depth annotations (stereo disparity, LiDAR, SfM, synthetic), use a **scale-and-shift-invariant loss**. Align prediction and ground truth by solving for the optimal scale $s$ and shift $t$ in closed form (least squares), then compute the error on the aligned pair:

$$
\mathcal{L} = \rho\bigl(s\cdot d_{\text{pred}} + t - d_{\text{gt}}\bigr), \qquad (s,t) = \arg\min_{s,t}\ \|s\,d_{\text{pred}} + t - d_{\text{gt}}\|^2
$$

This makes the loss indifferent to exactly the ambiguity the task cannot resolve, which is what allows mixing datasets — and mixing datasets is what produced the generalisation. **The lesson generalises: when your task has an unresolvable ambiguity, build it into the loss rather than asking the model to guess.** (Compare: gauge freedom in bundle adjustment, 5.8.)

**Depth Anything (v1/v2)** scaled this with a large **pseudo-labelling** pipeline — train a teacher on labelled data, pseudo-label ~62M unlabelled images, train a student on the union with strong perturbations. The result is a robust relative-depth foundation model; metric variants are fine-tuned per domain.

**Self-supervised monocular depth** (Monodepth2 and successors) trains without any depth labels at all, using **view synthesis as the supervision**: predict depth and ego-motion, warp a neighbouring video frame into the current view, and minimise photometric error. Elegant, and it inherits every brightness-constancy failure from 5.7, plus a specific one — **static objects moving with the camera** (a car ahead at the same speed) are explained by infinite depth, which the "auto-masking" trick handles.

### Where it's used, and the trap

Depth Anything and its kin are now a default building block: 3DGS and NeRF initialisation, ControlNet depth conditioning (6.15), AR occlusion, VLM spatial reasoning, and image editing.

**The interview trap:** "we'll use monocular depth to measure the object." You cannot — not without resolving scale. A beautiful, sharp, confident relative-depth map measures nothing. Say that, then give the fixes: known camera intrinsics plus a ground-plane assumption, a reference object of known size, a second view, or an actual depth sensor.

---

## A5.3 From Sparse to Dense: MVS, ICP, and Surface Reconstruction

Module 5.8 stops at bundle adjustment, which yields **camera poses and a sparse point cloud**. Getting from there to a usable surface is the other half of the reconstruction pipeline.

### Multi-View Stereo

Given known poses, compute **dense** depth per view.

- **Plane-sweep stereo:** hypothesise a set of fronto-parallel depth planes, warp neighbouring views onto each plane, and score photometric consistency. The depth with the best consistency wins. This produces a **cost volume** — the same structure as learned stereo (5.6) and as MVSNet-style learned MVS.
- **PatchMatch MVS** (COLMAP's dense stage): initialise depth and normal randomly per pixel, then alternate **propagation** (neighbours' hypotheses are probably good here too — spatial coherence) with **random refinement**. Converges remarkably fast and is the practical workhorse.
- **Photometric consistency needs texture**, so MVS fails on the same surfaces stereo does: blank walls, glass, water, specular metal.

### Fusion into a surface

- **Depth-map fusion / TSDF:** accumulate depth maps into a volumetric **truncated signed distance field** — each voxel stores the signed distance to the nearest surface, truncated near zero. Averaging TSDFs across views is a principled, noise-suppressing fusion (this is KinectFusion's core), and the surface is the zero level-set.
- **Marching cubes** extracts a triangle mesh from that level-set.
- **Poisson surface reconstruction** takes oriented points (positions + normals) and solves for an indicator function whose gradient matches the normals — producing a **watertight** mesh, which is what you need for 3-D printing or simulation.

### ICP — aligning two point clouds

The standard registration algorithm. Alternate:
1. **Correspondence:** for each point in $P$, find the closest point in $Q$ (accelerated with a k-d tree).
2. **Alignment:** solve for the rigid $(R, \mathbf{t})$ minimising the distance over those correspondences — in closed form via SVD (the orthogonal Procrustes problem, and the same "SVD gives the constrained optimum" pattern as 1.9 and 5.5).
3. Repeat until convergence.

**Point-to-plane ICP** minimises the distance along the *target's surface normal* rather than point-to-point, which lets surfaces slide over each other and **converges far faster** — usually the right default.

**ICP's real weakness is that it is local.** It converges to the nearest local minimum, so it needs a decent initialisation (from an IMU, odometry, or a global registration step such as FPFH features + RANSAC). Also known: it needs sufficient geometric constraint — two flat planes can slide freely and ICP will not object.

---

## A5.4 When the Sensor Isn't a Colour Camera

| Sensor | What it gives | Failure surface |
|---|---|---|
| **RGB-D** (structured light / ToF) | per-pixel metric depth at 30 Hz | structured light fails in sunlight and on dark/specular/transparent surfaces; ToF suffers multipath on corners and flying pixels at depth edges; both are short-range |
| **LiDAR** | accurate metric 3-D, long range, illumination-independent | sparse at range, no colour/texture, expensive, degraded by rain, fog, spray and retroreflectors |
| **Radar** | range **and radial velocity** directly (Doppler), works through weather | very low angular resolution; multipath and clutter |
| **Thermal (LWIR)** | emitted heat — sees people and engines in total darkness | low resolution, no colour, poor texture, needs periodic shutter calibration |
| **Multispectral / hyperspectral** | reflectance in many bands — material identification | expensive, low spatial resolution, huge data volume |
| **Event camera** | asynchronous **per-pixel brightness-change events**, microsecond latency, >120 dB dynamic range | no absolute intensity, no output from static scenes, and **standard CNNs do not apply** — you need voxel-grid/time-surface representations or spiking networks |

**Two things worth saying about event cameras specifically**, because they are the most conceptually different: they output a stream of $(x, y, t, \text{polarity})$ tuples rather than frames, so the natural representations are time surfaces or event voxel grids; and their advantage — no motion blur, microsecond latency, extreme dynamic range — makes them genuinely superior for high-speed robotics and driving out of tunnels, which is why they keep appearing despite the tooling cost.

**The unifying practitioner point:** every one of these is a different sampling of the same physical scene, and the engineering question is always the same — *what does this sensor measure that the others don't, and what does it fail at?* Fusion exists because the failure surfaces are largely **uncorrelated**: LiDAR fails in fog where radar works; camera fails in darkness where thermal works; radar has no angular resolution where camera has plenty. **That uncorrelated-failure argument is the real justification for multi-sensor systems**, and it is a better answer than "more data is better."

### ✅ Mastery check

(a) A warehouse robot must detect pallets in an aisle with no windows and harsh overhead lighting. Choose a sensor suite and justify it by failure surfaces, not capabilities.
(b) Your RGB-D camera returns holes on the black rubber wheels of every trolley. Explain the physics and give two fixes.
(c) Why can't you run a standard CNN on an event camera's output?

<details><summary>Answer sketch</summary>
(a) <b>Camera + short-range LiDAR or RGB-D, and skip thermal and radar.</b> Reasoning by failure surface: lighting is controlled and constant indoors, so a camera's main weakness (illumination variation) is absent and its strength (angular resolution, texture, reading labels and barcodes) is fully available. LiDAR/RGB-D adds metric geometry for pallet-pocket localisation and forklift approach, and its own weaknesses — sunlight interference, rain, fog — do not exist indoors. Thermal adds nothing (pallets are ambient temperature and there is no darkness problem); radar's angular resolution is far too coarse for pocket-level precision. The general move: <b>enumerate each sensor's failure modes, then ask which of them your environment actually triggers.</b> Indoors eliminates most of the reasons the automotive stack is expensive.
(b) Black rubber has very low <b>reflectance in the near-infrared</b> that both structured-light and ToF sensors emit and measure, so almost no light returns and the sensor reports no valid depth — the same reason those sensors fail on dark clothing, hair, and glass (which transmits) and polished metal (which reflects specularly away from the sensor). Fixes: (i) <b>fill from a different modality</b> — stereo from the RGB pair, or a learned depth-completion network conditioned on the RGB image, which is the standard production answer; (ii) <b>change the sensing geometry</b> — shorter range, or a sensor with higher illumination power, or a wavelength with better return; (iii) <b>fill from context</b> — fit the known geometry (a trolley wheel is a cylinder of known radius) rather than trusting per-pixel returns; (iv) simply <b>mark the holes as unknown</b> and make downstream planning treat unknown as occupied, which is the safe engineering answer.
(c) Because a CNN expects a <b>dense, synchronous array</b> of intensity values on a fixed grid at a fixed time, and an event camera produces none of those things — it emits a <b>sparse, asynchronous stream</b> of $(x, y, t, \text{polarity})$ tuples with no absolute intensity and no frames at all. There is nothing to convolve. The standard solutions build an intermediate representation: accumulate events over a window into an <b>event voxel grid</b> or a <b>time surface</b> (per-pixel timestamp of the most recent event) and convolve that — which works but discards temporal precision, the sensor's main advantage; or use <b>graph/point-based networks</b> treating events as a spatio-temporal point cloud (A5.1 applies directly); or <b>spiking neural networks</b>, which match the sensor's event-driven nature and its power profile but lack mature training methods and hardware. The honest summary: the representation question is unresolved, and it is why event cameras remain a research modality despite compelling physics.
</details>

---

## Appendix — Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Why not CNNs on point clouds | **unordered** (needs permutation invariance), **irregular** (no grid), **extremely sparse** (>99% empty) |
| PointNet | $f = \gamma(\max_i h(x_i))$ — shared per-point MLP + a **symmetric** aggregator; max gives the **critical point set** and robustness to dropout |
| PointNet's flaw | **no local neighbourhoods** — every point processed alone, then pooled globally |
| PointNet++ | **sample** (farthest-point) → **group** (ball query, fixed spatial scale) → **PointNet** per group; multi-scale grouping handles density variation |
| Sparse conv | compute only at occupied voxels; **submanifold** variant refuses to dilate occupancy, or sparsity decays away in a few layers |
| PointPillars | pillars in $x$–$y$ only → PointNet per pillar → scatter to a 2-D **pseudo-image** → ordinary 2-D CNN. **Reduces 3-D to the solved 2-D problem** |
| BEV | objects **don't overlap** and object scale is **constant with range**; the natural fusion and planning space |
| Fusion | early (raw) / **mid, in BEV (standard)** / late (per-modality detections, most robust to sensor failure) |
| 3-D box | 7 DoF $(x,y,z,l,w,h,\theta)$; **yaw is periodic** ⇒ regress $(\sin,\cos)$ or bin+residual; 3-D IoU is harsh |
| Monocular depth | **not geometry — learned priors**; scale-ambiguous by construction; relative vs metric is the distinction that matters |
| MiDaS | **scale-and-shift-invariant loss** (solve $s,t$ in closed form, then compare) is what allows mixing incompatible datasets |
| Depth Anything | massive **pseudo-labelling** on unlabelled images; relative depth foundation model |
| Self-supervised depth | view synthesis + photometric loss; inherits brightness-constancy failures; static-relative-motion objects appear infinitely far |
| MVS | plane sweep or **PatchMatch** (propagate + random refine); builds a **cost volume**; fails on textureless/specular/transparent |
| Fusion to surface | **TSDF** volumetric averaging → **marching cubes**; **Poisson** for watertight meshes from oriented points |
| ICP | alternate closest-point correspondence with a closed-form SVD alignment; **point-to-plane converges much faster**; **local** — needs initialisation |
| Sensors | fusion works because failure surfaces are **uncorrelated** — not because "more data is better" |
| Event cameras | asynchronous $(x,y,t,p)$ events, µs latency, >120 dB range, **no absolute intensity**; needs voxel-grid/time-surface/graph representations |

### 🔨 Appendix build tasks

**1 — PointNet from scratch (one sitting, GPU optional).** Implement PointNet classification on **ModelNet40** (12k CAD models, small enough for a laptop GPU; `torch_geometric` provides a loader). Then run the experiment that proves the theorem: **randomly permute the input points at test time** and confirm the output is unchanged to numerical precision, then **randomly drop 25 / 50 / 75% of points** and plot accuracy — you should see graceful degradation, and you can visualise the critical point set by tracking which points win the max.

**2 — ICP (one sitting, no GPU).** Implement point-to-point and point-to-plane ICP from scratch, register two partial scans of the same object (Open3D ships suitable data), and plot convergence — iterations to converge and final RMSE — for both variants from several initial offsets. You will see point-to-plane converge in a fraction of the iterations, and you will see both fail beyond some initial misalignment, which is the local-minimum lesson.

**3 — Sparse-to-dense (half a day).** Take your COLMAP reconstruction from 5.11, run its dense MVS stage, fuse to a mesh with Poisson reconstruction in Open3D, and compare against a 3DGS reconstruction of the same images. Compare on: surface quality, watertightness, novel-view rendering quality, and wall-clock. That single comparison answers "NeRF/3DGS or classical?" with your own evidence.

**Read:** Qi et al., "PointNet" (CVPR 2017) — read §4.3, the theoretical analysis; it is short and genuinely illuminating. Lang et al., "PointPillars" (CVPR 2019). Ranftl et al., "Towards Robust Monocular Depth Estimation" (MiDaS, PAMI 2020) §3 for the scale-and-shift-invariant loss. Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View Stereo" (COLMAP MVS, ECCV 2016). Rusinkiewicz & Levoy, "Efficient Variants of the ICP Algorithm" (2001) — the definitive comparison.
