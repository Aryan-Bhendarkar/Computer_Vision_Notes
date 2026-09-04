# Notation

One page. Read it once; return when a symbol surprises you.

## Global conventions

| Symbol | Meaning |
|---|---|
| $I(x,y)$ | image intensity at pixel $(x,y)$ |
| $I_x, I_y, I_t$ | partial derivatives of $I$ (spatial, temporal) |
| $\nabla I$ | image gradient $(I_x, I_y)^\top$ |
| $\nabla^2$ | Laplacian $\partial^2/\partial x^2 + \partial^2/\partial y^2$ |
| $*$ | convolution · $\star$ correlation |
| $\sigma$ | Gaussian standard deviation, in pixels (**not** a singular value unless subscripted $\sigma_i$) |
| $G_\sigma$ | Gaussian kernel of scale $\sigma$ |
| $\tilde{\mathbf{x}}$ | homogeneous coordinates |
| $\hat{\mathbf{x}}$ | normalised (calibrated) image coordinates, $K^{-1}\tilde{\mathbf{x}}$ |
| $\simeq$ | equal **up to scale** (homogeneous equality) |
| $[\mathbf{t}]_\times$ | the $3\times3$ skew-symmetric matrix with $[\mathbf{t}]_\times\mathbf{v} = \mathbf{t}\times\mathbf{v}$ |
| $\|\cdot\|_2$, $\|\cdot\|_1$ | L2 / L1 norm · $[\cdot]_+ = \max(0,\cdot)$ |
| $\mathrm{sg}[\cdot]$ | stop-gradient |
| $\mathbb{1}[\cdot]$ | indicator function |

## ⚠️ Symbols that mean different things in different modules

**Read this table.** These are the collisions that cause genuine confusion.

| Symbol | Module 1 / 5 (geometry) | Module 2–4, 6 (deep learning) |
|---|---|---|
| $H$ | **homography** ($3\times3$, 8 DoF) | image **height**; also the BA Hessian in older texts — **this curriculum writes the BA Hessian $\mathcal{H}$** to avoid the clash (5.8) |
| $K$ | **camera intrinsics** matrix | number of **classes**; also codebook size in 6.15b; also $k$-NN's $k$ |
| $E$ | **essential matrix** (5.5) | the camera–point coupling block in BA (5.8); expectation $\mathbb{E}$ elsewhere |
| $F$ | **fundamental matrix** (5.5) | a generic learned function $\mathcal{F}$ in ResNet (2.6) |
| $\sigma$ | Gaussian scale (1.2, 1.4) | activation function; standard deviation in BN (2.4); singular value $\sigma_i$ |
| $t$ | **translation vector** $\mathbf{t}$ (5.2) | **timestep** in diffusion (6.13); time in optical flow (5.7); temperature in some texts (this curriculum uses $\tau$) |
| $s$ | RANSAC **minimal sample size** (1.8) | scale factor; guidance scale in CFG (6.14); a similarity score |
| $\alpha$ | triplet-loss margin (3.5) | $\alpha_t$ = diffusion noise schedule (6.13); focal-loss class weight (4.6); blending weight |
| $\lambda$ | **eigenvalue** (1.5); LM damping (5.8) | loss weight; also depth scale in 5.5 |
| $C$ | camera **centre** $\mathbf{C}$ (5.2) | **channel** count (2.2); the point block of the BA Hessian (5.8) |
| $D$ | disparity (5.6); DoG response (1.6) | embedding **dimension** (3.1); the discriminator (6.12) |
| $\gamma$ | gamma encoding (1.1) | focal-loss exponent (4.6); BN scale parameter (2.4); positional encoding $\gamma(\cdot)$ (5.10) |
| $\tau$ | RANSAC/ratio threshold (1.7) | **temperature** in InfoNCE (3.6) and distillation (S.14) |

**Rule of thumb:** if a symbol appears inside a geometry section (Modules 1.9, 5.x), read it geometrically. Inside a network section, read it as a network quantity. The three that trip people most often are $H$, $K$, and $t$.

## Module-specific

**Module 1** — $M$ structure tensor · $R$ Harris response · $w$ RANSAC inlier ratio · $p$ RANSAC success probability · $N$ RANSAC iterations · $k$ DoG scale ratio $2^{1/s}$
**Module 2** — $C_{\text{in}}, C_{\text{out}}$ channels · $k$ kernel size · $s$ stride · $p$ padding · $d$ dilation · $r_l, j_l$ receptive field and jump · $\gamma,\beta$ BN affine parameters
**Module 3** — $z$ embedding · $\tau$ temperature · $m$ PQ sub-vectors · $M$ HNSW connections · `nprobe`, `efSearch` search dials
**Module 4** — $p_t$ focal-loss target probability · $\gamma$ focal exponent · $\alpha_t$ focal class weight · $t_x,t_y,t_w,t_h$ box deltas · $N$ DETR query count · $\varnothing$ the "no object" class
**Module 5** — $K$ intrinsics · $R,\mathbf{t}$ extrinsics · $\mathbf{C}$ camera centre · $E,F$ essential/fundamental · $\mathbf{e},\mathbf{e}'$ epipoles · $B$ stereo baseline · $d$ disparity · $\mathcal{H}$ BA Hessian · $\mathbf{v},\mathbf{w}$ camera/point blocks of $-g$
**Module 6** — $P$ ViT patch size · $D$ embedding dim · $\beta_t,\alpha_t,\bar\alpha_t$ diffusion schedule · $\epsilon_\theta$ noise prediction network · $s$ CFG guidance scale · $c$ conditioning · $\varnothing$ null conditioning
