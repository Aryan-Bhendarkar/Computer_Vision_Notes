# Formula Sheet

Every equation worth having at hand, by module. **If you can reproduce this file from memory, you know the material.**

---

## Module 1 — Classical CV

**Gaussian and scale-space**
$$G_\sigma(x,y) = \tfrac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2} \qquad G_{\sigma_1}*G_{\sigma_2} = G_{\sqrt{\sigma_1^2+\sigma_2^2}}$$
$$\frac{\partial G}{\partial\sigma} = \sigma\nabla^2 G \quad\Longrightarrow\quad G(k\sigma) - G(\sigma) \approx (k-1)\,\sigma^2\nabla^2 G$$
Blob radius from detected scale (2-D): $R = \sqrt{2}\,\sigma$. Pyramid: $k = 2^{1/s}$, need $s+3$ blurred and $s+2$ DoG images per octave.

**Structure tensor and Harris**
$$M = \sum_{x,y} w(x,y)\begin{bmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{bmatrix} \qquad R = \det M - \kappa\,(\operatorname{tr}M)^2,\quad \kappa\in[0.04,0.06]$$
Shi–Tomasi: $R = \min(\lambda_1,\lambda_2)$. SIFT edge rejection: $\dfrac{\operatorname{tr}(H)^2}{\det(H)} < \dfrac{(r+1)^2}{r}$, $r=10$.

**Matching**
Lowe's ratio test: accept if $d_1/d_2 < 0.8$ (removes ~90% of false matches, loses ~5% of true).
For L2-normalised vectors: $\|a-b\|_2^2 = 2 - 2\cos\theta$.

**RANSAC**
$$N = \frac{\log(1-p)}{\log(1-w^s)} \qquad t^2 = \chi^2_{m,\alpha}\,\sigma^2 \ \ (3.84\sigma^2 \text{ for } m{=}1,\ 5.99\sigma^2 \text{ for } m{=}2)$$
At $p=0.99, w=0.5$: homography ($s{=}4$) 72 · 5-pt E 146 · 7-pt F 588 · 8-pt F 1177.

**Homography** — 8 DoF. DLT: $A\mathbf{h}=\mathbf{0}$, $\mathbf{h}$ = right singular vector of the smallest singular value. **Hartley-normalise first** (centroid at origin, RMS distance $\sqrt2$).

**Classical IP (Appendix)**
Otsu: maximise $\sigma_B^2(t) = \omega_0\omega_1(\mu_0-\mu_1)^2$. Hough: $\rho = x\cos\theta + y\sin\theta$.
NCC: mean-subtract and variance-normalise ⇒ invariant to $I \to aI+b$.

---

## Module 2 — Deep Learning for Vision

**Conv arithmetic**
$$H' = \left\lfloor\frac{H + 2p - d(k-1) - 1}{s}\right\rfloor + 1$$
$$\text{params} = k^2 C_{\text{in}}C_{\text{out}} + C_{\text{out}} \qquad \text{MACs} = k^2 C_{\text{in}}C_{\text{out}}H'W'$$

**Receptive field** — $j_l = j_{l-1}s_l$, $\;r_l = r_{l-1} + (k_l-1)j_{l-1}$, with $r_0=1,j_0=1$. **Effective** RF is Gaussian and grows as $O(\sqrt{L})$.

**BatchNorm**
$$\hat x = \frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\quad y = \gamma_c\hat x + \beta_c \qquad\text{(statistics over } N,H,W \text{ per channel)}$$
Scale invariance: $L(aW)=L(W) \Rightarrow \boxed{\nabla_{aW}L = \tfrac{1}{a}\nabla_W L}$, so effective LR $\propto 1/\|W\|^2$.
BN folding: $W_{\text{fold}} = \frac{\gamma W}{\sqrt{\sigma^2+\epsilon}}$, $b_{\text{fold}} = \beta - \frac{\gamma\mu}{\sqrt{\sigma^2+\epsilon}}$.

**ResNet** — $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_i)$, so
$$\frac{\partial\mathcal{L}}{\partial\mathbf{x}_l} = \frac{\partial\mathcal{L}}{\partial\mathbf{x}_L}\left(\mathbf{1} + \frac{\partial}{\partial\mathbf{x}_l}\sum \mathcal{F}\right)$$

**Depthwise-separable cost ratio** — $\dfrac{1}{C_{\text{out}}} + \dfrac{1}{k^2} \approx \dfrac19$ for $k=3$.
**EfficientNet** — $d=\alpha^\phi, w=\beta^\phi, r=\gamma^\phi$ with $\alpha\beta^2\gamma^2\approx2$.
**Softmax + CE gradient** — $\partial\mathcal{L}/\partial z_k = p_k - y_k$.
**Label smoothing** — $y^{LS}_k = (1-\varepsilon)y_k + \varepsilon/K$.
**Grad-CAM** — $\alpha_k^c = \frac1Z\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}$, $L = \mathrm{ReLU}(\sum_k\alpha_k^c A^k)$.

---

## Module 3 — Representations

**Contrastive loss** — $\mathcal{L} = Y D^2 + (1-Y)\max(0, m-D)^2$
**Triplet loss** — $\mathcal{L} = \bigl[\|f(a)-f(p)\|^2 - \|f(a)-f(n)\|^2 + \alpha\bigr]_+$, $\alpha \approx 0.2$
**ArcFace** — $-\log\dfrac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)} + \sum_{j\ne y}e^{s\cos\theta_j}}$, $s{=}64$, $m{=}0.5$
**InfoNCE** — $\mathcal{L}_{i,j} = -\log\dfrac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i,z_k)/\tau)}$
**MoCo momentum** — $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, $m = 0.999$
**GeM pooling** — $f_c = \bigl(\frac{1}{|\mathcal{X}|}\sum x^p\bigr)^{1/p}$, $p$ learnable ($\to$ 3)
**PQ** — $m$ sub-vectors × 256 centroids ⇒ $m$ bytes/vector. ADC: $\hat d^2 = \sum_j T[j][\text{code}_j(x)]$.
**HNSW layer assignment** — $\ell = \lfloor -\ln(U(0,1))\cdot m_L\rfloor$
**IVF tuning** — `nlist` $\approx \sqrt{N}$ to $4\sqrt{N}$; `nprobe`/`efSearch` are the query-time recall dials.

---

## Module 4 — Detection & Segmentation

**Box encoding** — $t_x = \frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$
**Smooth L1** — $0.5x^2$ if $|x|<1$, else $|x|-0.5$
**FPN merge** — $P_l = \mathrm{conv}_{3\times3}\bigl(\mathrm{conv}_{1\times1}(C_l) + \mathrm{up}_{2\times}(P_{l+1})\bigr)$
**FPN level assignment** — $k = \bigl\lfloor k_0 + \log_2(\sqrt{wh}/224)\bigr\rfloor$, $k_0=4$
**FCOS centre-ness** — $\sqrt{\dfrac{\min(l,r)}{\max(l,r)}\cdot\dfrac{\min(t,b)}{\max(t,b)}}$
**Focal loss** — $\mathrm{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$, $\gamma{=}2$, $\alpha{=}0.25$
**Prior bias init** — $b = -\log\bigl(\frac{1-\pi}{\pi}\bigr)$, $\pi = 0.01$ ⇒ $b \approx -4.595$
**IoU / GIoU** — $\mathrm{IoU} = \frac{|A\cap B|}{|A\cup B|}$; $\mathrm{GIoU} = \mathrm{IoU} - \frac{|C\setminus(A\cup B)|}{|C|}$
**DETR matching** — $\hat\sigma = \arg\min_\sigma\sum_i \bigl[-\hat p_{\sigma(i)}(c_i) + \mathcal{L}_{\text{box}}\bigr]$ (Hungarian, $O(N^3)$)
**Dice ↔ IoU** — $\mathrm{Dice} = \dfrac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$, $\ \mathrm{IoU} = \dfrac{\mathrm{Dice}}{2-\mathrm{Dice}}$
**Dice loss** — $1 - \dfrac{2\sum p_ig_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$
**Panoptic Quality** — $\mathrm{PQ} = \underbrace{\frac{\sum_{TP}\mathrm{IoU}}{|TP|}}_{SQ}\times\underbrace{\frac{|TP|}{|TP|+\frac12|FP|+\frac12|FN|}}_{RQ}$
**OKS** (pose) — $\sum_i \exp\bigl(-d_i^2/2s^2\kappa_i^2\bigr)\delta(v_i>0)\ /\ \sum_i\delta(v_i>0)$

---

## Module 5 — Geometry

**Projection** — $x = fX/Z$; $\ \lambda\tilde{\mathbf{x}} = K[R\mid\mathbf{t}]\tilde{\mathbf{X}}$; $\ P$ has 11 DoF
**Camera centre** — $\mathbf{C} = -R^\top\mathbf{t}$
**Intrinsics** — $K = \begin{bmatrix}f_x & s & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix}$
**Distortion** — $x_d = x(1+k_1r^2+k_2r^4+k_3r^6) + \text{tangential}$, in **normalised** coords, $r^2=x^2+y^2$
**Plane-induced homography** — $H = K'\bigl(R - \frac{\mathbf{t}\mathbf{n}^\top}{d}\bigr)K^{-1}$
**Epipolar** — $\hat{\mathbf{x}}'^\top E\,\hat{\mathbf{x}} = 0$, $\ E = [\mathbf{t}]_\times R$, $\ F = K'^{-\top}EK^{-1}$
DoF: $F$ 7, $E$ 5. Both rank 2. $E$ has two equal non-zero singular values. $F\mathbf{e}=\mathbf{0}$.
**Stereo** — $Z = \dfrac{fB}{d}$, $\qquad \delta Z \approx \dfrac{Z^2}{fB}\,\delta d$ **(quadratic in depth)**
**Optical flow** — $I_xu + I_yv + I_t = 0$; LK solves $M\mathbf{v} = -\mathbf{b}$ with the **same $M$** as Harris
**Bundle adjustment** — $\min\sum_{i,j} v_{ij}\,\rho\bigl(\|\pi(\mathbf{c}_j,\mathbf{X}_i) - \mathbf{x}_{ij}\|^2\bigr)$
**Schur complement** — from $\mathcal{H}\Delta = -g$ with $-g = [\mathbf{v};\mathbf{w}]$:
$$(B - EC^{-1}E^\top)\Delta\mathbf{c} = \mathbf{v} - EC^{-1}\mathbf{w}, \qquad \Delta\mathbf{X} = C^{-1}(\mathbf{w} - E^\top\Delta\mathbf{c})$$
Gauge freedom ⇒ $\mathcal{H}$ rank-deficient by exactly **7**.
**NeRF** — $C(\mathbf{r}) = \int T(t)\sigma(\mathbf{r}(t))\mathbf{c}\,dt$, $\ T(t) = \exp\bigl(-\int_{t_n}^t\sigma\,ds\bigr)$
**PointNet** — $f(\{x_i\}) \approx \gamma\bigl(\max_i h(x_i)\bigr)$

---

## Module 6 — Modern & Foundation Models

**Attention** — $\mathrm{softmax}\bigl(QK^\top/\sqrt{d_k}\bigr)V$, cost $O(N^2 d)$
**ViT patch embed** — `Conv2d(3, D, kernel_size=P, stride=P)`; $N = HW/P^2$ tokens
**CLIP** — symmetric InfoNCE over the $N\times N$ similarity matrix with a learned temperature
**VAE ELBO** — $\log p(x) \ge \mathbb{E}_q[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|p(z))$
**Reparameterisation** — $z = \mu + \sigma\odot\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$
**GAN** — $\min_G\max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$
**Diffusion forward** — $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, $\ \bar\alpha_t = \prod_{s\le t}(1-\beta_s)$
**Diffusion loss** — $\mathcal{L}_{\text{simple}} = \mathbb{E}\bigl\|\epsilon - \epsilon_\theta(x_t, t)\bigr\|^2$ (a **reweighted**, not exact, ELBO)
**Score identity** — $\nabla_{x_t}\log p(x_t) = -\frac{1}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)$
**$v$-prediction** — $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$
**CFG** — $\tilde\epsilon = \epsilon_\theta(x_t,\varnothing) + s\bigl[\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\varnothing)\bigr]$, $s\approx7$
**Latent diffusion** — $512^2\times3 \to 64^2\times4$, a 48× reduction in elements
**VQ-VAE** — $z_q = e_k$, $k=\arg\min_j\|z_e - e_j\|$; straight-through: `z_q = z_e + (z_q - z_e).detach()`
**Distillation** — $\alpha T^2\,\mathrm{KL}\bigl(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\bigr) + (1-\alpha)\mathcal{L}_{CE}$

---

## Supplement

**FGSM** — $x_{\text{adv}} = x + \varepsilon\,\mathrm{sign}\bigl(\nabla_x\mathcal{L}\bigr)$
**Energy OOD score** — $-T\log\sum_k e^{z_k/T}$
**Effective-number reweighting** — $w_c \propto \dfrac{1-\beta}{1-\beta^{n_c}}$
**Logit adjustment** — subtract $\tau\log\pi_c$ from the logits
**Quantisation** — $q = \mathrm{round}(r/S) + Z$; use **per-channel** scales for conv weights
**Training memory** — $\approx16$ bytes/parameter for fp32 + Adam, **plus activations** (usually dominant in vision)
**Linear LR scaling** — multiply LR by the batch-size factor, with warmup
