# Module 2 — Drills, Interview Bank & Spaced Repetition

---

## Part A — Rapid-fire

1. Why does a CNN generalise better than an MLP on 800 images, given universal approximation?
2. Equivariance vs invariance — which does convolution have, which does pooling give, and why does it matter for detection?
3. Are CNNs shift-invariant in practice? What breaks it and what fixes it?
4. Give the output-size formula for a conv with padding, stride and dilation.
5. Params and MACs for a $3\times3$ conv, 256→512, on a $28\times28$ input.
6. Why do params and FLOPs peak in *different* parts of a network?
7. Three uses of a $1\times1$ convolution.
8. What do first-layer CNN filters look like, and which Module 1 concept does that echo?
9. Write the Grad-CAM formula. Why the ReLU?
10. Receptive field after conv7/s2 → pool3/s2 → conv3/s1. Show $j$ and $r$.
11. What is the effective receptive field and how does it scale with depth?
12. Why did pooling largely disappear from modern CNNs?
13. Three roles of global average pooling.
14. Write BatchNorm's four equations. Which axes are reduced for a $(N,C,H,W)$ tensor?
15. Why does BN have learnable $\gamma,\beta$?
16. Is "internal covariate shift" the reason BN works? What is?
17. Show that BN makes the loss scale-invariant in $W$, and derive the effective-LR consequence.
18. Why does BN fail at batch size 2? Does gradient accumulation fix it?
19. Why do transformers use LayerNorm instead of BatchNorm? Give three reasons.
20. What is shuffling BN in MoCo and what problem does it solve?
21. How do you fold BN into a conv at inference? Write the formulas.
22. Why is `bias=False` standard before a BN layer?
23. Why should BN parameters be excluded from weight decay?
24. State the VGG $3\times3$ argument with parameter counts.
25. Why did GoogLeNet have auxiliary classifiers, and why don't modern nets?
26. State the degradation problem *correctly*.
27. Derive $\partial\mathcal{L}/\partial\mathbf{x}_l$ for a residual stack. What does the "1" mean?
28. Why is zero easy to learn and identity hard?
29. What is the ensemble interpretation of ResNet? What experiment supports it?
30. Why is a ResNet-50 bottleneck cheaper than a ResNet-34 basic block stack?
31. What changed in pre-activation ResNet (v2) and why does it matter theoretically?
32. Derive the depthwise-separable cost ratio.
33. Explain MobileNetV2's inverted residual and linear bottleneck. Why no ReLU at the narrow end?
34. State EfficientNet's compound-scaling constraint and explain the exponents.
35. Why is MobileNet's real speedup on GPU much less than its FLOP reduction?
36. What is structural re-parameterisation (RepVGG) and why does it help?
37. Why do dropout and BatchNorm interact badly?
38. Why AdamW instead of Adam + L2?
39. When does label smoothing hurt?
40. Why cross-entropy and not MSE for classification? Give the gradient argument.
41. Which augmentation matters most for ImageNet-scale training?
42. What is Mixup's theoretical framing?
43. Name three domains where horizontal flip is a *bug*.
44. How does augmentation relate to contrastive self-supervised learning?
45. Give the transfer-learning decision matrix.
46. What did "Rethinking ImageNet Pre-training" conclude?
47. Why freeze BN statistics when fine-tuning?
48. What is layer-wise LR decay and a typical $\xi$?
49. Why are frozen backbones the 2026 production default?
50. Name three evaluation outputs beyond top-1 accuracy that you'd put in a model report.

---

## Part B — Whiteboard problems

**B1 — Cost accounting.** For ResNet-50: compute params and MACs of (a) conv1, (b) one stage-3 bottleneck block, (c) the final FC. Then state which of the three dominates params and which dominates FLOPs, and explain the pattern.

**B2 — BatchNorm backward.** Derive $\partial\mathcal{L}/\partial x_i$ for BatchNorm, remembering that $\mu$ and $\sigma^2$ both depend on every $x_i$. (Classic interview question; the three-term structure is the point.)

**B3 — Receptive field design.** You must detect 12-px objects and 300-px objects in the same 1024-px image. Design a backbone + head attachment strategy. State the stride and RF at each attachment point and justify.

**B4 — Residual derivation.** Write the forward for a pre-activation residual stack, derive the gradient w.r.t. an early activation, and explain precisely why the plain-network version vanishes.

**B5 — Efficiency trade study.** A model must run at 30 FPS on a device with 100 GFLOPS peak compute and 25 GB/s memory bandwidth. Given a candidate architecture's FLOPs and activation-memory traffic, use a roofline argument to predict whether it's compute- or bandwidth-bound, and choose between (i) halving channels, (ii) int8 quantisation, (iii) replacing $3\times3$ with depthwise-separable. Justify with the roofline.

**B6 — Debugging.** Training loss decreases smoothly; validation loss decreases then increases sharply at epoch 12 while validation *accuracy* keeps improving. Explain how both can be true, and what you'd do.

---

## Part C — Traps

| Trap | Common wrong answer | Correct answer |
|---|---|---|
| "Why does ResNet work?" | "It prevents overfitting in deep nets." | Degradation is an **optimisation** failure — plain-56 has higher *training* error than plain-20. The identity path gives an additive gradient term. |
| "Why does BatchNorm work?" | "It reduces internal covariate shift." | Santurkar et al. disproved that. It smooths the loss landscape / improves gradient predictiveness, and makes the loss scale-invariant in $W$. |
| "CNNs are translation invariant." | agreeing | Conv is **equivariant**; invariance comes from pooling/GAP. And strided downsampling **aliases**, so they aren't even reliably shift-invariant. |
| "Fewer FLOPs = faster." | "Yes." | Depthwise convs are bandwidth-bound. Optimise measured latency on the target device. |
| "Gradient accumulation fixes small-batch BN." | "Yes." | No — BN still normalises over the micro-batch. Use GroupNorm or SyncBN. |
| "Add dropout to regularise the CNN." | "Sure, 0.5 everywhere." | Dropout + BN causes variance shift; conv dropout is structurally weak (adjacent activations are correlated). Use stochastic depth / DropBlock / more augmentation. |
| "Adam with weight_decay=1e-4." | "Standard." | That's L2-inside-Adam, which is scaled by the adaptive term. Use **AdamW**, and exclude norms/biases. |
| "Bigger receptive field ⇒ sees more context." | "Yes." | Only *theoretically*. The effective RF is Gaussian and grows as $\sqrt{L}$. Dilation or explicit multi-scale buys far more per FLOP. |
| "Pretraining always improves accuracy." | "Yes." | With enough target data it mainly improves *convergence speed* (He et al. 2019). And negative transfer is real for distant domains. |
| "Label smoothing is a free win." | "Yes." | It hurts when the model will be used as a distillation teacher. |
| "ViT beat CNNs, so CNNs are obsolete." | agreeing | ConvNeXt showed much of the gap was the training recipe. The real axis is data regime and scaling behaviour. |

---

## Part D — Spaced-repetition cards

```
Q: Conv output size formula
A: H' = floor((H + 2p − d(k−1) − 1)/s) + 1

Q: Conv params and MACs
A: params = k²·C_in·C_out + C_out ; MACs = k²·C_in·C_out·H'·W'

Q: Receptive field recursion
A: j_l = j_{l−1}·s_l ; r_l = r_{l−1} + (k_l − 1)·j_{l−1}   (r_0 = 1, j_0 = 1)

Q: Effective receptive field
A: Influence over the theoretical RF is ~Gaussian, concentrated centrally; ERF grows as O(√L), not O(L). (Luo et al. 2016)

Q: BatchNorm equations
A: μ_c, σ²_c over (N,H,W) per channel; x̂ = (x−μ)/√(σ²+ε); y = γx̂ + β. Inference uses EMA running stats.

Q: Why BN really works
A: NOT internal covariate shift (Santurkar 2018). It smooths the loss landscape (better gradient Lipschitzness ⇒ larger stable LR), makes the loss scale-invariant in W (auto-decaying effective LR), and adds minibatch-noise regularisation.

Q: BN scale invariance
A: With BN after a linear layer, W→aW leaves output unchanged and ∇_{aW}L = (1/a)∇_W L. Effective LR ∝ 1/||W||² and self-decays.

Q: BN folding
A: W_fold = γW/√(σ²+ε) ; b_fold = β − γμ/√(σ²+ε). Removes BN at inference; 10–30% latency win.

Q: Normalisation axes
A: BN over (N,H,W) per channel · LN over (C,H,W) per sample · IN over (H,W) per sample+channel · GN over (C/G,H,W). Only BN is batch-dependent.

Q: Why transformers use LN not BN
A: (1) variable sequence length / padding makes batch stats ill-defined, (2) autoregressive inference has batch 1, (3) NLP batch statistics are far noisier. Plus Pre-LN trains without warmup.

Q: Degradation problem
A: 56-layer plain CNN has HIGHER TRAINING error than 20-layer. An optimisation failure, not overfitting.

Q: ResNet gradient
A: x_L = x_l + Σ F(x_i) ⇒ ∂L/∂x_l = (∂L/∂x_L)(1 + ∂/∂x_l Σ F). The additive "1" is an unimpeded gradient path; plain nets have a product instead.

Q: Why is identity hard and zero easy?
A: ReLU zeroes half its domain, so passing a signal unchanged needs an unstable weight configuration; driving F→0 is the direction weight decay already pushes. Modern recipes init the last BN γ=0 so blocks start as exact identities.

Q: ResNet ensemble view
A: n blocks ⇒ 2^n paths; deleting one block barely hurts (unlike VGG); gradient dominated by short paths. Effective depth ≪ nominal. (Veit 2016)

Q: VGG 3×3 argument
A: Three 3×3 = one 7×7 receptive field, 27C² vs 49C² params (45% fewer) and 3 non-linearities instead of 1.

Q: Depthwise-separable cost ratio
A: (k²C_in + C_in C_out)/(k² C_in C_out) = 1/C_out + 1/k² ≈ 1/9 for k=3.

Q: MobileNetV2 inverted residual
A: narrow→expand(t=6)→depthwise 3×3→project narrow, skip over the NARROW ends. Linear bottleneck = no ReLU after projection, because ReLU on a low-dim tensor destroys information irrecoverably.

Q: EfficientNet compound scaling
A: d=α^φ, w=β^φ, r=γ^φ with α·β²·γ²≈2 (FLOPs ∝ d·w²·r²). Grid search on B0 → α=1.2, β=1.1, γ=1.15.

Q: FLOPs vs latency
A: Depthwise conv has low arithmetic intensity (k² MACs/element vs k²C_out) ⇒ memory-bandwidth-bound ⇒ FLOP cuts don't translate. Optimise measured latency; fuse ops; quantise.

Q: Dropout + BN
A: Variance shift — dropout changes activation variance between train and test while BN's running stats assume the train variance. Use stochastic depth instead.

Q: AdamW
A: Decouples weight decay from the adaptive gradient scaling: θ ← θ − ηλθ applied directly. Adam + L2 regularises large-gradient params less. Exclude norms and biases.

Q: Softmax + CE gradient
A: ∂L/∂z_k = p_k − y_k. Clean and non-vanishing when confidently wrong. MSE adds a σ'(z) factor that vanishes exactly then.

Q: Label smoothing
A: y_LS = (1−ε)y + ε/K, ε=0.1. Improves calibration and generalisation but ERASES relative class similarity in the penultimate layer ⇒ a worse distillation teacher (Müller 2019).

Q: Mixup
A: x̃ = λx_i+(1−λ)x_j, ỹ likewise, λ~Beta(α,α). Framing: vicinal risk minimisation. CutMix pastes a patch and mixes labels by area.

Q: RandAugment
A: Two hyperparameters, N (ops per image) and M (magnitude), sampled from a fixed op list. Matches AutoAugment at zero search cost.

Q: Rethinking ImageNet Pre-training
A: He et al. 2019 — training COCO detection from scratch matches pretrained IF trained long enough with proper normalisation. Pretraining buys convergence speed, not ceiling, when target data is plentiful.

Q: Layer-wise LR decay
A: η_l = η_base · ξ^(L−l), ξ ≈ 0.65–0.9. Early layers barely move; standard for ViT fine-tuning.

Q: Grad-CAM
A: α_k^c = (1/Z)ΣΣ ∂y^c/∂A^k_ij ; L = ReLU(Σ_k α_k^c A^k). ReLU keeps only positive evidence for the class.
```

---

## Part E — Self-assessment rubric

| Skill | 1 | 3 | 5 (top 1%) |
|---|---|---|---|
| Conv mechanics | knows what conv is | computes params/FLOPs | knows params/FLOPs decouple, conv→GEMM, and can reason about kernel-launch and bandwidth cost |
| Receptive field | "deeper = more context" | computes theoretical RF | knows ERF is Gaussian and $\sqrt{L}$; designs attachment points for object scale |
| BatchNorm | knows the formula | knows train/test difference | debunks ICS, derives scale invariance, names 4 failure modes with fixes, folds BN for deployment |
| ResNet | "skip connections help" | knows degradation + formula | derives the gradient, states degradation as optimisation, gives ensemble + landscape views |
| Efficiency | "MobileNet is fast" | derives the cost ratio | explains arithmetic intensity/roofline, linear bottleneck, RepVGG, and measures instead of estimating |
| Regularisation | lists techniques | knows the vision-specific ordering | knows dropout/BN clash, AdamW decoupling, label-smoothing/distillation conflict |
| Augmentation | flip and crop | knows RandAugment/Mixup | knows augmentation defines SSL invariances, task-dependent harms, FixRes |
| Transfer | freeze vs fine-tune | knows the decision matrix | cites He 2019, freezes BN, uses LLRD, argues frozen backbones from a serving perspective |

---

## Part F — Module 2 exit criteria

- [ ] Derive BN's scale-invariance result and state the real reason BN works.
- [ ] Derive the ResNet gradient and state the degradation problem correctly.
- [ ] Compute params/MACs and receptive field for an arbitrary conv stack from memory.
- [ ] Derive the depthwise-separable cost ratio and explain why the latency win is smaller.
- [ ] Reproduced plain-vs-ResNet training-error curves yourself.
- [ ] Reproduced the BN-vs-GroupNorm batch-size sweep yourself.
- [ ] Shipped the 2.11 classification pipeline with an ablation table, calibration plot, Grad-CAM error analysis, and an int8 latency measurement.
