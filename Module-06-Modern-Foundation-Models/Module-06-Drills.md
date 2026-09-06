# Module 6 — Drills, Interview Bank & Spaced Repetition

> **Currency note:** version numbers in this module change fast. The *mechanisms* are what you're drilling; verify version specifics against primary sources before quoting them.

---

## Part A — Rapid-fire

1. Name three ways self-attention behaves differently from convolution when applied to images.
2. What is the quadratic cost problem, and name three architectural answers to it.
3. What operation is ViT's patch embedding, literally, in code?
4. Why does ViT need positional encoding at all?
5. Why do learned 1-D position embeddings work as well as 2-D?
6. What must you do to position embeddings when changing input resolution?
7. State ViT's data-regime result and the correct interpretation.
8. What did DeiT show, and what does its distillation token transfer?
9. What is the compute effect of going from ViT-B/16 to ViT-B/8?
10. Why can't you attach an FPN to a plain ViT? Name the two solutions.
11. Texture bias vs shape bias — which model class, and who showed it?
12. Why is "attention maps explain the model" an overclaim?
13. Why 75% masking for MAE and 15% for BERT?
14. Why does MAE's encoder not see mask tokens? Give both reasons.
15. Why per-patch normalised pixel targets?
16. Why does MAE underperform contrastive methods on a linear probe but beat them on fine-tuning?
17. Which paradigm is better for dense tasks and why?
18. What does DINOv2 combine, and what does that buy?
19. Write CLIP's loss. What determines the number of negatives?
20. Why did CLIP use a contrastive rather than a generative objective?
21. Why does "a photo of a {label}" beat "{label}"? What principle is that?
22. What is CLIP's most important robustness result?
23. Explain CLIP's compositionality failure mechanistically. Name a benchmark.
24. What does SigLIP change and what does that fix?
25. What are DINO's two anti-collapse mechanisms, and what does each prevent?
26. What is DINO's multi-crop strategy and why is local-to-global the point?
27. What is DINO's most striking emergent property?
28. What is Gram anchoring and what problem does it solve?
29. Name the three defining properties of a foundation model, plus two consequences.
30. Why is homogenisation a risk?
31. Describe SAM 1's encoder/decoder asymmetry and why it exists.
32. Why does SAM output 3 masks, and how is it trained?
33. What was SAM's actual contribution?
34. What is SAM 1's core limitation, and what follows from it?
35. What capability does SAM 3 add? What are its three components?
36. How does open-vocabulary detection replace a fixed classifier head?
37. Where does Grounding DINO fuse language, and how many places?
38. What does YOLO-World re-parameterise, and what does that buy?
39. Name three failure modes of open-vocabulary detection.
40. State the dominant production pattern for open-vocab models in one sentence.
41. Frame any VLM in terms of two questions.
42. What is the Q-Former, how many queries, and what property does that give?
43. What does LLaVA use as a connector, and what was its real contribution?
44. How many visual tokens does each approach send to the LLM?
45. Why did the field abandon the Q-Former?
46. What is the current answer to the token-count problem at high resolution?
47. What is the third connector design, and what is its distinctive advantage?
48. What is object hallucination, and which benchmark measures it?
49. Write the ELBO. What does each term do?
50. Why is the reparameterisation trick necessary?
51. Why are VAE samples blurry? Why does that not matter in latent diffusion?
52. Why are GANs sharp, and why are they unstable? (Same answer.)
53. What is mode collapse? Name two stabilisation techniques.
54. What does FID measure, and name two of its flaws.
55. Where do GANs still win in 2026?
56. Write the DDPM closed-form forward jump. Why does it matter?
57. Write $\mathcal{L}_\text{simple}$. Why is diffusion's simplicity the reason it beat GANs?
58. What does DDIM change, and what does determinism enable besides speed?
59. What is the score-matching connection?
60. What is flow matching, and which models use it?
61. What is DiT?
62. State latent diffusion's perceptual-vs-semantic compression argument.
63. Why is the SD VAE trained with an adversarial loss?
64. What is the general conditioning mechanism in the SD U-Net?
65. Why did SD3/Flux add a T5 text encoder?
66. Derive CFG. What does the scale trade, and what does it cost?
67. What is ControlNet's zero-init trick, and which earlier idea does it echo?
68. Name three evaluation metrics for text-to-image and what each misses.

---

## Part B — Whiteboard problems

**B1 — ViT accounting.** ViT-L/14 at $336\times336$.
(a) Number of tokens. (b) Parameters in the patch-embedding layer. (c) Attention FLOPs per layer per head. (d) Compare to ViT-L/14 at $224^2$.

**B2 — MAE design.** You are pretraining on 500k medical images.
(a) Choose a masking ratio and justify it from the redundancy argument, considering that medical images are often *more* redundant than natural ones.
(b) Design the evaluation protocol. Which metric would mislead you?
(c) Would you use MAE, DINOv2-style, or both? Defend it.

**B3 — CLIP loss.** Write the full symmetric InfoNCE in equations, then in ~10 lines of PyTorch. Then: what changes with SigLIP's sigmoid loss, and why does that remove the large-batch requirement?

**B4 — VLM design.** You must build a VLM for chart and table understanding.
(a) Vision encoder, resolution strategy, connector, and what you train — with justification for each.
(b) Estimate the visual token count for a 1600×1200 chart under your design.
(c) Which of the three connector families is worst here, and why?

**B5 — Diffusion derivation.** Starting from $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$, derive the closed form for $q(x_t|x_0)$. Then show how $\mathcal{L}_\text{simple}$ follows from reparameterising the variational bound's mean-matching term.

**B6 — CFG behaviour.** Sketch, on one axis, how prompt adherence and sample diversity vary with the guidance scale $s$. Mark the usual operating point. Then explain, mechanically, what causes the over-saturation at high $s$.

**B7 — System design.** Design a production visual-search-and-tagging service for a 20M-image stock photo library. Requirements: text search, image search, automatic tagging with an evolving tag vocabulary, sub-200 ms p99, and the ability to add 100k images/day. Name every model, every index, and every measurement you'd track.

---

## Part C — Traps

| Trap | Common wrong answer | Correct answer |
|---|---|---|
| "ViT beat CNNs, so transformers are better for vision." | agreeing | Only above a data threshold. DeiT (recipe) and ConvNeXt (recipe) both showed the gap was largely training methodology. |
| "ViT has no inductive bias." | "None at all." | It has patch-level locality (the patch embedding *is* a strided conv) and whatever the position embeddings learn. "Weaker prior," not "no prior." |
| "MAE is worse than contrastive — its linear probe is lower." | agreeing | You measured linear separability, which is what contrastive optimises and MAE doesn't. Report fine-tuning and dense-task transfer. |
| "Mask 15% like BERT." | "Standard." | Images are redundant; 15% is solvable by interpolation. 75% is the point. |
| "CLIP understands images." | "Yes." | It matches images to captions. It fails at counting, spatial relations, and attribute binding — a bag-of-concepts representation satisfies its loss. |
| "Use CLIP zero-shot for our specialist domain." | "It's zero-shot, it'll work." | Fine-grained and jargon vocabulary are exactly where CLIP fails. Baseline it, then use few-shot on frozen features. |
| "SAM segments objects." | "Yes." | SAM 1/2 segment *regions from prompts* and assign **no semantics**. Naming requires CLIP/a VLM, or SAM 3's concept prompting. |
| "Just deploy Grounding DINO — open vocabulary is strictly better." | "Right." | It's slow, prompt-sensitive, and poorly calibrated across prompts. Production uses it as an **auto-labeller** and deploys a distilled closed-vocab model. |
| "BLIP-2's Q-Former is more advanced than LLaVA's linear layer." | "Yes, it's more sophisticated." | The field converged on the simple projector — the Q-Former's bottleneck destroys the detail needed for OCR/documents, which is the dominant use case. |
| "The VAE in Stable Diffusion generates the image." | "Yes." | It's a near-deterministic **compressor** (tiny KL weight, adversarial loss for sharpness). The diffusion model generates; the VAE decodes. |
| "Higher CFG = better images." | "Yes." | It trades diversity for fidelity and, past ~10–12, produces over-saturation and artefacts. It also costs 2× compute per step. |
| "Diffusion is slow because the model is big." | "Yes." | It's slow because you run the model **20–50 times per image** (×2 with CFG). Hence step distillation and guidance distillation. |
| "GANs are obsolete." | "Yes." | They survive in super-resolution, inside the SD VAE decoder, and — importantly — as the **adversarial objective for distilling diffusion to 1–4 steps**. |
| "FID measures image quality." | "Yes." | FID measures the distance between feature distributions — quality *and* diversity together — and says nothing about prompt adherence. It's also sample-size dependent. |

---

## Part D — Spaced-repetition cards

```
Q: ViT patch embedding, in code
A: Conv2d(3, D, kernel_size=P, stride=P). Non-overlapping patches + linear projection = a strided convolution.

Q: Why ViT needs positional encoding
A: Self-attention is permutation-equivariant — shuffle the tokens and the output shuffles identically. ALL spatial structure must be injected. Learned 1-D suffices; the model recovers 2-D locality on its own. Must be 2-D interpolated when resolution changes.

Q: ViT data-regime result
A: ImageNet-1k: ViT loses to ResNet. ImageNet-21k: comparable. JFT-300M: ViT wins. Interpretation: inductive bias is a DATA-EFFICIENCY prior — worth more than flexibility when data is scarce, a constraint when it's plentiful.

Q: DeiT
A: ViT trainable on ImageNet-1k alone with heavy augmentation + stochastic depth + a DISTILLATION TOKEN from a CNN teacher (which literally transfers the convolutional prior). The data requirement was partly a recipe requirement.

Q: MAE masking ratio
A: 75% (vs BERT's 15%). Images are highly redundant — at 15% you can reconstruct by local interpolation, requiring no understanding. You must break interpolation to force reasoning about object extent.

Q: MAE asymmetric encoder
A: The encoder sees ONLY visible patches; mask tokens are inserted in a lightweight decoder (~9% of encoder compute), discarded after pretraining. Reasons: ~3× faster pretraining, and the encoder becomes a representation model rather than an inpainter.

Q: Contrastive vs masked SSL
A: Contrastive → strong LINEAR PROBE (optimises a globally discriminable embedding), weaker dense transfer (augmentation-invariance discards position/scale). Masked → weak linear probe, better FINE-TUNING and DENSE tasks (reconstruction preserves spatial detail). DINOv2 combines both.

Q: CLIP loss
A: Symmetric InfoNCE over the N×N cosine similarity matrix with a LEARNED temperature: (CE over rows + CE over columns)/2. Batch size = number of negatives (CLIP used 32,768).

Q: Why contrastive over generative for CLIP
A: Compute efficiency — predicting the exact caption is far harder. ~4× more efficient than a bag-of-words predictive baseline, ~12× than generative. "Learn to match, don't learn to generate."

Q: CLIP prompt engineering
A: "A photo of a {label}." beats "{label}" because the text encoder was trained on caption sentences — a bare word is out of distribution. Prompt ensembling over ~80 templates adds ~3.5 points on ImageNet. Principle: match inference distribution to pretraining distribution.

Q: CLIP's compositionality failure
A: The contrastive loss only needs the right pair to beat other pairs IN THE BATCH; random batches almost never differ by argument order, so nothing forces relational encoding. A bag-of-concepts representation suffices. Benchmarks: Winoground, ARO, SugarCrepe.

Q: SigLIP
A: Replaces softmax-InfoNCE with a PAIRWISE SIGMOID loss — no global normalisation over the batch, so it trains well at small batch and scales better. Now the default image encoder in many VLMs.

Q: DINO anti-collapse
A: CENTRING (subtract an EMA of the teacher's mean) prevents one dimension dominating but alone pushes toward uniform. SHARPENING (low teacher temperature) prevents uniform collapse but alone pushes toward one dimension. Two OPPOSING forces, one per collapse mode.

Q: DINO multi-crop
A: Teacher sees 2 global crops (≥50%); student sees those plus several small local crops. Student must predict the GLOBAL representation from a LOCAL view — local-to-global correspondence.

Q: DINO emergent property
A: ViT attention maps contain explicit semantic segmentation of the foreground object, with no segmentation supervision. Also strong k-NN classification (78.3% ImageNet) with no training.

Q: DINOv2 / DINOv3
A: v2 = curated 142M dataset + DINO self-distillation + iBOT masked patch objective ⇒ strong FROZEN features. v3 = 7B params, 1.7B images, GRAM ANCHORING (keeps the patch-feature Gram matrix anchored to an earlier checkpoint, preventing dense-feature degradation at long schedules). First SSL model to beat weakly-supervised across the board, frozen.

Q: Foundation model definition
A: Trained on broad data at scale, adaptable to many downstream tasks. Requires scale + self/weak supervision + adaptability. Consequences: emergence and HOMOGENISATION (a systemic risk — one backbone's biases propagate everywhere).

Q: SAM 1 architecture
A: Heavy MAE-pretrained ViT-H image encoder (~0.15 s, run ONCE per image) + prompt encoder + LIGHTWEIGHT mask decoder (~50 ms on CPU, run per prompt). Deliberate asymmetry enabling interactive prompting. Outputs 3 masks (whole/part/subpart), trained with MIN loss over the three.

Q: SAM's real contribution
A: The DATA ENGINE. SA-1B: 1.1B masks on 11M images in three stages (assisted-manual → semi-automatic → fully automatic with a 32×32 point grid). The model and dataset bootstrapped each other.

Q: SAM 1's limitation
A: It segments but does NOT name — no semantics. Hence every combined pipeline pairs it with CLIP, a VLM, or an open-vocab detector.

Q: SAM 3
A: PROMPTABLE CONCEPT SEGMENTATION from open-vocabulary text or image exemplars — segments EVERY instance of a concept. Components: DETR-based detector + SAM 2 memory-bank tracker + text/image encoders + a PRESENCE HEAD (decouples "does it appear" from "where"). Benchmark: SA-Co. SAM 3.1 adds object multiplexing (~16→32 FPS on H100).

Q: Open-vocabulary detection mechanism
A: Replace logits = W @ features with logits = text_embeddings @ region_features — CLIP's idea at region level. Grounding DINO fuses language at 3 points (feature enhancer, language-guided query selection, cross-modality decoder); ~52.5 zero-shot COCO AP.

Q: YOLO-World's trick
A: Re-parameterisable vision-language path — for a FIXED vocabulary known at deploy time, the text embeddings are pre-computed and baked in, so text costs nothing at inference. ~52 FPS.

Q: The open-vocab production pattern
A: COMPOSE TO DISCOVER, DISTIL TO DEPLOY. Use Grounding DINO + SAM as an auto-labeller over unlabelled data, verify a sample, train a fast closed-vocab YOLO/RF-DETR on the result.

Q: How to frame any VLM
A: Two questions: what is the CONNECTOR, and what gets TRAINED?

Q: BLIP-2
A: Q-Former: ~188M-param transformer with 32 LEARNED QUERIES that cross-attend to frozen image features. Fixed 32 visual tokens into the LLM regardless of image size. Two-stage training; only the Q-Former trains. Weak at OCR/counting — the bottleneck destroys detail.

Q: LLaVA
A: Connector = a single linear layer (1.5: 2-layer MLP). One visual token per patch (576 for ViT-L/14 @336²). Stage 1: align the projection on captions. Stage 2: instruction-tune projection + LLM on ~158k GPT-4-SYNTHESISED multimodal instructions. The DATA RECIPE was the contribution.

Q: Why the field abandoned the Q-Former
A: (1) The bottleneck loses detail needed for OCR/charts/documents — the dominant commercial use case. (2) Extra component with its own multi-objective pretraining. (3) The constraint it solved (context/compute) stopped binding. Architectural cleverness that buys efficiency loses to simplicity once the efficiency stops binding.

Q: Three VLM connector families
A: (1) Projection (LLaVA, Qwen-VL, InternVL) — simple, dominant. (2) Query-based (BLIP-2 Q-Former, Flamingo Perceiver Resampler) — fixed cheap context, loses detail. (3) Cross-attention inside the LLM (Flamingo, Llama-3.2-Vision) — doesn't consume the LLM's context window, but requires modifying the LLM.

Q: Object hallucination
A: A VLM confidently describing objects not present, driven by language priors over weak visual evidence. Benchmark: POPE (also CHAIR). Mitigations: counterfactual/negative training data, visual contrastive decoding, requiring grounded boxes, stronger/higher-res vision encoders.

Q: VAE ELBO + reparameterisation
A: log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x) || p(z)). Reparameterise z = μ + σ⊙ε, ε~N(0,I) — you can't backprop through sampling, so move the stochasticity into a parameter-free variable.

Q: Why VAE samples are blurry
A: The Gaussian likelihood (MSE reconstruction) makes the decoder output the MEAN of all plausible reconstructions rather than committing. Irrelevant in latent diffusion, where the VAE is only a compressor and the diffusion model supplies the detail.

Q: GAN sharpness and instability
A: SAME ROOT — the discriminator is a learned, adaptive loss. It punishes blur (⇒ sharp), but it's a two-player game with no single progress metric (⇒ unstable, mode collapse). Fixes: WGAN-GP, spectral norm, progressive growing. StyleGAN3 fixed texture-sticking via ANTI-ALIASED resampling (Nyquist again).

Q: DDPM forward closed form
A: x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε, with ᾱ_t = Π(1−β_s). You can jump to ANY timestep in one shot — which is what makes training tractable (never simulate the chain).

Q: DDPM training objective
A: L_simple = E‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t)ε, t)‖². A plain MSE regression. That simplicity — no adversarial game, no mode collapse, a loss that measures progress — is why diffusion beat GANs.

Q: DDIM
A: Non-Markovian, DETERMINISTIC reverse process ⇒ skip steps (20–50 instead of 1000) AND get a meaningful latent↔image correspondence, which is what enables editing and interpolation.

Q: Score-matching connection
A: ε_θ(x_t,t) ≈ −√(1−ᾱ_t)·∇_{x_t} log p(x_t). The network learns the SCORE; sampling is Langevin dynamics. DDPM and score matching are the same theory, unified by the SDE formulation.

Q: Latent diffusion's argument
A: Split generative learning into PERCEPTUAL compression (remove imperceptible high-frequency detail — an autoencoder does this efficiently, 512×512×3 → 64×64×4, 48× fewer elements) and SEMANTIC compression (the actual data structure — what the expensive diffusion model should spend capacity on). Pixel-space diffusion wastes capacity on the first.

Q: The SD VAE
A: A near-deterministic COMPRESSOR, not a generator. Tiny KL weight; trained with perceptual (LPIPS) + PATCH ADVERSARIAL loss to avoid MSE blur, because the decoder is the final stage of every generation.

Q: SD conditioning
A: CROSS-ATTENTION at multiple U-Net resolutions: Q from the latent, K/V from the text encoder τ(y). Swap τ and you condition on anything — that generality is why the architecture spread.

Q: Classifier-free guidance
A: Train one model with conditioning dropped ~10% of the time (learns both conditional and unconditional score). Sample with ε̃ = ε_θ(x_t,∅) + s·[ε_θ(x_t,c) − ε_θ(x_t,∅)]. s≈7–8 typical. Trades DIVERSITY for FIDELITY; costs 2× forward passes per step.

Q: ControlNet
A: A trainable copy of the U-Net encoder joined by ZERO-INITIALISED convolutions, conditioned on edges/depth/pose/scribble. Zero-init means the branch contributes nothing at step 0, so training starts from the pretrained model's exact behaviour — the same "start as the identity" principle as ResNet's zero-init final BN.

Q: Why SD3/Flux added T5
A: CLIP's text encoder was trained on short web captions and has a 77-token limit; T5 handles long, compositional prompts far better, which markedly improves prompt adherence and in-image text rendering. SD3/Flux also swap the U-Net for a DiT and DDPM for rectified flow matching.

Q: Text-to-image evaluation
A: FID (distribution quality + diversity; sample-size dependent, Inception-biased, says NOTHING about prompt adherence) · CLIPScore (prompt adherence; says nothing about quality) · HPSv2/PickScore/ImageReward (learned human preference) · T2I-CompBench/GenEval (compositionality). No single metric suffices.
```

---

## Part E — Self-assessment rubric

| Skill | 1 | 3 | 5 (top 1%) |
|---|---|---|---|
| ViT | "transformers on images" | describes patches + CLS + pos-embed | patch embed = strided conv, permutation-equivariance argument, data-regime interpretation, resolution interpolation, DeiT/ConvNeXt caveats |
| SSL | "self-supervised is unlabelled" | knows contrastive and masked exist | explains the 75%/15% asymmetry, the linear-probe/fine-tune divergence, and why DINOv2 combines both |
| CLIP | "matches images and text" | writes the loss | prompt-distribution argument, robustness result, compositionality failure mechanism with benchmarks, SigLIP's fix |
| DINO | "self-supervised ViT" | knows teacher/student + EMA | centring vs sharpening as opposing forces, multi-crop's local-to-global point, Gram anchoring |
| SAM | "segments anything" | knows the prompt types | encoder/decoder asymmetry, 3-mask min-loss, the data engine as the contribution, SAM 3's concept prompting + presence head |
| Open-vocab | "detects from text" | knows Grounding DINO | the text-embedding-as-classifier mechanism, YOLO-World's re-parameterisation, and the auto-label-then-distil production pattern |
| VLMs | "vision + LLM" | knows LLaVA and BLIP-2 | frames everything as connector + what's trained, explains the Q-Former bottleneck's OCR failure, knows why the field converged, names the cross-attention third option |
| Generative | knows GAN vs diffusion | writes the DDPM objective | derives the closed-form jump, explains why MSE simplicity beat GANs, knows DDIM's determinism, $v$-prediction, flow matching, DiT |
| Latent diffusion | "SD works in latent space" | names the four components | the perceptual/semantic compression argument, VAE-as-compressor with adversarial loss, cross-attention as a general interface, derives CFG with its trade and cost |

---

## Part F — Module 6 exit criteria

- [ ] Explain the ViT data-regime result and the ConvNeXt/DeiT caveats without picking a side.
- [ ] Explain the MAE 75%/BERT 15% asymmetry, and the linear-probe vs fine-tune divergence.
- [ ] Write CLIP's loss from memory and explain its compositionality failure mechanistically.
- [ ] Name DINO's two anti-collapse forces and DINOv3's Gram anchoring.
- [ ] Frame any VLM as connector + trained components, and explain why MLP projectors won.
- [ ] Derive DDPM's closed-form forward jump and $\mathcal{L}_\text{simple}$.
- [ ] Derive CFG and state its trade and its compute cost.
- [ ] Have implemented: ViT from scratch, MAE with a masking-ratio ablation, the CLIP loss, DDPM + DDIM, and a minimal LLaVA-style projector.
- [ ] Have shipped one capstone (6.16) with a baseline, an ablation, a latency measurement, and a documented failure analysis.
