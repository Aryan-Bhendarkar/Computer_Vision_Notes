# Module 6 — Modern & Foundation Models

> **This is the module closest to the job you want.** Everything here — ViT, CLIP, self-supervised pretraining, VLMs, diffusion — is the working vocabulary of GenAI engineering. You already know Transformers, so this module spends its budget on *what changes when the tokens are pixels*, and on the specific architectures you will be asked to name, compare, and combine.
>
> **Currency warning.** This is the fastest-moving module. Version numbers here are accurate as of **September 2026**; check the primary sources (Meta AI blog, arXiv, HF model cards) before quoting a version in an interview. Understanding the *mechanism* is durable; version numbers are not.

**Concept map**

```
6.1 Attention for vision ──> 6.2 ViT 🟡 ──> 6.3 CNN vs ViT
                                  │
              ┌───────────────────┼──────────────────┐
              v                   v                  v
   6.4 MAE (masked) 🟡     6.5 CLIP (contrastive)   6.6 DINO lineage 🟢
              └───────────────────┼──────────────────┘
                                  v
                        6.7 What is a foundation model
                                  │
              ┌───────────────────┼───────────────────┐
              v                   v                   v
   6.8 SAM 1→3 🟢        6.9 Grounding DINO      6.10 VLMs: BLIP-2, LLaVA 🟡
              └───────────────────┼───────────────────┘
                                  v
                     6.11 Combined promptable pipelines
                                  │
   6.12 AE→VAE→GAN ──> 6.13 DDPM ──> 6.14 Latent diffusion 🟡 ──> 6.15 Applications
                                  │
                                  v
                          6.16 CAPSTONE
```

---

## 6.1 Attention, Applied to Vision

**The question this section answers: you already know self-attention cold from language. What actually changes when the tokens are pieces of an image rather than words — and which of those changes are opportunities, which are problems?** We will not re-derive attention; here it is for reference only:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The productive way to hold this operator is that it is **completely blind to where its inputs came from**. It sees a *set* of vectors and computes weighted averages over that set. In language that blindness costs you word order, which you patch with a positional encoding and then largely stop thinking about. In vision the same blindness costs you *two dimensions* of structure, and it collides with the fact that images produce far more tokens than sentences do. Three consequences follow, and each one drives a design decision later in this module.

**1. Global receptive field from layer 1.** Every token attends to every other token immediately — there is no notion of "nearby" available to restrict it. Compare with what you derived in 2.3: a CNN's receptive field grows by only $(k-1)$ jumps per layer, and its *effective* receptive field — where gradient actually flows — grows more slowly still, roughly $O(\sqrt{L})$ in depth. So a CNN needs many layers before two distant pixels can influence one another; a ViT has that in layer 1. This is the single biggest representational difference between the families, and it is why ViTs are better at relating two ends of a long object, or a small object to the scene context that disambiguates it.

**2. Permutation equivariance — the load-bearing one.** Shuffle the input tokens and the output tokens shuffle identically; nothing else changes. Convince yourself rather than accepting it: the softmax weights depend only on the *pairwise* dot products $q_i \cdot k_j$, and reordering the set reorders those scores in exactly the same way, so the weighted sums come out permuted but otherwise untouched. The operator therefore has no concept of "above," "left of," or "adjacent to." **Every bit of spatial structure a ViT knows must be injected by the positional encoding**, because nothing else in the architecture can supply it. A CNN gets locality for free from the shape of its kernel; a ViT must be *told* what "next to" means. This is why 6.2 spends so long on positional encodings — there they are not a detail, they are the entire spatial prior.

**3. Quadratic cost in token count.** $O(N^2 d)$ for $N$ tokens, because you form all $N^2$ pairwise scores. For a $224\times224$ image at $16\times16$ patches, $N = 196$ — trivially fine. At $1024\times1024$ with the same patch size, $N = 4096$ and attention is $16.8$M pairs per head per layer.

**Pause:** a sentence runs to a couple of hundred tokens and a $224^2$ image is 196 tokens — so why is the quadratic cost a much bigger deal in vision than in language?

Because in language, token count grows with how much you want to *say*, and you control that. In vision, token count grows with **resolution**, and resolution is dictated by the task — a chest X-ray, a document scan or a satellite tile is not optional about needing fine detail. Worse, the pain compounds: halving the patch side quadruples $N$, which multiplies attention cost by **16**. Language rarely asks you to lengthen a sequence 16× for the same content; vision asks constantly. **This is why high-resolution vision transformers need windowed (Swin), deformable (4.8), or linear attention** — the quadratic cost is the central engineering constraint of the whole family, and you will watch it force architectural choices in 6.2, 6.8 and 6.10.

**Attention in vision predates ViT**, and knowing the lineage prevents a naive story. SENet's channel attention (2.5, 2017) reweights channels using a globally-pooled context; CBAM adds a spatial version of the same idea; and Non-local Neural Networks (Wang et al. 2018) is literally self-attention applied to CNN feature maps, framed as a "non-local mean" operation. Transformers did not arrive in vision in 2020 — attention had been leaking into CNNs for three years. 2020 is when someone removed the CNN.

**In your own words:** why does a convolution need no positional encoding while attention cannot function without one?

### ✅ Mastery check

(a) A $512\times512$ image. Compute the number of self-attention pairs per head per layer at patch size 16 and at patch size 8. Why does this make high-resolution vision transformers hard?
(b) Convolution needs no positional encoding; attention cannot function without one. State the property that causes this difference.
(c) Name three architectural answers to the quadratic cost, and say what each one gives up.

<details><summary>Answer sketch</summary>
(a) Patch 16: $(512/16)^2 = 32^2 = \mathbf{1024}$ tokens ⇒ $1024^2 \approx \mathbf{1.05}$M pairs. Patch 8: $64^2 = \mathbf{4096}$ tokens ⇒ $4096^2 \approx \mathbf{16.8}$M pairs — <b>16× the cost for 4× the tokens</b>, because cost is quadratic while resolution is linear in token count. Halving the patch size quadruples tokens and multiplies attention cost by 16. That is the entire reason plain ViTs stall at moderate resolution while dense tasks (segmentation, detection, document OCR) want the opposite.
(b) <b>Permutation equivariance.</b> Attention computes a weighted sum over a <i>set</i> — shuffle the tokens and the outputs shuffle identically, so the operation carries no notion of "adjacent" or "above". Convolution is defined by a spatial kernel over a grid, so locality and relative position are baked into the operator itself. All of a ViT's spatial structure must therefore be injected through positional encoding; a CNN gets it for free.
(c) <b>Windowed attention</b> (Swin): attend only within local windows, shifted between layers — linear cost, but gives up true global receptive field at any single layer and needs the shifting trick to propagate information across windows. <b>Deformable attention</b> (Deformable DETR, 4.8): each query samples $K$ learned offsets — linear cost and a useful locality prior, but gives up dense all-pairs interaction and adds a sampling mechanism to learn. <b>Linear / kernelised attention</b> (Performer, linear attention, and state-space models like Mamba): approximate or reformulate the softmax so cost is linear — gives up exact attention, and empirically loses some of the sharp retrieval behaviour full attention has. (Also acceptable: pooling/downsampling tokens, and hierarchical patch merging.)
</details>


### 🔨 Build + read

**Build:** You already know attention mechanics from language models — so make the *vision-specific* cost visible rather than re-deriving attention itself. On a small image (e.g. 224×224 patchified at 16×16 → 196 tokens), compute the exact FLOP count of self-attention's $O(N^2)$ term versus a convolution's $O(N)$ term at matched channel width, then repeat the calculation for a naive pixel-level attention (no patchification, $N=224^2$) and observe why patchification is not a design nicety but a computational necessity — plot cost vs. patch size.

**Read:** Vaswani et al., *Attention is All You Need* (2017) — skim only for notation, since you know this. Then jump straight to Dosovitskiy et al., *An Image is Worth 16×16 Words* (2021), §3.1, to see the minimal change that turns attention into a vision architecture.

---

## 6.2 Vision Transformer (ViT): Patch Embedding & Positional Encoding 🟡

### Intuition

**Start from the obstacle, because it explains the whole design.** You want to feed an image to a Transformer. The Transformer wants a sequence of vectors. The obvious move — one token per pixel — dies instantly on the arithmetic from 6.1: a $224^2$ image is 50,176 pixels, so 2.5 *billion* attention pairs per head per layer. Completely impossible. So the real question is not "how do I turn an image into tokens" but **"how do I turn an image into a manageable number of tokens without a convolutional hierarchy doing the downsampling for me?"**

The answer is embarrassingly blunt: cut the image into a grid of small square patches, flatten each into a vector, treat the sequence of patch vectors exactly like a sequence of word tokens, and run a standard Transformer encoder. That's it — a $16\times16$ patch grid turns 50,176 pixels into 196 tokens, a 256× reduction, in one step and with no hand-designed structure beyond "the grid." The famous line from the paper title is the whole idea: *"An Image is Worth 16×16 Words."*

**You might expect** the patch grid to be the crude part that a better design would replace — cutting an object in half at a patch boundary looks like vandalism. It isn't, and the reason is worth holding on to: attention is global from layer 1, so the two halves of that object can talk to each other immediately. Patching destroys nothing that the first attention layer cannot reassemble. What patching *does* destroy is the knowledge of which halves were adjacent — which is exactly what point 3 below has to restore.

### The architecture, precisely

**1. Patch embedding.** Split a $H\times W\times C$ image into $N = HW/P^2$ non-overlapping $P\times P$ patches, flatten each to $P^2C$ dimensions, and project linearly to $D$:

$$
\mathbf{z}_0 = [\,\mathbf{x}_{\text{cls}};\ \mathbf{x}_p^1E;\ \mathbf{x}_p^2E;\ \dots;\ \mathbf{x}_p^NE\,] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{(P^2C)\times D}
$$

**Implementation note worth knowing:** this linear projection over non-overlapping patches **is exactly a `Conv2d(3, D, kernel_size=P, stride=P)`**. Work out why rather than taking it — the equivalence is mechanical. A `Conv2d` with kernel size $P$ takes a $P\times P\times C$ window, flattens it against the kernel weights, and produces one output vector of length $D$ (one per output channel). Setting the stride equal to the kernel size means the windows tile the image without overlap — one window per patch, exactly. So "flatten each patch and multiply by a shared $E$" and "slide a $P\times P$ stride-$P$ conv with $D$ output channels" are the *same arithmetic written two ways*: the conv kernel tensor, reshaped, **is** $E$. That one line is how every ViT implementation actually does it, and saying it shows you've read the code, not just the paper. It also puts the "ViTs have no convolutions" claim in its place: a ViT's very first operation is a convolution — it just happens to be the one convolution that has no overlap, and therefore no spatial inductive bias to speak of.

For ViT-B/16 at $224^2$: $N = 196$ patches, $D = 768$, and $E$ is $768\times768$. (Sanity-check that shape: a patch is $16\times16\times3 = 768$ input dimensions, projected to $D=768$ — the square matrix is a coincidence of ViT-B's particular numbers, not a rule.)

**2. The `[CLS]` token.** A learned embedding prepended to the sequence, whose final-layer output is used for classification. Borrowed from BERT. The rationale is that you need *one* vector to classify, and rather than choosing a pooling rule by hand, you add a token that belongs to no patch and let attention learn how to aggregate into it. (Later work — and DINOv2 — often finds **mean-pooled patch tokens** work as well or better for dense and retrieval tasks; the `[CLS]` token is a convention, not a necessity. This should sound familiar from 3.1, where mean-pooled patch tokens appeared in the same table as `[CLS]` for exactly this reason.)

**3. Positional encoding.** Now pay the debt from 6.1. Once you flatten to a set, the model has no way to know that patch 15 sits directly above patch 29 — that fact lives nowhere in the tensor. So ViT **adds** a learned vector per position to each patch embedding, letting the network read position off the residual stream. ViT uses **learned 1-D** position embeddings — one vector per patch index. Two surprising facts:
- **1-D beats nothing by a large margin, but learned 2-D encodings give no further gain.** This is genuinely counter-intuitive: an image is 2-D, so surely telling the model "row 3, column 7" should beat telling it "index 29"? It doesn't, and the reason is that the embeddings are *learned*. The mapping from raster index to grid position is deterministic and simple, so the network just learns it. Visualising the trained embeddings shows the proof: they develop 2-D locality on their own — nearby patches acquire similar embeddings, arranged in a grid pattern that nobody put there. The lesson generalises. **A hand-designed structure only helps if the model cannot cheaply learn it**, which is the same theme that will decide the data-regime result below.
- **They don't transfer across resolutions.** Here is the practical consequence people get bitten by. Changing input resolution changes $N$, and you have exactly $N$ learned vectors — so at a new resolution some positions have no embedding. The fix is to reshape the embeddings back into their original $\sqrt{N}\times\sqrt{N}$ grid, **2-D interpolate** that grid to the new size, and flatten again. This is a standard step when fine-tuning at higher resolution and a common source of silent bugs, because forgetting it produces a model that runs fine and scores badly.
- Modern alternatives: **relative position bias** (Swin), **RoPE** (2-D rotary, increasingly standard in 2025–26 vision backbones and VLMs). Both encode *relative* offsets rather than absolute slots, which is precisely why they handle variable resolution more gracefully — a relative offset of "two patches to the left" means the same thing at any grid size, while absolute slot 29 does not.

**Pause:** before reading on — if you removed positional encodings entirely and trained a ViT on ImageNet, would accuracy collapse to chance?

No, and understanding why is the point. A bag of unordered patches still tells you a great deal: patch-level texture, colour and local shape are strongly diagnostic of class, and "contains fur-textured patches and eye-like patches" gets you a long way on ImageNet. So you lose several points, not everything. But shuffle the patches at *test* time and the no-position model's accuracy is completely unchanged — which is the actual demonstration that it learned no spatial structure at all. That is the ablation in this section's build task, and it is a much sharper probe than accuracy alone.

**4. Transformer encoder.** $L$ blocks of **Pre-LN** (2.4) multi-head self-attention + MLP with residuals:
$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \qquad \mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
$$
The MLP is a 2-layer expansion (typically $4D$) with GELU. Note there is nothing vision-specific in this block — it is the language Transformer encoder unchanged. Everything image-specific in a ViT happens in steps 1–3, before the first block. That is the cleanest one-sentence summary of the architecture, and it is why ViT was such a portable idea.

**Standard sizes:** ViT-B/16 = 12 layers, $D{=}768$, 12 heads, 86M params. ViT-L/16 = 24 layers, $D{=}1024$, 307M. ViT-H/14 = 32 layers, $D{=}1280$, 632M. The `/P` is the patch size — **smaller patch = more tokens = quadratically more compute but better fine detail**. Walk the arithmetic once so the notation stops being decorative: ViT-B/8 halves the patch side, so it has $2^2 = 4\times$ the tokens of ViT-B/16, and since attention is quadratic in token count, ~$4^2 = 16\times$ the attention cost. **Patch size, not depth or width, is the compute knob that bites hardest** — and it is the knob you are forced to turn for any dense task.

### The headline result and its caveat

Now the result that made ViT contentious rather than obvious. ViT beats CNNs — **but only when pretrained on enough data.** From the paper:

| Pretraining data | Result |
|---|---|
| ImageNet-1k (1.3M) | ViT **underperforms** a comparable ResNet |
| ImageNet-21k (14M) | roughly comparable |
| JFT-300M (300M) | ViT **clearly wins** |

**The wrong reading is "ViTs are just data-hungry," said as if it were a flaw.** The correct reading is that a CNN's inductive bias — locality, translation equivariance — is a **prior**, and a prior is a claim you make on the model's behalf before seeing data. Priors trade off against evidence in the way you already know from every Bayesian argument: when data is scarce, a decent prior beats flexibility, because flexibility with nothing to constrain it just fits noise. When data is abundant, the evidence can specify better structure than you guessed, and the prior turns from a helpful shortcut into a *constraint* — it forbids the model from learning anything a convolution cannot express. The table above is that curve, measured. **This is the cleanest illustration in all of deep learning of the bias/flexibility trade-off, and it is the answer to 6.3.**

**And then the story gets a twist worth knowing, because it is the sort of thing an interviewer uses to see if you read past the headline.** **DeiT** (Touvron et al. 2021) showed ViT *can* be trained on ImageNet-1k alone and be competitive, given the right *recipe* — heavy augmentation (RandAugment, Mixup, CutMix, Random Erasing), stochastic depth, repeated augmentation, and a **distillation token** learning from a CNN teacher. Look at what that last item really is: augmentation and distillation are both ways of *injecting a prior without hard-coding it into the architecture*. Augmentation tells the model "these transformations don't change the label," which is translation-equivariance and more, supplied as data. Distillation from a CNN teacher literally transfers the convolutional prior through the teacher's outputs. So the data requirement was substantially a recipe requirement — exactly the symmetric point ConvNeXt later made in the other direction (2.5), where a plain ConvNet given ViT's *training recipe* matched ViT. Two papers, one lesson: **a chunk of what people attribute to architecture is attributable to recipe**, and separating those two is a genuinely top-1% habit.

**In your own words:** why does more pretraining data flip the sign of the CNN's inductive bias from asset to liability?

### 🎯 Top-1% distinction

1. **Patch embedding = strided convolution.** Implementation-level knowledge.
2. **Attention is permutation-equivariant, so positional encoding carries all spatial structure** — and 1-D learned embeddings suffice because the model recovers 2-D structure.
3. **Position embeddings must be interpolated when changing resolution.**
4. **The data-regime result and its correct interpretation** as bias vs flexibility.
5. **DeiT's counterpoint** — much of the data requirement was a recipe requirement.
6. **Patch size is the key compute knob** and it scales quadratically.
7. **ViT has no feature pyramid**, which is why detection/segmentation on plain ViTs needs adaptation (ViTDet, 4.4) or a hierarchical variant (Swin).

### ✅ Mastery check

(a) ViT-B/16 at $384\times384$ instead of $224\times224$: how many tokens, and how does attention FLOPs scale? What must you do to the position embeddings?
(b) Why does ViT underperform on small datasets, and give two ways to fix it without more data.
(c) You need a ViT backbone for semantic segmentation. What is the structural problem and how do the two main solutions differ?

<details><summary>Answer sketch</summary>
(a) $384/16 = 24$, so $24^2 = 576$ patches (+1 CLS = 577) vs 196 (+1) at $224^2$ — about <b>2.9× the tokens</b>. Self-attention is $O(N^2)$, so the attention cost rises by ~$2.9^2 \approx \mathbf{8.6\times}$ (the MLP cost rises only linearly, so overall it's somewhere between 3× and 8.6× depending on the layer mix). The position embeddings were learned for a $14\times14$ grid and must be <b>reshaped to $14\times14\times D$, bilinearly interpolated to $24\times24$, and flattened back</b>. Forgetting this silently destroys accuracy.
(b) It has no built-in locality or translation-equivariance prior, so it must learn from data what a CNN gets for free — higher sample complexity. Fixes without more data: (i) <b>the DeiT recipe</b> — heavy augmentation (RandAugment + Mixup + CutMix + Random Erasing), stochastic depth, long schedules, and <b>distillation from a CNN teacher</b> via a distillation token, which literally transfers the convolutional prior; (ii) <b>use a pretrained backbone</b> (DINOv2/CLIP) and linear-probe or LoRA-tune it — 2.10's decision matrix says small data ⇒ don't fine-tune 86M parameters; (iii) <b>hybrid or hierarchical architectures</b> (Swin, ConvNeXt, or a CNN stem before the transformer) that reintroduce locality architecturally.
(c) <b>Problem:</b> a plain ViT produces a <b>single-scale</b> feature map at stride 16 (or 14), with no hierarchy — but segmentation and detection heads expect a multi-scale pyramid (FPN, 4.4), and stride 16 is too coarse for fine boundaries. <b>Solution A — change the backbone:</b> Swin Transformer uses windowed attention (linear cost) with patch merging between stages, producing a genuine CNN-like hierarchy at strides 4/8/16/32; drop-in compatible with FPN. <b>Solution B — keep the plain ViT and adapt:</b> ViTDet builds a simple pyramid from the <i>last</i> layer alone using strided convs and deconvs, plus a few global-attention layers among mostly windowed ones. The difference matters: A modifies the pretraining architecture (so you need Swin-pretrained weights); B lets you use any plain-ViT checkpoint — including CLIP, MAE, or DINOv2 — which in the foundation-model era is a decisive practical advantage.
</details>

### 🔨 Build + read

**Build:** Implement ViT from scratch (patch embed as a `Conv2d`, CLS token, learned pos-embed, Pre-LN blocks) and train it on CIFAR-10. Then run the ablation that teaches the lesson: (i) no positional encoding, (ii) learned 1-D, (iii) fixed 2-D sinusoidal — and shuffle the patch order at test time in each case. Then implement position-embedding interpolation and fine-tune at 2× resolution.

**Read:** Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021) §3 and Figure 7 (the data-regime plot). Touvron et al., "DeiT" (ICML 2021). Liu et al., "Swin Transformer" (ICCV 2021) §3.

---

## 6.3 CNN vs ViT — the Inductive-Bias Trade-off

### The comparison

**The question this section answers — and the first move is to notice it's a trap.** "CNN or ViT?" is asked constantly in interviews, and it has no answer, because the two differ along several axes that point in different directions depending on your data scale, your task's need for multi-scale features, and your inference budget. What a strong candidate does is decompose the question into those axes before answering any of them. The table is that decomposition; read each row as *a condition under which one of them wins*, not as a scorecard.

| | **CNN** | **ViT** |
|---|---|---|
| Built-in priors | locality, translation equivariance, hierarchy | **almost none** (only patch-level locality) |
| Receptive field | grows with depth; **effective** RF ~$\sqrt{L}$ | global from layer 1 |
| Cost | linear in pixels | **quadratic in tokens** |
| Small-data regime | **wins** | overfits without a strong recipe |
| Large-data regime | saturates earlier | **scales better** |
| Multi-scale | natural hierarchy | needs Swin or ViTDet-style adaptation |
| Robustness | more texture-biased; sensitive to some corruptions | more shape-biased; better on occlusion and some OOD shifts |
| Interpretability | feature visualisation, Grad-CAM | attention maps (with caveats — attention ≠ explanation) |

### The three findings that make a strong answer

1. **The prior is a data-efficiency trade, not a quality ceiling.** (6.2's data-regime table.) Inductive bias helps when data is scarce and constrains when data is plentiful. Note the shape of this claim: it says the CNN is not *better*, it is *cheaper in data* — which means the correct question is never "which architecture" but "which architecture **at my data scale**."
2. **ConvNeXt (2.5) showed much of the measured gap was the training recipe.** When you give a ResNet the ViT recipe — AdamW, cosine schedule, 300 epochs, heavy augmentation, LN, GELU, large depthwise kernels, inverted bottleneck — it closes most of the gap. The honest statement: **architecture differences are smaller than the literature's headline numbers suggested, because the comparisons weren't controlled.** This is a methodological point as much as a technical one, and stating it as such lands well: when two things differ in five ways and you attribute the outcome to one of them, you have not run an experiment.
3. **They fail differently, and this is the most useful finding of the three.** Geirhos et al. showed ImageNet-trained CNNs are strongly **texture-biased** — shown a cat-shaped image rendered with elephant skin texture, they say "elephant." ViTs are measurably more **shape-biased**, closer to human judgement, which follows naturally from global attention: shape is a long-range relationship between distant parts, exactly the thing a CNN's slowly-growing receptive field is worst at and attention gets for free. ViTs are also markedly more robust to occlusion and patch removal — delete a large fraction of patches and a ViT degrades gracefully rather than falling over.

**Pause:** that last property — a ViT still works when most patches are missing — is not just a robustness curiosity. What training method does it make possible?

Masked pretraining. If removing 75% of the patches still leaves a coherent representation, you can *deliberately* remove 75% and ask the model to reconstruct what's gone — which is precisely MAE, the very next section. Note the direction of the argument: the robustness property is what makes the training method viable, not a consequence of it. Keep that link in view when you read 6.4.

**The 2026 practical answer:** the question is nearly moot for most applications, because you use a **pretrained foundation backbone** and the architecture is whatever that backbone is. The real decision axes are: what was it pretrained on and how, does it produce the feature structure your head needs, and what is its inference cost on your hardware. Notice that all three are questions about the *checkpoint*, not the architecture — which is what "foundation model era" actually means in practice (6.7).

**In your own words:** state the CNN-vs-ViT trade-off as a conditional — "X wins when…, Y wins when…" — without using the word "better."

### 🎯 Top-1% distinction

- **Refuse the framing "which is better."** Answer with the data-regime trade, the ConvNeXt controlled-experiment caveat, and the different failure modes. Interviewers asking this are usually testing whether you'll pick a side or reason about conditions.
- **Texture bias vs shape bias**, citing Geirhos.
- **Attention maps are not explanations** — attention weights don't reliably indicate causal importance (the "Attention is not Explanation" line of work). A candidate who says "I'd visualise the attention maps to interpret it" without that caveat is over-claiming.
- **Hybrid is the default in practice** — a conv stem, or windowed attention, or ConvNeXt-style large depthwise kernels. Pure architectures are a research abstraction.

### ✅ Mastery check

(a) You have 800 labelled medical images. CNN or ViT? Now you have 80 million unlabelled images from the same scanner and 800 labels. Does your answer change?
(b) A colleague says "the attention maps show the model is looking at the tumour, so it's interpretable and we can trust it." Refute this precisely.
(c) What did ConvNeXt actually prove, and what did it *not* prove?

<details><summary>Answer sketch</summary>
(a) <b>800 labels: neither, trained from scratch.</b> Both overfit catastrophically. The right answer is a <b>pretrained backbone + a light head</b> — and at that data scale a frozen backbone with a linear or k-NN probe beats fine-tuning 86M parameters (2.10's decision matrix). If forced to choose an architecture to train, a small CNN, because its locality and translation-equivariance priors substitute for data. <b>With 80M unlabelled in-domain images the answer changes completely:</b> now do <b>self-supervised pretraining on your own data</b> (MAE or DINOv2-style, 6.4/6.6), which is exactly the regime where a ViT's weaker prior stops being a liability — you have enough data to learn the structure rather than assume it — and then fine-tune on the 800 labels. That is ViT's data-regime result applied in the direction that actually matters: the relevant axis is <i>total data seen</i>, not <i>labels held</i>.
(b) Two problems. <b>Attention weights are not attributions.</b> A high attention weight means a token contributed to a weighted sum; it does not establish that the output <i>depends</i> on that token causally — you can often permute or ablate high-attention tokens with little effect, and the "Attention is not Explanation" line of work showed different attention distributions can yield identical predictions. <b>And plausibility is not correctness:</b> a map that lands on the tumour is consistent with the model reading the tumour <i>and</i> with it reading a correlated artefact nearby (a marker, a scanner-specific border, a resolution cue). The test is causal: occlude the region and measure the score drop; run the background-only shortcut test (S.2); and run Adebayo's sanity check — randomise the weights and see whether the map changes. If it doesn't, the method is describing the input, not the model.
(c) It proved that <b>a large part of the measured CNN-vs-ViT gap was the training recipe, not the architecture</b> — AdamW, long schedules, heavy augmentation, LayerNorm, GELU, fewer activations, inverted bottlenecks, large depthwise kernels — because applying them to a ResNet recovers most of the difference. What it did <b>not</b> prove: that the architectures are equivalent at all scales. The data-regime and scaling behaviour still differ, attention still gives a global receptive field from layer 1 that no depthwise kernel replicates, and ConvNeXt's own large-kernel design was itself borrowed <i>from</i> the transformer literature. The honest reading is that the comparison had been uncontrolled, not that architecture is irrelevant.
</details>


### 🔨 Build + read

**Build:** Train a small ViT and a small ResNet (matched parameter count, e.g. ~5M) on CIFAR-100 from scratch — no pretraining — under an identical schedule and augmentation policy. Plot both learning curves and final accuracy. Then repeat with a stronger augmentation policy (RandAugment + mixup) applied only to the ViT run. Confirm the folk result: ViT underperforms without inductive bias substitutes at small data/compute scale, and augmentation partially closes the gap. This is Dosovitskiy's data-scale crossover (Fig. 3 in the ViT paper) reproduced in miniature.

**Read:** Raghu et al., *Do Vision Transformers See Like Convolutional Neural Networks?* (2021, NeurIPS) for representation-similarity evidence of the inductive-bias difference, rather than benchmark numbers alone. Then Touvron et al., *DeiT* (2021, ICML) for the augmentation/distillation recipe that made ViT viable without JFT-300M-scale data.

---

## 6.4 Self-Supervised Learning: Masked Autoencoders 🟡

> The gap analysis flagged this because DINO/contrastive learning is only *half* of the self-supervised story. MAE is the other half, and the two paradigms have genuinely different strengths.

### Intuition

**The question this section answers: contrastive learning (3.6) got its supervision signal by declaring two augmented views of the same image "the same thing." That works, but it required you to hand-design the augmentations, mine negatives, and defend against collapse. Is there a way to get supervision out of an unlabelled image that needs none of that?**

There is, and you already know it from language: **hide part of the input and predict it**. That is BERT, and the supervision is free — the answer is the data you removed. Applied to pixels: hide 75% of an image and train the model to paint the missing parts back in. To do that it must understand what objects are, how they extend behind occlusions, and what textures belong where. Nothing to mine, nothing to collapse into, no augmentation policy to tune.

So the idea transfers in one line. **The interesting part — and the part interviews probe — is why the *recipe* cannot transfer.** BERT masks 15% of tokens; MAE masks 75%. That is not a hyperparameter that happened to land differently. It is a statement about how images differ from text, and it is the heart of this section.

### MAE (He et al., CVPR 2022)

**Three design decisions, each with a reason. Take the reasons seriously — every one of them is a "why not the obvious thing" answer.**

**1. A very high masking ratio — 75%, versus BERT's 15%.**

**Pause:** why would masking 15% of an image's patches produce a nearly useless training signal?

Because you could solve it without looking at the image *as an image at all*. Ask what a masked patch's neighbours tell you. In text, the word missing from "the capital of France is ___" is not recoverable from adjacent characters — language is information-dense and each token carries semantic content its neighbours do not. In an image, a missing $16\times16$ patch is usually surrounded by eight patches of nearly the same colour, texture and gradient, so **bilinear interpolation is a decent solution** and the model that learns it has understood nothing. Natural images are enormously redundant at the scale of a patch; that redundancy is precisely what makes JPEG work.

So the masking ratio has to be pushed until *local interpolation stops being sufficient* and the only way to fill a hole is to know what object is there and how it continues. At 75%, an average masked patch has no visible immediate neighbours at all — the model has to reason across the image about content, which is exactly the behaviour you were trying to buy. (The paper ablates this: accuracy peaks around 75% and stays good up to 85% — note that the curve is broad, which tells you the mechanism is "enough to break interpolation," not a delicate tuning.) **This asymmetry between language and vision is the single most quotable point about MAE**, because it reframes a number as an argument about the statistics of the two modalities.

**2. An asymmetric encoder–decoder.** The **encoder sees only the visible 25% of patches** — masked patches are simply not passed in, not even as placeholder tokens. Mask tokens are inserted only in a **lightweight decoder** (8 blocks, 512-dim, ~9% of the encoder's compute) that reconstructs pixels.

Two reasons, and they are of different kinds. The first is **efficiency, and it is where the 75% ratio pays for itself twice**. Attention is quadratic (6.1), so dropping 75% of tokens cuts the encoder's attention cost by roughly $4^2 = 16\times$ and its total cost by ~3×, which is what made ViT-Huge pretraining affordable at all. Notice the pleasing inversion: in most masked-prediction setups a higher mask ratio is a harder task you pay for; here it is a harder task that is *cheaper*, because the discarded tokens are discarded from the compute too.

The second reason is **representational**, and it is the subtler one. If mask tokens went into the encoder, the encoder's job would partly be inpainting — allocating capacity to "what colour goes in this hole." By never showing it a mask token, you make its job strictly "represent what is here," and push all the reconstruction machinery into a decoder you are going to throw away. **This is the same structural move as SimCLR's projection head (3.1): when your training objective demands something your downstream task does not want, put a discardable module between them to absorb it.** The decoder is discarded after pretraining; only the encoder ships.

**3. Reconstruct raw pixels, with per-patch normalisation.** Simple MSE on normalised pixel values — pleasantly boring, given how much machinery BEiT needed for discrete targets. The one refinement that mattered: normalising each patch's target by its *own* mean and std improved representation quality. The reason is a capacity argument. Without it, a large share of the MSE is explained by each patch's overall brightness — a low-frequency, nearly-semantic-free quantity — so the model earns most of its loss reduction by predicting "this region is dark," and spends capacity on it. Standardising per patch removes that term from the target, leaving the model to be scored on *structure within the patch*, which is where the semantics live.

**Results:** ViT-Huge fine-tuned to 87.8% ImageNet top-1 — the first purely-ImageNet self-supervised method to clearly beat supervised pretraining. And it scales: bigger models keep improving, where contrastive methods saturated.

**In your own words:** why does an image need a far higher mask ratio than a sentence?

### Contrastive vs Masked — the comparison that matters

These are the two families of self-supervision, and the whole table falls out of one difference in what they ask the model to do. Contrastive learning asks for **invariance**: two views must map to the same vector, so anything that differs between the views must be *discarded*. Masked modelling asks for **reconstruction**: the missing pixels must be recoverable, so anything needed to place them must be *kept*. Invariance destroys information by design; reconstruction preserves it by design. Read every row below as a consequence of that.

| | **Contrastive** (SimCLR, MoCo, DINO) | **Masked** (MAE, BEiT, SimMIM) |
|---|---|---|
| Signal | invariance to augmentation | reconstruction of hidden content |
| Needs | strong augmentation, many negatives (or an anti-collapse trick) | just masking — **no augmentation design, no negatives, no collapse risk** |
| Batch size | large (or a queue) | small is fine |
| **Linear probe** | **strong** — features are already linearly separable | **weak** — features are rich but not linearly organised |
| **Fine-tuning** | good | **better**, especially for dense tasks |
| Dense tasks (det/seg) | weaker — augmentation-invariance discards position and scale | **stronger** — reconstruction requires spatial detail |
| Failure mode | collapse; shortcut exploitation (colour histograms, 3.6) | can learn low-level texture statistics rather than semantics |

**The linear-probe-vs-fine-tune gap is the key diagnostic**, and being able to explain it — rather than just report it — is the difference between having read the table and having understood it. A linear probe measures one specific thing: *is the class information already laid out along linear directions?* Contrastive learning optimises a global embedding explicitly for discriminability between instances, so it produces exactly that geometry and a linear classifier works immediately. MAE never asks for a globally discriminative vector at all — it asks for enough information to redraw pixels, which can be encoded in any distributed, tangled form the network finds convenient. The information is present; it is simply not *linearly organised*. Fine-tuning (or an attentive probe) can untangle it; a linear layer cannot.

**So: if you evaluate MAE with a linear probe and conclude it's worse, you've measured the wrong thing.** Generalise the lesson — **a probe measures the geometry of a representation, not its information content**, and choosing a probe that mismatches your training objective will reliably tell you the wrong story. That is a very quotable point in an interview about evaluating pretrained backbones.

**The synthesis:** modern top backbones combine both. **DINOv2** (6.6) uses a DINO-style self-distillation objective *plus* iBOT's masked-image-modelling patch-level objective — global and local, discriminative and reconstructive. That combination is why DINOv2/v3 features are strong on both classification *and* dense tasks.

### Related methods to name

- **BEiT** (2021): predict discrete visual tokens (from a pretrained dVAE) instead of pixels — closer to BERT's discrete targets. Preceded MAE.
- **SimMIM**: like MAE but a simpler design (mask tokens in the encoder, a single linear prediction head) — shows the asymmetric encoder is an efficiency win more than an accuracy necessity.
- **iBOT**: online tokeniser + self-distillation on masked patches — the bridge that DINOv2 builds on.
- **I-JEPA** (2023): predict in **representation space** rather than pixel space, avoiding the need to model pixel-level detail that is irrelevant to semantics. The motivation is the objection you should already have to MAE: reconstructing exact pixels forces the model to predict things that are genuinely unpredictable and genuinely unimportant — the precise grain of a patch of grass — and capacity spent there is wasted. Predicting a *representation* of the hidden region asks only for what a representation retains. The catch is that a predicted target which is itself learned can collapse (both sides can agree on a constant), so I-JEPA needs anti-collapse machinery of the sort contrastive methods needed and MAE did not. LeCun's preferred direction, and an important conceptual alternative.

**In your own words:** why does contrastive pretraining win on linear probes while masked pretraining wins on fine-tuned dense tasks?

### 🎯 Top-1% distinction

1. **The 75%-vs-15% masking ratio and the redundancy argument.**
2. **The asymmetric encoder — why the encoder must not see mask tokens.**
3. **The linear-probe vs fine-tune gap** and what it says about each paradigm.
4. **Contrastive discards what augmentation removes; masked preserves spatial detail** — which is why masked methods win on dense tasks.
5. **DINOv2 combines both**, and that's why it's the current default backbone.
6. **I-JEPA's predict-in-latent-space** framing as the conceptual next step.

### ✅ Mastery check

(a) Why 75% masking for images and 15% for text? Be precise about the mechanism.
(b) You pretrain with MAE and evaluate with a linear probe; it loses badly to a CLIP baseline. What did you measure wrong, and what should you report?
(c) You need a backbone for a dense task (segmentation) with 3000 labelled images and 500k unlabelled in-domain images. Design the pretraining.

<details><summary>Answer sketch</summary>
(a) The difference is <b>information density and redundancy</b>. A text token carries a lot of information and is hard to infer from neighbours — masking 15% already creates a genuinely hard prediction problem requiring syntax and semantics. An image patch is highly redundant with its neighbours: at 15% masking you can reconstruct almost perfectly by <b>local interpolation</b>, which requires no semantic understanding, so the pretext task teaches nothing useful. You must remove enough that interpolation fails and the model has to reason about object extent and identity. MAE's ablation shows accuracy peaking around 75%.
(b) You measured <b>linear separability of a global embedding</b> — which is exactly what CLIP's contrastive objective optimises and exactly what MAE's reconstruction objective does not. MAE's features are rich but not linearly organised. Report instead: <b>fine-tuned</b> accuracy (MAE's headline metric), and — since you presumably care about dense tasks — <b>transfer to detection/segmentation</b> (COCO AP, ADE20k mIoU). An <b>attentive probe</b> (a small attention-pooling head instead of a linear layer) is the fairer middle ground and is now commonly reported for exactly this reason.
(c) 500k unlabelled in-domain images is a strong asset — this is the regime SSL exists for. <b>Design:</b> start from a public DINOv2 (or MAE) checkpoint rather than from scratch — 500k is not enough to pretrain from random init competitively. Then run <b>continued self-supervised pretraining on your unlabelled in-domain data</b> with a masked objective (or a DINOv2-style combined objective), because the task is <b>dense</b> and masked pretraining preserves the spatial detail segmentation needs. Then fine-tune on the 3000 labelled images with a light decoder head, layer-wise LR decay, frozen early layers, and heavy augmentation. <b>Evaluate</b> with mIoU <i>and</i> Boundary IoU (4.12), with the split made by site/session, not randomly. Also worth stating the cheaper baseline you should run first: frozen DINOv2 + a linear or lightweight decoder head — it often gets most of the way there for a fraction of the effort, and it's the honest thing to compare against.
</details>

### 🔨 Build + read

**Build:** Implement MAE on CIFAR-10 or a small ImageNet subset. Run the masking-ratio ablation (25 / 50 / 75 / 90%) and report **both** linear-probe and fine-tuned accuracy for each — the divergence between those two curves is the lesson of this section. Visualise reconstructions at each ratio.

**Read:** He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022) — short, exceptionally clear, read all of it. Then skim I-JEPA (Assran et al., CVPR 2023) §3 for the latent-space-prediction argument.

---

## 6.5 CLIP: Contrastive Vision–Language Pretraining

### Intuition

**The question this section answers: every classifier you have built so far was born with a fixed label set. Train on ImageNet and you can say 1000 things, forever. Where does that limit actually come from, and can it be removed?**

Trace it back and the limit is not in the network — it is in the *supervision format*. A one-hot label over $K$ classes is a supervision signal with exactly $\log_2 K$ bits and a closed vocabulary; the final linear layer has $K$ rows because the labels had $K$ values. So the fixed label set is downstream of choosing categorical labels in the first place. Replace the label with something open-ended and the limit dissolves.

CLIP's answer: supervise with **the caption**. Train on 400 million (image, caption) pairs scraped from the web, and teach the model to match images to their captions. Natural language is unbounded, compositional, and — crucially — already attached to billions of images on the internet, so this is simultaneously a richer signal *and* a cheaper one than hand-labelling. At the end you have a model that can classify into **any** set of categories you can describe in words, including ones it was never explicitly trained on, because the "classifier" is now a sentence you write at inference time.

**You have seen the mechanism already.** In 3.6 you learned contrastive self-supervision: two augmented views of the same image are a positive pair, everything else in the batch is a negative, InfoNCE pulls positives together and pushes negatives apart. CLIP changes exactly one thing — **the second view is not an augmented image, it is the caption.** Everything else is the machinery you already know. Hold that mapping while you read the method; the rest of this section is mostly noticing which term corresponds to which.

### The method

Two encoders — an image encoder (ViT or ResNet) and a text encoder (Transformer) — projected into a **shared embedding space**. Note why a shared space is the whole trick: once an image vector and a text vector live in the same space with the same metric, "does this caption describe this image" becomes a dot product, and so does every downstream use (retrieval, zero-shot classification, guiding a diffusion model in 6.14).

For a batch of $N$ pairs, compute the $N\times N$ cosine-similarity matrix and apply a **symmetric InfoNCE** loss (3.6): cross-entropy over rows (image→text) plus over columns (text→image), with a **learned temperature** $\tau$ (parameterised as $\log(1/\tau)$ and clipped, for stability).

**Map it term for term onto 3.6 before reading the code.** In SimCLR's InfoNCE the numerator was $\exp(\text{sim}(z_i, z_j)/\tau)$ for the positive pair and the denominator summed over all other views in the batch. Here the positive is $\exp(\text{sim}(I_i, T_i)/\tau)$ — image $i$ with *its own* caption — and the denominator sums over $\exp(\text{sim}(I_i, T_j)/\tau)$ for every other caption in the batch. The $N\times N$ similarity matrix is that computation for all $i$ at once: **the diagonal is the positives, everything off-diagonal is a negative.** The symmetric part is simply that the matrix can be normalised along either axis — softmax over a row asks "which caption goes with this image," softmax over a column asks "which image goes with this caption" — and there is no reason to prefer one, so you average both. And $\tau$ is the same temperature knob from 3.6, controlling how sharply the loss concentrates on the hardest negatives; CLIP just learns it instead of tuning it.

```python
# The entire CLIP loss, essentially:
I_e = l2_normalize(image_encoder(images) @ W_i)      # N × d
T_e = l2_normalize(text_encoder(texts)  @ W_t)       # N × d
logits = (I_e @ T_e.T) * exp(t)                      # N × N, t = learned temperature
labels = arange(N)
loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```

Six lines. Read `labels = arange(N)` carefully, because it carries the entire idea: **the correct answer for row $i$ is column $i$** — the supervision is nothing but "each image goes with its own caption," which requires no annotation at all, only that the pairs arrived together.

**Three details that matter:**

1. **Batch size is the number of negatives** — CLIP used 32,768. This is 3.6's lesson at scale, and it is worth restating the mechanism: a negative is what tells the model which *distinctions* to preserve, so with 128 negatives the model only needs to tell a dog from 128 random things, while with 32,768 it must tell this dog from thousands of near-misses. Batch size is therefore not a throughput knob here — it is part of the objective, which is why CLIP-scale training is a distributed-systems problem as much as a modelling one.
2. **The contrastive objective was chosen over generative captioning**, and the reason is the most transferable idea in the paper. Generating the exact caption forces the model to model everything about how the sentence is phrased — word choice, ordering, style — almost none of which is about the image. Matching only requires the model to tell the right caption from the wrong ones, a far weaker demand that still extracts the semantic content. CLIP's paper measures it: contrastive is ~4× more compute-efficient than a predictive (bag-of-words) baseline and ~12× than generative. **"Learn to match, don't learn to generate" is the efficiency insight** — and it is the same argument, in different clothes, as I-JEPA's objection to pixel reconstruction in 6.4.
3. **Prompt engineering matters, and it is not a hack.** "A photo of a {label}." beats a bare label by several points. The reason is a distribution-matching argument you should be able to give unprompted: the text encoder was trained exclusively on *web captions*, which are sentences. A bare token like "dachshund" is out of distribution for it — nothing in training looked like that — so its embedding sits in a poorly-modelled region of the space. Wrapping it in a caption-shaped template moves the query back onto the manifold the encoder actually learned. **Prompt ensembling** (averaging the text embeddings of ~80 templates) adds ~3.5 points on ImageNet, for the same reason ensembling usually helps: it averages away template-specific noise while keeping the shared class signal.

**Pause:** CLIP's zero-shot ImageNet number is quoted constantly. But "zero-shot" plainly cannot mean "never saw dogs." What is it actually zero-shot *with respect to*?

With respect to **the label set and the task**, not the content. CLIP saw enormous numbers of dogs, and captions naming breeds. What it never saw was ImageNet's particular 1000-way categorisation, or any training example formatted as that task. So the honest claim is "no task-specific supervised training," and the honest caveat — which the paper itself raises — is that web-scale data likely contains material overlapping the *evaluation* sets, making "zero-shot" a statement about protocol rather than a guarantee of novelty. Being precise about this distinction is a small thing that reads as real fluency.

### Zero-shot classification

Encode each candidate class name as a prompt, encode the image, and take the highest cosine similarity. **The classifier is synthesised from text at inference time** — no training, no fixed label set.

It is worth seeing that this is not an analogy to a classifier but *literally* one. An ordinary linear classification head computes $W h$ and takes the argmax; row $k$ of $W$ is a learned vector for class $k$. CLIP computes the dot product of the image embedding with the *text embedding of each class name* and takes the argmax. Same operation — the only difference is where the class vectors came from: gradient descent on labelled data in one case, the text encoder reading a sentence in the other. **The text encoder is a classifier-weight generator**, and once you say it that way, open-vocabulary detection (6.9) stops being surprising: it is the same substitution applied to a detector's classification head.

CLIP zero-shot matched a fully-supervised ResNet-50 on ImageNet (76.2%), and — the more important result — was dramatically more robust on distribution-shifted variants (ImageNet-R, ImageNet-A, ObjectNet), where supervised models collapse. **Natural-language supervision at web scale buys robustness, not just convenience.**

### Limitations to be able to state

- **Poor at counting, spatial relations, and compositionality.** "A red cube on top of a blue sphere" vs "a blue cube on top of a red sphere" — CLIP struggles badly. Do not memorise this as a quirk; it is a direct prediction from the objective. Ask what the loss actually demanded: that image $i$ score higher against caption $i$ than against 32,767 *randomly drawn other captions*. Random captions almost never share a scene's object set, so **the bag of concepts — "cube, sphere, red, blue" — is virtually always enough to win**, and the model is never penalised for ignoring which adjective binds to which noun. The training signal contains no hard negatives that differ only in binding, so no pressure exists to represent binding. (Benchmarks: Winoground, ARO, SugarCrepe.) **This is the most important known weakness and a favourite interview probe** — and the strong answer names the missing hard negatives, not just the symptom.
- **Fine-grained discrimination is weak** (specific bird species, aircraft variants) unless those distinctions are common in web captions.
- **Inherits web biases** — CLIP's own paper documents demographic bias in zero-shot classification.
- **Not a detector or segmenter** — it produces a single global embedding, which is why open-vocabulary detection (6.9) needs additional machinery.

### The lineage

**OpenCLIP** (open reproduction, LAION-2B/5B data), **SigLIP** (replaces the softmax-InfoNCE with a **pairwise sigmoid** loss — no global normalisation over the batch, so it trains well at small batch and scales better; now the default image encoder in many VLMs), **EVA-CLIP**, **MetaCLIP** (open data-curation recipe). **SigLIP is the one to name for currency**, and the mechanism is worth one extra sentence because it explains *why* the change helps. Softmax-InfoNCE normalises each row over the whole batch, so the loss for one pair depends on every other pair — which is what makes huge batches necessary and forces expensive all-gathers across devices. A sigmoid loss scores each of the $N^2$ pairs independently as a binary "match / no match," so the objective decomposes; no global normalisation, no cross-device coupling, good behaviour at small batch. Knowing that "the batch-size problem was solved by changing the loss from softmax to sigmoid, which removes the batch-wide normalisation term" is exactly the kind of specific, mechanistic detail interviewers reward.

**In your own words:** in one sentence, what did CLIP replace the one-hot label with, and what capability did that replacement buy?

### 🎯 Top-1% distinction

1. **Symmetric InfoNCE with a learned temperature**; batch size = negatives.
2. **The efficiency argument for contrastive over generative.**
3. **Prompt engineering and ensembling** with the distribution-matching explanation.
4. **The robustness result** (distribution-shifted ImageNet variants), not just the zero-shot accuracy.
5. **Compositionality failure** with named benchmarks — this is the sharpest known limitation.
6. **SigLIP's sigmoid loss** and why it removes the large-batch requirement.
7. **CLIP is the bridge to nearly everything downstream** — it is the text encoder in Stable Diffusion (6.14), the vision encoder in many VLMs (6.10), and the scoring function in open-vocabulary detection (6.9).

### ✅ Mastery check

(a) Why does "a photo of a {label}" beat "{label}"? What general principle does this instantiate?
(b) CLIP fails to distinguish "the dog chasing the cat" from "the cat chasing the dog." Explain mechanistically.
(c) You need a zero-shot classifier for 40 industrial defect types with very specific names. Predict CLIP's performance and give your actual plan.

<details><summary>Answer sketch</summary>
(a) The text encoder was trained on web <b>captions</b> — full sentences — so a bare word like "dog" is out of distribution for it and lands in a poorly-modelled region of the embedding space. Wrapping it in a caption-like template puts the query back in the training distribution. The general principle: <b>match the inference-time input distribution to the pretraining distribution.</b> Same idea as FixRes (2.9's train/test resolution mismatch) and as choosing augmentations that mirror the deployment domain. Prompt ensembling over ~80 templates then averages away template-specific noise, worth ~3.5 points on ImageNet.
(b) The contrastive objective only requires that the correct (image, caption) pair scores higher than the other pairs <i>in the batch</i>. Since batches are random, the other captions in a batch almost never differ from the correct one only by argument order — so <b>the loss never applies pressure to encode relational structure</b>. A "bag of concepts" representation ({dog, cat, chasing}) solves the training task essentially as well as a relational one, and it's easier to learn. Hence CLIP behaves like a bag-of-words model for relations. Benchmarks that isolate this: Winoground, ARO, SugarCrepe. Fixes explored in the literature: hard negatives constructed by permuting captions, and generative or captioning objectives that force word order to matter.
(c) <b>Prediction: poor.</b> Industrial defect terminology ("porosity", "lack of fusion", "hot tear", "orange peel") barely appears in web alt-text paired with the corresponding close-up imagery, and defect discrimination is <b>fine-grained</b> and often texture-based — two of CLIP's known weak spots. Zero-shot will be near-useless. <b>Actual plan:</b> (i) baseline it anyway — it costs an hour and calibrates expectations; (ii) the real approach is <b>few-shot with a frozen backbone</b>: use CLIP or DINOv2 features and train a linear/kNN classifier on whatever labelled examples you have — this is where frozen foundation features genuinely shine with 20–50 examples per class; (iii) if you have unlabelled in-domain images, do continued SSL pretraining first (6.4); (iv) if you must keep the text interface, <b>fine-tune CLIP on your own (image, description) pairs</b> — even a few thousand pairs adapts the text encoder to your vocabulary. Also worth saying: the defect task is probably <b>detection/segmentation</b>, not classification, since defect location and extent matter.
</details>

### 🔨 Build + read

**Build:** Implement the CLIP loss from scratch and verify it against `open_clip`. Then, using a pretrained CLIP: (i) reproduce the prompt-ensembling gain on a small dataset; (ii) build a text-to-image search over 10k of your own images (this is Module 3's retrieval pipeline with a CLIP encoder — connect them explicitly); (iii) construct 20 compositional test pairs ("red cube on blue sphere" / "blue cube on red sphere") and measure CLIP's accuracy. Getting ~chance on (iii) yourself is worth more than reading the papers.

**Read:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021) — §2 and §3.1. Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP, ICCV 2023).

---

## 6.6 DINO → DINOv2 → DINOv3 🟢

### DINO (Caron et al., ICCV 2021) — self-**di**stillation with **no** labels

**The question this section answers: contrastive learning (3.6) needed negatives, and negatives are expensive — huge batches, memory queues, careful mining. Can you learn a good representation using only positives?**

The immediate objection is the one you should already be raising: with no negatives, nothing stops the model outputting the same constant vector for every image. That is *collapse*, and it satisfies a positives-only objective perfectly. So the real question is narrower and sharper — **what can play the role negatives played, i.e. what else can prevent collapse?** DINO's answer is asymmetry between two copies of the network, and the anti-collapse machinery below is the actual content of the method.

A student network and a teacher network with the **same architecture**; the teacher is an **EMA** of the student. Both see different augmented crops; the student is trained to match the teacher's output distribution via cross-entropy on softmax-ed features. The EMA matters: the teacher changes more slowly than the student, so the target is not something the student can instantly agree with by degenerating — it lags, and that lag is a source of signal.

**Multi-crop:** the teacher sees 2 **global** crops (≥50% of the image); the student sees those plus several **local** crops (small, <50%). The student must predict the *global* view's representation from a *local* view — a "local-to-global" correspondence objective that is where much of DINO's power comes from.

**Two anti-collapse mechanisms** (3.6's central question, answered a third way):
- **Centring**: subtract an EMA of the teacher's mean output. Prevents one dimension dominating — but alone would push toward a uniform distribution.
- **Sharpening**: a low teacher temperature. Prevents collapse to uniform — but alone would push toward one dominant dimension.
- **They are deliberately opposing forces**, and their balance is what keeps the output distribution non-degenerate. Spell out the logic, because it is the whole design: there are exactly two ways a softmax output distribution can go degenerate — it can become **the same peak for every image** (one dimension always dominant), or it can become **flat for every image** (uniform). Centring subtracts a running mean, which suppresses any dimension that is consistently large, killing the first mode but pushing toward the second. Sharpening lowers the teacher's temperature, which forces peaks, killing the second but pushing toward the first. Neither is safe alone; run together, each one's failure direction is the other's target. That framing — **two collapse modes, one mechanism each, tuned against each other** — is elegant and very quotable.

**Pause:** BYOL, MoCo and DINO all avoid collapse without negatives. What structural feature do all three share?

An **asymmetry between the two branches** that prevents the trivial solution being reachable by gradient descent: a momentum/EMA target branch that receives no gradient (so the network cannot move the target to meet itself), plus an extra predictor or centring/sharpening operation on one side only. The general principle — worth stating in exactly these terms — is that **you can drop negatives if you break the symmetry between the branches**; negatives and asymmetry are two different ways of making the constant solution unattractive.

**The famous emergent property, and it is less magical than it looks once you connect it to multi-crop.** DINO's ViT attention maps contain **explicit semantic segmentation** of the foreground object, with no segmentation supervision whatsoever. Why would that emerge? Because the local-to-global objective repeatedly demands that a small crop predict the whole image's representation — and a small crop only supports that prediction if it landed on the *object*, not the background. The training pressure is therefore "figure out which regions are the object," which is a segmentation signal in everything but name. Emergence here is a consequence of the objective, not an accident. Its features also enable strong k-NN classification (78.3% ImageNet top-1 with a simple k-NN classifier — no training at all).

### DINOv2 (2023) — the backbone that made frozen features work

Not a new objective so much as **an engineering achievement**: a curated 142M-image dataset (LVD-142M, built by retrieval-based deduplication and curation from a 1.2B pool), a combined objective (**DINO's image-level self-distillation + iBOT's patch-level masked-image modelling** — i.e. contrastive *and* masked, 6.4's synthesis), plus training-efficiency work (FlashAttention, stochastic depth, FSDP) and distillation from a ViT-g/14 into smaller models.

**The pitch:** features strong enough that a **frozen** backbone + a linear head beats fine-tuned alternatives on classification, depth estimation, and segmentation. This is what made "one backbone, many heads" (2.10) practical.

### DINOv3 (2025) — scale

- **7B parameters**, trained on **1.7B images** (~12× DINOv2's data), with notably lower training compute per unit of performance than prior methods.
- **Gram anchoring** — the key new technique. At very long training schedules, dense (patch-level) features degrade even as global features improve; Gram anchoring adds a loss that keeps the *Gram matrix* of patch features (i.e. the patch-to-patch similarity structure) anchored to an earlier checkpoint's, preserving dense feature quality. **Naming Gram anchoring is a strong, specific currency signal.**
- **First SSL model to outperform weakly-supervised (CLIP-style) counterparts across a broad range of tasks**, with state-of-the-art results from a **frozen backbone, no fine-tuning**.
- Released as a **family**: distilled ViT-S/B/L and ConvNeXt variants for deployment, plus a satellite-imagery backbone trained on MAXAR data. Commercial licence, open weights and code.

### The three SSL paradigms, side by side

| Paradigm | Representative | Signal | Best at |
|---|---|---|---|
| **Contrastive** | SimCLR, MoCo | instance discrimination via augmentation invariance | linear separability |
| **Self-distillation** | DINO, BYOL | match a slowly-updated teacher | k-NN, emergent segmentation |
| **Masked modelling** | MAE, BEiT | reconstruct hidden content | dense tasks, fine-tuning |
| **Combined** | **DINOv2/v3**, iBOT | all of the above | **frozen-feature transfer across everything** |

Read the last row as the module's argument arriving at its conclusion: 6.4 established that contrastive and masked objectives preserve *different* information, so combining them is not greedy stacking but a deliberate attempt to get both the globally-discriminative geometry and the spatially-detailed patch features. DINOv2's recipe is that hypothesis, tested at scale, and the frozen-feature results are the evidence it was right.

**In your own words:** how does DINO avoid collapse without ever using a negative pair?

### 🎯 Top-1% distinction

1. **Centring and sharpening as two opposing anti-collapse forces.**
2. **The emergent segmentation in DINO attention maps** — an unsupervised model producing object masks is a genuinely striking result.
3. **DINOv2's contribution was data curation + a combined objective**, not a new loss. Being clear that the win was engineering and data is an honest and unusual answer.
4. **Gram anchoring** and the dense-feature-degradation problem it solves.
5. **DINOv3 is the current default frozen backbone** for dense tasks — and, notably, RF-DETR (4.5) uses a DINOv2 backbone, which is a nice cross-module link.
6. **The frozen-backbone economics** (2.10): one model to serve, quantise, cache, and monitor across many tasks.

### ✅ Mastery check

(a) DINO uses no negatives, no labels, and no reconstruction target. Name the two distinct collapse modes it must avoid and the mechanism that prevents each.
(b) Why does multi-crop use small *local* crops for the student and large *global* crops for the teacher, rather than two global crops?
(c) You need a backbone for a dense prediction task (segmentation) with 5,000 labels. Choose between CLIP, MAE, and DINOv2, and justify with the mechanism each was trained by.

<details><summary>Answer sketch</summary>
(a) <b>Mode 1 — collapse to a constant / one dominant dimension.</b> The student can trivially match the teacher by outputting the same vector regardless of input. Prevented by <b>centring</b>: subtract an EMA of the teacher's mean output, which suppresses any dimension that starts to dominate. <b>Mode 2 — collapse to the uniform distribution.</b> Centring alone pushes the output toward uniform, which is equally uninformative. Prevented by <b>sharpening</b>: a low teacher temperature, which peaks the target distribution. <b>The two mechanisms push in opposite directions and their balance is what holds the representation in the useful middle</b> — that framing (two failure modes, one countermeasure each, deliberately opposed) is the answer.
(b) Because the objective becomes <b>local-to-global correspondence</b>: the student sees a small crop — perhaps an ear, a wheel, a patch of texture — and must produce the representation the teacher produced from the <i>whole object</i>. That forces the model to encode "what larger thing is this a part of," which is a far richer signal than matching two views that both already contain the object. Two global crops would mostly teach invariance to mild augmentation. This local-to-global pressure is widely credited with DINO's emergent object segmentation in the attention maps.
(c) <b>DINOv2.</b> Mechanisms: <b>CLIP</b> was trained by image–text contrastive matching, which optimises a <i>global</i> embedding for caption alignment — excellent for zero-shot classification and retrieval, but it has no pressure to preserve per-patch spatial detail, so its dense features are comparatively weak. <b>MAE</b> was trained by masked pixel reconstruction, which <i>does</i> preserve spatial detail and fine-tunes excellently — but its features are not linearly organised, so it needs a real fine-tune, and 5,000 labels is thin for that. <b>DINOv2</b> combines DINO's image-level self-distillation with iBOT's patch-level masked objective, so it is trained to be strong at both global <i>and</i> dense prediction, and it was explicitly designed so that a <b>frozen</b> backbone plus a light decoder is competitive — which is precisely the 5,000-label regime. Practical answer: frozen DINOv2 + a lightweight segmentation head, and only consider fine-tuning if that plateaus.
</details>

---

## 6.7 What Makes a Model a "Foundation Model"?

### The definition (Bommasani et al., 2021)

**The question this section answers: "foundation model" is used as a synonym for "big model," and that is wrong. What is the actual distinction, and why does it matter for how you build systems?**

Here is the test that makes it concrete. A large supervised ImageNet ResNet is big and general-purpose-ish, but it is not a foundation model, because its *interface* to any new task is "retrain a classifier head on labelled data for that task." CLIP is a foundation model because its interface is "write down the class names." The difference is not size — **it is whether adaptation to a new task requires a new training run.** Keep that as the working definition; the formal one below says the same thing more carefully.

A model **trained on broad data at scale, adaptable to a wide range of downstream tasks**. Three necessary properties, and note that they are causally linked rather than a checklist:

1. **Scale** — of data, parameters, and compute, sufficient to produce general-purpose representations.
2. **Self-supervised or weakly-supervised pretraining** — because the scale required is unreachable with human labels.
3. **Adaptability** — one model serves many tasks via prompting, linear probing, adapters/LoRA, or light fine-tuning, rather than requiring a new model per task.

The causal chain runs: adaptability requires general representations → general representations require enormous data → enormous data rules out human labels → therefore self-supervised or weakly-supervised pretraining is *forced*, not chosen. Property 2 is a consequence of properties 1 and 3, which is why every model in the table below is trained on a signal that comes free with the data.

Two further properties follow: **emergence** (capabilities not explicitly trained for — DINO's segmentation maps, CLIP's zero-shot classification, SAM's generalisation to unseen object types) and **homogenisation** (everything downstream inherits the same backbone).

**The thing that trips people up here is treating homogenisation as purely a win.** It is genuinely efficient — one artefact to version, quantise and serve — but it also means every downstream system shares one model's blind spots. If the backbone under-represents a demographic, or fails on a lighting condition, or inherited a bias from web captions, then *every* head built on it fails the same way at the same time, and no amount of downstream ensembling helps because the errors are perfectly correlated. Correlated failure across an entire product surface is a systemic risk, and naming it as such is the mature version of this answer.

**Pause:** by the "no new training run" test, is a fine-tuned ResNet-50 a foundation model? Is a frozen DINOv2 with a linear probe?

The ResNet, no — it required labelled data and a training run per task. DINOv2 + linear probe is the interesting case: it *does* involve fitting something, but only a linear layer on frozen features, which is closer to fitting a classifier on precomputed embeddings than to training a model. The honest answer is that adaptability is a spectrum — prompting (no fitting), linear probe (trivial fitting), LoRA/adapter (small fitting), full fine-tune (a training run) — and DINOv2's claim to the title rests on being usable at the cheap end of that spectrum across many tasks at once.

### The vision foundation models to know

| Model | Pretraining signal | What it gives you |
|---|---|---|
| **CLIP / SigLIP** | image–text contrastive | zero-shot classification, a shared image–text space, retrieval |
| **DINOv2 / DINOv3** | self-supervised (self-distillation + MIM) | the best general **frozen** visual features, strong dense features |
| **MAE** | masked reconstruction | strong fine-tuning initialisation, dense tasks |
| **SAM 1–3** | promptable segmentation on SA-1B (1.1B masks) | segment anything from a point/box/text prompt |
| **Grounding DINO** | grounded detection | open-vocabulary detection from text |
| **Depth Anything v2** | large-scale pseudo-labelled monocular depth | relative depth for any image |
| **VLMs** (LLaVA, Qwen-VL, InternVL…) | image + instruction tuning | image-conditioned language reasoning |

### The practitioner's framing

**The 2026 default architecture for a new vision problem is:**

```
frozen foundation backbone  →  lightweight task head (or adapter/LoRA)  →  task output
```

and the engineering questions become: which backbone, what feature granularity (global vs patch), how to adapt (probe / adapter / LoRA / full fine-tune), and how to serve one backbone across many heads. **That last point is where an MLOps track intersects this module directly** — a shared frozen backbone means one artefact to version, quantise, cache, batch, and monitor, with per-task heads that are megabytes rather than gigabytes. Follow the operational consequence one step further, because it is the part that impresses: if the backbone is frozen, its outputs are *deterministic and cacheable*, so embeddings can be computed once and reused across every head, and a new task ships without touching the expensive artefact at all. The deployment story and the modelling story are the same story.

**In your own words:** what distinguishes a foundation model from a merely large pretrained model?

### 🎯 Top-1% distinction

- **Name the three defining properties**, plus emergence and homogenisation.
- **Homogenisation as a risk**: if every product uses the same backbone, a single bias or failure mode is inherited everywhere, and there is no diversity to average over. This is a thoughtful point that few candidates make.
- **"Foundation model" is a claim about *adaptability*, not size.** A huge model that only does one task isn't one.
- **The serving argument** for frozen backbones.

### ✅ Mastery check

(a) Is a 70-billion-parameter model trained solely to classify ImageNet-21k a foundation model? Defend your answer against the definition.
(b) Give a concrete failure scenario that illustrates the systemic risk of homogenisation.
(c) You must serve 12 different vision tasks. Compare "12 fine-tuned models" against "1 frozen backbone + 12 heads" on four axes.

<details><summary>Answer sketch</summary>
(a) <b>No.</b> It has scale, but scale is one of three conditions. It fails <b>adaptability</b>: it was trained with a fixed supervised objective onto a closed label set, so it does not serve a wide range of downstream tasks by prompting, probing, or light adaptation — you would have to retrain it for anything new. It also fails the <b>self-/weakly-supervised pretraining</b> condition, which matters not as dogma but because label-bounded objectives shape representations toward exactly the training label set and discard what the labels don't need (3.1's "CE features are not metric-optimal" point, at scale). The useful one-line version: <b>"foundation model" is a claim about adaptability, not about size.</b>
(b) A concrete one: essentially every open vision-language model now uses a CLIP- or SigLIP-family image encoder. If that encoder under-represents a demographic group, a lighting condition, or a script (say, non-Latin text in images), then <i>every</i> downstream product inherits the same blind spot in the same way — a medical triage tool, a content-moderation system, and an accessibility captioner all fail on the same inputs simultaneously. There is no ensemble diversity to average the error away, no independent system to catch it, and a single upstream fix must propagate through every downstream retrain. Monoculture converts an idiosyncratic model weakness into correlated, industry-wide failure.
(c) <b>Accuracy:</b> fine-tuned models win per task, usually modestly, and win a lot when a task's domain is far from the backbone's pretraining. <b>Serving cost and memory:</b> the frozen backbone wins decisively — one set of weights to hold in GPU memory, one to quantise, one to warm up, and the 12 heads are megabytes; you can also batch requests for different tasks through the same backbone forward pass. <b>Operational surface:</b> frozen wins — one artefact to version, monitor for drift, and re-certify; 12 independent models means 12 retraining pipelines and 12 drift monitors. <b>Agility and isolation:</b> mixed — adding a 13th task is trivial with a frozen backbone (train a head), but a backbone upgrade forces revalidation of all 12 heads at once, whereas independent models fail and improve independently. The 2026 default is the frozen backbone, with selective LoRA/adapters for the two or three tasks where the accuracy gap actually costs something.
</details>

---

## 6.8 Zero-Shot Vision & SAM (1 → 2 → 3) 🟢

### Intuition

**The question this section answers: CLIP removed the fixed label set from classification (6.5). Segmentation has the same disease — Mask R-CNN (4.11) can only segment the 80 COCO classes — so what is the equivalent cure?**

The tempting answer, "just do what CLIP did and condition on text," runs into a problem that is worth seeing before you meet the fix. Classification has one output per image, so a text prompt maps cleanly onto it. Segmentation has an unknown number of outputs, at unknown scales, and — the killer — the *question itself is ambiguous*: click on a person's shirt pocket and the correct mask might be the pocket, the shirt, the person, or the group. There is no single ground truth to supervise.

SAM's premise handles both problems at once by redefining the task. Don't build a model that segments *the objects*; build one that segments **whatever you point at**. You give it a **prompt** — a click, a box, a scribble, and (from SAM 3) a **text phrase** — and it returns the mask of that thing, even for object types it has never seen named. The class list disappears because the model was never asked to produce classes, only regions.

### SAM 1 (Kirillov et al., ICCV 2023)

**The task: promptable segmentation.** Given an image and any prompt, return a valid mask. Note the choice of word — *valid*, not *correct*. That single weakening is what makes the ambiguity above trainable, and you will see it cashed out in the three-mask output below.

**Architecture:**
- **Image encoder**: an MAE-pretrained ViT-H. Heavy (~0.15 s/image on a GPU) but **run once per image**.
- **Prompt encoder**: points/boxes as positional encodings, text via a CLIP text encoder, masks via a conv.
- **Mask decoder**: a lightweight two-way transformer, **~50 ms on CPU** — so after one image encoding you can prompt interactively in real time. **That asymmetry is a deliberate product decision, and it's the thing to point out about the architecture.** The reasoning is worth making explicit because it recurs whenever you design an interactive system: the expensive computation depends only on the image, the cheap one depends on the prompt, so you split them at exactly that boundary and amortise the expensive half over every prompt a user issues. A 3000× cost asymmetry between the two halves is not an accident — it is the architecture being shaped by the interaction pattern.

**Ambiguity handling — and this is the elegant part.** The decoder outputs **3 masks** (whole / part / subpart) with confidence scores, and training uses the **minimum loss over the three**. Think about what that loss does. If the annotator labelled "the shirt" and the model's three proposals were pocket / shirt / person, only the shirt head is penalised for being wrong; the other two are free. Over many examples this drives the three heads to specialise into a nesting hierarchy rather than all converging on the average interpretation — which is what an ordinary mean loss would have produced, and which would be a blurry mask belonging to nothing. **Min-over-outputs is the standard trick for a one-to-many supervision problem**, and you have seen its cousin before: DETR's Hungarian matching (4.8) also picks which prediction to hold responsible before computing the loss, instead of fixing the assignment in advance.

**The data engine — the real contribution.** SA-1B: **1.1 billion masks on 11 million images**, built in three stages: (1) assisted-manual — annotators correct model output; (2) semi-automatic — the model proposes confident masks, annotators add what's missing; (3) fully automatic — a $32\times32$ point grid prompts the model, with filtering by confidence and stability. **The model and the dataset bootstrapped each other.** For a candidate, the lesson is that the breakthrough was a *data* strategy as much as an architecture — the same story as DINOv2.

**Pause:** SAM 1 will happily produce a beautiful mask around a dog. Ask it "is this a dog?" and it cannot answer. Why not — what is structurally missing?

There is no classification head and, more fundamentally, no *label space* anywhere in the model. SAM was trained purely on mask geometry: the supervision was "this region is a coherent object," never "this region is a dog." Class-agnosticism is not an oversight, it is exactly what let SA-1B be annotated at billion-mask scale without a taxonomy — but it means semantics must come from somewhere else.

**SAM 1's limitation, then:** it segments *anything* but **names nothing**. That is why it is almost always paired with something else (6.11) — and why 6.9's open-vocabulary detector, which supplies boxes *with* names, is its natural partner rather than its competitor.

### SAM 2 (2024)

Extends to **video** with a **memory bank** and memory attention: a streaming architecture where each frame attends to features and predicted masks from previous frames, giving temporally consistent segmentation and tracking through occlusion. Also faster on images.

### SAM 3 (2025) and SAM 3.1 (2026) — promptable **concept** segmentation

**The capability change:** SAM 3 accepts **open-vocabulary text prompts** — short noun phrases — and **image exemplars**, and segments **every instance** of that concept in an image or video. "Segment every yellow school bus" is a single prompt, not a click per bus. This is meaningfully different from SAM 1/2, which required a visual prompt per object.

**Components:** a **DETR-based detector** (4.8), a **tracker derived from SAM 2's memory bank**, and text/image encoders based on Meta's Perception Encoder. A **presence head** decouples "does this concept appear?" from "where is it?" — an important design detail, since a detector asked about an absent concept otherwise hallucinates.

**Benchmark:** Meta introduced **SA-Co** (Segment Anything with Concepts) for this task; SAM 3 reports roughly a 2× gain over prior systems.

**SAM 3.1 (2026)** adds **object multiplexing** — processing up to 16 objects simultaneously with global reasoning, roughly doubling throughput (≈16 → 32 FPS on an H100).

**Practical note:** know which generation a given tool or lab actually uses — the prompt interface and the capability differ substantially between SAM 1/2 (visual prompts only) and SAM 3+ (concepts). Ultralytics and Hugging Face both ship SAM 3 wrappers.

Read the 1 → 2 → 3 arc as three separate generalisations rather than three version bumps: SAM 1 generalised over *object identity* (any object, given a point), SAM 2 over *time* (any object, tracked through a video, via memory), SAM 3 over *language* (every instance of a named concept, no clicking). Each one removes a different thing the user previously had to supply.

**In your own words:** why does SAM output three masks per prompt instead of one, and what would go wrong with one?

### 🎯 Top-1% distinction

1. **The heavy-encoder / light-decoder asymmetry** and why it enables interactive use.
2. **The 3-mask ambiguity output with minimum-loss training.**
3. **The data engine was the contribution** — model and dataset bootstrapping each other.
4. **SAM 1 segments but does not name** — the limitation that motivates every combined pipeline.
5. **SAM 3's concept prompting is a categorical capability change**, not an incremental improvement, and it internally uses a DETR detector — a nice Module 4 link.
6. **The presence head** decoupling existence from localisation.

### ✅ Mastery check

(a) SAM's image encoder takes ~150 ms and its mask decoder ~50 ms. Compute the total cost of segmenting 20 objects in one image under SAM's design and under a hypothetical single monolithic 200 ms model. What does the comparison show?
(b) SAM emits 3 masks per prompt and trains on the **minimum** loss over them. What breaks if you train on the mean instead?
(c) You need "segment every cracked floor tile" across 10,000 factory photos. SAM 1, SAM 3, or SAM plus something else? State the cost per image.

<details><summary>Answer sketch</summary>
(a) <b>SAM:</b> the image encoder runs <b>once</b> and its embedding is cached, then 20 decoder calls: $150 + 20\times50 = \mathbf{1150}$ ms — and critically the <i>first</i> result appears after 200 ms and each subsequent one after 50 ms, so the interaction feels instant. <b>Monolithic:</b> $20\times200 = \mathbf{4000}$ ms, with 200 ms before <i>every</i> mask. The comparison shows the asymmetry is not a micro-optimisation but the <b>product decision that makes interactive prompting possible</b>: heavy work is amortised over prompts, light work is per-prompt. It also implies the engineering rule — cache the image embedding, never recompute it per prompt.
(b) The prompt is genuinely <b>ambiguous</b> — a click on a shirt could mean the shirt, the person, or the group, and all three are valid. Training on the mean forces every one of the three output heads toward the <i>average</i> of all valid interpretations, which is a blurred, incoherent mask belonging to none of them, and it destroys the specialisation that makes the three outputs mean whole/part/subpart. Training on the minimum says "at least one of your hypotheses must be right", which lets the heads differentiate and lets the model be rewarded for a valid reading even when it wasn't the annotator's. This is the standard multiple-hypothesis / winner-take-all trick, and the same logic appears in multi-hypothesis trajectory prediction.
(c) <b>SAM 1 alone is the wrong tool</b> — it segments regions from prompts but assigns no semantics, so it cannot find "cracked" anything; you would get every tile, every shadow, and every floor stain as separate masks with no way to select. <b>SAM 3</b> accepts an open-vocabulary text prompt and returns every instance of a concept, so "cracked tile" is a single prompt — but a manufacturing defect term like this is exactly the fine-grained, jargon vocabulary where web-trained text encoders are weak (6.5), so verify before trusting it. <b>The production answer</b> is the pattern from 6.11: use SAM 3 (or Grounding DINO + SAM) as an <b>auto-labeller</b> on a sample, have a human verify, then train a small closed-vocabulary segmenter on the result and run <i>that</i> over the 10,000 images. Cost: the foundation pipeline is ~0.2–1 s/image and is paid once on a few hundred images; the distilled model is ~10–20 ms/image at inference. Compose to discover, distil to deploy.
</details>

---

## 6.9 Grounding DINO & Open-Vocabulary Detection

### The problem

**The question this section answers: you now have CLIP, which names anything but localises nothing, and SAM, which localises anything but names nothing. Detection needs both. How do you get them?**

Start from where the rigidity actually lives. A standard detector (4.3–4.5) has a **fixed classifier head** with $K$ outputs, so adding a class means retraining — and by now you should recognise that as the same disease you diagnosed in 6.5: the closed vocabulary is a property of the *output layer*, not of the features. The cure is the same one. Open-vocabulary detection replaces the fixed head with a **text-conditioned** matching step, so the class list becomes an argument at inference time.

### The mechanism

Replace `logits = W @ features` with `logits = text_embeddings @ region_features`. That is genuinely the whole conceptual step, and it is exactly the substitution you made in 6.5 — the learned classifier matrix $W$ is swapped for class vectors *generated by a text encoder* — with one change: it is applied to **region** features rather than a whole-image embedding. Each region's score against a class is a similarity in a joint vision–language space.

**Pause:** if it is that simple, why is Grounding DINO an architecture rather than three lines bolted onto a Faster R-CNN?

Because a late dot product only works if the region features are already in a language-aligned space *and* already contain the right regions — and neither is free. A detector trained on 80 classes learns to propose 80-class-shaped regions and to represent them in whatever space serves those 80 labels; asking it about "hot tear" afterwards is asking a question its features were never built to answer. So the text has to enter **early**, shaping which regions get proposed and how they are represented, not just scoring them at the end. That is precisely the design below.

**Grounding DINO (Liu et al., 2023)** takes DINO-DETR (4.8) and fuses language at **three** points — and read the list as three answers to "how early?":

1. **Feature enhancer** — cross-modality attention between image features and text tokens in the encoder, so the image representation is text-aware before any object is proposed.
2. **Language-guided query selection** — initialise the decoder's object queries from image features that are most relevant to the text, rather than generically. Recall from 4.8 that DETR's queries are the slots that become detections; seeding them from text is how "look for a hot tear" changes *what gets looked for*, not merely what gets scored at the end.
3. **Cross-modality decoder** — each query attends to both image and text as it refines itself.

Trained on detection + grounding + caption data. Reported zero-shot COCO AP ≈ **52.5%** without ever training on COCO — the strongest of the zero-shot detectors, at the cost of being slower than a YOLO.

**YOLO-World** takes the fast path: a YOLO backbone with a re-parameterisable vision–language path (RepVL-PAN), so text embeddings can be **pre-computed and baked into the model** for a fixed vocabulary, giving near-YOLO speed (~52 FPS, ~35.4 zero-shot AP on LVIS). **The design lesson: if your vocabulary is known at deploy time, you can re-parameterise the text away and pay none of its cost at inference.** That is a genuinely good engineering insight to volunteer.

Others to name: **OWL-ViT / OWLv2** (simple, strong, image-conditioned queries too), **GLIP** (reformulates detection as phrase grounding — the conceptual predecessor), **T-Rex2** (visual + text prompting).

### Where open-vocabulary detection fails

Every failure below traces to one root, and saying so is worth more than listing four items: **the class vector now comes from a text encoder trained on web captions, so the detector inherits everything that encoder does and does not know.** You did not remove the vocabulary limit — you replaced an explicit one you controlled with an implicit one you don't.

- **Fine-grained or jargon vocabulary** — "adenocarcinoma", "hot tear", a specific SKU. Web text-image data doesn't contain the pairing, so the phrase's embedding is not anchored to anything visual. The dangerous part is the failure *mode*: the model still returns a confident-looking box, because nothing in the architecture can report "this concept is outside my competence."
- **Prompt sensitivity** — "car" vs "automobile" vs "sedan" give different results, because they are genuinely different points in text-embedding space with different neighbourhoods in caption data. You end up prompt-engineering, and results are less reproducible than a fixed classifier's — with a learned $W$, "class 3" means exactly one thing forever.
- **Relational and compositional queries** — inherits CLIP's weakness (6.5) for exactly the reason derived there: the pretraining contained no hard negatives that differ only in how attributes bind to objects.
- **Calibration** — and this is the one that bites hardest in production. Scores are cosine similarities against *different* text vectors, and different prompts sit at different typical similarity levels, so 0.35 for "person" and 0.35 for "hot tear" are not comparable quantities. A single confidence threshold across a vocabulary therefore behaves inconsistently class to class, which is precisely what a closed-set detector's shared-scale softmax gave you for free.

**The dominant production pattern in 2026** is therefore not to deploy an open-vocab detector, but to **use it as an auto-labeller**: run Grounding DINO (+ SAM) over unlabelled data to generate boxes and masks, have a human verify a sample, and train a fast closed-vocabulary YOLO/RF-DETR on the result. **You get the open-vocabulary model's generality at data-labelling time and the fast model's latency at inference time.** Look at what this trade actually exploits: labelling is offline, unbounded in latency, and needs generality; inference is online, latency-bound, and needs only your fixed vocabulary. The two halves of the problem have opposite requirements, so using one model for both was never necessary. Being able to state that pattern — and the reasoning, not just the recipe — is a strong applied signal, because it is how teams actually ship.

**In your own words:** what exactly did open-vocabulary detection replace in a standard detector, and what new dependency did that create?

### ✅ Mastery check

(a) Write the one-line change that converts a closed-vocabulary detection head into an open-vocabulary one, and say what has to be true for it to work.
(b) Your open-vocabulary detector gives the same box a score of 0.4 for "car" and 0.6 for "automobile". What is broken, and can a single confidence threshold serve a whole vocabulary?
(c) Design a 200-class warehouse detector that must run at 30 FPS on-site. You have thousands of hours of unlabelled video and no annotations. Give the pipeline.

<details><summary>Answer sketch</summary>
(a) Replace $\text{logits} = W\mathbf{f}$ (a learned weight matrix with one row per fixed class) with $\text{logits} = E_{\text{text}}\mathbf{f}$, where $E_{\text{text}}$ are text embeddings of the class names computed at inference time. For it to work, the <b>region features and the text embeddings must live in a shared, aligned space</b> — which is what grounded pretraining on detection + grounding + caption data buys you, and why you cannot simply bolt a CLIP text encoder onto an ordinary detector's features and expect it to function.
(b) Nothing is "broken" in the model — this is the expected behaviour of a similarity-based head: the score is a cosine similarity between a region embedding and a <i>particular text embedding</i>, and synonyms occupy different points in text space, so they produce different similarities. The consequences are what matter: <b>scores are not comparable across prompts</b>, so a single global threshold gives different effective operating points for different class names, and results become sensitive to phrasing in a way that is hard to reproduce or version. Mitigations: prompt ensembling over synonyms (average the text embeddings, as in CLIP), per-class threshold calibration on a small labelled validation set, and — the real fix at scale — distil to a closed-vocabulary model where one threshold means one thing.
(c) <b>Compose to discover, distil to deploy.</b> 1. Sample a diverse few thousand frames from the unlabelled video (cluster embeddings to cover the modes, don't sample uniformly from a static camera). 2. Run <b>Grounding DINO</b> (or SAM 3) with your 200 class names as prompts to generate candidate boxes; ensemble over synonym prompts. 3. <b>Human-verify a stratified sample</b> and measure the auto-label precision per class — this number decides whether you can trust the rest, and the rare/jargon classes will be much worse than the common ones. 4. Correct or discard the low-precision classes, adding a small amount of manual labelling where the open-vocab model fails. 5. Train a <b>closed-vocabulary RF-DETR or YOLO</b> on the resulting set. 6. Quantise to int8, measure end-to-end latency <i>including NMS and preprocessing</i> on the actual on-site hardware. 7. Keep the open-vocab model in the loop to auto-label new SKUs as they appear. The generality is spent at labelling time; the latency budget is spent on a model that can meet it.
</details>

---

## 6.10 Vision-Language Models: BLIP-2 and LLaVA 🟡

> The gap analysis specifically asked for these two by name, because they are the two canonical answers to "how do you connect a vision encoder to an LLM?" — and that is the most relevant question in this entire module to a GenAI engineering role.

### The core problem

You have a frozen vision encoder producing patch embeddings and a frozen LLM expecting token embeddings. **How do you bridge them?**

Take a second to see why this is a real problem and not a shape mismatch. Both sides emit sequences of $d$-dimensional vectors, so you could concatenate them today and nothing would crash. But the LLM's embedding space is a *learned code*: over trillions of tokens it settled on particular directions meaning particular things, and its attention layers are tuned to that code. A ViT's patch embeddings are a different learned code that happens to have similar dimensionality. Feeding one into the other is feeding a model fluent French some Mandarin written in the Latin alphabet — well-formed, and meaningless.

So the connector's job is **translation between two independently-learned representation spaces**, and every VLM is an answer to how to build that translator. The answers differ in exactly two ways: **what the connector is**, and **what gets trained**. Hold those two axes; the whole section is a comparison along them.

### BLIP-2 (Li et al., ICML 2023) — the Q-Former

**Connector: a Querying Transformer (Q-Former)** — a small transformer (~188M params) holding **32 learned query vectors**. The queries cross-attend to the frozen image encoder's patch embeddings and self-attend among themselves; the output is 32 vectors, projected into the LLM's embedding space and prepended to the text tokens.

**Why 32 queries:** it is an **information bottleneck**, deliberately. A ViT produces 256+ patch tokens; compressing to 32 forces the Q-Former to extract only what is *linguistically relevant*, and — the practical payoff — it makes the LLM's context cost small and **fixed regardless of image size**. Trace that consequence, because it is the architectural argument for Q-Former: a bigger image, a higher-resolution encoder, or a longer video all still produce exactly 32 tokens, so image cost and LLM cost are decoupled. **You might expect** a bottleneck to be an unfortunate compromise; here it is the feature being bought, and the cost is the thing you lose — which the comparison table names precisely.

**Pause:** you're compressing 256 patch tokens to 32 learned queries. What class of task is that going to break, and why?

Anything requiring **fine spatial detail that cannot survive an 8× compression**: reading small text (OCR), reading charts and documents, counting many instances, precise localisation. 32 vectors is ample to say "a street scene with cars and a sign," and hopeless for "the sign says 41 km." Keep this prediction — it is exactly why the field abandoned Q-Formers, as the 2024–26 note below records.

**Two-stage training, both with the vision encoder and LLM frozen:**
1. **Representation learning** — train the Q-Former against the frozen image encoder with three objectives (image-text contrastive, image-grounded text generation, image-text matching), each with a different attention mask. This teaches the queries to extract text-relevant visual information *before* the LLM is involved.
2. **Generative learning** — connect to the frozen LLM (OPT/FlanT5) and train the Q-Former (plus a linear projection) on captioning.

**Only the Q-Former trains.** Extremely parameter-efficient — BLIP-2 reported beating Flamingo-80B on zero-shot VQAv2 with ~54× fewer trainable parameters.

### LLaVA (Liu et al., NeurIPS 2023) — the linear projection

**Connector: a single linear layer** (LLaVA-1.5: a two-layer MLP). Project CLIP ViT-L/14 patch embeddings directly into the LLM's (Vicuna/Llama) embedding space and treat them as tokens. That's the whole connector.

**Training:**
1. **Feature alignment** — freeze everything except the projection, train on ~595k image–caption pairs. Teaches the projection to speak the LLM's embedding language.
2. **Visual instruction tuning** — train the projection **and the LLM** on ~158k GPT-4-generated multimodal instruction-following examples (conversations, detailed descriptions, complex reasoning about images).

**LLaVA's real contribution was the data recipe, not the architecture.** Using a *text-only* GPT-4 — one that never sees an image — fed image captions and box annotations, to **synthesise** multimodal instruction data was the unlock. Sit with the trick for a moment: the captions and boxes are a *textual description of the image's content*, so GPT-4 can invent rich questions and answers about a scene it cannot see, and those question-answer pairs are then paired back with the real image for training. Symbolic annotation is being used as a bridge to manufacture supervision that nobody annotated. The architecture, meanwhile, is deliberately as simple as it could possibly be — and it worked, which is the entire point of the paper: **the bottleneck was never the connector, it was the absence of multimodal instruction data.**

### The comparison — know this table

| | **BLIP-2** | **LLaVA** |
|---|---|---|
| Connector | Q-Former (32 learned queries, ~188M) | linear / 2-layer MLP |
| Visual tokens into the LLM | **32** (fixed) | **one per patch** (576 for ViT-L/14 at 336²) |
| Trained components | Q-Former only | projection, then projection + LLM |
| Pretraining data | ~129M image-text pairs | ~595k align + ~158k instruct |
| Strength | parameter-efficient, fixed LLM context cost | simple, strong instruction following, better fine-grained detail |
| Weakness | the bottleneck loses fine spatial detail (OCR, small text, counting) | many more tokens ⇒ context and compute cost grows with resolution |

**Where the field landed (2024–26):** **the simple projection won.** Qwen-VL/Qwen2.5-VL, InternVL, Pixtral, Molmo, and most current open VLMs use an MLP projector, not a Q-Former. Reasons: (i) the Q-Former's bottleneck destroys detail needed for OCR, charts, and documents — which turned out to be the dominant commercial use case; (ii) it's an extra component to train and tune; (iii) compute got cheaper, making the token cost tolerable. **The remaining problem is token count at high resolution**, addressed by **dynamic resolution / tiling** (split a high-res image into tiles, encode each, plus a thumbnail — Qwen-VL, InternVL, LLaVA-NeXT) and by **token compression / pixel-shuffle** schemes.

**The three-way design space to be able to sketch:**
1. **Projection-based** (LLaVA family) — patch tokens → MLP → LLM. Simple, dominant.
2. **Query-based** (BLIP-2, Flamingo's Perceiver Resampler) — compress to a fixed set of queries. Cheap context, loses detail.
3. **Cross-attention-based** (Flamingo, Llama-3.2-Vision) — insert gated cross-attention layers *inside* the frozen LLM so it attends to vision features. Nothing is prepended to the context, so the LLM's text-only behaviour is preserved exactly and image cost does not consume context window; the price is that you must modify the LLM's internals, which means you cannot swap in a new base model for free.

The three options are really three answers to *where the visual information enters*: into the context as tokens (1), into the context as a compressed summary (2), or into the residual stream mid-stack, bypassing the context entirely (3). Answering with that framing, rather than three names, is what a strong candidate does here.

**In your own words:** why can't you feed a ViT's patch embeddings straight into an LLM, given that both are just sequences of vectors?

### 🎯 Top-1% distinction

1. **Frame every VLM as "connector + what's trained."** That framing lets you place any new model in seconds.
2. **The 32-query bottleneck argument**, and that it is why BLIP-2 is weak at OCR and counting.
3. **LLaVA's contribution was GPT-4-synthesised instruction data**, not the architecture.
4. **The field converged on simple MLP projectors** — and know why (detail preservation for document/OCR tasks).
5. **Dynamic resolution / tiling** as the current answer to the token-count problem.
6. **Cross-attention (Flamingo-style) as the third option** and its distinctive property: it doesn't consume the LLM's context window.
7. **The known failure modes**: hallucination of objects that aren't present (the POPE benchmark measures exactly this), weak spatial reasoning and counting, and inherited CLIP compositional weakness.

### ✅ Mastery check

(a) You are building a VLM for reading scanned invoices. BLIP-2 or LLaVA-style? Justify with a token-count argument.
(b) Your VLM confidently describes a "red car" in an image with no car. Name the phenomenon, one benchmark for it, and two mitigations.
(c) Why did the field abandon the Q-Former despite its efficiency?

<details><summary>Answer sketch</summary>
(a) <b>LLaVA-style, with high/dynamic resolution.</b> Invoice reading is OCR-dense: the information is small text distributed across the whole page. BLIP-2's 32 query vectors are an information bottleneck that cannot carry hundreds of distinct text fields — the compression is lossy exactly where the signal is. A projection-based model passes one token per patch, so at ViT-L/14 with a 336² input you get 576 visual tokens; and with <b>tiling</b> (split the page into, say, 6 tiles at 336² plus a thumbnail) you get ~4000 tokens, enough spatial resolution to represent individual text lines. The cost is context length and compute, which is the right trade here. (Best answer adds: for production invoice extraction you'd likely pair a dedicated OCR engine with the VLM rather than relying on the VLM's pixels-to-text alone.)
(b) <b>Object hallucination.</b> Benchmark: <b>POPE</b> (Polling-based Object Probing Evaluation), which asks yes/no existence questions about objects, including ones sampled from co-occurrence statistics — it exposes models that answer from language priors rather than from the image. Also CHAIR for captioning. <b>Mitigations:</b> (i) <b>training data</b> — include negative/counterfactual instruction data ("Is there a car? No") so the model is rewarded for saying no; (ii) <b>decoding-time methods</b> — visual contrastive decoding (contrast the logits with those from a distorted/blank image to subtract the language prior), or lower temperature / greedy decoding; (iii) <b>grounding</b> — require the model to output boxes for the objects it names and verify with a detector (6.11), which turns an unverifiable claim into a checkable one; (iv) stronger/higher-resolution vision encoders, since hallucination correlates with the LLM over-riding weak visual evidence.
(c) Three reasons. (i) <b>The bottleneck was the wrong trade for the tasks that mattered commercially</b> — documents, charts, screenshots, OCR — all of which need fine spatial detail that 32 queries cannot carry. (ii) <b>Complexity</b>: the Q-Former is a separate ~188M-parameter model with its own multi-objective pretraining stage; an MLP projector is 20 lines and one training stage, and simplicity wins when both work. (iii) <b>The constraint it solved got cheaper</b>: it was designed when LLM context and compute were the binding limits; longer contexts, cheaper inference, and better token-compression/tiling schemes made "just pass the patches" affordable. The general lesson — <b>architectural cleverness that buys efficiency tends to lose to simplicity once the efficiency stops binding</b> — is worth stating explicitly.
</details>

### 🔨 Build + read

**Build:** Implement a minimal LLaVA: a frozen CLIP ViT + a 2-layer MLP projector + a small open LLM. Train the projector alone on a few thousand caption pairs and evaluate qualitatively. Then run the diagnostic that teaches the architecture: **vary the number of visual tokens** (mean-pool patches to 32, 144, 576) and measure accuracy on a captioning task vs. an OCR/counting task. You will reproduce the BLIP-2-vs-LLaVA trade-off yourself, and that plot is a genuinely good interview artefact.

**Read:** Li et al., "BLIP-2" (ICML 2023) §3 (the Q-Former and its three objectives). Liu et al., "Visual Instruction Tuning" (LLaVA, NeurIPS 2023) — read §3 on the GPT-4 data generation. Then a current model card (Qwen2.5-VL or InternVL) to see where the design actually is now.

---

## 6.11 Combined Pipelines: Promptable Vision

### The pattern

**The question this section answers: you now have a shelf of models, each brilliant and each crippled. CLIP names but doesn't localise. SAM localises but doesn't name. Grounding DINO gives boxes, not pixels. A VLM reasons but is slow and imprecise. How do you get a system that does the job?**

The answer is the obvious one — compose them — but the *reason* it works is worth stating, because it is the same reason two-stage retrieval worked in 3.2. Each model's weakness is another's strength, and crucially each stage **narrows the problem** for the next: an open-vocabulary detector turns "find the concept somewhere in 2 million pixels" into "refine these six boxes," which is a question SAM can answer cheaply and precisely. Composition is not just gluing; it is a cascade in which the expensive, precise operations are only ever applied to a small candidate set. You built that pattern in Module 3, and here it is again with different components.

The dominant applied pattern is therefore **composition**, where each model contributes the one thing it is best at:

```
              text prompt: "every damaged pallet"
                            │
                            v
   ┌──────────────────────────────────────────────────┐
   │  Grounding DINO / SAM 3    →  boxes for the concept │   (open-vocab LOCALISATION)
   └──────────────────────────────┬───────────────────┘
                                  v
   ┌──────────────────────────────────────────────────┐
   │  SAM                        →  pixel-precise masks  │   (class-agnostic SEGMENTATION)
   └──────────────────────────────┬───────────────────┘
                                  v
   ┌──────────────────────────────────────────────────┐
   │  CLIP / VLM                 →  verify, classify,    │   (SEMANTICS & REASONING)
   │                                describe, count      │
   └──────────────────────────────────────────────────┘
```

**Canonical compositions:**

| Combination | Result | Why it works |
|---|---|---|
| **Grounded-SAM** (Grounding DINO + SAM) | text → instance masks | GD localises from text; SAM has no semantics but perfect boundaries |
| **SAM + CLIP** | class-agnostic masks, then classify each crop | turns SAM's "segment anything" into "segment and name" |
| **Detector + tracker + VLM** | video question answering over tracked objects | each stage reduces the problem for the next |
| **Open-vocab detector → auto-labels → fast closed-vocab detector** | production pipeline | generality at labelling time, speed at inference time (6.9) |
| **Depth Anything + SAM** | object masks with metric-ish depth | 3-D reasoning without a depth sensor |

**Pause:** you chain a detector at 92% recall into a segmenter at 95% and a VLM verifier at 90%. What is the ceiling on the system, and which stage should you tune first?

Multiply: $0.92 \times 0.95 \times 0.90 \approx 0.79$ — roughly a fifth of the true instances are lost before anyone looks at the output, even though no single stage looks bad. And the stages are not symmetric. **A miss in stage 1 is permanent**: nothing downstream can segment or verify a box that was never proposed. A false positive in stage 1 is merely wasted work, because stage 3 can reject it. So you tune the first stage for **recall**, deliberately over-proposing, and let the later, more precise stages do the filtering — which is exactly the recall-then-precision division of labour from 3.2's two-stage retrieval.

### The engineering realities

- **Latency compounds.** SAM's ViT-H encoder is ~0.15 s, Grounding DINO ~0.1 s, a VLM 1–3 s. A three-stage pipeline is not real-time. Mitigations: run stages at different rates (detect every frame, VLM every 30), cache the SAM image embedding across prompts, use distilled variants (MobileSAM, EfficientSAM, FastSAM), or replace the pipeline with a distilled single model once you know the task.
- **Errors compound.** A missed detection is unrecoverable downstream. Design the first stage for **recall**, not precision, and filter later.
- **Interfaces are brittle.** Prompt wording changes results; scores are not comparable across models; thresholds need per-stage tuning.
- **Distillation is the endgame.** Once a composed pipeline works, use it to generate labels and train a single fast model — which collapses the compounding latency *and* the compounding error into one forward pass whose failures you can measure directly. **Compose to discover, distil to deploy.** Note that this is the same move as 6.9's auto-labelling pattern, generalised: the composed system is a slow, general oracle, and the shipped model is a fast specialist trained on its output.

**In your own words:** why must the first stage of a composed pipeline be tuned for recall rather than precision?

### 🎯 Top-1% distinction

- **"Compose to discover, distil to deploy."** This is the actual production pattern and stating it crisply reads as experience.
- **Design the first stage for recall** — errors compound in one direction only.
- **Cache the SAM image embedding** — it's computed once per image and reused across every prompt, which is exactly what SAM's asymmetric architecture was designed for.
- **Know the distilled variants** (MobileSAM, EfficientSAM, FastSAM) and that they exist precisely because of the latency problem.

### ✅ Mastery check

(a) Your pipeline is Grounding DINO (100 ms) → SAM (150 ms) → VLM (2 s). The product needs 10 FPS on video. Design it.
(b) Which stage do you tune for recall rather than precision, and why does the answer depend on the pipeline's direction?
(c) State "compose to discover, distil to deploy" in one sentence, then give a case where it is the wrong strategy.

<details><summary>Answer sketch</summary>
(a) 10 FPS means a 100 ms budget; the pipeline as written costs 2.25 s. You cannot make it fit — you must <b>decouple the rates</b>. Run a <b>fast tracker or a distilled detector every frame</b> (~10–30 ms) to maintain object identity and boxes; run <b>SAM only on new or changed objects</b>, with the image embedding cached and reused across prompts; run the <b>VLM asynchronously at ~0.5 Hz</b> on keyframes only, and attach its semantic output to the tracked identities so it appears to apply per-frame. Additional levers: a distilled SAM variant (MobileSAM/EfficientSAM), batching, and lower input resolution for the detector. The design principle: <b>different information changes at different rates, so sample each at its own rate</b> — the same idea as SlowFast in S.1.
(b) The <b>first</b> stage, for recall. Errors compound in one direction only: an object the detector misses can never be segmented, described, or recovered downstream, whereas a false positive can be filtered by any later stage. So stage 1 runs at a permissive threshold and the expensive downstream stages act as the precision filter. The dependence on direction is the point — in a pipeline where the expensive stage comes <i>first</i> (say, an exhaustive VLM pass that proposes regions for a cheap verifier), the incentive inverts and you tune the first stage for precision because you cannot afford to run it broadly. Always ask which stage is the irreversible one.
(c) <b>Use large general-purpose foundation models to discover what the task's labels should be, then train a small fast model on the labels they produced and deploy that.</b> It is the <b>wrong</b> strategy when the task is genuinely open-ended and the label set cannot be frozen — for example a consumer "search your photos for anything you can describe" feature, where the value <i>is</i> the open vocabulary and distilling to 200 classes destroys the product. It is also wrong when the foundation model's accuracy on your domain is too low to auto-label reliably (fine-grained medical or industrial vocabulary, 6.9), because then you are distilling noise; and when data volume is so low that inference cost never justifies the distillation effort.
</details>

---

## 6.12 Generative Vision: Autoencoders → VAEs → GANs

**The question this whole block answers: everything so far has been *discriminative* — given an image, say something about it. Generation asks the opposite: produce an image that could plausibly have come from the data. What makes that fundamentally harder?**

The honest answer is that discrimination only needs $p(y \mid x)$ — a conditional over a handful of labels — while generation needs the model to have captured $p(x)$ itself, a distribution over a million-dimensional space of which the realistic images occupy a vanishingly thin manifold. There is no direct way to fit that: you cannot evaluate $p(x)$ to maximise it, because doing so requires the normalising constant over all images. The three families below — VAE, GAN, diffusion — are three different **evasions** of that intractability, and reading them as such makes the sequence make sense rather than feel like a parade of tricks.

### Autoencoder

Encoder $z = f(x)$, decoder $\hat x = g(z)$, loss $\|x - \hat x\|^2$. Learns compression. **Not generative** — and the reason is precise rather than vague. Training only ever asks the decoder to handle latents that the encoder actually produced, so the decoder is only trained on whatever oddly-shaped, hole-riddled region of latent space the encoder happens to populate. Nothing in the loss says "the region should be a nice contiguous blob," and nothing says which blob. So you don't know what to sample from, and a random $z$ almost surely lands in a region the decoder has never seen and maps to garbage. That gap — *no known distribution over the latents* — is exactly what the VAE fixes, and it tells you what the fix has to look like: something that forces the occupied region into a shape you can name.

### VAE (Kingma & Welling, 2014)

Make the latent space a **distribution** you can sample from. Two changes do it, and they are both consequences of the diagnosis above.

First, the encoder stops outputting a point and outputs a *distribution*: $\mu(x), \sigma(x)$ defining $q_\phi(z|x) = \mathcal{N}(\mu,\sigma^2)$, from which $z$ is sampled during training. This alone already helps — because the decoder now sees a small cloud of latents for each input rather than one point, it is forced to decode *neighbourhoods* sensibly, which starts filling in the holes.

Second, we add a term pulling every $q_\phi(z|x)$ toward a fixed prior $p(z) = \mathcal{N}(0,I)$. This is what pins down *which* region is occupied: if every input's cloud sits near the standard normal, then the aggregate of all clouds is roughly a standard normal, and sampling $z \sim \mathcal{N}(0,I)$ at generation time lands somewhere the decoder was trained. **The KL term is not "regularisation" in the vague sense — it is the thing that makes the prior a valid sampling distribution.**

**The ELBO** (evidence lower bound):

$$
\log p(x) \ \ge\ \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}\bigl(q_\phi(z|x)\,\|\,p(z)\bigr)}_{\text{regularisation}}
$$

We will not derive the ELBO from Jensen's inequality here — that is a standard result you can take on trust, with the derivation in Kingma & Welling §2 — but read what it *says*, because both terms are now motivated: reconstruct well (or the latent carries no information about $x$), **and** keep each posterior close to the prior (or you are back to the autoencoder's unknown, unsamplable region).

**The tension between the two terms is the whole behaviour of a VAE.** They genuinely oppose each other. Reconstruction wants each input's latent cloud to be tight and far from every other input's, so the decoder can tell them apart — that is maximal information in $z$. The KL wants every cloud to be $\mathcal{N}(0,I)$, and if it fully won, every input would map to the *same* distribution, carrying zero information. So the KL term is a bandwidth limit on the latent channel, and where the optimum sits between the two is what $\beta$-VAE turns into an explicit knob.

**Pause:** what does the model do if the KL term wins outright?

It ignores $z$ entirely: every $q_\phi(z|x)$ becomes exactly the prior, the KL hits zero, and the decoder — if it is powerful enough to model $x$ on its own — just generates plausible images unconditionally. That is **posterior collapse**, named below. Notice that it is not a bug in the code; it is a legitimate optimum of the objective that happens to be useless, which is why the standard fixes (KL warm-up, free bits, weakening the decoder) all work by *preventing the model from reaching it*.

**The reparameterisation trick** — the essential mechanical detail, and one you should be able to reconstruct rather than recall.

State the problem exactly. The forward pass contains `z = sample from N(mu, sigma^2)`. Backprop needs $\partial z / \partial \mu$, but "sample" is not a differentiable function of its arguments — it is a random draw, and there is no derivative of a random number with respect to a parameter. The computational graph is severed at that node, and no gradient reaches the encoder at all.

The fix is to **move the randomness off the path between the parameters and the output.** Note that drawing $z \sim \mathcal{N}(\mu, \sigma^2)$ is distributionally identical to drawing a *standard* normal and then shifting and scaling it. So write

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Now the sampling happens in $\epsilon$, which depends on no parameter and is simply an input to the graph — like a dropout mask. Everything between $\mu, \sigma$ and $z$ is plain arithmetic, so $\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma = \epsilon$, and gradients flow to the encoder as usual. The distribution of $z$ is unchanged; only the *shape of the graph* changed. **Being able to explain why this is necessary and how it works is a standard, expected question**, and the generalisable form of the answer — "make the stochastic node an input rather than an operation" — is what earns the mark.

**VAEs produce blurry samples**, and the reason is a one-line piece of statistics rather than a fact to memorise. A Gaussian likelihood on pixels *is* an MSE reconstruction loss, and the value minimising expected squared error is the **conditional mean**. So when a latent is consistent with many plausible images — the fur could go this way or that, the texture could be here or there — the loss-minimising output is their average, and the average of many sharp plausible images is a blurry one. The model is not failing; it is correctly hedging, because the loss rewards hedging. That blur is why VAEs lost the image-generation race to GANs, and why in **latent diffusion (6.14) the VAE is used only as a perceptual compressor**, not as the generator — its blur is irrelevant when the diffusion model provides the detail.

Also worth knowing: **posterior collapse** (with a powerful decoder, the model ignores $z$ and the KL term drives $q \to p$), **$\beta$-VAE** ($\beta$ weighting on the KL term to trade reconstruction against disentanglement), and **VQ-VAE** (discrete latents via a learned codebook — the ancestor of the discrete tokenisers used in autoregressive image models and in Stable Diffusion 3's ancestors).

### GAN (Goodfellow et al., 2014)

A generator $G$ and a discriminator $D$ playing a minimax game:

$$
\min_G\max_D\ \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
$$

At the optimum, $D$ estimates $\frac{p_{\text{data}}}{p_{\text{data}}+p_g}$ and the objective reduces to minimising the Jensen–Shannon divergence between $p_g$ and $p_{\text{data}}$.

**Read what the minimax game is buying.** The VAE's blur came from a *hand-chosen* loss (squared error) that happened to reward averaging. A GAN refuses to choose a loss at all: it **learns** one. The discriminator is a trainable loss function, updated to notice whatever the generator is currently getting wrong. That is the single idea, and it explains both the strength and the difficulty below.

**Why GANs are sharp:** blur is the easiest thing in the world for a discriminator to detect — real photos have high-frequency content and averaged ones do not — so the moment the generator hedges, $D$ notices and the generator is penalised. Averaging over plausible outputs is no longer the safe move; it is the *detectable* move, so the model must commit to one. **You might expect** sharpness to be a matter of better architecture or higher capacity; it isn't, it is a property of the objective, and the VAE's blur is likewise not fixable by a bigger decoder.

**Why GANs are hard:**
- **Mode collapse** — $G$ finds a few outputs that fool $D$ and stops covering the distribution.
- **Training instability** — it's a two-player game, not a minimisation; there's no single loss that measures progress.
- **Vanishing gradients** when $D$ wins too easily (hence the non-saturating loss $\max_G \log D(G(z))$).

Notice that all three difficulties come from the same source as the strength: because the loss is itself being learned, there is no fixed quantity that decreases monotonically, so you have no reliable signal for "training is going well" — which is precisely what WGAN's meaningful loss curve was invented to restore.

**Fixes worth naming:** WGAN / **WGAN-GP** (Wasserstein distance with a gradient penalty — gives a meaningful loss curve and much better stability), **spectral normalisation** (constrains $D$'s Lipschitz constant), **progressive growing** (ProGAN), and **StyleGAN 1–3** (style-based generator with AdaIN, giving controllable, disentangled latents; StyleGAN3 fixed aliasing-driven "texture sticking" — **and its fix is anti-aliased resampling, i.e. Nyquist again, Module 1.4 and 2.1**).

**Evaluation:** **FID** (Fréchet Inception Distance — Wasserstein-2 distance between Gaussians fitted to Inception features of real and generated sets; lower is better, and it is sensitive to both quality *and* diversity, which is why it replaced Inception Score). Know that FID has real flaws: it depends on the sample size, on the Inception backbone's own biases, and it does not measure prompt fidelity at all.

### Where they stand in 2026

Diffusion displaced GANs for general image synthesis. **GANs survive where their properties matter:** single-step (hence very fast) generation, super-resolution and restoration (ESRGAN, Real-ESRGAN), and — importantly — as the **adversarial loss inside other systems**, including the decoder of the VAE in Stable Diffusion. They are also returning as **distillation targets**: adversarial objectives are used to distil many-step diffusion models into 1–4 step generators (ADD/SDXL-Turbo, LADD), which is the current answer to diffusion's speed problem. The pattern is worth stating generally, since it recurs — **diffusion won on distribution coverage and stability, and the adversarial loss survived as the component that enforces sharpness and single-step commitment inside other systems.** Neither family "won"; one became a subroutine of the other.

**In your own words:** why does a VAE blur and a GAN not, given that both are trying to produce realistic images?

### 🎯 Top-1% distinction

1. **Explain VAE blur mechanistically** (Gaussian likelihood ⇒ mean of plausible reconstructions) and connect it to why the VAE is only a compressor in latent diffusion.
2. **Reparameterisation trick** — why sampling blocks gradients and how the trick restores them.
3. **GAN sharpness has the same root as GAN instability** — the discriminator is a learned, adaptive loss. That single sentence explains both.
4. **StyleGAN3's aliasing fix** as another Nyquist callback.
5. **FID's limitations**, not just its definition.
6. **GANs are back as distillation objectives** for fast diffusion — that's the 2026-aware note.

### ✅ Mastery check

(a) A plain autoencoder is not generative. State precisely what is missing, and what the KL term in the VAE's ELBO buys you.
(b) Explain why GANs produce sharp images *and* why they are unstable to train — using one mechanism for both.
(c) Your GAN's FID improves from 18 to 12, but human raters prefer the earlier model's images. Give two explanations.

<details><summary>Answer sketch</summary>
(a) What is missing is a <b>known distribution over the latent space</b>. An autoencoder learns an encoder and decoder that reconstruct well, but nothing constrains <i>where</i> in latent space the data lands — the encoded points may occupy a complicated, disconnected, arbitrarily-scaled region, so there is no distribution you can sample $z$ from and expect $g(z)$ to be a plausible image. The <b>KL term</b> $D_{KL}(q_\phi(z|x)\,\|\,p(z))$ pulls every per-example posterior toward a fixed prior $\mathcal{N}(0,I)$, which does two things: it makes the aggregate latent distribution approximately match a distribution you can sample from, and it makes the space <b>smooth and continuous</b>, so interpolating between two latents passes through valid images rather than off-manifold noise. The price is the reconstruction/regularisation trade — push the KL too hard and you get posterior collapse.
(b) <b>The discriminator is a learned, adaptive loss function.</b> Sharpness: a blurry output is exactly what a discriminator finds easiest to detect, so blur is penalised directly and the generator must commit to a single plausible high-frequency realisation rather than hedging across all of them — which is precisely what an MSE-trained decoder cannot do. Instability: because that loss is <i>itself being learned</i> and changes every step, the optimisation is a two-player game with no fixed objective and no scalar that measures progress; the generator chases a moving target, can collapse onto a few modes that currently fool the discriminator, and receives vanishing gradients whenever the discriminator wins too easily. <b>One mechanism, both consequences</b> — that is the answer to give.
(c) 1. <b>FID measures distribution distance, not per-image quality.</b> A model that produces slightly worse individual images but covers the data distribution more evenly scores better than one producing beautiful images with reduced diversity — FID punishes mode collapse and rewards coverage, which is not what a rater scoring individual images perceives. 2. <b>FID is computed in Inception feature space</b>, which has its own biases: it is sensitive to textures and high-frequency statistics that correlate imperfectly with human preference, and is known to be gameable by artefacts that shift feature statistics favourably while looking wrong. (Also valid: FID is <b>sample-size dependent</b>, so the two numbers may not be comparable if computed on different set sizes; and FID says nothing about prompt adherence or semantic correctness — for which you need CLIPScore or a human-preference model like HPSv2.)
</details>


### 🔨 Build + read

**Build:** Implement a VAE from scratch (encoder → reparameterisation trick → decoder, on MNIST or Fashion-MNIST) and separately a small GAN (DCGAN-style) on the same dataset. Compare samples qualitatively (VAE: blurrier, more diverse; GAN: sharper, more prone to mode collapse) and quantitatively if you can (FID on a held-out set, understanding its limitations as you do). Then ablate the VAE's KL-term weight ($\beta$ in a $\beta$-VAE) across 3–4 values and observe the reconstruction-sharpness/latent-regularity trade-off directly.

**Read:** Kingma & Welling, *Auto-Encoding Variational Bayes* (2014, ICLR) for the reparameterisation trick and the ELBO derivation — this is the single most load-bearing derivation for everything in 6.13–6.14. Then Goodfellow et al., *Generative Adversarial Networks* (2014, NeurIPS), §4, for the minimax optimality proof.

---

## 6.13 Diffusion Models: DDPM Fundamentals

### Intuition

**The question this section answers: 6.12 left you with two unsatisfying options — a VAE that is stable but blurry, and a GAN that is sharp but a two-player game with no loss curve. Is there a way to get a *single, stable regression objective* that still produces sharp images?**

The obstacle is the one identified at the top of 6.12: mapping noise to a photograph in one shot is an absurdly hard function to learn, and every attempt to supervise it directly either hedges (VAE) or needs an adversary (GAN). Diffusion's escape is to refuse to learn that map at all, and instead **decompose it into a thousand steps each of which is nearly trivial.**

Here is the intuition. Take a photo and add a tiny bit of noise. Repeat 1000 times and you have pure static. Now train a network to undo *one* step of that. Chain it 1000 times starting from pure static, and you have generated an image.

**The genius is in what the decomposition does to the difficulty.** "Turn static into a photograph" is hopeless. "Given this slightly-noisy image, estimate the noise that was added" is an ordinary supervised regression with a known ground-truth target — you added the noise, so you have the label for free. The hard, multi-modal, hedging-prone part of generation has been amortised across a thousand easy steps, none of which individually has to choose between plausible images. That is the whole idea; everything below is the machinery that makes it computable.

### The forward process

The destruction direction is fixed and has no learned parameters — it is defined, not trained. A Markov chain adding Gaussian noise over $T$ steps with a variance schedule $\beta_t$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\bigr)
$$

Read the mean: $\sqrt{1-\beta_t}\,x_{t-1}$, not $x_{t-1}$. Each step slightly *shrinks* the signal as well as adding noise, and that shrinkage is not decoration — it is what makes the chain converge to $\mathcal{N}(0,I)$ rather than a Gaussian of ever-growing variance. Without it you could not sample your starting point at generation time, because you would not know what distribution the chain ends in.

**Pause:** to train on timestep $t = 700$, you appear to need to simulate 700 sequential noising steps per training example. That would be ruinous. What property of Gaussians rescues you?

**The key closed form** — with $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$, you can jump to any timestep in one shot:

$$
\boxed{\ q(x_t\mid x_0) = \mathcal{N}\bigl(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\bigr)
\quad\Longrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\ }
$$

**Where that comes from — two lines.** Write one step, then substitute the one before it:

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}}_{\text{two independent Gaussians}}
$$

Independent zero-mean Gaussians add in variance, $\mathcal{N}(0,\sigma_1^2 I) + \mathcal{N}(0,\sigma_2^2 I) = \mathcal{N}(0,(\sigma_1^2+\sigma_2^2)I)$, and here $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$. So the two-step form is $x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon$ — the same shape with $\alpha$'s multiplied. Induct down to $x_0$ and the product telescopes into $\bar\alpha_t$. **The whole trick is that a sum of Gaussians is a Gaussian**, which is precisely why the forward process was chosen to be Gaussian in the first place.

**This is what makes training tractable**, and it is the answer to the Pause: you never simulate the chain. You sample a random $t$, jump straight to $x_t$ in one line, and train on that — so a training step costs the same whether $t$ is 1 or 999, and the 1000-step chain exists only conceptually during training. It is genuinely simulated only at sampling time.

Look at the closed form once more as a statement about *what $x_t$ is*: a weighted blend of the clean image and pure noise, with $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$ as the mixing weights, and $\bar\alpha_t$ sliding from ≈1 (all signal) to ≈0 (all noise) as $t$ grows. The noise schedule $\beta_t$ is nothing more than the choice of *how fast* that slider moves — which is why schedule design (linear, cosine, zero-terminal-SNR) is a real research topic rather than a hyperparameter: it decides how much training time is spent at each difficulty level. Being able to state *why* the closed form matters is the difference between describing diffusion and understanding it.

### The reverse process and the training objective

Now the direction that must be learned. We want $p_\theta(x_{t-1}\mid x_t)$: given a noisier image, produce a slightly less noisy one.

**Why is this even possible?** In general, reversing a stochastic process is intractable — the true reverse $q(x_{t-1}\mid x_t)$ requires knowing the data distribution. The saving fact is a result from stochastic processes: **when the forward step is Gaussian and $\beta_t$ is small, the reverse step is also approximately Gaussian.** That is the licence for the parametric form below, and it is the second place (after "sums of Gaussians are Gaussian") where the choice of Gaussian noise pays for itself. Small $\beta_t$ is thus not a tuning preference — it is a *requirement* of the approximation, and it is why $T$ has to be large: the total corruption has to be spread thinly enough that every individual step stays near-Gaussian in reverse.

So: learn $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t), \Sigma_t)$, with only the mean predicted (DDPM fixes $\Sigma_t$ to a schedule constant).

**The path from the variational bound to the loss — four moves, worth following once.** The destination is a plain MSE on noise, and the point of walking it is that the destination looks *too* simple to be a variational method; seeing where the simplicity comes from is what makes it yours.

1. **The bound decomposes per timestep.** The usual variational upper bound on $-\log p_\theta(x_0)$ splits into $L_T + \sum_{t>1} L_{t-1} + L_0$, where the interesting terms are
   $$L_{t-1} = D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)$$
   ($L_T$ has no parameters — the forward process is fixed — so it is a constant.)
2. **The forward posterior is Gaussian in closed form.** The raw reverse $q(x_{t-1}\mid x_t)$ is intractable, but $q(x_{t-1}\mid x_t, x_0)$ — reverse *given we also know the clean image* — is available in closed form: $\mathcal{N}\bigl(\tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\bigr)$. This is the step that makes everything tractable, and it is why the loss conditions on $x_0$. During training $x_0$ is in hand (it is the training image), so we can supervise against an exactly-known target; at sampling time we do not have $x_0$, which is precisely the gap the network is being trained to fill.
3. **KL between two Gaussians with fixed variance is a squared difference of means.** So $L_{t-1} = \frac{1}{2\sigma_t^2}\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\|^2 + C$.
4. **Reparameterise the mean in terms of the noise** — this is the move that turns a variational objective into a regression, so do it deliberately. We currently have "predict $\tilde\mu_t$." But $\tilde\mu_t$ is a function of $x_t$ (which the network already sees) and $x_0$ (which it does not), so the only genuinely unknown quantity is $x_0$ — and by the closed form, $x_0$ and $\epsilon$ determine each other given $x_t$: invert $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ to get $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$. Substituting, the $\tilde\mu$ and $\mu_\theta$ terms collapse onto the same expression in $x_t$ plus a term in the noise, everything the network already knows cancels, and their difference becomes proportional to $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ with a $t$-dependent weight $\dfrac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$.

   **So "why does predicting the noise work?" has a precise answer**, and it is worth having ready because it is asked constantly: predicting $\epsilon$, predicting $x_0$, and predicting the reverse mean $\tilde\mu_t$ are **the same prediction in three coordinate systems**, related by invertible affine maps given $x_t$ and $t$. Nothing deep separates them mathematically. What separates them is *conditioning*: at large $t$ the image is nearly pure noise, so predicting $x_0$ means amplifying a tiny signal by a huge factor and the target's scale explodes, while $\epsilon$ stays unit-variance at every $t$ — a well-scaled regression target across the whole schedule, which is exactly what you want an MSE loss to see.

**Then DDPM's actual simplification: throw the weight away** (set it to 1). That gives:

$$
\boxed{\ \mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\bigl\|\epsilon - \epsilon_\theta\bigl(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t\bigr)\bigr\|^2\right]\ }
$$

**Note what just happened: $\mathcal{L}_{\text{simple}}$ is not the ELBO** — it is a *reweighted* ELBO, and the reweighting is deliberate. Dropping the weight up-weights the large-$t$ (heavily noised, harder, more perceptually important) terms relative to the near-clean ones, and empirically it produces markedly better samples than optimising the true bound. **"The principled objective is not the one that works best, and they knew it" is a genuinely good thing to be able to say about diffusion.**

**It's an MSE regression.** The training loop, in full: sample an image, sample $t \sim U(1,T)$, sample $\epsilon \sim \mathcal{N}(0,I)$, form $x_t$ in one line with the closed form, ask the network to predict $\epsilon$ from $(x_t, t)$, take an MSE step. That's it. Five lines, one loss, no adversary, no collapse mode, a number that goes down and means something — **and that stability, not sample quality per se, is why diffusion beat GANs.** Come back to the 6.12 comparison and the contrast is stark: the GAN got sharpness by learning its loss and paid for it with instability; diffusion gets sharpness with a *fixed* MSE loss, because the decomposition into small steps means no single step ever has to choose between plausible images, so hedging never becomes the optimal move.

**Parameterisation choices** worth naming, and now you can read them as choices of coordinate system rather than as different models: predicting $\epsilon$ (DDPM's choice, well-scaled at high noise), predicting $x_0$ directly (well-scaled at *low* noise, where the image is nearly clean and $\epsilon$ is the hard thing to isolate), or **$\mathbf{v}$-prediction** ($v = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$), which interpolates between the two so that neither end of the schedule is badly conditioned — which is why it is standard for distillation and for high-resolution models, both of which stress the extremes of the noise range.

**In your own words:** why is predicting the added noise an easier learning problem than predicting the clean image, when the two are equivalent given $x_t$?

### Sampling

Training was cheap because of the closed form; **sampling has no such shortcut**, because each reverse step needs the network's output on the *previous* step's result. This asymmetry — cheap training, expensive inference — is diffusion's defining engineering problem, and everything in this subsection is an attack on it.

**DDPM ancestral sampling** needs all $T$ (=1000) steps — slow. **DDIM** (Song et al., 2021) reformulates the reverse process as **non-Markovian and deterministic**, allowing you to skip steps: 20–50 steps with little quality loss, and — because it's deterministic — it gives a meaningful **latent↔image correspondence**, which is what makes image editing and interpolation possible.

Faster solvers: **DPM-Solver / DPM-Solver++** (treat it as an ODE and use a high-order solver — 10–20 steps), **Euler / Heun** samplers, and **distillation** to 1–4 steps (Progressive Distillation, Consistency Models, LCM, **ADD/SDXL-Turbo** — which uses an adversarial loss, closing the loop with 6.12).

### The connections worth knowing

- **Score-based view** (Song & Ermon): $\epsilon_\theta(x_t,t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p(x_t)$ — the network is learning the **score** (the gradient of the log-density), and sampling is Langevin dynamics. This is worth pausing on, because it reframes what the model is: a noise predictor is, up to scaling, an estimate of *which direction increases the log-probability of the data* at this noise level. Sampling is then gradient ascent on log-density with noise injected to keep it exploring — which also tells you why the sign flips matter and, in 6.14, why you can steer generation simply by *adding another gradient* to the score. The DDPM and score-matching literatures are the same theory in two languages, unified by the SDE formulation (Song et al., ICLR 2021).
- **Flow matching / rectified flow** — the current reformulation (Stable Diffusion 3, Flux). Learn a velocity field transporting noise to data along straighter paths, which needs fewer sampling steps. **Naming flow matching as where the field is now is a strong currency signal.**
- **The architecture is a U-Net** (4.11) with residual blocks, self-attention at lower resolutions, and **timestep conditioning** via sinusoidal embeddings added into each residual block. Recently, **DiT** (Diffusion Transformer) replaced the U-Net with a transformer, which scales better and underlies SD3 and Sora-class video models.

### 🎯 Top-1% distinction

1. **The closed-form jump to $x_t$** and why it makes training tractable.
2. **The objective is plain MSE noise prediction** — and that simplicity is *the* reason diffusion beat GANs (no adversarial game, no mode collapse, a loss that actually measures progress).
3. **DDIM's determinism enables editing**, not just speed.
4. **The score-matching equivalence.**
5. **$v$-prediction and flow matching** as the modern parameterisations.
6. **DiT replacing the U-Net** — the architecture is not essential to the method.
7. **Diffusion's real cost is inference**: 20–50 forward passes vs a GAN's one. That's why distillation to 1–4 steps is the hottest practical area.

### ✅ Mastery check

(a) Why does DDPM train the network to predict $\epsilon$ rather than $x_0$ directly, given that the two are algebraically interchangeable?
(b) DDPM needs 1000 sampling steps; DDIM achieves comparable quality in 50. What did DDIM change, and what did it gain *besides* speed?
(c) Your diffusion model's training loss decreases smoothly but samples are structured noise. Give three candidate causes and the diagnostic for each.

<details><summary>Answer sketch</summary>
(a) They are interchangeable via $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$, but <b>the conditioning of the regression differs wildly across $t$</b>. At large $t$, $x_t$ is nearly pure noise and $\bar\alpha_t \to 0$, so recovering $x_0$ means dividing by a tiny number — the target has enormous variance and the network is being asked to hallucinate an entire image from noise, a badly-scaled and unstable objective. Predicting $\epsilon$ has <b>unit-variance targets at every $t$ by construction</b>, so the loss is comparably scaled across the whole schedule and a single network handles all timesteps. (The corollary is that $\epsilon$-prediction is poorly conditioned at <i>small</i> $t$, where $x_t$ is nearly clean and the noise is a small residual — which is exactly why $\mathbf{v}$-prediction exists, interpolating between the two so that neither end degenerates. That is the strong close.)
(b) DDIM reformulated the reverse process as <b>non-Markovian and deterministic</b>: it defines a family of processes sharing DDPM's marginals but with zero (or tunable) noise injection at each step, which means the trajectory no longer needs every intermediate step to be valid and you can skip timesteps. Beyond speed, determinism gives a <b>meaningful, invertible correspondence between latents and images</b> — the same $x_T$ always yields the same image, so you can interpolate in latent space, invert a real image to its latent (DDIM inversion) and re-generate it with a modified prompt. <b>That is what makes editing possible</b>, and it is the more important of the two gains.
(c) 1. <b>Noise-schedule or $\bar\alpha$ bug</b> — e.g. the schedule doesn't reach near-pure noise at $t=T$, so sampling starts from a distribution the model never saw. Diagnostic: plot $\bar\alpha_t$ and check $\bar\alpha_T \approx 0$; visualise $q(x_T|x_0)$ and confirm it looks like pure noise. 2. <b>Train/sample mismatch</b> — the sampler uses a different schedule, timestep indexing (off-by-one, 0- vs 1-indexed), or normalisation than training. Diagnostic: run the sampler starting from a <i>noised real image</i> at small $t$; if it can't denoise even that, the mismatch is in the loop, not the model. 3. <b>Under-training or too-small a model for the data</b> — the loss decreases because predicting the mean noise is easy, but the model hasn't learned structure. Diagnostic: check the loss <i>per timestep bucket</i> — a model failing to learn structure has flat loss at small $t$ where the task should be easy; also sanity-check by overfitting a single image and confirming it can reproduce it.
</details>


### 🔨 Build + read

**Build:** Implement DDPM training and sampling from scratch on a small dataset (MNIST or a 32×32 crop of CIFAR-10) with a small U-Net. Verify the closed-form forward jump ($q(x_t|x_0)$ in one step) against literally simulating $t$ sequential noising steps — they should match to floating-point precision, which is the best way to confirm you actually understand why the closed form is valid. Then sample with the full $T$-step reverse process and separately with DDIM's fewer-step deterministic sampler on the *same trained model*, and compare sample quality and wall-clock at matched step count.

**Read:** Lilian Weng, *What are Diffusion Models?* (`lilianweng.github.io/posts/2021-07-11-diffusion-models/`) for the cleanest single derivation reconciling DDPM/DDIM notation. Then Ho et al., *Denoising Diffusion Probabilistic Models* (2020, NeurIPS), §3.2, for where $\mathcal{L}_\text{simple}$ actually comes from and why it's a reweighted, not exact, ELBO.

---

## 6.14 Latent Diffusion (Stable Diffusion): VAE + U-Net + CLIP + CFG 🟡

### The problem it solves

**The question this section answers: 6.13 gave you a training objective that works. Why could almost nobody run it, and what had to change for Stable Diffusion to fit on a consumer GPU?**

Do the arithmetic that forces the design. Pixel-space diffusion at $512\times512\times3$ means every one of 50 denoising steps operates on 786,432 dimensions — and the U-Net's self-attention layers are quadratic in spatial positions (6.1 again). You are running a large network 50 times over three-quarters of a million values per image. That is why DALL·E 2 and Imagen didn't do it either: they used cascades of low-resolution diffusion plus separate super-resolution stages — expensive, complicated, and several models to train and keep in sync.

**Latent Diffusion (Rombach et al., CVPR 2022)** moves diffusion into a **compressed latent space**. A pretrained autoencoder maps $512\times512\times3 \to 64\times64\times4$ — a **48× reduction** in elements. Diffusion runs entirely there.

**Pause:** compressing 48× before generating sounds like it must cost you image quality. Why doesn't it?

Because of *what* is being thrown away. Ask which bits of a photograph the 786,432 numbers are actually spending themselves on: overwhelmingly, high-frequency detail — the exact grain of a texture, sensor noise, the precise pixel values along an edge. That detail is perceptually near-invisible and, crucially, it is **reconstructible from context** rather than semantic. A decoder can hallucinate plausible grain; it cannot hallucinate that the scene contains a cathedral. So a compressor can strip the first kind at 48× with almost no perceptual loss, leaving the diffusion model to work only on the part where the difficulty actually lives.

**The justification (this is the part to say):** the authors split generative learning into two phases. A **perceptual compression** phase removes high-frequency detail that is imperceptible but consumes most of the bits — and an autoencoder does this efficiently. A **semantic compression** phase learns the actual structure of the data — and this is what the diffusion model should spend its capacity on. Pixel-space diffusion wastes most of its capacity and compute on the first phase. **Latent diffusion separates the two so the expensive model only does the part that needs it.** Note the division of labour this implies, because it explains every design choice in the next subsection: the autoencoder is a *deterministic detail-restorer* trained once and frozen, and the diffusion model is a *stochastic semantic generator*. Neither is asked to do the other's job — which is exactly why the VAE's famous blurriness (6.12) stops being a problem here.

### The four components

**1. The VAE (autoencoder) — and the thing that trips people up is the name.** This component is called a VAE and is *not* being used as one. In 6.12 a VAE was a generative model: you sampled $z$ from the prior and decoded. Here you never sample from it at all; you only ever encode a real image and decode a latent the diffusion model produced. It is a codec. That is why the KL weight is set almost to zero — the whole point of a strong KL was to make the prior samplable, and nothing here samples the prior.

Encoder $\mathcal{E}: x \to z$ (downsample 8×), decoder $\mathcal{D}: z \to \hat{x}$. Trained with a perceptual loss (LPIPS) plus a **patch-based adversarial loss** — i.e. a GAN discriminator, 6.12 — to avoid the blurriness a plain MSE VAE would give. Regularised either by a very weak KL term or by vector quantisation. **The KL weight is deliberately tiny** — this is *not* a generative VAE, it is a near-deterministic compressor, and its latent space is not meant to be sampled from directly. And look at where the adversarial loss went: the GAN from 6.12, which lost the generation race, is here doing the one job it is unbeatable at — forcing sharp high-frequency output — inside a system generated by something else entirely. That is the "compose, don't crown a winner" pattern in its cleanest form.

**2. The denoising U-Net.** Operates on $64\times64\times4$ latents. Residual blocks, self-attention at lower resolutions, and — the crucial addition — **cross-attention** to the conditioning embedding at multiple resolutions:
$$
\text{Attention}(Q,K,V), \quad Q = W_Q\,\varphi(z_t),\quad K = W_K\,\tau_\theta(y),\quad V = W_V\,\tau_\theta(y)
$$
Read the assignment of roles, because it is the reason this generalises: the **queries come from the image latent** and the **keys and values come from the conditioning**. So each spatial position in the latent asks "which parts of the prompt are relevant to me?" and pulls in that information. Nothing in that arrangement cares what the conditioning *is* — it only has to be a sequence of vectors. **Cross-attention is therefore the general conditioning mechanism** — swap $\tau_\theta$ and you condition on text, layouts, semantic maps, or anything else. That generality is why the architecture spread everywhere.

**3. The text encoder.** SD 1.x uses **frozen CLIP ViT-L/14** text embeddings; SD 2.x uses OpenCLIP ViT-H; SDXL concatenates **two** text encoders; SD3/Flux add a **T5** encoder, which markedly improves long-prompt and text-rendering fidelity (CLIP's text encoder was trained on short captions and has a 77-token limit).

**4. Classifier-free guidance (CFG).** The single most important sampling-time knob.

### Classifier-free guidance — derive it

**Start with the problem, because CFG looks like an arbitrary formula until you know what it is for.** Train a conditional diffusion model honestly and sample from it, and the results are disappointingly *loose*: the prompt is respected in spirit, but the model keeps producing plausible images that only partly match. Nothing is broken. The model is sampling faithfully from $p(x \mid c)$, and that distribution genuinely contains a great many mediocre matches alongside the good ones. **We do not actually want to sample from $p(x\mid c)$ — we want to sample from a sharpened version of it**, one where the probability mass has been concentrated on images that match $c$ strongly. So the real question is: how do you sharpen a distribution you can only access through its score?

Earlier work answered it with *classifier* guidance: train a separate classifier on noisy images and push the sample toward a class using $\nabla_{x_t}\log p(y|x_t)$ — literally nudge the sample in the direction that makes the classifier more confident. It works, but it requires training an extra classifier that operates on *noisy* images at every noise level, which is a whole second training pipeline and one that does not exist for free-form text.

**Ho & Salimans' trick — and it is one of the great cheap ideas in the field.** Train a single conditional model, but **randomly drop the conditioning** (replace it with a null/empty embedding) ~10% of the time. That one network then learns *both* the conditional and the unconditional score, because 10% of its training batches were unconditional. Once you have both, the classifier can be eliminated **algebraically**: you never train it, you compute its gradient as a difference of two things you already have.

**The derivation — four steps.** Follow it once and the formula stops being something to memorise.

*The goal, stated precisely, so you can recognise it when it appears:* obtain the score of a sharpened conditional distribution using only the two scores the network provides.

*Step 1: the network is a score estimator.* From 6.13, $\epsilon_\theta$ and the score of the noisy marginal are the same object up to a scale:

$$
\nabla_{x_t}\log p(x_t) = -\frac{1}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)
$$

*Step 2: what classifier guidance does.* To sample from the conditional distribution you need $\nabla_{x_t}\log p(x_t \mid c)$. Bayes' rule, differentiated (the $\log p(c)$ term has no $x_t$ dependence and vanishes):

$$
\nabla_{x_t}\log p(x_t\mid c) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(c\mid x_t)
$$

Read that identity as a decomposition of forces: the first term pulls the sample toward "looks like a real image," the second toward "looks like *this prompt*." Classifier guidance sharpens by *scaling the second term* by a guidance weight $s > 1$ — turning up the prompt force relative to the realism force — which is precisely the sharpening we asked for. Its only defect is that it requires $p(c \mid x_t)$, a classifier on noisy images.

*Step 3: eliminate the classifier.* Rearrange the same Bayes identity to solve for the classifier term:

$$
\nabla_{x_t}\log p(c\mid x_t) = \nabla_{x_t}\log p(x_t \mid c) - \nabla_{x_t}\log p(x_t)
$$

**Both terms on the right are things our single dropout-trained network already estimates** — the conditional score (pass $c$) and the unconditional score (pass $\varnothing$). No classifier needed; the *implicit* classifier hiding inside the difference of the two scores is used instead. That substitution *is* the entire idea of classifier-*free* guidance, and it is worth appreciating that the dropout training in the previous paragraph exists solely to make this line legal.

*Step 4: substitute and collect.* The guided score is

$$
\tilde\nabla = \nabla\log p(x_t) + s\bigl[\nabla\log p(x_t\mid c) - \nabla\log p(x_t)\bigr]
$$

and converting back to $\epsilon$-space by multiplying through by $-\sqrt{1-\bar\alpha_t}$ (Step 1, which is linear, so the combination passes straight through):

$$
\boxed{\ \tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s\,\bigl[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\bigr]\ }
$$

Equivalently $\tilde\epsilon = (1-s)\,\epsilon_\varnothing + s\,\epsilon_c$. Look at the coefficients: they sum to 1, so for $s \in [0,1]$ this is an interpolation between the two predictions — but at $s = 7.5$ the unconditional term's coefficient is $-6.5$. **This is an extrapolation**, and that single observation explains every artefact CFG produces. You are stepping *past* the conditional prediction, along the line away from the unconditional one, into a region where neither of the two estimates was ever validated. The model was trained to be accurate at $\epsilon_c$, not at $7.5\epsilon_c - 6.5\epsilon_\varnothing$.

**Pause:** given that reading, predict what happens to colour and to diversity as you raise $s$, before looking at the list below.

- $s = 1$: ordinary conditional sampling.
- $s > 1$: push **away** from the unconditional prediction and **further** in the conditional direction — amplifying prompt adherence. Typical $s = 7$–8 for SD.
- $s$ too high: over-saturated colours, blown-out contrast, reduced diversity, and artefacts. **CFG trades diversity for fidelity**, and that trade is the thing to state.

Both of those follow from extrapolation: pushing past the valid region drives pixel values toward the extremes of their range (saturation, blown contrast), and amplifying the prompt direction while suppressing the generic direction collapses the sampler toward the single most prompt-typical image — fidelity bought with diversity, which is the trade to state out loud.

Cost: **two forward passes per step** (conditional and unconditional), so CFG doubles inference compute. That's why "CFG distillation" (training a student to reproduce the guided output in one pass) is a standard optimisation — and it is a good example of the general move of turning an *inference-time* computation into *training-time* weights.

**In your own words:** what is classifier-free guidance actually free *of*, and what does the guidance scale physically do to the sample?

### Conditioning and control

- **ControlNet** (2023): clone the U-Net encoder into a trainable copy connected by **zero-initialised convolutions**, and condition on an edge map, depth map, pose skeleton, or scribble. The zero-init is the clever part: at initialisation the branch contributes exactly nothing, so training starts from the pretrained model's exact behaviour and cannot damage it. **This is the same "start as the identity" principle as ResNet's zero-init final BN (2.6)** — a nice cross-module link.
- **LoRA** for cheap style/subject adaptation; **DreamBooth** for subject-driven generation; **IP-Adapter** for image prompting; **T2I-Adapter** as a lighter ControlNet.
- **Inpainting** by masked latent replacement; **SDEdit / img2img** by partially noising a real image and denoising with a new prompt (the noise level is the edit-strength knob).

### The 2026 landscape

**SD3 / Flux** replaced the U-Net with a **DiT/MMDiT transformer** and DDPM with **rectified flow matching**, and use multiple text encoders including T5. The results are notably better prompt adherence and — finally — reliable text rendering inside images. **Turbo/LCM/distilled variants** generate in 1–4 steps. Video diffusion (Sora-class, and open models) extends the same machinery with temporal attention and 3-D latents.

### 🎯 Top-1% distinction

1. **The perceptual-vs-semantic compression argument** for why latent space is the right place to diffuse.
2. **The VAE here is a compressor, not a generator** — tiny KL weight, adversarial loss for sharpness. Candidates who call it "the generative VAE" have not understood the design.
3. **Cross-attention is the general conditioning interface** — one mechanism, any modality.
4. **Derive CFG** and state the diversity/fidelity trade and its 2× compute cost.
5. **ControlNet's zero-initialised convolutions** and the identity-at-init principle.
6. **CLIP's 77-token limit and short-caption training** is why SD3/Flux added T5 — a specific, mechanistic explanation for a version change.
7. **The failure modes**: text rendering (largely fixed by 2026), hands and counting, compositional binding (inherited from CLIP, 6.5), and prompt-adherence limits that CFG papers over rather than solves.

### ✅ Mastery check

(a) Why does latent diffusion need a VAE trained with an adversarial loss rather than a plain MSE autoencoder?
(b) Derive CFG and explain what happens at $s = 1$, $s = 7$, and $s = 25$.
(c) Your generated images have the right objects but the wrong attribute bindings ("a red cube and a blue sphere" comes out reversed). Explain the root cause and give two fixes at different points in the stack.
(d) You must generate 1000 images per minute on one A100. What do you change?

<details><summary>Answer sketch</summary>
(a) A plain MSE autoencoder is <b>blurry</b> for exactly the VAE reason (6.12): MSE makes the decoder output the mean of plausible reconstructions, losing high-frequency texture. In latent diffusion the decoder is the <b>final</b> stage — every generated image passes through it — so its blur would cap the quality of the entire system no matter how good the diffusion model is. A perceptual (LPIPS) loss plus a patch-based adversarial loss forces the decoder to <i>commit</i> to plausible high-frequency detail. Note this is a GAN doing exactly what GANs are good at (sharpness) inside a diffusion system — a nice illustration that the paradigms compose.
(b) Train one model with the conditioning randomly dropped ~10% of the time, so $\epsilon_\theta$ learns both $p(x_t|c)$ and $p(x_t)$. Then $\tilde\epsilon = \epsilon_\theta(x_t,\varnothing) + s[\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\varnothing)]$ — extrapolate along the direction that the conditioning adds. $s=1$: plain conditional sampling; faithful to the learned distribution, but often weak prompt adherence and somewhat generic images. $s=7$: the usual operating point — strong prompt adherence, good quality, reduced diversity. $s=25$: severe over-saturation, blown highlights, high-contrast artefacts, and collapsed diversity (many prompts converge to near-identical compositions) — you have pushed far outside the region where the score estimate is valid.
(c) <b>Root cause: the text encoder.</b> CLIP's text embedding is trained contrastively on short web captions, and (6.5) that objective never forces it to encode <i>binding</i> between attributes and objects — a bag-of-concepts embedding satisfies the loss. The diffusion U-Net cross-attends to that embedding, so it inherits the ambiguity; attention maps for "red" and "blue" overlap both objects. <b>Fixes at different levels:</b> (i) <b>Text encoder</b> — use a model with a stronger text pathway: SD3/Flux add a <b>T5</b> encoder specifically because it handles compositional and long prompts far better. (ii) <b>Sampling / attention</b> — attention-guidance methods (Attend-and-Excite, structured diffusion guidance) explicitly manipulate the cross-attention maps during sampling to force each attribute token to attend to a distinct region. (iii) <b>Conditioning</b> — sidestep language entirely with <b>ControlNet</b> or regional prompting: supply a layout/segmentation map so the spatial assignment is given, not inferred.
(d) 1000 images/min ≈ 17 images/s — roughly two orders of magnitude beyond stock SD on one A100. Changes, in order of impact: (i) <b>Step distillation</b> — SDXL-Turbo / LCM / a consistency-distilled model at <b>1–4 steps</b> instead of 50, a 12–50× win, and the single biggest lever. (ii) <b>Eliminate CFG's second forward pass</b> via guidance distillation — an immediate 2×. (iii) <b>Reduce resolution / latent size</b> and batch aggressively to saturate the GPU. (iv) <b>Quantise and compile</b> — fp8/int8 weights, `torch.compile`, TensorRT, fused attention. (v) Re-examine the requirement: if the images are variations on a theme, cache and reuse latents or use img2img from a small set of bases. Worth stating the honest conclusion: at that throughput you are no longer running standard diffusion — you are running a distilled 1-step generator, which is architecturally much closer to a GAN, which is why adversarial distillation (ADD) is the technique that gets you there.
</details>

### 🔨 Build + read

**Build:** Implement DDPM from scratch on MNIST/CIFAR (forward closed form, $\epsilon$-prediction U-Net, ancestral sampling), then add DDIM sampling and compare quality vs. step count (1000 / 100 / 50 / 20). Then, with a pretrained Stable Diffusion: sweep CFG scale from 1 to 25 on a fixed seed and prompt, and produce the grid — seeing the saturation collapse yourself is the lesson. Finally, do a latent-space experiment: encode a real image with the VAE, decode it, and measure the reconstruction error, then diffuse in that latent space.

**Read:** Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020) §2–3. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022) §3. Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022) — 5 pages. Then Lilian Weng's "What are Diffusion Models?" blog post for the derivations laid out cleanly, and Umar Jamil's from-scratch Stable Diffusion video for the implementation.

---

## 6.15 Image Generation & Editing Applications

**The question this section answers: you have one machine — a conditional latent diffusion model. Almost every product you have seen built on generative vision is that same machine with a different thing plugged into the conditioning slot. Which slot does each application use?**

Read the table with that lens rather than as a list of products. Column three is the interesting one: it tells you *where in 6.14's architecture* the application intervenes — at the cross-attention conditioning, at the initial noise level, at the latent being replaced each step, or by adding a parallel branch. Recognising which of those four an unfamiliar new technique is doing is a genuinely reusable skill.

| Task | Method | Mechanism |
|---|---|---|
| **Text-to-image** | SD / SDXL / SD3 / Flux | cross-attention conditioning + CFG |
| **Image-to-image** | SDEdit / img2img | partially noise a real image, denoise with a new prompt; the noise level is the edit strength |
| **Inpainting** | masked latent diffusion | replace unmasked latent regions with the (noised) original at every step |
| **Structural control** | **ControlNet**, T2I-Adapter | condition on edges (Canny — Module 1.3!), depth, pose, segmentation |
| **Subject personalisation** | DreamBooth, LoRA, Textual Inversion | fine-tune (or learn a new token) from 3–5 images of a subject |
| **Image prompting** | IP-Adapter | a decoupled cross-attention path for image conditioning |
| **Instruction editing** | InstructPix2Pix, and modern instruction-tuned editors | trained on (image, instruction, edited image) triples |
| **Super-resolution** | SD upscalers, Real-ESRGAN | diffusion or GAN conditioned on the low-res image |
| **Video** | Sora-class, Stable Video Diffusion, open video models | temporal attention, 3-D latents, flow matching |
| **3-D** | 3DGS/NeRF from generated views; SDS-based methods | distil a 2-D diffusion prior into a 3-D representation |

**Pause:** img2img and inpainting both start from a real image. What is the single knob that distinguishes them, and what does each do to the starting latent?

img2img controls **how much noise you add before denoising** — noise the image a little and you get a light restyle, noise it heavily and you get something only loosely related, because you have destroyed more of the original before handing it back. Inpainting instead controls **where**: at every reverse step it overwrites the unmasked region of the latent with the (correspondingly noised) original, so those pixels are pinned to the truth while the masked region is free to be generated. One is a *time* intervention, one is a *space* intervention — and both operate on the same unmodified sampler, which is why neither needed a new model.

**Note the loop closing:** ControlNet's most-used conditioning input is a **Canny edge map** — a 1986 algorithm from Module 1.3 steering a 2026 generative model. That is a genuinely satisfying illustration that classical CV didn't get replaced; it got *repurposed as an interface*.

**Evaluation** (worth naming): **FID** for distribution quality, **CLIPScore** for prompt adherence, **HPSv2 / PickScore / ImageReward** for learned human preference, and **T2I-CompBench / GenEval** for compositionality. **No single metric is adequate** — FID says nothing about prompt fidelity and CLIPScore says nothing about image quality, so you report several and still look at images.

**Responsible-use notes** worth being able to raise unprompted, because interviewers for applied roles do care: provenance and watermarking (C2PA, SynthID), training-data consent and copyright, deepfake and likeness risk, and the fact that safety filters are bypassable and are not a complete control. The framing that lands best is that these are *system* properties, not model properties — a watermark that survives re-encoding and a provenance chain are engineering commitments across a pipeline, not a checkbox on a checkpoint.

**In your own words:** name the four places you can intervene in a diffusion sampler, and one application that uses each.

### ✅ Mastery check

(a) You must place one product photo into 50 different scenes with consistent lighting and correct perspective. Name the technique stack and the order.
(b) Why does ControlNet initialise its connecting convolutions to zero, and which earlier idea in this curriculum is that?
(c) Your text-to-image model reports FID 8 and CLIPScore 0.31. Which number tells you prompts are being followed, and what does neither number tell you?

<details><summary>Answer sketch</summary>
(a) Order matters. 1. <b>Segment the product</b> (SAM, prompted with a box) to get a clean alpha matte — everything downstream depends on this being pixel-accurate. 2. <b>Generate or select the 50 backgrounds</b> with text-to-image, or use real plates. 3. <b>Composite with structural control</b>: use <b>inpainting</b> with the product region masked as fixed, so the model generates the surrounding scene <i>conditioned on</i> the product rather than regenerating it — this is what preserves product fidelity, which is usually a legal requirement. 4. <b>Condition on depth or a normal map</b> via ControlNet so perspective and ground contact are plausible. 5. For lighting consistency, either relight the product (a relighting model or a learned harmonisation network) or generate scenes conditioned to match the product's existing lighting direction. 6. If the product must appear in genuinely novel poses, <b>DreamBooth/LoRA</b> on a handful of product photos to teach the model the object identity — but note this trades fidelity for flexibility and is the step most likely to alter the product. The general principle: <b>never let the generative model redraw the thing that must stay exact.</b>
(b) At initialisation a zero-weight convolution outputs zero, so the ControlNet branch contributes <b>nothing</b> to the frozen backbone — the model's behaviour at step 0 is bit-identical to the pretrained model. That means training starts from a known-good function and can only improve on it, rather than starting from a randomly perturbed version of a carefully pretrained model and having to recover. It also means the gradient signal flows cleanly into the branch from the first step without destabilising the base. This is <b>the same "start as the identity" principle as ResNet's zero-initialised final BatchNorm $\gamma$</b> (2.6), which makes each residual block an exact identity at initialisation — and, more broadly, the same idea as LoRA's zero-initialised $B$ matrix.
(c) <b>CLIPScore</b> measures prompt adherence — it is the cosine similarity between the generated image's embedding and the prompt's embedding in CLIP space. <b>FID</b> measures how close the distribution of generated images is to the distribution of real ones, and is entirely blind to whether the image has anything to do with the prompt (a model that ignores prompts and generates beautiful random images can score excellent FID). What <b>neither</b> tells you: per-image quality as a human perceives it (use HPSv2/PickScore/ImageReward, or actual raters); <b>compositional correctness</b> — whether "a red cube on a blue sphere" got the attributes bound to the right objects, which needs T2I-CompBench or GenEval and is CLIP's known blind spot (6.5), so CLIPScore is structurally unable to detect its own weakness; and anything about diversity within a single prompt, safety, or artefacts like malformed hands and text.
</details>

---

## 6.15b Discrete Visual Tokens, Autoregressive & Masked Generation, and Video 🟡

> Diffusion is not the only generative paradigm, and the alternative — **turning images into discrete tokens** — is the mechanism behind unified multimodal models, which is the most job-relevant corner of generative vision for an LLM-engineering track.

### The idea: make an image look like text

**The question this section answers: 6.10 bolted a vision encoder onto an LLM with a projection layer, which lets the LLM *read* images but not *write* them. What would it take for one model to do both, with one objective?**

An LLM operates on a sequence of discrete tokens from a finite vocabulary, and its entire training objective is "predict the next token." Images are continuous, so they don't fit — that is the mismatch the projector in 6.10 papered over on the input side and could do nothing about on the output side. But if you can convert an image into a genuine sequence of **discrete tokens**, the mismatch vanishes, and **every technique built for language applies to images unchanged** — autoregressive generation, masked prediction, in-context learning, KV caching, instruction tuning, and a single model that consumes and produces both modalities with one loss.

That is the prize, and vector quantisation is how you claim it. Note that the prize is *not* compression — a continuous latent compresses just as well. It is **discreteness**, because discreteness is what a softmax over a vocabulary requires.

### VQ-VAE (van den Oord et al., 2017)

An autoencoder whose latent is forced onto a **learned codebook** $\{e_1,\dots,e_K\}$. The encoder produces a continuous feature map; each spatial position is replaced by its **nearest codebook entry**:

$$
z_q(x) = e_k, \qquad k = \arg\min_j \|z_e(x) - e_j\|_2
$$

So a $256\times256$ image becomes, say, a $32\times32$ grid of integers from a vocabulary of 8192 — **1024 tokens, exactly like a paragraph of text.** At that point an image *is* a document, and you can concatenate it with real text and train one transformer over the lot.

**Pause:** an $\arg\min$ sits in the middle of the forward pass. What does that do to training, and where have you met this exact obstacle before?

It severs the graph — a lookup is piecewise constant, so its gradient is zero almost everywhere and the encoder receives no signal at all. You met it in 3.2: VLAD was untrainable for precisely this reason, and NetVLAD's fix was to replace the hard $\arg\min$ with a softmax so gradients could flow. VQ-VAE takes the *other* road, keeping the hard assignment (it has to — the whole point is discrete tokens) and faking the gradient instead. Two different escapes from the same trap; noticing that they are the same trap is the connection worth making.

Two mechanisms make it trainable:

- **The straight-through estimator.** Since $\arg\min$ has zero gradient, copy the gradient from $z_q$ straight back to $z_e$ as if quantisation were the identity: `z_q = z_e + (z_q - z_e).detach()`. Read that line carefully — it is a small piece of graph surgery worth being able to write from memory. Numerically it equals $z_q$, because the two $z_q$ terms differ only in whether gradient flows through them; but the *only* term with a live gradient path is the leading $z_e$, so backward sees $\partial z_q/\partial z_e = 1$. Forward pass quantises; backward pass pretends it didn't. A biased estimator — you are handing the encoder the gradient of a different, unquantised objective — that nonetheless works, because when the codebook fits well the quantisation error is small and the identity is a decent local approximation.
- **The two auxiliary losses.** A **codebook loss** $\|\mathrm{sg}[z_e] - e\|^2$ moves codebook entries toward the encoder outputs assigned to them (often replaced by an EMA update), and a **commitment loss** $\beta\|z_e - \mathrm{sg}[e]\|^2$ stops the encoder's output drifting away from the codebook faster than the codebook can follow. Without the commitment term the encoder's output space grows without bound.

**The characteristic failure is codebook collapse** — most entries go unused and the effective vocabulary shrinks to a handful. The mechanism is rich-get-richer and it is worth being able to state: an entry only receives a gradient when it is somebody's nearest neighbour, so an entry initialised far from the encoder's output distribution is never selected, never updated, and stays dead forever. Nothing in the objective rewards using the codebook evenly, so there is no force pulling dead codes back into play. Fixes: EMA codebook updates, restarting dead codes from encoder outputs, lower-dimensional codes (as in ViT-VQGAN), or replacing hard assignment entirely with **finite scalar quantisation (FSQ)**, which quantises each dimension independently to a small set of levels and has no codebook to collapse.

### VQGAN — the version that actually gets used

**VQ-VAE reconstructions are blurry** for the same reason plain VAEs are (6.12): an MSE reconstruction loss produces the mean of plausible outputs. **VQGAN** (Esser et al., CVPR 2021) fixes it with the same medicine latent diffusion uses (6.14): add a **perceptual (LPIPS) loss and a patch-based adversarial loss** to the decoder. The result is a tokeniser whose reconstructions are sharp enough that a generative model over its tokens produces good images — and it is the tokeniser lineage behind Stable Diffusion's autoencoder, Parti, and most discrete image generators.

### What you do with the tokens

| Paradigm | Mechanism | Character |
|---|---|---|
| **Autoregressive** (Image GPT, DALL·E 1, Parti) | predict tokens left-to-right with a transformer | exactly an LLM; scales predictably; **slow** — 1024 sequential steps per image |
| **Masked / parallel** (MaskGIT, MUSE) | BERT-style: mask a subset of tokens and predict them all at once, then iteratively unmask the most confident | **10–20 steps instead of 1024**; the confidence-ordered unmasking schedule is the clever part |
| **Unified multimodal** (Chameleon, Emu, and the current frontier) | one transformer over interleaved text and image tokens | the reason discrete tokenisation matters for a GenAI career — one model, one objective, both modalities |

Notice that the middle row is the same idea as 6.4's masked modelling, used generatively rather than for representation learning — mask, predict, repeat — and that the "unmask the most confident first" schedule is doing the same work CFG did in 6.14: spending the model's certainty where it is highest, and letting the easy decisions constrain the hard ones.

**The trade against diffusion, stated plainly:** diffusion currently produces better image quality at high resolution and dominates text-to-image products; discrete-token models integrate natively with LLMs and give you a single architecture for understanding *and* generation. Both are actively contested and the boundary is moving — which is itself the honest thing to say.

### Video generation

Extends the same machinery along time, with three additions:

1. **Spatio-temporal latents** — compress in time as well as space (a 3-D VAE), because 100 frames at full latent resolution is prohibitive.
2. **Temporal attention or 3-D attention** — factorised (spatial attention, then temporal attention) is the standard efficiency trick, the same factorisation idea as R(2+1)D in S.1.
3. **DiT backbones and flow matching** — the Sora-class recipe: patchify the spatio-temporal latent into tokens, run a transformer, train with rectified flow.

**The hard part is not quality, it is consistency** — object permanence, physical plausibility, and identity preservation across hundreds of frames. There is a structural reason to expect this: the per-frame objective rewards each frame looking real, and *nothing in an MSE-on-noise loss explicitly penalises a mug that changes handle between frame 40 and frame 90.* Consistency has to emerge from the temporal attention's receptive field, and over hundreds of frames that field is stretched thin. Evaluation is correspondingly immature: FVD is the standard metric and is widely acknowledged to correlate poorly with human judgement.

**In your own words:** why does turning an image into discrete tokens unlock the entire language-model stack, when a continuous latent — which compresses just as well — does not?

### 🎯 Top-1% distinction

1. **Say what tokenisation is *for*** — it makes images consumable by the entire language-model stack. That is the reason it matters, not the compression.
2. **The straight-through estimator** and *why* it is needed (arg-min has no gradient).
3. **The commitment loss stops the encoder outrunning the codebook** — most people list the losses without saying what each prevents.
4. **Codebook collapse** and the modern fixes (EMA, dead-code restarts, FSQ).
5. **VQGAN = VQ-VAE + perceptual + adversarial loss**, and it is the *same fix* as latent diffusion's autoencoder — a nice cross-section connection.
6. **MaskGIT's parallel decoding gets 1024 steps down to ~10** via confidence-ordered unmasking.
7. **Video's bottleneck is temporal consistency, and FVD is a weak metric** — currency plus honesty.

### ✅ Mastery check

(a) Why does VQ-VAE need a straight-through estimator, and what exactly is biased about it?
(b) Your VQ-VAE trains, reconstructions are recognisable but blurry, and only 300 of 8192 codebook entries are ever used. Diagnose both problems and fix each.
(c) You want one model that answers questions about images *and* generates them. Argue for discrete tokens over diffusion, then give the strongest counter-argument.

<details><summary>Answer sketch</summary>
(a) The quantisation step is $\arg\min_j\|z_e - e_j\|$ — a lookup, piecewise constant in $z_e$, so its gradient is <b>zero almost everywhere and undefined at the boundaries</b>. Backpropagation through it would deliver no signal to the encoder at all, and the encoder would never train. The straight-through estimator copies $\partial\mathcal{L}/\partial z_q$ directly to $z_e$, i.e. it pretends the forward operation was the identity. <b>What is biased:</b> the true Jacobian is zero and we substitute the identity, so the gradient the encoder receives is not the gradient of the actual loss it is being optimised against — it is the gradient of a <i>different</i>, unquantised objective. It works because the quantisation error is small when the codebook is well-fitted, so identity is a decent local approximation; it degrades when codes are far from encoder outputs, which is exactly the regime the commitment loss exists to prevent.
(b) <b>Blur:</b> the reconstruction loss is MSE (or L1), which is minimised by the conditional mean over all plausible reconstructions — the same mechanism as VAE blur (6.12), and it is not a bug in the quantiser. Fix: add a <b>perceptual (LPIPS) loss and a patch adversarial loss</b> to the decoder — i.e. make it a VQGAN. <b>Codebook collapse:</b> 300/8192 used means most entries are dead — they were initialised far from the encoder's output distribution, were never the nearest neighbour to anything, therefore never received a gradient, and stayed dead permanently. It is a rich-get-richer failure. Fixes, in order of practicality: <b>EMA codebook updates</b> instead of a gradient-based codebook loss; <b>dead-code restarts</b> (periodically reinitialise unused entries to random encoder outputs from the current batch); <b>lower-dimensional codes</b> with L2 normalisation (ViT-VQGAN's fix — lower dimension makes the nearest-neighbour assignment less degenerate); or switch to <b>FSQ</b>, which has no learned codebook and therefore cannot collapse. Also check the commitment weight $\beta$ — too low and the encoder drifts away from the codebook entirely.
(c) <b>For discrete tokens:</b> a single transformer over interleaved text and image tokens gives you one architecture, one objective, one set of weights for both understanding and generation — so the model can genuinely reason across modalities (answer a question, then draw the answer) rather than bolting a generator onto an encoder. You inherit the whole LLM stack for free: KV caching, in-context learning, instruction tuning, RLHF, existing serving infrastructure. And generation is <i>conditioned on the same representation</i> used for understanding, which is what makes multi-turn image editing by conversation natural. <b>The strongest counter-argument:</b> <b>the tokeniser is a lossy bottleneck you cannot train past.</b> Image quality is hard-capped by the VQ decoder's reconstruction fidelity — no amount of scaling the transformer fixes a tokeniser that discards fine detail and text — whereas diffusion operates in a continuous latent and has demonstrably higher ceilings at high resolution. Diffusion also parallelises across the spatial dimension while autoregressive decoding is sequential in it. The honest resolution: the field is currently split, hybrid systems (an LLM that <i>calls</i> a diffusion decoder, conditioned on its own embeddings) are the pragmatic middle, and anyone claiming the question is settled is ahead of the evidence.
</details>

---
## 6.16 Capstone: Combine Two or More Foundation Models

### Pick one and build it properly

**Option A — Promptable visual search (best fit for your RAG background).**
Text or image query → CLIP/SigLIP embedding → FAISS/HNSW index over 100k+ images (Module 3.7) → top-100 → **rerank with a VLM** or with geometric verification → results. Add filtered search, measure Recall@k vs QPS, and expose it as an API. **This is Modules 3 + 6 in one system, and it is the single most transferable capstone for a GenAI-engineering role.**

**Option B — Open-vocabulary auto-labelling pipeline.**
Grounding DINO (or SAM 3) + SAM over an unlabelled dataset → auto-generated boxes and masks → human verification of a sample → train a fast YOLO/RF-DETR on the result → compare against the zero-shot pipeline on accuracy *and* latency. **Deliverable: a quantified statement of exactly what you gained and lost by distilling.** This demonstrates the "compose to discover, distil to deploy" pattern end to end.

**Option C — A minimal VLM.**
Frozen SigLIP/CLIP encoder + MLP projector + a small open LLM; train the projector on captions, then instruction-tune. Evaluate on a captioning benchmark **and** on POPE (hallucination). Then run the visual-token-count ablation from 6.10.

**Option D — Controllable generation.**
Fine-tune SD with LoRA on a custom subject, add ControlNet conditioning from a Canny/depth map, and build an interface where a user sketches a layout and gets a generated image. Evaluate with CLIPScore and a small human preference study.

### What makes a capstone interview-grade

Not that it works — that it is **measured**. Every option above should produce: a quantitative evaluation with a baseline, an ablation showing what each component contributes, a latency/cost measurement on real hardware, a documented failure analysis with example images, and a short written report. **A measured project beats an ambitious unmeasured one in every interview.**

---

## Going Deeper — Papers, Sources and Research Scope

*This is where the field currently is, and therefore where this section is longest. At CVPR 2026, multimodal LLMs / VLMs were the largest and fastest-growing category of highlighted papers (4.9% → 10.6%), video generation and world models roughly doubled (3.8% → 8.8%), and embodied AI grew to 6.2%. If you intend to do research at all, it is here. Corollary: this section will date fastest — re-check it every few months.*

### A. The canonical papers

**Backbones and self-supervision**

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| **Dosovitskiy et al., *An Image is Worth 16×16 Words* (ViT)** | 2021, ICLR (2010.11929) | Patches as tokens; the finding that convolutional inductive bias is *replaceable given enough data* is the actual claim, and Fig. 3 (the data-scale crossover) is the paper. | arXiv 2010.11929 |
| Touvron et al., *DeiT* | 2021, ICML (2012.12877) | ViT without JFT-300M — distillation and augmentation substituting for data. The practical unlock. | arXiv 2012.12877 |
| Liu et al., *Swin Transformer* | 2021, ICCV (2103.14030) | Hierarchical windows — reintroducing locality and multi-scale structure into ViT. | arXiv 2103.14030 |
| **He et al., *Masked Autoencoders* (MAE)** | 2022, CVPR (2111.06377) | 75% masking, asymmetric encoder/decoder. The ablation on mask ratio (Fig. 5) is the argument. | arXiv 2111.06377 |
| Caron et al., *DINO* | 2021, ICCV (2104.14294) | Self-distillation with no labels; the emergent-attention-segmentation result was genuinely surprising. | arXiv 2104.14294 |
| Oquab et al., *DINOv2* | 2024, TMLR (2304.07193) | Curated data at scale; frozen features competitive with fine-tuned supervised models on dense tasks. | arXiv 2304.07193 |
| Meta AI, *DINOv3* | 2025 | The current iteration. Check the Meta AI blog directly — this line moves and the paper lags the release. | ai.meta.com |
| **Liu et al., *ConvNeXt*** | 2022, CVPR (2201.03545) | The re-examination: a ResNet modernised step-by-step matches Swin. Read it *against* ViT, not after. | arXiv 2201.03545 |

**Vision-language**

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| **Radford et al., *CLIP*** | 2021, ICML (2103.00020) | Contrastive image-text pretraining; the zero-shot-classifier-from-text-encoder trick is what makes open-vocabulary everything possible. | arXiv 2103.00020 |
| **Zhai et al., *SigLIP*** | 2023, ICCV (2303.15343) | Sigmoid loss decouples the objective from global batch size. The practical default. | arXiv 2303.15343 |
| Thrush et al., *Winoground* | 2022, CVPR (2204.03162) | **The critique.** Near-chance on relations that differ only by binding. The compositionality failure, measured. | arXiv 2204.03162 |
| Yuksekgonul et al., *When and why VLMs behave like bags-of-words* (ARO) | 2023, ICLR (2210.01936) | Diagnoses *why*: no training negative differs only by word order, so nothing forces the model to encode it. | arXiv 2210.01936 |
| **Li et al., *BLIP-2*** | 2023, ICML (2301.12597) | Q-Former as a learned bottleneck between frozen vision and frozen LLM. | arXiv 2301.12597 |
| **Liu et al., *LLaVA*** | 2023, NeurIPS (2304.08485) | The simpler answer that won: a linear projector plus instruction tuning. Read against BLIP-2 and ask what the Q-Former was buying. | arXiv 2304.08485 |
| **Kirillov et al., *SAM*** | 2023, ICCV (2304.02643) | Promptable segmentation and the data engine — the data engine is the real contribution. | arXiv 2304.02643 |
| Ravi et al., *SAM 2* | 2024 (2408.00714) | Streaming memory for video. | arXiv 2408.00714 |
| Meta AI, *SAM 3 / SAM 3D* | Nov 2025 | Promptable **concept** segmentation (text prompts, not just points/boxes) and 3D reconstruction. SAM 3.1 adds multiplexing and improved video tracking. Check the Meta blog for current state. | ai.meta.com |
| Liu et al., *Grounding DINO* | 2024, ECCV (2303.05499) | Open-vocabulary detection by fusing text into the detector at multiple stages. | arXiv 2303.05499 |
| *Vision Language Models: A Survey of 26K Papers* | 2025 (2510.09586) | The map of the whole area. Read first if you're lost. | arXiv 2510.09586 |

**Generative**

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Kingma & Welling, *Auto-Encoding Variational Bayes* | 2014, ICLR (1312.6114) | The reparameterisation trick and the ELBO. Everything in diffusion descends from this. | arXiv 1312.6114 |
| Goodfellow et al., *GANs* | 2014, NeurIPS (1406.2661) | Read for the minimax formulation and §4's optimality proof. | arXiv 1406.2661 |
| van den Oord et al., *VQ-VAE* | 2017, NeurIPS (1711.00937) | Discrete latents and the straight-through estimator — the ancestor of every image tokeniser. | arXiv 1711.00937 |
| **Ho et al., *DDPM*** | 2020, NeurIPS (2006.11239) | The forward closed form, the reverse parameterisation, and $\mathcal{L}_\text{simple}$. | arXiv 2006.11239 |
| Song et al., *DDIM* | 2021, ICLR (2010.02502) | Non-Markovian sampling; deterministic, and 20 steps instead of 1000. | arXiv 2010.02502 |
| Song et al., *Score-Based Generative Modeling through SDEs* | 2021, ICLR (2011.13456) | Unifies DDPM and score matching as one SDE. This is the framing modern work uses. | arXiv 2011.13456 |
| **Rombach et al., *Latent Diffusion*** | 2022, CVPR (2112.10752) | Diffusion in a VAE latent — the cost argument that made Stable Diffusion possible. | arXiv 2112.10752 |
| **Ho & Salimans, *Classifier-Free Guidance*** | 2022 (2207.12598) | Two forward passes, one extrapolation, enormous quality gain — and the diversity cost nobody has cleanly fixed. | arXiv 2207.12598 |
| Peebles & Xie, *DiT* | 2023, ICCV (2212.09748) | Replaces the U-Net with a Transformer; the backbone of most current video/image generators. | arXiv 2212.09748 |
| **Lipman et al., *Flow Matching*** | 2023, ICLR (2210.02747) | **The gap this curriculum has.** Straight probability paths, simulation-free training; largely displaced ε-prediction DDPM in new work. Read this and Rectified Flow (2209.03003) to be current. | arXiv 2210.02747 |
| Song et al., *Consistency Models* | 2023, ICML (2303.01469) | One-to-four-step generation via a distilled consistency constraint. | arXiv 2303.01469 |

### B. The single best source, per hard topic

- **ViT, mechanically (6.2).** *The Annotated ViT* / the `vit-pytorch` repo read beside the paper. The patch embedding is literally a strided conv — see it in code once and it stops being mysterious.
- **CNN vs ViT inductive bias (6.3).** Raghu et al., *Do Vision Transformers See Like Convolutional Neural Networks?* (NeurIPS 2021, 2108.08810). Representation-similarity analysis rather than benchmark numbers.
- **MAE vs contrastive (6.4).** The MAE paper §4 ablations, then the "what do SSL objectives learn" discussion in DINOv2 §5.
- **CLIP, and why the text encoder is a classifier generator (6.5).** The paper's §2.2 plus `openai/CLIP`'s `notebooks/Prompt_Engineering_for_ImageNet.ipynb` — the prompt-ensembling notebook makes the mechanism concrete.
- **Diffusion, the whole derivation (6.13).** Lilian Weng, *What are Diffusion Models?* (`lilianweng.github.io/posts/2021-07-11-diffusion-models/`) — the best single derivation anywhere, and it reconciles the notation across DDPM/DDIM/score-SDE. Then Calvin Luo, *Understanding Diffusion Models: A Unified Perspective* (2208.11970) for the full ELBO with every step shown.
- **Diffusion, in code (6.13–6.14).** Hugging Face *Diffusion Models Course*, plus Umar Jamil's *Coding Stable Diffusion from scratch* on YouTube — the only walkthrough that builds VAE, U-Net, CLIP conditioning and the sampler without hiding any of them.
- **Flow matching (the gap).** Lipman et al. §3, then the Meta AI *Flow Matching Guide and Code* (2412.06264) which is written as a tutorial.
- **VLM architecture (6.10).** Umar Jamil's *Coding a Multimodal (Vision) Language Model from scratch*. Then the LLaVA paper's §3, which is two pages.
- **SAM's data engine (6.8).** The SAM paper's §4 — the three-stage annotation loop is the actual innovation and it's the part everyone skips.

### C. Reference implementations worth reading

- **`huggingface/diffusers` → `schedulers/scheduling_ddpm.py` and `scheduling_ddim.py`.** Read them side by side. The diff *is* the DDIM paper. Then `scheduling_flow_match_euler_discrete.py` to see how much simpler flow matching is.
- **`huggingface/diffusers` → `pipelines/stable_diffusion/pipeline_stable_diffusion.py`.** Find where CFG happens: the batch is duplicated, conditional and unconditional are run together, and the two are combined with `guidance_scale`. Three lines, and it is the whole of 6.14.
- **`openai/CLIP` → `clip/model.py`.** Look at the logit scale (a *learned* temperature, clamped) and the symmetric cross-entropy over both axes.
- **`facebookresearch/mae` → `models_mae.py`.** `random_masking` uses argsort-of-noise to shuffle — a neat trick worth stealing. Note that the decoder is genuinely tiny.
- **`facebookresearch/dinov2` → `dinov2/loss/`.** The centring and sharpening that prevent collapse. Compare with SimSiam's stop-gradient: two different answers to the same failure mode.
- **`haotian-liu/LLaVA` → `llava/model/multimodal_projector/builder.py`.** The entire vision-language connector is an MLP. Sit with how little it is.
- **`facebookresearch/sam2`.** The memory attention module — how a promptable image model becomes a video model.

### D. Open research questions

The genuinely open ones, in the order I would attempt them.

1. **Can CFG's diversity collapse be fixed without a second forward pass, or without extrapolating outside the trained region?** *Why open:* at $s=7.5$ the unconditional score carries weight $-6.5$; existing fixes (dynamic thresholding, guidance intervals, autoguidance) are empirical patches. *Minimum experiment:* fixed prompt set, sweep scale and schedule, measure fidelity and diversity separately (diversity via pairwise embedding dispersion, not FID), compare 3–4 published fixes under one protocol. **Inference-only on a laptop. The single most feasible research project in this curriculum.**
2. **Where does VLM grounding actually live — encoder, projector, or LLM?** *Why open:* everyone assumes the encoder; the evidence is indirect. *Minimum experiment:* shuffle patch order; ablate positional information; swap encoders; train the projector on 1%/10%/100% of data. Measure spatial-relation QA separately from object naming. Projector-only training is cheap. **~10–30 GPU-hours. Highly topical.**
3. **Is CLIP's compositionality failure fixable by hard negatives alone?** *Why open:* the ARO diagnosis says the objective never requires binding. The obvious fix (mine caption negatives that differ only in word order) helps but does not close the gap, and nobody knows whether that's a data-quantity issue or an architectural one. *Minimum experiment:* fine-tune a small CLIP with synthetic order-swapped hard negatives; measure Winoground and check whether standard retrieval degrades. **Feasible.**
4. **Does the MAE 75% mask ratio hold at small data scale?** *Why open:* the ablation was at ImageNet scale, and the redundancy argument that justifies 75% depends on dataset statistics. *Minimum experiment:* sweep mask ratio on a 10k-image domain-specific set. **Very feasible.**
5. **Does compounding foundation models multiply their error rates, and is the loss predictable?** *Why open:* pipelines like Grounding DINO → SAM → VLM are everywhere and end-to-end error is rarely decomposed. *Minimum experiment:* measure each stage's recall independently, then the pipeline's, and check whether the product model predicts it. **Very feasible, and it makes an excellent capstone-adjacent result.**
6. **Are VLM benchmarks contaminated, and by how much?** *Why open:* pretraining corpora are public and benchmarks are public; near-duplicate detection between them is straightforward and almost nobody does it. *Minimum experiment:* perceptual-hash and CLIP-embedding near-duplicate search between a benchmark's images and LAION subsets; report the contamination rate and the score delta on the decontaminated subset. **CPU-heavy, GPU-light. Uncomfortable, cheap, and valuable.**
7. ~~A new attention variant, or a new VLM architecture.~~ **Trap.** Both need pretraining-scale compute to demonstrate anything, and the architecture question in VLMs is largely closed — LLaVA's linear projector beat BLIP-2's Q-Former, which was the field's answer.

---

## Module 6 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Attention in vision | global RF from layer 1; **permutation-equivariant** so position encoding carries all spatial structure; **quadratic in tokens** — the central engineering constraint |
| ViT | patch embed **= `Conv2d(3, D, kernel=P, stride=P)`**; `[CLS]` + learned 1-D pos-embed (must be **interpolated** on resolution change) |
| ViT data regime | loses on ImageNet-1k, ties at 21k, wins at JFT-300M ⇒ **inductive bias is a data-efficiency prior, not a quality ceiling** |
| DeiT | the data requirement was partly a **recipe** requirement (aug + distillation token from a CNN teacher) |
| CNN vs ViT | ConvNeXt showed much of the gap was the recipe; CNNs are **texture-biased**, ViTs more **shape-biased**; attention maps are not explanations |
| MAE | **75% masking** (images are redundant; 15% is solvable by interpolation); **encoder never sees mask tokens**; per-patch normalised pixel targets |
| Contrastive vs masked | contrastive ⇒ strong **linear probe**, weak dense; masked ⇒ weak linear probe, strong **fine-tuning and dense tasks**. DINOv2 combines both |
| CLIP | symmetric InfoNCE, learned temperature, batch = negatives; contrastive chosen over generative for **compute efficiency**; prompt templates + ensembling |
| CLIP's weakness | **compositionality/binding** — the loss never forces relational encoding (Winoground, ARO); also fine-grained and jargon vocabulary |
| SigLIP | pairwise **sigmoid** loss removes the global batch normalisation ⇒ no large-batch requirement |
| DINO | self-distillation, EMA teacher, **multi-crop local-to-global**; **centring + sharpening** are two opposing anti-collapse forces; emergent segmentation in attention maps |
| DINOv2 / v3 | curated data + DINO **+ iBOT masked** objective; DINOv3 = 7B params, 1.7B images, **Gram anchoring** preserves dense features at long schedules; SOTA from a **frozen** backbone |
| Foundation model | scale + self-supervision + **adaptability**; emergence and **homogenisation** (a systemic risk) |
| SAM 1 | heavy encoder (once/image) + **~50 ms decoder** (per prompt); **3 masks** for ambiguity with min-loss training; the **data engine** (SA-1B, 1.1B masks) was the contribution; **segments but doesn't name** |
| SAM 3 | **promptable concept segmentation** from text/exemplars; DETR detector + SAM 2 memory tracker + presence head; SA-Co benchmark; 3.1 adds object multiplexing |
| Open-vocab detection | replace `W @ features` with `text_emb @ region_features`; Grounding DINO fuses language at 3 points, ~52.5 zero-shot COCO AP; **YOLO-World re-parameterises text away** for a fixed vocabulary |
| The production pattern | **compose to discover, distil to deploy** — open-vocab model as auto-labeller, fast closed-vocab model at inference |
| VLM framing | every VLM = **connector + what's trained** |
| BLIP-2 | **Q-Former**, 32 learned queries = fixed-cost bottleneck; only the Q-Former trains; weak at OCR/counting because of the bottleneck |
| LLaVA | **linear/MLP projector**, one token per patch; contribution was **GPT-4-synthesised instruction data**; the field converged here |
| VLM current | MLP projectors won (detail for OCR/documents); token count handled by **dynamic resolution / tiling**; third option is **cross-attention** (Flamingo) which doesn't consume context |
| VLM failures | object **hallucination** (POPE), counting, spatial relations, inherited CLIP compositionality |
| VAE | ELBO = reconstruction − KL; **reparameterisation trick** $z = \mu + \sigma\epsilon$; blurry because Gaussian likelihood ⇒ mean of plausible reconstructions |
| GAN | sharp *because* the discriminator is a learned adaptive loss — **and unstable for the same reason**; mode collapse; WGAN-GP / spectral norm; StyleGAN3 fixed **aliasing** |
| Diffusion | closed form $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ makes training a **plain MSE** on $\epsilon$ — that simplicity is why it beat GANs |
| DDIM | non-Markovian, **deterministic** ⇒ few steps **and** meaningful latents for editing |
| Modern diffusion | $v$-prediction, **flow matching / rectified flow** (SD3, Flux), **DiT** replacing the U-Net, 1–4-step distillation (LCM, ADD) |
| Latent diffusion | separate **perceptual compression** (VAE, 48× fewer elements) from **semantic compression** (diffusion); the VAE is a **compressor, not a generator** (tiny KL, adversarial loss) |
| Cross-attention | the general conditioning interface — swap the encoder, condition on anything |
| CFG | drop conditioning ~10% in training; $\tilde\epsilon = \epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$; trades **diversity for fidelity**; costs **2× compute** |
| ControlNet | trainable encoder copy joined by **zero-initialised convs** ⇒ starts as the exact pretrained model (same principle as ResNet's zero-init BN) |
| VQ-VAE | codebook nearest-neighbour + **straight-through estimator** (arg-min has no gradient) + commitment loss (stops the encoder outrunning the codebook); **codebook collapse** is the failure |
| VQGAN | VQ-VAE + perceptual + adversarial loss — the *same* fix as latent diffusion's autoencoder |
| Why tokenise | it makes images consumable by the **entire LLM stack** — one model for understanding and generation |
| MaskGIT | parallel masked decoding with confidence-ordered unmasking: ~10 steps instead of 1024 |
| Video generation | spatio-temporal latents + factorised attention + DiT/flow matching; the bottleneck is **temporal consistency**, and FVD is a weak metric |
| The closing loop | ControlNet's most-used input is a **Canny edge map** — a 1986 algorithm steering a 2026 generative model |

---

*End of Module 6 notes. Drills in `Module-06-Drills.md`.*
