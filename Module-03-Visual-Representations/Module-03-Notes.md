# Module 3 — Visual Representations

> **Why this module is disproportionately valuable to you.** You have shipped a RAG system with a vector database. This module is *the same machinery applied to pixels*: an encoder produces embeddings, an ANN index makes them searchable, and a reranking stage restores precision. Every concept here has a one-to-one counterpart in what you already built, and being able to state that mapping fluently is a genuine interview advantage — it turns "I did a RAG project" into "I understand retrieval systems as a class."
>
> **Interview weight: very high**, and rising. Embedding/retrieval questions now appear in almost every GenAI-adjacent interview.

**Concept map**

```
3.1 Embeddings ──> 3.2 Learned vs handcrafted ──> 3.3 Similarity & retrieval
       │                                                     │
       v                                                     v
3.4 Siamese ──> 3.5 Triplet + mining 🟡 ──> 3.6 Contrastive SSL 🟡     3.7 ANN search 🔴
                                                   │                          │
                                                   └──────────┬───────────────┘
                                                              v
                                                   3.8 End-to-end retrieval system
```

---

## 3.1 What Is an Embedding (Vision Context)

### Intuition

**Start from the problem, not the word.** A user photographs a handbag and you must find that bag among 40 million catalogue images, in 30 milliseconds. You cannot compare pixels: the same bag under different lighting and viewpoint has almost no pixels in common with its catalogue photo, while two *different* bags shot in the same studio share thousands. Raw pixel space gets the ordering exactly backwards.

So you need a different space — one whose coordinates encode "what this is," not "what light hit the sensor." That is all an embedding is: a vector standing in for the image, chosen so that **distance in vector space means dissimilarity in the world**. Two photos of the same handbag from different angles land near each other; a photo of a shoe lands far away. Two questions follow immediately, and they are the whole module: how do you get vectors with that property, and how do you search 40 million of them fast?

**You have built this before.** Your RAG system ran a text chunk through a sentence transformer and got a vector whose neighbours were semantically related chunks, not chunks sharing characters. The move is identical; only the nuisance factors change — paraphrase and word order for text, lighting and viewpoint and background for images. In both cases the encoder's job is to be invariant to the nuisance and sensitive to the meaning. Hold that sentence: it is the thread through 3.4, 3.5 and 3.6, which are all different ways of *specifying* which is which.

Real-life anchor: the "shop the look" button in an e-commerce app. Your photo becomes a 512-dim vector; the catalogue is 40 million vectors; the top 20 nearest are your results — in 30 ms.

### Where an image embedding comes from

Which tensor inside a trained network should you actually take? Read the table as a ladder, from repurposed classifier internals at the top to spaces explicitly trained to be metric spaces further down.

| Source | What it is | Typical dim |
|---|---|---|
| **Penultimate layer of a classifier** (post-GAP) | the feature the linear classifier sees | 512 (ResNet-18) / 2048 (ResNet-50) |
| **`[CLS]` token of a ViT** | the aggregate token | 768 (ViT-B) / 1024 (ViT-L) |
| **Mean-pooled patch tokens** | often better than `[CLS]` for dense/retrieval tasks | same |
| **Projection-head output** (SimCLR/CLIP) | the space the contrastive loss lives in | 128–768 |
| **Aggregated local descriptors** (VLAD/NetVLAD, GeM pooling) | classical/hybrid global descriptor | 512–4096 |

**Never use logits as an embedding — and the reason is worth working out rather than memorising.** A logit vector is the penultimate feature $h$ multiplied by the classifier matrix $W \in \mathbb{R}^{K\times d}$: a *linear projection from $d$ dimensions down to $K$*. Projections destroy everything in the null space of $W$, and here that null space was chosen deliberately — training pushed $W$ to keep exactly the directions separating the $K$ labels and to be indifferent to every other direction. But "everything else" is precisely the within-class variation: which handbag, not just *that* it is a handbag. Two different-looking dogs get nearly identical logits **by design, not by accident**. The penultimate feature has not yet been projected, so it still carries those directions.

**You might expect** logits to be the *most* semantic thing in the network, since they are the layer that names things. That is the trap. Naming is a compression, and retrieval needs what was compressed away.

### Properties that matter

- **Dimensionality.** More dims ⇒ more expressive but more memory and slower search. Production sweet spot is 128–768. **The counter-intuitive part:** PCA/whitening down to 128 often *improves* retrieval rather than merely costing less. A 2048-d CNN feature does not have 2048 independent things to say — its dimensions are heavily correlated, and a handful of high-variance directions (typically illumination and background) dominate the L2 distance, drowning out the low-variance directions that actually distinguish one handbag from another. Whitening rescales every direction to unit variance so the discriminative ones get an equal vote; truncation then drops a tail that is mostly noise. Little signal lost, much noise lost, index 4–16× smaller. **Matryoshka representation learning (MRL)** goes further, training a single embedding whose *prefixes* are all valid embeddings, so you can truncate 768→128 at query time with graceful degradation — increasingly the modern default and a great thing to name.
- **Normalisation.** L2-normalising puts everything on the unit hypersphere and makes L2 and cosine equivalent. **Derive it in one line rather than trusting it:** $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2\,a\!\cdot\!b$; if both are unit-norm the first two terms are each 1, so $\|a-b\|^2 = 2-2\cos\theta$ (Module 1.7). Squared distance is now a strictly *decreasing* function of cosine and nothing else, so ranking by one is ranking by the other. Almost always normalise — it makes a whole class of index questions disappear (3.3).
- **Anisotropy / hubness.** High-dimensional learned embeddings tend to occupy a narrow cone rather than filling the sphere (anisotropy), so every pair already has high cosine similarity and the *differences* carrying your ranking signal are squeezed into a tiny range. Relatedly, a few points near the cone's centre of mass become **hubs** appearing in an absurd number of neighbour lists regardless of query. Both are failures of *spread*, and the fixes restore it: whitening, centring, or similarity normalisation (CSLS). Keep "spread" in mind — it returns as collapse in 3.5 and uniformity in 3.6.

### Why a classifier's features are not a metric space — argue it, don't assert it

The penultimate feature beats logits, but a deeper limitation remains, and it motivates the rest of the module. Ask what cross-entropy actually *demands* of the feature space.

CE is minimised when $W_y^\top h$ exceeds every other $W_j^\top h$ by a comfortable margin — a **linear separability** requirement, nothing more. It says nothing about how *tightly* a class clusters (a class can be a long thin filament and still be perfectly separable), nothing about *relative* distances between classes (so "poodle" may end up closer to "sofa" than to "labrador"), and nothing at all about classes absent from training, which have no $W_j$.

**Pause:** suppose a feature space perfectly separates 1000 ImageNet classes. Can you multiply every $h$ by 10, or stretch one class along some direction, without hurting the loss?

Yes to both. Scaling by 10 scales all logits by 10, which only *sharpens* the softmax — the loss goes *down*. Stretching a class along a direction orthogonal to every $W_j$ changes no logit at all. Both wreck the geometry a nearest-neighbour search depends on while the objective sits indifferent. **A loss invariant to a transformation cannot be relied on to control what that transformation destroys.** The fix is a loss that constrains distances directly: 3.4 and 3.5.

**In your own words:** why is the layer *before* the classifier a better retrieval space than the classifier's output, even though the output is the "semantic" one — and why is even that layer not yet a metric space?

### 🎯 Top-1% distinction

**Cross-entropy embeddings are not metric-optimal — and here is why.** Softmax + CE only requires that classes be *linearly separable* in the penultimate space. It imposes no constraint on intra-class compactness or on the relative distances between different classes. So a classifier's features are excellent for *classification* and merely adequate for *retrieval*, especially for **open-set** problems where query classes were never in the training set (face recognition, product search, person re-identification). That gap is the entire reason metric learning (3.4, 3.5) exists.

**The projection-head finding (SimCLR, and it generalises).** When you train with a contrastive loss through a small MLP head $g$, the representation $h$ *before* the head transfers better than $z = g(h)$ *after* it — by a large margin (SimCLR: ~+10% linear-probe accuracy). Explanation: the contrastive objective demands invariance to the augmentations, so $z$ must **destroy** information about colour, orientation, and crop. Downstream tasks often need that information. The head acts as a *buffer* that absorbs the task-specific invariance, leaving $h$ richer. **This is one of the most quotable practical findings in representation learning** and it generalises: whenever your training objective demands an invariance your downstream task doesn't want, put a head between them.

### ✅ Mastery check

You build a visual product-search system. You take an ImageNet ResNet-50, use its 2048-d penultimate features, L2-normalise, and index them. Recall@10 is 38% — disappointing.

(a) Give three distinct reasons this underperforms, in order of expected impact.
(b) Your colleague suggests using the 1000-d logits instead "because they're semantic." Refute.
(c) You reduce 2048→128 with PCA and recall goes *up*. Explain.

<details><summary>Answer sketch</summary>
(a) 1. <b>Objective mismatch</b> — CE features are optimised for separating 1000 ImageNet classes, not for instance-level similarity within "handbag." Product search is an <b>open-set instance retrieval</b> problem; you need metric learning (triplet/ArcFace) or a contrastive/self-supervised backbone (DINOv2, CLIP). 2. <b>Domain gap</b> — catalogue images are clean studio shots on white backgrounds; queries are phone photos with clutter and odd lighting. The embedding is dominated by nuisance factors. Fix: fine-tune on in-domain pairs, or augment to bridge the gap. 3. <b>Pooling and granularity</b> — GAP over the whole image mixes the product with the background; fine-grained attributes (a buckle, a logo) get averaged away. Fix: GeM pooling, higher resolution, or a detection/segmentation crop before embedding.
(b) Logits are a learned linear projection onto a 1000-class simplex, trained to discard exactly the within-class variation that instance retrieval depends on. Two different handbags that are both "purse" will have nearly identical logits by design. Also, logits are unbounded and axis-aligned to an arbitrary label set that has nothing to do with your catalogue. This is precisely the "CE features are not metric-optimal" point, taken to its extreme.
(c) Three mechanisms. (i) <b>Whitening/decorrelation</b> — the 2048-d features have strongly correlated dimensions; a few high-variance directions (often illumination and background) dominate the L2 distance. PCA-whitening rebalances so that discriminative low-variance directions contribute. (ii) <b>Noise removal</b> — trailing components are mostly noise; dropping them improves signal-to-noise in the distance. (iii) <b>Concentration of measure</b> — distances are better separated in 128-d than 2048-d for the same intrinsic dimensionality, so nearest-neighbour ranking is more reliable. (Bonus: it also makes the index 16× smaller and faster, so it's a strict win.)
</details>

### 🔨 Build + read

**Build:** On a dataset with instance-level labels (Stanford Online Products, CUB-200, or DeepFashion), extract embeddings from (a) an ImageNet ResNet-50 penultimate layer, (b) its logits, (c) a frozen CLIP image encoder, (d) a frozen DINOv2 ViT-B. Compute Recall@1/@10 for each with exact search. Then add PCA-whitening to 128-d for each and recompute. One table, four rows, and you will have a permanent, personally-verified answer to "which embedding should I use?"

**Read:** Radford et al., CLIP §3 (for what a retrieval-oriented embedding space looks like). Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022).

---

## 3.2 Learned vs. Handcrafted Representations

### The historical arc (and why it matters)

**The question this section answers: how did anyone do image retrieval before there was a network to produce embeddings?** You need one vector per image with the property from 3.1, and in 2003 you had no way to *learn* one. What you did have, from Module 1, was hundreds of SIFT descriptors per image — local, invariant, excellent. But "hundreds of 128-d descriptors, a different number per image" is not a vector; you cannot index it or dot-product it. So the whole classical era is a search for one move: **how do you aggregate a variable-length bag of local descriptors into a single fixed-length global vector?** Every method below is a different answer, and they get progressively more informative per byte. It is archaeology with a purpose: the *architecture* of those systems is still the architecture of modern retrieval.

**Bag of Visual Words (BoVW, Sivic & Zisserman 2003).** The first answer, stolen wholesale from text search — which is why it will feel familiar. A document is also a variable-length bag, and text search solved that by counting words against a fixed vocabulary. So invent a vocabulary for images, and count. Literally the text-search recipe applied to images:
1. Extract SIFT descriptors from all images.
2. k-means them into a "visual vocabulary" of $k$ words ($k = 10^4$–$10^6$).
3. Represent each image as a **histogram** of visual-word counts.
4. Weight with **tf-idf**, L2-normalise, and index with an **inverted file**.
5. Retrieve by cosine similarity on the sparse histogram, then **geometrically verify** the top candidates with RANSAC (Module 1.8).

**What is wrong with counting.** A histogram records *how many* descriptors landed in each vocabulary cell and throws away *where inside the cell* they landed. With $k=10^5$ words the cells are small enough that the loss is tolerable — but you are storing an enormous sparse vector to say very little per dimension. The fix is to keep the discarded information.

**VLAD (Vector of Locally Aggregated Descriptors, Jégou 2010).** Instead of counting, accumulate **residuals**: for each visual word $c_i$, sum $(x - c_i)$ over descriptors assigned to it. The residual is exactly the "where inside the cell" BoVW dropped, so each cell contributes a $D$-dim direction rather than one integer — and because each cell now carries $D$ numbers, you need far fewer cells. Concatenate → $k \times D$ vector; typically $k=64$, $D=128$ → 8192-d dense, then PCA to 128, against BoVW's $10^5$ sparse dimensions. Far more informative per byte than a count histogram.

**Fisher Vectors.** The probabilistic generalisation, and the natural next step once you read VLAD as "record the first moment of the descriptors in each cell." Replace hard k-means cells with a GMM and take gradients of the log-likelihood w.r.t. its parameters: the gradient w.r.t. the means recovers something very like VLAD's residuals, and the gradient w.r.t. the variances adds how *spread out* the descriptors in a cell are. First and second order statistics. State of the art immediately before CNNs.

**NetVLAD (Arandjelović et al., CVPR 2016)** — the bridge, and the place to slow down. VLAD is untrainable for exactly one reason: assigning a descriptor to a visual word is a hard `argmin`, which has zero gradient almost everywhere, so no error signal can flow back to the features. That single non-differentiable step is the whole obstacle — and the fix is the standard one you know from attention and from every discrete choice inside a network: **replace the hard argmin with a softmax over the same scores**, giving each descriptor a soft weight on every cluster. Differentiable soft assignment:

$$
\bar{a}_k(x_i) = \frac{e^{\mathbf{w}_k^\top x_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^\top x_i + b_{k'}}}, \qquad
V(k,j) = \sum_i \bar{a}_k(x_i)\,\bigl(x_i(j) - c_k(j)\bigr)
$$

so the whole pipeline — CNN features → VLAD aggregation → normalisation — becomes end-to-end trainable, and the cluster centres $c_k$ are no longer fixed by an unsupervised k-means run that never saw your task; they are learned *for retrieval*. Still a strong baseline for **visual place recognition** in 2026.

**Pooling for retrieval.** With the aggregator inside the network, which pooling? The default, global average pooling, is wrong here. Averaging asks "what is this image like on the whole," but retrieval asks "does this image contain the distinctive thing the query contains?" One very strong response to a rare discriminative pattern — a logo, a buckle — is exactly the evidence you want, and averaging it against thousands of bland background positions dilutes it to nothing. Max pooling has the opposite flaw: it keeps only the single strongest response and is fragile to noise. You want a point between the two, chosen by data rather than by you. That is **GeM (Generalised Mean) pooling**:

$$
f_c = \left(\frac{1}{|\mathcal{X}_c|}\sum_{x\in\mathcal{X}_c} x^p\right)^{1/p}
$$

Check the two ends yourself: at $p=1$ the exponents vanish and you have a plain mean — average pooling. As $p\to\infty$ the largest term dominates the sum so completely that the $1/p$ root returns it alone — max pooling. Every $p$ between is a soft emphasis on the strongest activations, and $p$ is *learnable*, typically converging to ~3 — distinctly closer to max than mean, which is the network telling you retrieval wants evidence over averages. GeM consistently beats GAP for retrieval for exactly that reason.

### The two-stage retrieval paradigm — the structural insight

**The tension that forces this design.** You want the accuracy of comparing every query against every database image with an expensive, precise method, and the speed of a single dot product. Incompatible at 40 million items — but only if you assume one comparison method for all items. Relax that and the answer appears: use the cheap method to cut 40 million to a thousand, then spend the expensive method on the thousand. Cost is now (cheap × 40M) + (expensive × 1000), and if the cheap stage's recall is high the accuracy is nearly the expensive method's. Classical and modern systems land on the same two stages independently, because the arithmetic forces it:

```
STAGE 1 — RECALL              STAGE 2 — PRECISION
global descriptor             local features + geometric verification
+ ANN index                   (or a cross-encoder reranker)
→ top-1000 candidates    →    → re-ranked top-10
cheap, approximate            expensive, accurate
```

**This is exactly your RAG pipeline, and the arithmetic above is why — not coincidence.** Bi-encoder + vector DB for recall, cross-encoder reranker for precision. The bi-encoder is cheap because it encodes query and document *separately*, so documents are encoded offline and the online cost is one dot product; the cross-encoder is expensive because it must see the pair *together*, so nothing can be precomputed. Vision splits identically: a global descriptor is computed per image offline, while geometric verification needs both images' local features in hand at once. Same shape, different modality:

| Retrieval concept | Text/RAG | Vision |
|---|---|---|
| Unit | text chunk | image (or region) |
| Encoder | sentence-transformer / bi-encoder | CNN/ViT image encoder |
| Index | Qdrant / pgvector / FAISS | FAISS / HNSW |
| Sparse counterpart | BM25 / tf-idf on tokens | tf-idf on **visual words** (BoVW) |
| Hybrid search | dense + BM25 fusion | global descriptor + BoVW |
| Reranker | cross-encoder | **RANSAC geometric verification** on local features |
| Query expansion | HyDE / multi-query | **α-query expansion**, diffusion on the neighbour graph |

**Say this mapping out loud in an interview.** It converts a "did a RAG project" line on your CV into evidence that you understand retrieval as a general system pattern.

**In your own words:** why can't you simply make stage 1 good enough to delete stage 2 — what kind of evidence does stage 2 supply that a better embedding cannot?

### 🎯 Top-1% distinction

1. **BoVW is tf-idf with an inverted index** — i.e. classical image retrieval literally borrowed the text-search stack, and the modern dense stack borrowed it back. Naming that round trip is a strong framing.
2. **Geometric verification is the original reranker**, and it's *stronger* than a learned reranker for instance retrieval because it enforces a hard geometric constraint rather than a soft similarity.
3. **GeM pooling over GAP for retrieval**, with the learnable $p$ and the reason.
4. **NetVLAD as the differentiable-classical bridge** — it shows you know the field didn't discontinuously jump.
5. Handcrafted features still win when: no training data, need for interpretability/auditability, and instance-level matching where geometry is verifiable.

### ✅ Mastery check

Design a landmark-recognition system (query photo → "which of 5M landmarks is this?"). You have 5M reference images and a 200 ms budget.

(a) Sketch the two stages and name a concrete method for each.
(b) Why not just do stage 1 with a bigger, better embedding and skip stage 2?
(c) How would you handle the case where the query is of a landmark not in the database?

<details><summary>Answer sketch</summary>
(a) <b>Stage 1:</b> global descriptor from a retrieval-trained backbone (DINOv2 or a GeM-pooled ResNet trained with ArcFace on Google Landmarks) → PCA-whiten to 256-d → HNSW or IVF-PQ index → top-100 candidates in ~10–20 ms. <b>Stage 2:</b> extract local features (SuperPoint or SIFT) for query and the 100 candidates, match with LightGlue or a ratio test, run <b>MAGSAC++</b> to count geometric inliers, rank by inlier count. Accept the top match only if inliers exceed a threshold. ~150 ms for 100 candidates with batching, or fewer candidates if tighter.
(b) Because a global descriptor is a lossy summary that cannot distinguish "similar-looking building" from "the same building." Two different neo-classical facades produce near-identical global embeddings; only <b>point-level geometric consistency</b> can separate them. Global similarity measures <i>appearance</i>; RANSAC inlier count measures <i>whether a single rigid/projective transform explains the correspondences</i>, which is a categorically stronger evidence type. Also practically: stage 2 gives you a calibrated confidence (inlier count) that a cosine similarity does not.
(c) This is the <b>distractor / open-set</b> problem, and it's the hard part of the real Google Landmarks benchmark. Handle it with (i) a <b>rejection threshold on the stage-2 inlier count</b>, not on the stage-1 cosine similarity (inlier count is far better calibrated); (ii) a large set of <b>distractor images</b> in your evaluation set so your threshold is tuned against realistic negatives; (iii) metrics that penalise false positives — use <b>µAP / GAP (global average precision)</b> rather than Recall@k, since Recall@k can't express "should have returned nothing." Naming the open-set problem explicitly is the mark of someone who has actually built one of these.
</details>

### 🔨 Build + read

**Build:** Implement BoVW end to end on a small dataset (Oxford5k or a few thousand of your own photos): SIFT → k-means vocabulary (k=10k) → tf-idf histograms → inverted index → retrieve → RANSAC rerank. Measure mAP with and without the geometric verification stage. The size of that gap is the lesson.

**Read:** Arandjelović & Zisserman, "NetVLAD" (CVPR 2016). Radenović et al., "Fine-tuning CNN Image Retrieval with No Human Annotation" (PAMI 2018) — the GeM pooling paper.

---

## 3.3 Similarity Metrics & Image Retrieval

### The metrics

**The question first: you have two vectors and need one number saying how alike they are. Does the choice matter?** In your RAG work you probably picked cosine because the library defaulted to it — and for normalised embeddings that was genuinely fine, for a reason we are about to derive. But if your vectors are *not* normalised the choice changes your results substantially, and once they are in an approximate index it changes whether the index is even *correct*. Hence: which of these five are true metrics, and which are not.

| Metric | Formula | Notes |
|---|---|---|
| **Euclidean (L2)** | $\|a-b\|_2$ | a true metric; what most ANN indexes are built for |
| **Cosine similarity** | $\dfrac{a\cdot b}{\|a\|\|b\|}$ | scale-invariant; **not** a metric (no triangle inequality) |
| **Angular distance** | $\arccos(\cos\theta)/\pi$ | the metric version of cosine |
| **Inner product (MIPS)** | $a\cdot b$ | not a metric; norm matters, so "nearest" ≠ "largest IP" |
| **Hamming** | popcount(XOR) | for binary codes (ORB, LSH, binary hashing) |

**The identity to have permanently loaded — two lines of algebra, so derive it rather than memorising it.** Expand the squared distance exactly as you would for scalars:
$$
\|a-b\|_2^2 = (a-b)\cdot(a-b) = \|a\|^2 + \|b\|^2 - 2\,(a\cdot b)
$$
True for any vectors. Now impose L2 normalisation, $\|a\| = \|b\| = 1$, so the first two terms are each exactly 1 and everything collapses to
$$
\|a-b\|_2^2 = 2 - 2\,(a\cdot b) = 2 - 2\cos\theta
$$
Read that as a function: $2-2x$ is strictly decreasing in $x$, and a strictly monotone transformation cannot reorder anything — so sorting ascending by L2 gives the *identical list* as sorting descending by cosine or inner product. All three induce the **same ranking**. Consequence: normalise once at index time and use whichever kernel your index implements fastest. Without normalisation the $\|a\|^2 + \|b\|^2$ terms do not cancel and vary per item, so the three genuinely disagree — with MIPS the odd one out, because L2 *penalises* a large database norm through $+\|b\|^2$ while inner product simply *rewards* it.

**Pause:** you switch a proximity-graph index from L2 to inner product on unnormalised vectors. Beyond "results change", why might the index now be *wrong* rather than merely different?

Because its correctness argument silently assumed a metric. Every pruning structure — trees, and the greedy graph traversal of 3.7 — rests on "this whole region is too far to contain the answer, skip it," which is the triangle inequality in disguise: if $d(q,c)$ is large and every point in the region is within $r$ of $c$, then every point is at least $d(q,c)-r$ from $q$. Maximum inner product has no triangle inequality, so that bound does not exist and the pruning arguments that make trees and proximity graphs *correct* don't hold — you can greedily walk away from the true maximum and never come back. Standard fixes: L2-normalise (turning MIPS into NN, since the identity then applies), or apply an asymmetric transformation embedding MIPS into an L2 problem. FAISS and HNSW support IP metrics but with weaker guarantees. **Knowing that MIPS ≠ NN is a genuine discriminator** and it comes up constantly in vector-DB work.

### Retrieval evaluation — and the mAP confusion

A retrieval system returns a *ranked list*, not a decision, so the metric must say something about ordering — not merely whether the right answer is in there somewhere. Given a query with a set of relevant items:

- **Precision@k / Recall@k** — the basics. Recall@1 and Recall@10 are the standard headline numbers for instance retrieval. Note the sharp edge: they see only a cut-off, blind to the order within it and everything past it.
- **Average Precision (AP)** for one query: $\text{AP} = \frac{1}{R}\sum_{k} P@k \cdot \mathbb{1}[\text{item } k \text{ is relevant}]$, where $R$ = number of relevant items. **mAP** = mean over queries. The indicator is doing the work: it fires only at ranks where a relevant item sits, recording the precision *up to that point*. A relevant item at rank 1 contributes $P@1 = 1$; the same item at rank 50 with nothing relevant above it contributes $1/50$. So AP is "the average precision you were enjoying at the moments you found something" — it rewards putting relevant items *early*, which Recall@k cannot see.
- **µAP / GAP (global AP)** — used in Google Landmarks; treats all (query, prediction) pairs globally so that confidence is comparable *across* queries. Essential when the system must sometimes return nothing.
- **nDCG** when relevance is graded rather than binary.
- **MRR** when there is exactly one right answer.

> ⚠️ **Retrieval mAP ≠ detection mAP.** They share a name and an averaging step and nothing else. Retrieval AP averages precision over the ranked list of *database items for a query*. Detection mAP (4.7) averages precision over a PR curve built by sweeping a *confidence threshold*, computed per class at an IoU threshold, then averaged over classes (and over IoU thresholds for COCO's AP@[.5:.95]). Conflating them in an interview is a visible error; distinguishing them unprompted is a nice signal.

### Making retrieval better without retraining

Retraining the encoder is the expensive lever. Before pulling it, note that a *single* query vector is a thin summary of what the user wants, and that most of the tricks below are one idea: **use the neighbourhood you just retrieved to build a better query than the user gave you.** If that sounds like HyDE and multi-query expansion from your RAG stack, it is the same idea — invented in vision first.

- **PCA-whitening** — decorrelate and rescale; consistently improves retrieval, often by several mAP points (3.1), for the reason argued there: it stops a few high-variance nuisance directions monopolising the distance.
- **α-Query Expansion (αQE):** re-query using a weighted average of the query and its top-$k$ results, $q' = \sum_i (\text{sim}_i)^\alpha x_i$. The logic: if the top few results are the same object seen differently, their average is a *less viewpoint-specific* description than any one of them — you have synthesised a canonical view. $\alpha$ is the safety valve: raising it concentrates weight on the most similar results, so a wrong item scraping in at rank 5 contributes almost nothing. Cheap, large gains, standard in landmark retrieval.
- **Database-side augmentation / diffusion:** propagate similarity over the k-NN graph of the database. This handles what αQE cannot — query and target so far apart (a 90° viewpoint change) that the target never enters the top-$k$ at all. If some third image sits between them, a 45° view, then query→intermediate→target is a short path even though query→target is a long edge. Strong for large viewpoint changes.
- **Re-ranking with local features** (3.2) — the biggest single win for instance-level tasks, because it supplies a different *kind* of evidence rather than more of the same.

**In your own words:** why does normalising your vectors make the L2-vs-cosine-vs-inner-product question disappear, and what breaks when they are not normalised?

### 🎯 Top-1% distinction

1. **L2 ≡ cosine ≡ IP for normalised vectors; MIPS is genuinely different otherwise, and MIPS breaks index pruning guarantees.**
2. **Retrieval mAP vs detection mAP.**
3. **Cosine is not a metric** (fails the triangle inequality); angular distance is. This matters for any algorithm that assumes metric structure.
4. **Query expansion and diffusion** — cheap post-hoc gains that most candidates have never heard of.
5. **Choose the metric that matches your evaluation:** if your product must sometimes say "no match," Recall@k is the wrong metric and you need a global-AP-style measure with a calibrated threshold.

### ✅ Mastery check

(a) Your embeddings are **not** normalised and you switch your index from L2 to inner product. Results change substantially. Explain what happened geometrically and which is "right."
(b) You report Recall@10 = 0.92 and your PM asks "so it's right 92% of the time?" Correct him.
(c) Give the one-line difference between retrieval mAP and detection mAP.

<details><summary>Answer sketch</summary>
(a) With unnormalised vectors, $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a\!\cdot\!b$. L2 search penalises database vectors with large norm; IP search <b>rewards</b> them. So switching to IP biases results toward high-norm items — which, for a CNN embedding, typically means images with strong, confident activations (large, centred, high-contrast objects). Neither is universally "right": IP is correct if your training objective was an inner-product objective (e.g. a dot-product-trained two-tower model, where norm encodes something like confidence/popularity); L2/cosine is correct if you want pure directional similarity. The real answer: <b>match the index metric to the metric your loss was trained with</b>, and if you don't know, L2-normalise so the question disappears.
(b) No. Recall@10 = 0.92 means that for 92% of queries, at least one relevant item appeared somewhere in the top 10. It says nothing about (i) whether the relevant item was ranked <i>first</i> (that's Recall@1), (ii) how many irrelevant items were also shown (precision), or (iii) how the system behaves on queries with <b>no</b> correct answer, which Recall@k cannot express at all. If the product shows one result, report Recall@1; if it must sometimes decline, report a precision/recall curve over a confidence threshold, or global AP.
(c) Retrieval mAP averages precision over the <b>ranked database list per query</b>; detection mAP averages precision over a <b>confidence-threshold sweep</b>, per class, at a given IoU, then averages over classes (and IoU thresholds for COCO).
</details>

### 🔨 Build + read

**Build:** Implement Recall@k, AP, mAP and nDCG from scratch and validate against a reference implementation. Then, on your 3.1 embeddings, measure the effect of (i) L2 normalisation, (ii) PCA-whitening to 128-d, (iii) αQE with $k=5,\alpha=3$. Three interventions, one table.

**Read:** Radenović et al., "Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking" (CVPR 2018) — the evaluation-protocol paper, unusually clear about what these metrics do and don't measure.

---

## 3.4 Siamese Networks

### Intuition

3.1 ended with a diagnosis: cross-entropy never constrains distances, so it cannot be trusted to produce a metric space. The remedy is to stop hoping and **write distance into the loss directly** — and once you do, something surprising falls out for free.

Train one network, apply it twice. Feed it two images, compare the outputs, and train so that "same" pairs come out close and "different" pairs come out far apart. Notice what is *missing*: at no point did you name a class. The supervision is a binary "same or not," so the network never learns a fixed label set — it learns a *notion of sameness* that generalises to categories it has never seen. That is not a bonus; it is the direct consequence of removing class identity from the objective.

Real-life anchor: face unlock. Your phone has exactly one photo of you enrolled. No classifier can be trained on one example — there is nothing to fit a $W_j$ to. But a network that learned "same person / different person" from millions of *other* people transfers immediately, because at enrolment it does not learn you, it merely *embeds* you.

### Architecture and loss

Two (or more) branches with **shared weights** — it is *one* network evaluated multiple times, which is why the gradient from both branches accumulates into the same parameters. **Why sharing is not optional, and why the name misleads:** "Siamese" suggests twins, i.e. two things, and beginners implement two networks. But then branch A develops a coordinate system unrelated to branch B's, and comparing $f_A(x_1)$ to $f_B(x_2)$ is comparing measurements in different units — meaningless even for identical inputs. One shared encoder guarantees both inputs land in a *common* space, which is the only thing that makes a distance between them mean anything.

**Contrastive loss** (Hadsell, Chopra & LeCun, CVPR 2006), with $D = \|f(x_1) - f(x_2)\|_2$ and $Y=1$ for a similar pair:

$$
\mathcal{L} = Y\,D^2 + (1-Y)\,\max(0,\; m - D)^2
$$

Read the two terms as two separate jobs, since $Y$ switches one on and the other off:

- Similar pairs ($Y=1$): the loss is $D^2$, pull together, quadratically, with no floor — the loss keeps pulling all the way to $D=0$.
- Dissimilar pairs ($Y=0$): the loss is $\max(0, m-D)^2$, push apart **only until** $D \ge m$. At that point $m - D$ goes negative, the hinge clips it to zero, and the gradient vanishes — the pair is "solved" and shouldn't waste capacity.

**Why the margin — try designing the loss without it and watch it fail.** The natural first attempt at "push dissimilar things apart" is $-D^2$. But that has no minimum; it decreases without bound as $D\to\infty$, so the cheapest way to reduce the loss is not to organise anything, it is to scale every output up by a constant. Distances all grow, the loss falls forever, and the *relative* geometry — the only thing you cared about — never improves. The hinge stops this: past $m$ the term is exactly zero, so inflating buys nothing. The margin makes the objective bounded below *and* supplies a definition of "far enough."

### The problem with contrastive loss

But look at what the margin costs. $m$ is in **absolute distance units**, so writing it down requires knowing in advance what scale your embedding space will settle at — and that scale drifts, so a value that was demanding at epoch 1 may be trivial at epoch 50. Worse, the space is not homogeneous: some classes are naturally tight (photos of one document scan), others genuinely diffuse (photos of "a chair"). One global $m$ is simultaneously too loose for the tight and too strict for the diffuse.

The way out is noticing you never cared about absolute distances. Retrieval needs the right answer to rank *above* the wrong ones — a statement about ordering, which is scale-free. **Triplet loss (3.5) fixes exactly this by making the constraint *relative*.**

**In your own words:** what goes wrong if you drop the margin, and what goes wrong if you keep it?

### Where Siamese networks show up

- **Signature verification** — the original 1993 application (Bromley et al., at Bell Labs, with LeCun).
- **Face verification** — DeepFace, DeepID, FaceNet.
- **One-shot / few-shot learning** — Koch et al.'s Omniglot Siamese net; then Matching Networks, Prototypical Networks (which use class prototypes = mean embeddings, and a softmax over negative distances).
- **Visual object tracking** — **SiamFC / SiamRPN / SiamRPN++**: embed the target template once, then cross-correlate it against each new frame's feature map. Tracking becomes template matching *in embedding space*. (Forward link to 5.7.)
- **Change detection, duplicate detection, patch matching** (MatchNet, HardNet).

### 🎯 Top-1% distinction

1. **"Siamese" means shared weights, i.e. one network applied twice** — not two networks. And the reason is that a shared encoder guarantees a *common* embedding space.
2. **The margin's role and its weakness** — absolute-distance margins don't transfer across classes or across training time. That's the motivation for triplet, and later for angular-margin losses.
3. **Pseudo-Siamese / asymmetric variants** exist deliberately: BYOL and SimSiam (3.6) use an *asymmetric* architecture (a predictor on one branch, stop-gradient on the other) specifically to avoid collapse without negatives. So "shared weights" is a design choice with a purpose, and breaking it is also a design choice with a purpose.
4. **Siamese trackers are the same idea in a different costume** — that connection across Modules 3 and 5 is a good one to volunteer.

### ✅ Mastery check

You train a Siamese net with contrastive loss for signature verification. Loss goes to near zero, but at test time the system accepts forgeries.

(a) Give two mechanisms that produce "low loss, bad verification."
(b) How would switching to triplet loss help, and what new problem does it introduce?
(c) Your positive pairs are two scans of the *same physical signature*. What's wrong with that, and what should a positive pair be?

<details><summary>Answer sketch</summary>
(a) (i) <b>Easy negatives.</b> If dissimilar pairs are sampled randomly, almost all are trivially far apart and already beyond the margin, contributing zero gradient. The loss is low because the task is easy, not because the embedding is good — and the decision boundary was never trained near the region that matters (skilled forgeries). (ii) <b>Margin/scale degeneracy.</b> The network can satisfy the loss by inflating the overall embedding scale so that all negatives exceed $m$, without improving the <i>relative</i> geometry. Also possible: <b>collapse of the positive term</b> — mapping everything from one writer to a single point regardless of content, which satisfies $D^2 \to 0$ but throws away discriminative structure.
(b) Triplet replaces the absolute constraint with a <b>relative</b> one: the negative must be farther than the positive <i>by a margin</i>, so the scale degeneracy disappears and the margin has a consistent meaning across classes. New problem: <b>triplet mining</b> — with $N$ samples there are $O(N^3)$ triplets and the overwhelming majority are already satisfied (zero gradient), so naive sampling stalls training. You must mine semi-hard or batch-hard triplets, which adds machinery and its own failure mode (collapse under too-hard negatives).
(c) Two scans of the same physical signature only teach invariance to <b>scanner noise</b> — a nuisance factor, not the real intra-class variation. A signature verification system must be invariant to the natural variation in how the <i>same person signs on different occasions</i>. Positive pairs must be <b>two genuine signatures by the same writer, written at different times</b>. And critically, the hard negatives must be <b>skilled forgeries of that writer</b>, not signatures by random other people. The rule generalises: <b>your positive pairs define the invariances you learn, and your negative pairs define the discriminations you learn</b> — get either wrong and the loss is measuring the wrong thing. This is the same principle that makes augmentation choice decisive in SimCLR (3.6).
</details>

### 🔨 Build + read

**Build:** Train a Siamese net with contrastive loss on Omniglot or a face dataset. Then run the diagnostic: plot the *distribution* of positive-pair and negative-pair distances at epochs 1, 5, 20. You should see the negative distribution pile up just past the margin (zero gradient) — a direct visualisation of why mining matters.

**Read:** Hadsell, Chopra & LeCun, "Dimensionality Reduction by Learning an Invariant Mapping" (CVPR 2006). Koch et al., "Siamese Neural Networks for One-shot Image Recognition" (ICML workshop 2015).

---

## 3.5 Triplet Loss & Hard-Negative Mining 🟡

### Intuition

Instead of saying "same pairs must be closer than 0.3 and different pairs farther than 1.0" (which requires knowing the right numbers), say: **"whatever the scale, an anchor must be closer to a positive than to a negative, by a comfortable margin."** That's a relative constraint, and it transfers.

Notice the structural change: contrastive loss compares a pair against an *external* reference, the number $m$; triplet loss compares two distances *to each other*. The reference has moved inside the data, so no prior knowledge of scale is needed. That is also exactly what your metric measures — Recall@1 asks whether the positive outranks the negatives, not how far away anything is — so this is the first loss in the module that optimises the thing you actually evaluate.

**And here the section's real subject appears.** The loss is four lines and you will understand it immediately. What makes 3.5 a *topic* is that writing it down does not give you a trainable system, for reasons that are combinatorial rather than mathematical. Read the loss quickly; spend your time on the mining.

### The loss

Given anchor $a$, positive $p$ (same identity), negative $n$ (different identity):

$$
\mathcal{L}(a,p,n) = \bigl[\,\|f(a)-f(p)\|_2^2 - \|f(a)-f(n)\|_2^2 + \alpha\,\bigr]_+
$$

where $[\cdot]_+ = \max(0,\cdot)$ and $\alpha$ is the margin (FaceNet used $\alpha = 0.2$ with L2-normalised embeddings). Read the bracket as a condition, not a formula: the loss is zero exactly when $d(a,n)^2 \ge d(a,p)^2 + \alpha$. Any triplet satisfying that contributes no loss and, because the hinge is flat there, **no gradient either**. Hold onto that; it is the entire problem.

**Embeddings are L2-normalised onto the unit hypersphere**, and this plugs a specific hole. Without it the network has a cheap way to satisfy $d(a,n)^2 - d(a,p)^2 \ge \alpha$: scale all embeddings by $c$ and both squared distances scale by $c^2$, so their *difference* does too — any positive gap can be inflated past $\alpha$ by growing norms, geometry unchanged. That is the contrastive loss's scale degeneracy reappearing in relative form, and the unit sphere removes the free parameter $c$ entirely. It also makes $\alpha$ interpretable: on the sphere $\|a-b\|^2 = 2-2\cos\theta$ (3.3), so a squared-L2 margin is a cosine margin — $\alpha = 0.2$ means the negative's cosine must be at least 0.1 below the positive's, an angular statement valid identically everywhere.

### The mining problem — this *is* the topic

**Do the counting first, because the numbers are the argument.** Choose an anchor ($N$ ways), a positive from its identity, a negative from any other: $O(N^3)$ triplets, on the order of $10^{15}$ for a modest $N = 10^5$ re-ID dataset. That alone is not fatal — SGD is happy with a random sample of an enormous set. The fatal part is *which* ones random sampling gives you.

Picture the space after a few hundred steps. The network already has the easy structure: faces far from cars, one person's photos broadly clumped. Draw a random triplet — anchor a face, negative a uniformly random other image, almost certainly a completely different-looking person. Is that negative further than the anchor's own positive by 0.2? Almost certainly yes, because that is exactly the structure the network already has. The hinge clips, and the gradient is **exactly zero**, not small.

That distinction is what separates this from ordinary diminishing returns. An easy classification example still contributes a small gradient. A satisfied triplet contributes *nothing* — not a weak vote, an absent one. So a batch of 128 in which 127 are satisfied is not a batch with noisy gradients; it is effectively a **batch of size one** with the learning rate calibrated for 128. The estimate is dominated by whichever single triplet happened to be hard, variance is enormous, and the effective step collapses. And it worsens monotonically: every triplet the network solves is permanently removed from the pool of ones that could teach it anything.

Random sampling therefore produces a signal that decays to nothing within an epoch or two, while the loss curve tells a comforting lie — it goes to zero because the *sampled task* became trivial, not because the embedding became good. **Triplet loss without mining does not work**; the counting above is why. Everything below is machinery for finding the informative $10^{-6}$ of triplets without enumerating the other $10^{15}$.

**Pause:** so why not always pick the *hardest* available negative, the one closest to the anchor? It has the largest gradient by construction. Predict what goes wrong before reading on.

Categorise negatives relative to a given $(a,p)$:

$$
\underbrace{d(a,n) < d(a,p)}_{\textbf{hard}} \quad\big|\quad \underbrace{d(a,p) < d(a,n) < d(a,p)+\alpha}_{\textbf{semi-hard}} \quad\big|\quad \underbrace{d(a,n) > d(a,p)+\alpha}_{\textbf{easy (zero gradient)}}
$$

These are just the three ways the hinge can behave. **Easy**: the constraint already holds with room to spare — flat part, zero gradient, useless. **Hard**: the ordering is outright wrong, the negative *closer* than the positive, so the triplet is a live retrieval error carrying the largest possible gradient. **Semi-hard**: the ordering is correct but not by enough — right yet under-confident, gradient real but bounded.

**Mining strategies:**

| Strategy | What it does | Trade-off |
|---|---|---|
| **Random** | uniform triplets | ~all easy ⇒ no learning |
| **Batch-all** | average loss over all non-zero triplets in the batch | many near-zero terms dilute the average; use *batch-all with only non-zero terms averaged* |
| **Semi-hard** (FaceNet) | pick the negative that is farther than the positive but within the margin | avoids the collapse that hardest-negative mining causes; FaceNet's choice |
| **Batch-hard** (*In Defense of the Triplet Loss*, Hermans et al. 2017) | for each anchor, take the **hardest positive and hardest negative *within the mini-batch*** | strong and simple; works because "hardest in a batch of 128" is moderately hard, not globally hardest |
| **Distance-weighted sampling** (Wu et al., ICCV 2017) | sample negatives inversely to the density of pairwise distances on the hypersphere | corrects the fact that in high dimensions almost all random pairs sit at $\sqrt{2}$; gives a well-conditioned spread of difficulties |

**Why the *hardest* negative is dangerous — the mechanism, not the verdict.**

Ask what an image must look like to be the globally hardest negative for anchor $a$: labelled a *different* identity, yet sitting closer to $a$ than $a$'s own positives. Two kinds qualify. Genuinely difficult ones — an identical twin, the same handbag in another colourway. And **mislabelled** ones: photos that actually *are* the anchor's identity but carry the wrong label. The second kind is not merely present in the hard set, it is *concentrated* there, because a mislabelled image is by construction about as close to the anchor as an image can get. Every real dataset has a percent or two of label noise, and hardest-negative mining is a machine for finding precisely those and nothing else.

Now follow the gradients. Each such triplet demands, with maximum confidence, "push these apart" — about two images of the same person. Huge gradient, wrong direction. The encoder now faces contradictory demands: separate $x$ from $x'$ while other triplets insist on keeping them together. One arrangement makes every contradictory demand equally satisfiable, and gradient descent finds it — send *everything* to the same point. If $f(x)=\text{const}$ then $d(a,p)=d(a,n)=0$ for every triplet, the hinge evaluates to $0-0+\alpha = \alpha$, and the loss parks at a flat, stable $\alpha$: no triplet is violated more than any other, so nothing pulls it apart again. That is the **collapse**, and its signature is unmistakable — training loss falls, stops at exactly the margin value, and never moves.

The fix follows from the diagnosis: the problem was sampling the extreme tail where the noise lives, so sample just below it. Semi-hard mining takes negatives that violate the margin but not the ordering, excluding the pathological cases by definition. Batch-hard takes the hardest negative *within a batch of 128*, and the batch limit is doing statistical work — the hardest of 128 random samples comes from the bulk of the difficulty distribution, whereas the hardest of $10^5$ comes from the tail. Difficulty high but bounded.

**Batch construction: PK sampling.** In-batch mining only works if the batch contains something to mine. Draw 128 images uniformly from 10,000 identities and most anchors will have *no* positive present — not a single valid triplet. So build each batch as $P$ identities × $K$ images (e.g. 32×4 = 128). Every anchor then has $K-1$ positives and $(P-1)K$ negatives across $P-1$ identities, which is what makes "hardest in this batch" meaningful rather than accidental. **Batch construction is part of the loss** in metric learning — most candidates miss this, because in ordinary supervised training the batch really is just a random sample.

### The successors — and the important currency correction

Triplet loss was the standard for face recognition around 2015. **It largely isn't any more**, and the reason is a lesson in reading a problem correctly. All the machinery above — mining, PK sampling, collapse diagnostics — exists to solve one problem: *choosing which comparisons to make*. So the field asked whether there is a formulation where you never have to choose, and found one by going back to the softmax classifier rejected in 3.1 and fixing it rather than replacing it. A softmax over $K$ classes already compares each sample against *every* class simultaneously, in the denominator — no sampling problem because there is no sampling. What it lacked was any constraint on distances. So put one in. Hence **margin-based softmax losses**, which get the metric-learning benefit *without any mining*:

$$
\mathcal{L}_{\text{softmax}} = -\log\frac{e^{W_{y}^\top f}}{\sum_j e^{W_j^\top f}}
\;\xrightarrow{\text{normalise } W, f}\;
-\log\frac{e^{s\cos\theta_y}}{e^{s\cos\theta_y} + \sum_{j\ne y} e^{s\cos\theta_j}}
$$

Follow the arrow — the normalisation step is the whole trick. In a plain softmax the logit is $W_j^\top f = \|W_j\|\,\|f\|\cos\theta_j$, contaminated by two magnitudes that have nothing to do with similarity. Force $\|W_j\| = \|f\| = 1$ and it becomes *purely* $\cos\theta_j$. The classifier now lives on the unit hypersphere, exactly where triplet loss lived, and each $W_j$ is a learned **prototype** for that identity. The scale $s$ is reintroduced because a softmax over values confined to $[-1,1]$ is far too soft to drive the loss down. Then insert a margin:

| Loss | Margin form | Note |
|---|---|---|
| **SphereFace** (A-Softmax) | $\cos(m\theta_y)$ — multiplicative angular | first of the family; hard to optimise |
| **CosFace** | $\cos\theta_y - m$ — additive cosine | stable, simple |
| **ArcFace** | $\cos(\theta_y + m)$ — **additive angular** | the standard; margin is a constant *angular* distance, geometrically the cleanest |

All three margins do the job $\alpha$ did in the triplet hinge: make the correct class's logit *artificially worse* during training, so the network must overshoot to be right, and the extra angle it is forced to buy is the inter-class separation you wanted. The rows differ only in *how* the angle is penalised, and ArcFace wins because $\cos(\theta_y+m)$ subtracts a constant number of radians — a fixed angular distance meaning the same everywhere on the sphere, unlike CosFace's fixed cosine offset (a large angle near $\theta=0$, a small one near $\pi/2$) or SphereFace's scale-dependent multiplicative $m\theta$. Typical ArcFace settings: scale $s=64$, margin $m=0.5$ rad.

**Why these won:** no triplet mining, no batch-construction machinery, stable training, and a margin with a consistent geometric meaning everywhere on the hypersphere. Every serious face-recognition system in 2026 uses an ArcFace-family loss (often with sub-centre ArcFace for noisy data), not triplet loss. **Triplet loss remains relevant** where you cannot enumerate classes — very large or open/unbounded label sets, ranking with graded relevance, or when identities appear and disappear continuously.

Note the trade being made, since it explains exactly when triplet survives: ArcFace buys "no sampling" by keeping a weight vector $W_j$ for every training identity, so it needs the identity set to be **finite and enumerable in advance**. Triplet loss needs no such table — it only ever compares samples to samples — which is why it remains the right tool wherever the label set is unbounded or continuously changing.

**Being able to say "the field moved from triplet to ArcFace-style margin softmax, and here's why" is a strong currency signal** — most candidates learn triplet loss from a 2018 blog post and stop.

**In your own words:** why is a triplet whose loss is zero worse for training than a triplet whose loss is merely small — and what does that force you to build?

### 🎯 Top-1% distinction

1. **State that mining is the crux**, not the loss formula. "Triplet loss without mining does not train."
2. **Explain why hardest-negative mining collapses** (label noise → huge wrong gradients → constant embedding at loss $\alpha$) and why semi-hard/batch-hard avoids it.
3. **PK batch construction is part of the method.**
4. **L2 normalisation removes the scale degeneracy and makes the margin angular.**
5. **Know the ArcFace family and why it displaced triplet loss** — and when triplet is still right.
6. **The $O(N^3)$ complexity and the fact that almost all triplets are trivially satisfied.**

### ✅ Mastery check

You train a person re-identification model with triplet loss. Training loss plateaus at exactly the margin value $\alpha$ and never decreases.

(a) Diagnose precisely. What has the embedding become?
(b) Three fixes, ranked, with the reasoning for each.
(c) Your colleague proposes switching to ArcFace. Under what condition is that a bad idea for *this* task?

<details><summary>Answer sketch</summary>
(a) The loss sitting exactly at $\alpha$ means $d(a,p) = d(a,n)$ for every triplet — the model has <b>collapsed to a constant (or near-constant) embedding</b>. All distances are zero (or identical), so the hinge evaluates to $0 - 0 + \alpha = \alpha$. This is the degenerate optimum of triplet loss and it is the classic symptom of over-aggressive hardest-negative mining, too large a learning rate early on, or a margin that is too large relative to what the encoder can achieve.
(b) 1. <b>Switch from hardest-negative to semi-hard or batch-hard mining</b>, and check for label noise in the identities — collapse is usually driven by mislabelled samples appearing as impossibly hard negatives. Highest-impact and addresses the root cause. 2. <b>Fix batch construction (PK sampling) and reduce the margin</b> ($\alpha$ 0.2–0.3 with normalised embeddings); also add a warmup where you train with an auxiliary classification (ID) loss so the embedding has structure before metric training begins — the standard re-ID recipe is in fact <b>ID loss + triplet loss jointly</b>, precisely because the classification term prevents collapse. 3. <b>Lower the learning rate and add BNNeck</b> (a BN layer between the triplet-loss feature and the ID-loss classifier — a standard re-ID trick that resolves the conflict between the two losses' preferred geometries).
(c) ArcFace requires a <b>fixed, enumerable set of training identities</b>, since it maintains a weight vector $W_j$ per class. It is a bad idea if (i) the number of identities is enormous (the classifier weight matrix becomes the memory bottleneck — millions of identities × 512 dims), or (ii) identities are <b>open-ended and continuously arriving</b>, e.g. a re-ID system deployed across new camera networks where you cannot retrain per new person. Note ArcFace still generalises open-set at *inference* (you discard $W$ and use the embedding), so the constraint is on <i>training</i>, not deployment — that distinction is worth stating.
</details>

### 🔨 Build + read

**Build:** Implement triplet loss with three mining strategies (random, semi-hard, batch-hard) and PK batch sampling. Train on CUB-200 or Market-1501 and plot Recall@1 vs epoch for all three on one figure — random will flatline. Then implement ArcFace and add it as a fourth curve. That single figure is a complete, defensible answer to any metric-learning interview question.

**Read:** Schroff et al., "FaceNet" (CVPR 2015) §3.2 (triplet selection). Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification" (2017) — the batch-hard paper, very practical. Deng et al., "ArcFace" (CVPR 2019).

---

## 3.6 Contrastive Learning Foundations: SimCLR & MoCo 🟡

> The gap analysis flagged this because DINO (6.6) otherwise arrives out of nowhere. It is also the paradigm that produced CLIP (6.5), so this section is load-bearing for Module 6.

### Intuition

3.4 and 3.5 both needed someone to tell you which pairs are "same." That is the bottleneck — labelled identity data is expensive, and nobody labelled the internet. So: **is there any way to obtain a correct "these two are the same thing" label without a human?**

There is, and it is almost embarrassingly cheap. Take one photo and produce two differently-augmented views — crop differently, jitter the colours, flip one. You *know* they depict the same thing, because you made them from the same file: a free, guaranteed-correct positive pair. Any view from a *different* photo is almost certainly not the same thing, giving free negatives. Declare "these two are the same; everything else in the batch is not," pull the pair together and push the rest apart, and after enough images the network has learned what visual similarity means with zero annotation.

The supervision has not vanished, it has moved. In 3.5 a human specified the invariances by curating positive pairs; here **you** specify them by choosing the augmentations. That relocation is the most important thing in this section, and it is what makes the colour-jitter story below more than a hyperparameter anecdote.

Real-life anchor: this is how modern vision backbones are pretrained. DINOv3 saw 1.7 billion images with no labels at all.

### The InfoNCE loss

You could plug these free pairs straight into the triplet machinery from 3.5 — and then you would inherit its mining problem. Contrastive learning takes the other route, the one 3.5 ended on: compare against *everything* at once, in a softmax. Here is how that is built.

For a batch of $N$ images, produce $2N$ augmented views. For a positive pair $(i,j)$:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp\bigl(\text{sim}(z_i, z_j)/\tau\bigr)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\ne i]}\exp\bigl(\text{sim}(z_i, z_k)/\tau\bigr)}
$$

with $\text{sim}(u,v) = u^\top v/(\|u\|\|v\|)$ (cosine).

**Don't memorise this — recognise it.** Write ordinary softmax cross-entropy for a $C$-way problem with logits $\ell_c$ and correct class $y$: $\mathcal{L} = -\log\frac{e^{\ell_y}}{\sum_c e^{\ell_c}}$. Now match terms. The candidate set is the other $2N-1$ views, so $C = 2N-1$. The "logit" for candidate $k$ is $\text{sim}(z_i,z_k)/\tau$ — a similarity standing in for $W_k^\top f$, which after 3.5's normalisation argument is exactly what a logit *is* on the hypersphere. The "correct class" is $j$, the other view of the same photo. Substitute and you have written InfoNCE. It is not a new kind of objective at all.

**So read it as a classification problem:** given anchor $i$, pick its partner out of $2N-1$ candidates — a softmax cross-entropy over a $(2N-1)$-way task **whose labels are free**, since the label is "whichever view I cropped from the same file" and the data loader knows it for nothing. That reframing also tells you what the negatives are doing: they are the denominator, the classes you must rank below the right one, and there are $2N-2$ because the batch supplied them. Batch size *is* the number of classes here, which is why SimCLR cares about it so much.

**Temperature $\tau$ is not a minor hyperparameter.** In the classification reading it is the inverse of the logit scale $s$ from ArcFace, playing the same role from the other direction. It scales the logits before the softmax:
- **Small $\tau$** (0.05–0.1) ⇒ sharp distribution ⇒ gradient concentrates on the **hardest negatives** ⇒ strong uniformity on the hypersphere, but sensitive to false negatives (semantically identical images labelled as negatives).
- **Large $\tau$** (0.5–1.0) ⇒ soft ⇒ treats all negatives similarly ⇒ more tolerant, less discriminative.

The mechanism behind those bullets is just the softmax gradient. Differentiating softmax cross-entropy weights each negative by its softmax probability, so negatives the model currently believes *are* the answer receive nearly all the gradient and the rest almost none. Dividing by a small $\tau$ magnifies differences between similarities, sharpening that distribution and handing the gradient almost entirely to the closest negative. **Small $\tau$ is therefore automatic hard-negative mining** — the thing you built by hand in 3.5, obtained free as a side effect of the softmax. And it inherits the same danger: the hardest negative may be a semantically identical image that merely came from a different file (a **false negative**), and a sharp $\tau$ punishes the model hardest for getting exactly that one "wrong."

Wang & Isola (ICML 2020) decomposed the contrastive objective into **alignment** (positives close) and **uniformity** (embeddings spread evenly on the hypersphere) and showed $\tau$ trades between them. Being able to name that decomposition is a strong signal. InfoNCE is also a lower bound on mutual information (van den Oord et al., CPC), though later work showed the MI framing doesn't fully explain why it works.

### SimCLR (Chen et al., ICML 2020) — four ingredients, all necessary

**Start with an experiment that failed, and work out why.** You implement everything above faithfully, with random resized crop plus horizontal flip — the two most standard, most obviously correct vision augmentations there are. The contrastive loss trains beautifully and goes very low: the model matches views to their partners with high accuracy. You linear-probe the backbone and the features are close to useless.

The loss is genuinely being minimised, so the optimiser is not at fault; the task must be. **Pause: what strategy could match two crops of the same photo while requiring no understanding of objects whatsoever?**

Ask what two random crops of one photograph reliably share. Not shape — they may barely overlap, showing a wheel and a windscreen. Not position, which crop randomises by design. But they came from one exposure, one white balance, one illumination, while two *different* photographs almost always differ in overall colour cast. The network can therefore achieve near-perfect InfoNCE by computing something close to a mean RGB and matching on it. Not a bug in the loss; a correct solution to a badly specified task — and the fix is forced, namely **break the shortcut by making what it relies on unreliable.**

1. **Composition of strong augmentations.** The ablation is the paper's most important figure: no single augmentation suffices, and the critical pair is **random crop + colour distortion** — exactly what the argument above predicts, since crop supplies the difficulty and colour jitter closes the escape hatch. Why colour jitter is essential: two crops of the *same* image share almost identical colour histograms, so the network can solve the contrastive task by comparing colour statistics — a **shortcut** that yields useless features. Colour jitter destroys the shortcut and forces the network to use shape and structure. **This example is the perfect illustration of "the augmentation set defines what the representation learns"** (2.9) — including the cost: the model is now deliberately colour-blind, a disaster if your task is retrieving red handbags. There is no free invariance.
2. **A non-linear projection head** $z = g(h) = W_2\,\sigma(W_1 h)$. Train the loss on $z$; **use $h$ for downstream tasks.** SimCLR measured ~+10 points of linear-probe accuracy from this alone. The explanation (3.1): the contrastive task demands invariance to colour and orientation, so $z$ must discard that information; $h$, one layer upstream, retains it.
3. **Large batches** (up to 8192) and long training. Negatives come from the batch, so batch size *is* the number of negatives. Needs LARS to train stably at that scale.
4. **Careful normalisation** — cosine similarity, i.e. L2-normalised embeddings.

### MoCo (He et al., CVPR 2020) — decoupling negatives from batch size

SimCLR's dependence on 8192-sample batches means TPU pods, and the coupling is structural: batch size *is* the number of classes, so more negatives means a bigger batch means more accelerator memory. MoCo attacks the coupling itself. **The negatives never needed to come from the current batch** — they only need to be embeddings of other images. So keep a **queue** of previously computed keys, making the number of negatives ($K = 65536$) a free parameter of queue length rather than a consequence of batch size (256). Storage is trivial: 65k vectors of 128 dims is tens of megabytes, versus the activations of a 65k-image forward pass.

The obvious objection: those keys were embedded by *older* versions of the encoder, thousands of steps ago. If the encoder has moved since, comparing today's query against a stale coordinate system is meaningless — worse, the model can reduce the loss simply by *drifting* away from where the old keys sit, which teaches it nothing. The negatives must remain comparable to the query.

You cannot re-embed 65k keys every step, so the only lever left is to make the key encoder change slowly enough that a key from 65k samples ago is still approximately valid. Hence a **momentum encoder** for keys:

$$
\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q, \qquad m = 0.999
$$

Put a number on "very slowly": with $m = 0.999$ the key encoder retains $0.999^t$ of its state after $t$ steps, and $0.999^{1000}\approx 0.37$, so substantial change takes on the order of a thousand steps — which at batch 256 is exactly the timescale over which the 65k queue turns over. The slowness is not a regularisation nicety; it is tuned so the oldest key is still comparable to the newest query. **The large $m$ is essential** — the ablation shows $m=0.9$ (a memory of ~10 steps) performs far worse than $m=0.999$, and $m=0$ (sharing the encoder) fails outright. **The counter-intuitive bit worth remembering:** the branch that lags furthest behind is the one that makes the method work.

**Shuffling BN — a second shortcut, subtler than colour.** Recall from 2.4 that BatchNorm makes each sample's output depend on the *other samples in its batch*, a strange property we normally tolerate. Here it is fatal. If a query and its matching key are computed in the same GPU's batch they are normalised by the same mean and variance, leaving a faint common signature no other pair shares — and the network discovers it can identify the correct key by detecting "these two were normalised together," a fact about the data loader rather than the image. So the model can **cheat**: BN's batch statistics leak information that identifies which key belongs to which query, and the network exploits it instead of learning features. MoCo shuffles the sample order across GPUs before the key encoder so that query and key see different BN statistics. **This is the cleanest concrete example of "BN's batch dependence causes subtle bugs" (2.4) and it is a great cross-module connection to volunteer.**

**MoCo v2** adds SimCLR's MLP projection head and stronger augmentation to MoCo's framework — better than both predecessors at 256 batch size. **MoCo v3** adapts it to ViTs and identifies a training-instability failure (patch-embedding gradient spikes), fixed by freezing the patch projection.

### Negative-free methods — and the collapse problem

Step back and ask what the negatives were actually *for*. The tempting answer is "so the model learns what differs," but the real one is defensive. Take the objective "make two views of an image land close together" alone and find its global optimum: $f(x) = \text{const}$ for every image. Perfect alignment, exactly zero loss, and a representation that has discarded the entire dataset. This is the same collapse that ambushed us in 3.5, arriving from a different direction — there it was contradictory gradients, here a degenerate optimum the loss actively rewards. **Negatives are one way to prevent it**: you cannot map everything to one point while also being required to push different images apart, so negatives supply *spread* (Wang & Isola's uniformity term).

That reframing — negatives as an anti-collapse device rather than an end in themselves — raises the question the next generation asked: **if spread is all we need, is there a cheaper way to get it?** The table is a catalogue of answers:

| Method | Mechanism against collapse | Notes |
|---|---|---|
| **BYOL** (2020) | asymmetry: online branch has an extra **predictor** MLP; target branch is an **EMA** of the online one with **stop-gradient** | no negatives at all; the discovery that this doesn't collapse was genuinely surprising. Early analyses implicated BN as an implicit contrastive term; later work showed the predictor + EMA dynamic is sufficient |
| **SimSiam** (2021) | predictor + **stop-gradient only** — no momentum encoder, no negatives, small batches | the minimal ablation: it proves **stop-gradient is the essential ingredient**. The clearest paper in the family |
| **Barlow Twins** (2021) | make the cross-correlation matrix between the two views' embeddings equal the identity — on-diagonal ⇒ invariance, off-diagonal ⇒ **redundancy reduction** | no negatives, no asymmetry, no momentum |
| **VICReg** (2022) | explicit **V**ariance (hinge on per-dim std), **I**nvariance (MSE between views), **C**ovariance (decorrelation) terms | makes the anti-collapse mechanism explicit and tunable |
| **DINO** (2021) | self-distillation, student/teacher with **centring + sharpening** to prevent both collapse modes | leads directly into 6.6 |

**The unifying question — "what prevents collapse?" — is the single best interview question in this area**, and the answer is: something must impose *spread* on the embedding distribution. Negatives do it explicitly (uniformity term); stop-gradient + predictor does it through the optimisation dynamics; Barlow Twins/VICReg do it through explicit statistical constraints; DINO does it with centring and sharpening.

**In your own words:** InfoNCE is a classification loss — what are its classes, and where does the label come from?

### How SSL representations are evaluated

There are no labels in training, so "does the loss go down?" tells you almost nothing — the colour-jitter story had a low loss and a worthless model. Evaluation must be *external*: freeze the representation and ask a downstream task how useful it is. Each protocol below asks that differently, and the differences matter.

- **Linear probe** — freeze the backbone, train a linear classifier on ImageNet, report top-1. The standard headline number.
- **k-NN probe** — no training at all; classify by nearest neighbours in feature space. Measures whether the *metric structure* is good, which is exactly what retrieval cares about. Often more informative than the linear probe for our purposes.
- **Semi-supervised fine-tuning** with 1% / 10% of ImageNet labels.
- **Transfer** to detection/segmentation — the test that matters most, and where contrastive methods historically underperformed masked methods (6.4).

### 🎯 Top-1% distinction

1. **The colour-jitter shortcut** — a concrete, memorable instance of shortcut learning, and it proves you read the ablations rather than the abstract.
2. **The projection-head finding and its explanation.**
3. **MoCo's queue + momentum as the answer to "negatives require huge batches"**, with $m=0.999$ and *why* slowness is the point.
4. **Shuffling BN** — the BN batch-leak, connecting Module 2 to Module 3.
5. **"What prevents collapse?"** answered across all five method families.
6. **Frame InfoNCE as $(2N{-}1)$-way classification** and mention the alignment/uniformity decomposition and $\tau$'s role.
7. **Know the limitation:** contrastive SSL learns *augmentation-invariant* features, so it deliberately discards exactly what augmentation removes — colour, precise position, scale. That makes it weaker for **dense** tasks (segmentation, depth) than masked-modelling approaches (MAE, 6.4), which is precisely why both paradigms exist and why DINOv2/v3 combine ideas from both.

### ✅ Mastery check

You pretrain a ResNet-50 with SimCLR on 500k unlabelled product photos, then linear-probe on 5k labelled ones. It underperforms a plain ImageNet-pretrained baseline.

(a) Give four plausible causes, with a diagnostic for each.
(b) Your augmentation set is crop + flip + rotation, no colour jitter. What specifically will the model have learned, and how would you confirm it?
(c) You have one GPU. Which method do you choose over SimCLR, and why?

<details><summary>Answer sketch</summary>
(a) 1. <b>Batch size too small</b> ⇒ too few negatives ⇒ weak uniformity. Diagnostic: sweep batch size and watch linear-probe accuracy; also measure the uniformity metric (log of average pairwise Gaussian potential on the hypersphere). 2. <b>Under-training</b> — SSL needs far longer schedules than supervised (SimCLR used 800–1000 epochs; 100 epochs is not a fair comparison). Diagnostic: plot probe accuracy vs epoch; if still rising, you simply stopped early. 3. <b>Wrong augmentation set for the domain</b> — see (b). Diagnostic: the ablation, one augmentation at a time. 4. <b>Probing the wrong layer</b> — if you're probing $z$ (after the projection head) instead of $h$, you'll lose ~10 points for free. Diagnostic: probe both. 5. <b>500k images is not that many</b> — SSL's advantage over supervised ImageNet pretraining typically needs millions; with 500k in-domain images, <i>fine-tuning</i> an ImageNet model on your labels may simply be the better recipe. Diagnostic: run that baseline too.
(b) Without colour distortion, two crops of the same image share nearly identical colour histograms, so the network can minimise InfoNCE by <b>matching low-level colour statistics</b> — a shortcut requiring no shape or semantic understanding. For product photos this is especially bad, since colour is often the dominant per-image signature. Confirm it by: (i) evaluating retrieval on a set where the same product appears in multiple colourways — the model will confuse colourways with different products and vice versa; (ii) computing the linear-probe accuracy of a trivial baseline (a 3D colour histogram) and seeing that your embedding barely beats it; (iii) visualising nearest neighbours — they'll be colour-matched rather than shape-matched.
(c) <b>MoCo v2</b> (or BYOL/SimSiam). MoCo v2 gets its 65k negatives from a queue rather than the batch, so it trains well at batch 256 on a single GPU — that is exactly the problem it was designed to solve. SimSiam is an even simpler alternative (no queue, no momentum encoder, no negatives — just predictor + stop-gradient) and is explicitly designed to work at small batch. Either is a much better fit than SimCLR, which needs batches in the thousands to be competitive.
</details>

### 🔨 Build + read

**Build:** Implement SimCLR on CIFAR-10 (small enough to actually finish). Then run the two experiments that carry the whole concept: (i) **the augmentation ablation** — crop only, crop+flip, crop+colour, full — and report linear-probe accuracy for each; (ii) **probe $h$ vs $z$** and measure the gap. Both should reproduce the paper's qualitative findings on a laptop-scale budget. Bonus: implement SimSiam and verify that removing the stop-gradient causes immediate collapse (loss drops to its degenerate minimum and the embedding std goes to zero) — one of the most instructive five-line experiments in deep learning.

**Read:** Chen et al., "SimCLR" (ICML 2020) — §3 and Figures 4–5. He et al., "MoCo" (CVPR 2020) §3. Chen & He, "Exploring Simple Siamese Representation Learning" (CVPR 2021) — read the stop-gradient ablation. Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity" (ICML 2020).

---

## 3.7 Approximate Nearest-Neighbour Search: FAISS & HNSW 🔴

> **Flagged critical, and it is the highest-leverage section in this module for you specifically.** It is the "last mile" that turns an embedding into a product, it is a direct extension of your vector-DB work, and it is the most infrastructure-flavoured topic in the whole curriculum — which makes it prime material for the MLOps pivot.

### Intuition

You have 100 million image vectors, each 768-d, and a query must find its nearest neighbours in 10 milliseconds. **Work out whether that is even possible before reading the solution.** An exact scan is 100M × 768 multiply-adds ≈ $10^{11}$ FLOPs, and it must stream 100M × 768 × 4 bytes = 300 GB from memory *per query*. A CPU core does perhaps $10^{10}$ FLOPs/s. You are three to four orders of magnitude short, and the bandwidth is worse than the arithmetic. No engineering closes that; the algorithm must change.

So what can you give up? Not the data, not the query. Only **correctness** — and it turns out to be a superb thing to trade. Approximate nearest-neighbour search gives up the guarantee of finding the *exact* nearest neighbour for a 100–10,000× speedup, while still finding it 95–99% of the time. And note why the trade is nearly free *here specifically*: your embedding is already an imperfect proxy for human similarity, so the 3rd-nearest vector is often as good an answer as the 1st. You are adding a small approximation error on top of a much larger modelling error. **Every vector database on earth — Qdrant, pgvector, the one behind your RAG system — is a wrapper around this trade**, and the parameters you set there (`M`, `ef`, `nprobe`) are the dials that position you on it.

### The trade-off triangle

Every ANN design is a point in a three-way trade. The triangle is not a mnemonic — it records that each family below sacrifices exactly one of the three to buy the other two, and knowing *which* is how you pick an index:

```
                    RECALL
                   (accuracy)
                      /\
                     /  \
                    /    \
                   /      \
            LATENCY ────── MEMORY
             (QPS)         (RAM/disk)
```

You can have any two. Flat = perfect recall + high memory + terrible latency. PQ = great memory + good latency + reduced recall. HNSW = great recall + great latency + heavy memory. **Every interview question in this area is really asking you to pick a corner given a constraint.**

### The index families

#### 1. Flat (exact)

Brute-force scan, $O(ND)$. Perfect recall. On a GPU, FAISS can do exact search over ~10M × 768-d in a few milliseconds — **so don't reach for an approximate index too early.** Below ~1M vectors, flat is often the right answer and saves you an entire class of bugs.

#### 2. IVF — Inverted File (coarse quantisation)

**The idea, borrowed from the text search you already know.** BM25 does not score every document; it keeps an inverted index from term to documents and touches only documents sharing a term with the query. The saving comes from *not looking at most of the corpus*. Vectors have no terms — but you can invent some by carving the space into regions and calling each region a "term."

1. **Train:** k-means the dataset into `nlist` centroids (Voronoi cells).
2. **Index:** assign each vector to its nearest centroid; store per-cell posting lists.
3. **Search:** find the `nprobe` nearest centroids to the query, scan only those cells.

Do the arithmetic yourself, because it explains the tuning rule. Balanced cells hold $N/\texttt{nlist}$ vectors each and you scan `nprobe` of them, touching $N\cdot\texttt{nprobe}/\texttt{nlist}$ vectors — speedup $\approx$ `nlist / nprobe`. But finding *which* cells to probe costs a scan over the centroids, `nlist` distance computations. Total work is $\texttt{nlist} + N\cdot\texttt{nprobe}/\texttt{nlist}$: the first term grows with `nlist`, the second shrinks with it, and setting the derivative to zero balances them at $\texttt{nlist}\approx\sqrt{N\cdot\texttt{nprobe}}$ — which is where the folk rule comes from. Tuning: `nlist` $\approx \sqrt{N}$ to $4\sqrt{N}$ (for $N=10^6$, ≈ 1024–4096); `nprobe` is the **recall/latency dial**, swept at query time with no re-indexing — operationally valuable, since you can retune recall in production without rebuilding.

**Failure mode: the boundary problem — inherent, not a bug.** k-means commits every vector to one cell, but a query does not respect that partition. Picture a query just inside cell A, a hair from the boundary with B; its true nearest neighbour may sit a millimetre away on B's side. With `nprobe = 1` you scan A, never look at B, and return something worse while being entirely unaware. That is why recall saturates below 100% however well you tune, and why `nprobe` must grow for high recall — raising it buys insurance against the query being near a boundary. In high dimensions almost every point is near *some* boundary, so `nprobe` needs to be tens, not one or two.

#### 3. PQ — Product Quantisation (compression)

IVF reduced *how many* vectors you touch; it did nothing about their size, so those 300 GB are still 300 GB. PQ is the other half, and it is the cleverest idea in this module, so build it rather than state it.

**Attempt 1: plain vector quantisation.** Run k-means on the whole dataset with $k$ centroids and replace each vector by the id of its nearest centroid — $\log_2 k$ bits per vector, wonderful. Now ask how large $k$ must be to approximate well. You need centroids dense enough that the nearest one is genuinely close in 768 dimensions, and volume in $D$ dimensions grows like $r^D$, so covering the space at a fixed resolution needs a centroid count exponential in $D$. To spend 96 bytes per vector you would need $k = 2^{768}$ centroids — all of which you must *store*, and *scan* to encode. Plain VQ is not impractical, it is impossible by hundreds of orders of magnitude.

**The bind: you want an enormous codebook (accuracy) that is tiny to store and fast to search (practicality).** These look contradictory — but they are about different quantities. Accuracy wants many distinct *codewords*; cost is driven by how many centroids you *store and compare against*. Is there a construction where representable codewords vastly outnumber stored centroids?

**Attempt 2: chop the vector up.** Split the $D$-dim vector into $m$ sub-vectors of dimension $D/m$; run k-means with $k^*=256$ centroids **independently in each sub-space**; store each sub-vector as the 1-byte index of its nearest sub-centroid. A vector is now $m$ bytes.

$$
\text{bytes per vector} = m \qquad\text{(vs } 4D \text{ for float32)}
$$

**Now count the codewords — this is the payoff.** A reconstructed vector is one independent choice from each sub-space, so the effective codebook has $256^m$ entries — astronomically large — while you store only $m \times 256 \times (D/m)$ floats. The combinatorial product ("product" quantisation) buys exponential expressiveness at linear cost. What you sacrifice is that the codewords lie on a rigid grid, placeable only at the Cartesian product of per-subspace choices rather than wherever the data is densest; that is the entire quality cost.

**Walk the numbers.** $D=768$, float32: raw storage is $768\times 4 = 3072$ bytes. Choose $m=96$, so each sub-vector is $768/96 = 8$ dimensions coded by one of 256 centroids — one byte. Ninety-six sub-vectors, ninety-six bytes: **3072 bytes → 96 bytes, a 32× compression**, the ratio being just $4D/m$, so $m$ is your dial. Meanwhile the stored codebook is $96 \times 256 \times 8 = 196{,}608$ floats, under a megabyte, and the representable codewords number $256^{96} = 2^{768}$ — *exactly* the astronomical codebook attempt 1 demanded. At 200 million vectors the storage is 614 GB (does not fit on a machine) versus 19 GB (fits in RAM with room to spare): the difference between a distributed system and a single server.

**Asymmetric Distance Computation (ADC) — and the puzzle it solves.** You compressed the database but seem to have made search *harder*: comparing the query to a stored code apparently requires decompressing it back to 768 floats, which costs more than never compressing. So how is a PQ scan fast?

Use the fact that the sub-spaces partition the coordinates. Squared Euclidean distance is a sum over coordinates, so it splits cleanly along that partition:
$$
\|q - x\|^2 = \sum_{j=1}^{m} \|q_j - x_j\|^2
$$
Substitute the approximation: $x_j$ is not stored, but its centroid $c_{j,\text{code}_j(x)}$ is, so the term becomes $\|q_j - c_{j,\text{code}_j(x)}\|^2$. Here is the observation that makes everything work: **that term depends on the database vector only through one byte.** For a fixed query it can take exactly 256 values in sub-space $j$, however many database vectors you scan.

So compute all of them up front. Precompute, once per query, a lookup table $T[j][c] = \|q_j - c_{j,c}\|^2$ of size $m \times 256$ — 24,576 small distance computations here, negligible and paid once. Then the approximate distance to any database vector is

$$
\hat{d}(q,x)^2 = \sum_{j=1}^{m} T\bigl[j\bigr]\bigl[\text{code}_j(x)\bigr]
$$

— **$m$ table lookups and adds, no multiplications.** Compare the work: an exact 768-d distance is 768 multiply-adds over 3072 bytes of traffic; the ADC estimate is 96 lookups and 96 adds over 96 bytes, with the whole 96 KB table sitting in L2 cache. You have replaced arithmetic with lookups in a table small enough to stay hot, which on modern hardware is where the real speedup lives.

**Why "asymmetric," and why it wins.** The symmetric alternative quantises the *query* too and reads distances from a precomputed centroid-to-centroid table — marginally faster, but consider the error. Write $x = \hat{x} + \epsilon_x$, the database vector's quantisation error, which you cannot avoid since you threw $x$ away. Asymmetric computes $\|q - \hat{x}\|$, carrying only $\epsilon_x$. Symmetric computes $\|\hat{q} - \hat{x}\|$, carrying $\epsilon_x$ **and** $\epsilon_q$ — and $\epsilon_q$ is self-inflicted, because you had the exact query and chose to degrade it. The two errors are roughly independent, so squared error roughly doubles, for a saving of one cheap table build per query (not per database vector). **The general principle travels: never quantise something you have exactly, just because what you are comparing it against was quantised.**

**OPQ (Optimised PQ) — fixing the arbitrariness of the split.** Attempt 2 cut contiguous blocks of 8 dimensions, an arbitrary choice that costs accuracy twice. First, variance is unevenly distributed: if dimensions 0–7 carry huge variance and 300–307 almost none, the first sub-space's 256 centroids are hopelessly overstretched while the second's are wasted — and since total error sums over sub-spaces, it is dominated by the worst one, so you want variance *balanced*. Second, PQ assumes sub-spaces are independent (that is what makes the product codebook valid), but adjacent embedding dimensions are usually correlated, and correlation across a split is information the product structure cannot represent. One move fixes both: apply a learned rotation $R$ before splitting, balancing variance and decorrelating sub-spaces. A rotation preserves all distances, so it costs nothing in fidelity and one matrix multiply per query. Typically a free 1–3% recall gain; use it by default.

**In your own words:** how does PQ get a $2^{768}$-entry codebook while storing under a megabyte of centroids — and why is the lookup table rebuilt per query but only once per query?

#### 4. IVF-PQ — the billion-scale workhorse

The two ideas solve orthogonal halves of the problem and compose without interfering: IVF narrows *which* vectors to scan, PQ makes each comparison cheap and each vector small.

There is also a genuine synergy. IVF-PQ usually stores the PQ code of the **residual** $x - c_{\text{cell}}$ rather than $x$ itself, and the reason is that PQ's error depends on how spread out the things being quantised are — 256 centroids per sub-space must cover whatever range they are given. Raw vectors span the whole dataset. Residuals span only the *within-cell* spread, smaller by roughly a factor of `nlist` in volume. Same 256 centroids, a far smaller region to cover, so quantisation error drops substantially at zero extra storage: the coarse quantiser has done the large-scale work, leaving PQ to encode fine detail. This is what powers most billion-scale deployments.

#### 5. HNSW — Hierarchical Navigable Small World (graph)

The highest-recall-per-latency structure, and the default in Qdrant, Weaviate, Milvus, Elasticsearch, and pgvector — so if your RAG project used any of those with default settings, this is the algorithm that was actually answering your queries.

**Build it from the skip list you already know.** A sorted linked list searches in $O(N)$ because you can only step to the next element. A skip list stacks sparser copies above it — each level promoting roughly half the nodes — so you take long strides at the top until you overshoot, drop a level, take shorter strides, and so on, giving $O(\log N)$. The crucial detail is that levels are assigned **randomly** per node, buying tree-like balance with no rebalancing logic.

Now ask what breaks for vectors. A skip list rests entirely on a **total order**: "step forward while the next key is less than the target." In $\mathbb{R}^{768}$ there is no order, only distances. So replace both order-dependent parts with distance analogues. "Next in the list" becomes **a neighbour in a proximity graph**, each node linking to a handful of nearby points; "keep stepping while the key is smaller" becomes **greedily move to whichever neighbour is closest to the query, stopping when none improves**. The layered structure carries over unchanged, because its purpose — long strides first, short strides later — is geometric, not ordinal.

**Structure.** A multi-layer proximity graph. Each node gets a maximum layer $\ell = \lfloor -\ln(U(0,1)) \cdot m_L \rfloor$ — the continuous version of the skip list's coin flips, same exponential thinning — so layer 0 contains everything and higher layers are exponentially sparser. Sparsity is what creates long edges: if a layer holds $N/100$ of the points, its nearest neighbours are far apart in absolute terms, so an edge there covers a lot of ground. Higher layers act as an **express highway** — long-range links reaching the right neighbourhood in a few hops; layer 0 has short-range links for precise local search. A probabilistic skip-list generalised to metric space.

**Search.** Start at the entry point in the top layer. Greedily move to the neighbour closest to the query until no neighbour improves; drop a layer; repeat.

**Pause:** pure greedy descent has an obvious failure — what is it, and what does HNSW do about it?

It gets stuck in local minima: you reach a node whose neighbours are all further from the query, so you stop — but the true nearest neighbour lies beyond a slightly-worse intermediate node you refused to step onto. The remedy is to stop being greedy about a *single* current node: at layer 0, run a **beam search** maintaining a candidate set of size `efSearch`, returning the best $k$. Keeping `efSearch` candidates alive means several fronts stay open, so one dead end does not end the search. That is why `efSearch` is the recall dial, and why it must be at least $k$ — you cannot return $k$ results from a beam narrower than $k$.

**Parameters:**

| Parameter | Meaning | Effect |
|---|---|---|
| `M` | max neighbours per node (layer >0); layer 0 gets $2M$ | ↑ recall, ↑ memory, ↑ build time. 16–64 typical |
| `efConstruction` | beam width during build | ↑ graph quality, ↑ build time. 100–500 |
| `efSearch` | beam width at query | **the query-time recall/latency dial**; ≥ $k$ |

**Memory — notice which term dominates, because it decides the architecture.** The graph costs roughly $N \times M \times 2 \times \text{sizeof(int)}$ for edges: at $N=10^8$, $M=32$ that is ~25 GB, which sounds like the problem but is not. The real cost is that every greedy step computes an actual distance to a candidate, so the index needs **the full-precision vectors** — HNSW alone does not compress. For $10^8$ × 768-d float32 that's ~300 GB, an order of magnitude more than the graph. **Memory is HNSW's binding constraint**, and once you see that the vectors rather than the edges are the problem, the fix is obvious: compress the vectors, keep the graph. Hence production systems pairing it with PQ (Qdrant's scalar/product quantisation, FAISS's `IVF_HNSW`, `HNSWPQ`) — the two techniques address disjoint costs.

**Other weaknesses, all traceable to "it is a graph, not a partition."** Deletions are awkward: removing a node may sever the paths other nodes relied on to reach a region, so implementations tombstone and rebuild rather than truly delete. Build time is $O(N\log N)$ with a big constant, because inserting each node means running a full search to find its neighbours. And there is no natural sharding by data locality — IVF splits cleanly by cell, whereas cutting a graph in half severs edges and leaves the halves un-navigable.

**In your own words:** what does a skip list's "levels" become in HNSW, and what does its "step forward while smaller" become?

#### 6. Others worth naming

- **ScaNN** (Google): *anisotropic* vector quantisation — weights quantisation error by its effect on the **inner product** rather than treating all error equally, because for MIPS, error parallel to the query direction matters far more than error orthogonal to it. Strong on MIPS benchmarks.
- **DiskANN / Vamana** (Microsoft): a graph index designed so most of it lives on **SSD**, enabling billion-scale search on a single machine with modest RAM. The right answer when memory cost dominates.
- **LSH**: theoretically elegant, practically dominated by IVF/graph methods for real data. Still relevant for binary codes and for streaming/sublinear-memory settings.
- **Binary hashing / ITQ / deep hashing**: learn a binary code, search by Hamming distance. Extremely fast and compact; lower recall.

### The mapping to your RAG work — make this explicit

| Your RAG system | Vision retrieval | Same or different? |
|---|---|---|
| Chunk text → embed with a sentence transformer | Crop/resize image → embed with a ViT/CNN | **Same pattern**, different encoder |
| Qdrant / pgvector collection | FAISS index / Qdrant with image vectors | **Identical machinery** — Qdrant doesn't know or care what the vectors mean |
| HNSW index inside Qdrant | HNSW index | **The same algorithm**, same `M`/`ef` knobs |
| Cosine similarity on normalised embeddings | Cosine on L2-normalised image embeddings | Same, and the same $\|a-b\|^2 = 2-2\cos\theta$ identity |
| Metadata filtering (`where category = 'x'`) | Filter by product category, date, camera | **Same hard problem** — see below |
| Cross-encoder reranking of top-$k$ | RANSAC geometric verification of top-$k$ | Structurally identical two-stage design |
| Chunk-size / overlap tuning | Crop / resolution / pooling choice | Both are "what is the unit of retrieval?" |
| Hybrid dense + BM25 | Dense global descriptor + BoVW/tf-idf | Same fusion idea |

**Filtered search is the hard problem in both, and it is worth understanding why rather than just knowing that it is.** The index was built around one assumption, proximity in embedding space, and a filter introduces a second, orthogonal criterion the structure knows nothing about. There are only two places to apply it, and both break in complementary ways. Pre-filtering (restrict the candidate set, then search) sounds correct, but the surviving nodes are scattered across the graph and the edges connecting them ran through nodes you just removed — a filter keeping 1% of the data leaves a graph with 99% of its nodes gone, no longer navigable, and it can be slower than brute force. Post-filtering (search, then filter) leaves the index intact but discards results failing the predicate *after* retrieving $k$ by similarity, so you may return fewer than $k$, or none — and the more selective the filter the likelier that is, which is exactly backwards. Real systems use filter-aware traversal, or maintain separate indexes per high-cardinality filter value. **If you can discuss the pre-filter/post-filter trade-off, you are ahead of most candidates in any vector-search interview** — and you have already hit this in your RAG project, so say so.

### Tuning methodology (the practical answer)

The recurring mistake is treating index choice as a lookup ("100M vectors → IVF-PQ") when it is a measurement. Approximate means *you do not know your recall unless you measure it*, and you cannot measure it without something exact to compare against. Hence the order: build the truth, then the curve, then let the product decide where on the curve to sit.

1. **Build a ground-truth set**: exact top-$k$ for ~1000 held-out queries (flat index, run once, offline).
2. **Measure `Recall@k` vs `QPS`** by sweeping the query-time dial (`nprobe` or `efSearch`). This produces the canonical recall/QPS curve — the only honest way to compare indexes.
3. **Set a recall target from the product**, not from a benchmark: for a "top 20 similar products" carousel, Recall@20 = 0.95 is invisible to users; for de-duplication or biometric matching it isn't.
4. **Then** minimise memory subject to that recall.
5. Re-measure after every embedding-model change — index parameters do not transfer across embedding spaces.

**Rules of thumb:** < 1M vectors → flat, or HNSW if you want headroom. 1M–10M → HNSW (memory permitting) or IVF-Flat. 10M–100M → IVF-PQ or HNSW+PQ. > 100M → IVF-PQ sharded, or DiskANN. Always benchmark; these are starting points, not answers.

### 🎯 Top-1% distinction

1. **The recall/latency/memory triangle**, and picking a corner from a stated constraint. This is what the question is *actually* about.
2. **Explain PQ's ADC lookup-table trick** and why asymmetric beats symmetric.
3. **Know HNSW's memory is the binding constraint** and that production systems pair it with quantisation.
4. **MIPS ≠ NN**, and that inner-product search lacks the triangle inequality, weakening graph/tree pruning guarantees. Hence normalise, or use ScaNN's anisotropic quantisation.
5. **Filtered search: pre- vs post-filter**, and why selective filters break graph indexes.
6. **"Don't use an ANN index below ~1M vectors"** — knowing when *not* to reach for the complex tool is a seniority signal.
7. **ANN works only because real embeddings have low intrinsic dimension.** In truly uniform high-dimensional data, concentration of measure means every point is roughly equidistant and no index can help. Real data lies on a low-dimensional manifold — that's the assumption every ANN method quietly relies on.

### ✅ Mastery check

You must serve visual similarity search over **200 million** product images. Embeddings are 768-d float32. Requirements: p99 latency < 50 ms, Recall@10 ≥ 0.95, and results must be filterable by `category` (500 values) and `in_stock` (boolean).

(a) Compute the raw memory for flat storage. Is HNSW feasible on a 512 GB machine?
(b) Design the index. Justify each choice.
(c) How do you handle the two filters? Which is harder and why?
(d) You need to add 2M new products per day and remove 1M. What breaks?

<details><summary>Answer sketch</summary>
(a) $200\text{M} \times 768 \times 4\,\text{B} = \mathbf{614\ GB}$ of raw vectors alone — already over 512 GB. HNSW adds edges: with $M=32$, roughly $200\text{M}\times 32\times 2\times 4\,\text{B} \approx 51$ GB, plus overhead. So <b>plain HNSW is infeasible on one machine</b>; you either shard across machines or compress.
(b) <b>IVF-PQ with OPQ</b>, or <b>HNSW over PQ-compressed vectors</b>. Concretely: OPQ rotation → PQ with $m=96$ (96 bytes/vector) → $200\text{M}\times96\,\text{B} = \mathbf{19\ GB}$, comfortably in RAM with the IVF structure and centroids. `nlist` ≈ $4\sqrt{2\times10^8} \approx 56{,}000$; sweep `nprobe` (start ~64–128) against the recall target. To hit Recall@10 ≥ 0.95 with PQ compression you almost certainly need a <b>rerank stage</b>: retrieve top-200 by PQ distance, then recompute exact distances against full-precision vectors held on SSD/mmap for just those 200. That two-stage design is how you get both the memory saving and the recall. Alternative if budget allows: shard HNSW across 2–4 machines with full-precision vectors — simpler, higher recall, more expensive.
(c) <b>`in_stock` (boolean, low cardinality, high churn) is the harder one.</b> `category` has 500 values and is nearly static, so the clean solution is <b>partitioning</b>: maintain one index per category (or route by category to a shard), turning the filter into index selection with no recall cost. `in_stock` can't be partitioned that way — it flips constantly, and it's roughly 50/50 so post-filtering is affordable *on average* but has a bad tail (if a query's neighbourhood is mostly out-of-stock you may return < $k$). Practical approach: <b>filter-aware traversal</b> (the index checks the predicate during graph/list traversal and keeps searching until $k$ passing results are found — what Qdrant and Milvus implement), with an over-fetch factor and a fallback to a wider search. Worth stating the general rule: <b>low-cardinality, static filters → partition; high-churn or high-cardinality filters → filter-aware search with over-fetch.</b>
(d) (i) <b>IVF centroids drift.</b> They were trained by k-means on the original distribution; after months of 2M/day inserts the data distribution has moved and cells become unbalanced, degrading both recall and latency. Mitigation: monitor cell-size skew, retrain and rebuild periodically (e.g. weekly/monthly), or use a small held-out query set to alarm on recall regression. (ii) <b>Deletions.</b> Neither IVF nor HNSW supports true deletion cheaply — you tombstone and filter at query time, and the index gradually fills with dead entries, so you need periodic compaction. HNSW deletion is worse because removing a node can disconnect the graph. (iii) <b>Rebuild cost and availability</b> — you need a blue/green index-swap strategy so rebuilds don't take the service down, and a write-ahead buffer (a small flat index over the newest vectors, searched alongside the main index and merged in periodically) so new products are searchable immediately rather than at the next rebuild. That last pattern — <b>a small hot index + a large cold index</b> — is the standard production answer and is exactly what mature vector DBs do internally.
</details>

### 🔨 Build + read

**Build — the single highest-value lab in this module.** Take 1M image embeddings (SIFT1M/GIST1M from the FAISS benchmarks, or embed a large image set yourself). Build: `Flat`, `IVFFlat`, `IVFPQ`, `HNSW`, `HNSWPQ`. For each, sweep the query-time dial and plot **Recall@10 vs QPS** on one log-scale figure, and annotate each point with index memory. Then implement PQ *from scratch* (train sub-codebooks with k-means, encode, and implement ADC with the lookup table) and verify your recall matches FAISS's. **That figure plus that implementation is a portfolio piece** — it is directly transferable evidence for both CV and ML-infra roles.

**Read:** Jégou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search" (PAMI 2011) — read §III for ADC. Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW" (PAMI 2018) — read the algorithm boxes. The FAISS wiki's "Guidelines to choose an index." Then browse **ann-benchmarks.com** to see what the recall/QPS frontier actually looks like.

---

## 3.8 Practical: Build an Image Retrieval System End to End

Everything in 3.1–3.7 was a component. This section is about the one property of the assembled system that no component enforces alone: **the indexing path and the serving path must apply exactly the same transformation.** Notice that the diagram's right column repeats its left column verbatim for the first three steps — same preprocessing, same encoder, same PCA and normalisation. That repetition is the point, not redundancy in the drawing. If the offline job resizes to 256 and the online service to 224, or the PCA matrix is refitted at reindex while the query path holds the old one, the two sets of vectors live in subtly different spaces and every distance is quietly wrong. Recall falls, nothing crashes, no test fails. It is the most common way a retrieval system breaks in production, and it is a *systems* failure rather than a modelling one — which is why this section is its own concept.

### Architecture

```
INDEXING (offline)                          SERVING (online)
──────────────────                          ────────────────
images                                      query image
  │                                            │
  ├─ preprocess (resize, centre/GeM crop)      ├─ same preprocessing
  │                                            │
  ├─ encoder (DINOv2 / CLIP / fine-tuned)      ├─ same encoder
  │                                            │
  ├─ PCA-whiten → L2-normalise                 ├─ same transform
  │                                            │
  ├─ ANN index build (HNSW / IVF-PQ)           ├─ ANN search → top-200
  │                                            │
  └─ store full vectors + metadata             ├─ exact rerank on full vectors → top-50
                                               │
                                               ├─ (optional) local-feature geometric
                                               │   verification → top-10
                                               │
                                               └─ apply filters, return
```

### The build checklist

The ordering of these steps is itself the lesson: each one is cheap to change early and expensive to change late, and steps 1–2 determine the ceiling that steps 5–6 can only approach.

1. **Choose the encoder by measurement, not by reputation.** Baseline four candidates (ImageNet ResNet-50, CLIP ViT-B/32, DINOv2 ViT-B/14, and a fine-tuned version of the best) on *your* data with exact search. Only then pick.
2. **Decide the unit of retrieval.** Whole image? Detected object crop? Multiple regions per image? This choice usually matters more than the encoder — a cluttered scene embedded as one vector retrieves poorly, and running a detector first (Module 4) to crop the object often doubles recall.
3. **Post-process embeddings:** PCA-whiten (fit on the index set), L2-normalise. Measure the gain; it's usually real.
4. **Build ground truth** for ~500–1000 queries, ideally with human relevance labels rather than only exact-duplicate labels.
5. **Pick the index from the recall/QPS/memory curve**, per 3.7.
6. **Add a rerank stage** — exact distances on full-precision vectors for the top-200 costs almost nothing and recovers most of the recall lost to quantisation.
7. **Handle filters** — decide partition vs filter-aware search per filter (3.7).
8. **Instrument:** log recall against a shadow exact index on a sample of live traffic, p50/p99 latency, index memory, and embedding-distribution drift.
9. **Plan for updates** — hot index + cold index, periodic rebuild, blue/green swap.

**In your own words:** which single stage of this pipeline sets the ceiling on the system's quality, and which stages can only trade latency and memory against approaching that ceiling?

### Failure modes to design against

Every row below shares a shape worth naming: the system reports healthy numbers while producing bad results, because the thing being measured has drifted away from the thing that matters.

| Failure | Cause | Mitigation |
|---|---|---|
| Good offline recall, bad user results | evaluation set doesn't reflect real queries (clean catalogue images vs phone photos) | build the eval set from *real query logs* |
| Near-duplicates flood the top-k | catalogue contains many copies of the same item | diversity re-ranking / MMR; dedupe at index time |
| Fails on unseen categories | encoder fine-tuned too narrowly | keep a general backbone; evaluate on held-out categories |
| Latency spikes at p99 | selective filters force deep graph traversal | over-fetch, filter-aware search, per-filter partitions |
| Recall silently degrades over months | data drift vs. stale IVF centroids | shadow exact-search recall monitor + scheduled rebuild |

---

## Going Deeper — Papers, Sources and Research Scope

*This module has the highest research-per-page density of the six, because retrieval and self-supervision are both genuinely unsettled and both cheap to experiment on. It is also the module most directly transferable to your RAG/vector-database work.*

### A. The canonical papers

| Paper | Year / Venue | Why it matters | Where |
|---|---|---|---|
| Chopra, Hadsell & LeCun, *Learning a Similarity Metric Discriminatively* | 2005, CVPR | The original Siamese/contrastive formulation. The margin term's justification is clearer here than in any modern paper. | CVPR 2005 |
| **Schroff et al., *FaceNet*** | 2015, CVPR (1503.03832) | Triplet loss, and — the part people skip — §3.2 on *semi-hard* negative mining and why hardest-negative mining collapses. | arXiv 1503.03832 |
| Hermans et al., *In Defense of the Triplet Loss for Re-ID* | 2017 (1703.07737) | Batch-hard mining, and a much more careful account of the mining design space. | arXiv 1703.07737 |
| **Musgrave, Belongie & Lim, *A Metric Learning Reality Check*** | 2020, ECCV (2003.08505) | **The most important paper in this module.** A decade of claimed metric-learning gains largely evaporates under a fair protocol with equal hyperparameter budgets. Read it for the *methodology* — it is a template for the strongest kind of paper you could write. | arXiv 2003.08505 |
| Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search* | 2011, PAMI | Where PQ comes from, with the asymmetric-distance-computation trick that makes it work. | PAMI 33(1) |
| **Malkov & Yashunin, *HNSW*** | 2016/2018, PAMI (1603.09320) | Navigable small-world graphs with a skip-list-like hierarchy. The complexity argument in §4 is the part to understand. | arXiv 1603.09320 |
| Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs* (FAISS) | 2017 (1702.08734) | The systems paper behind FAISS — k-selection on GPU, and the IVF-PQ pipeline end to end. | arXiv 1702.08734 |
| Guo et al., *ScaNN* (*Accelerating Large-Scale Inference with Anisotropic Vector Quantization*) | 2020, ICML (1908.10396) | Quantisation error should be weighted by its effect on the *inner product*, not on reconstruction. A genuinely different idea from PQ. | arXiv 1908.10396 |
| **Chen et al., *SimCLR*** | 2020, ICML (2002.05709) | NT-Xent, the augmentation ablation (Fig. 5), and the projection head. Read the ablations, not just the method. | arXiv 2002.05709 |
| **He et al., *MoCo*** | 2020, CVPR (1911.05722) | The queue and the momentum encoder — i.e. how to decouple negative count from batch size. MoCo v2 (2003.04297) is the practical version. | arXiv 1911.05722 |
| **Grill et al., *BYOL*** | 2020, NeurIPS (2006.07733) | Works *without negatives*, which broke the field's mental model. | arXiv 2006.07733 |
| Chen & He, *SimSiam* | 2021, CVPR (2011.10566) | Strips BYOL to the minimum and isolates the stop-gradient as the thing preventing collapse. The cleanest ablation in SSL. | arXiv 2011.10566 |
| Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity* | 2020, ICML (2005.10242) | Decomposes InfoNCE into two interpretable terms. This is the theory paper that makes the loss stop feeling arbitrary. | arXiv 2005.10242 |
| Radford et al., *CLIP* | 2021, ICML (2103.00020) | Read here rather than only in M6: it is the same InfoNCE with "second view = caption". | arXiv 2103.00020 |
| Zhai et al., *SigLIP* | 2023, ICCV (2303.15343) | Sigmoid loss removes the global-batch normalisation, so batch size stops being a hyperparameter you can't afford. The practical default now. | arXiv 2303.15343 |
| Aumüller et al., *ANN-Benchmarks* | 2020, Information Systems (1807.05614) | The standard evaluation harness. Read the methodology section before you benchmark anything yourself. | arXiv 1807.05614 |

### B. The single best source, per hard topic

- **Contrastive learning, conceptually.** Lilian Weng, *Contrastive Representation Learning* (`lilianweng.github.io/posts/2021-05-31-contrastive/`) — the best single survey of the loss-function zoo, with consistent notation across papers that use none.
- **The self-supervised landscape.** Weng, *Self-Supervised Representation Learning*, same site. Read after the above.
- **Triplet mining, practically.** Olivier Moindrot, *Triplet Loss and Online Triplet Mining in TensorFlow* — despite the framework, this is still the clearest exposition of batch-all vs batch-hard, with the masking logic written out.
- **PQ and IVF, mechanically.** The FAISS wiki's *Guidelines to choose an index* plus *Faiss indexes* pages (`github.com/facebookresearch/faiss/wiki`). Read the index-string grammar (`IVF4096,PQ64`) — it forces you to understand the components.
- **HNSW, mechanically.** Pinecone's HNSW explainer for the picture, then Malkov §3–4 for the argument. In that order.
- **Recall/latency trade-offs, empirically.** `ann-benchmarks.com` — read the actual plots before forming opinions about which index wins.
- **Why the projection head is discarded.** SimCLR §4.2 plus the discussion in SimSiam. There is no better treatment; the honest summary is that the field has a plausible story, not a proof.

### C. Reference implementations worth reading

- **`facebookresearch/faiss` → `faiss/IndexIVFPQ.cpp`.** Find the ADC lookup-table construction — the moment where a distance computation becomes 64 table lookups and an addition. That is the whole idea of PQ, in about thirty lines.
- **`nmslib/hnswlib` → `hnswalg.h`.** `searchBaseLayerST` is the greedy-search-with-candidate-heap loop. Note how `ef` controls the candidate set size and therefore the entire recall/latency knob.
- **`KevinMusgrave/pytorch-metric-learning`.** The reference implementation of the *fair protocol* from the reality-check paper. Read `losses/` and `miners/` — every loss in the literature, in one consistent interface, which is itself the argument of the paper.
- **`facebookresearch/moco` → `moco/builder.py`.** ~100 lines. The queue enqueue/dequeue and `_momentum_update_key_encoder` are the entire contribution.
- **`lightly-ai/lightly` or `vturrisi/solo-learn`.** Both implement a dozen SSL methods under one training harness — the right starting point for any SSL experiment you run, because the harness is what makes the comparison fair.

### D. Open research questions

This is the module where I would actually encourage you to try something.

1. **Does the metric-learning reality check still hold, six years on?** *Why open:* Musgrave et al. evaluated the pre-2020 literature under 2020 training budgets. Since then, augmentation, schedules and backbones have all changed substantially — and nobody has redone the study. Either answer is interesting: if the gains reappear, the 2020 finding was budget-limited; if they don't, that's a decade-and-a-half of literature. *Minimum experiment:* 6–8 losses, one backbone, one augmentation policy, equal hyperparameter search budget per method, on CUB-200 and Cars196 (both small). ~40–60 GPU-hours. **This is the strongest research idea in the entire curriculum and it is within your budget.** See `Supplement-Capstone-Projects.md` #3.
2. **What does SSL do below 10,000 images?** *Why open:* every SSL result is at ImageNet scale or larger; the small-data regime is where actual products live and is barely studied. *Minimum experiment:* SimCLR and MAE from scratch on 1k/5k/10k/50k subsets vs supervised-from-scratch vs ImageNet-transfer; find the crossover point. **Feasible; negative results publishable.**
3. **Is the projection head's benefit explained by augmentation-invariance absorption, or by something simpler?** *Why open:* the standard story is a story. *Minimum experiment:* vary projection-head depth and width, and measure augmentation-parameter *decodability* from the representation before and after the head. If the story is right, decodability should drop sharply across the head. **Feasible, cheap, and a clean falsifiable prediction — which is rare.**
4. **Where exactly is the HNSW/IVF-PQ crossover, and what does it depend on?** *Why open:* the folk answer is "IVF-PQ above a few million" but the dependence on intrinsic dimensionality and clustering structure is not well characterised. *Minimum experiment:* fixed recall target, sweep N and intrinsic dimensionality on synthetic data plus two real embedding sets, plot the crossover surface. CPU-only. **Very feasible, and it doubles as content for capstone #2.**
5. **Do quantised embeddings degrade *uniformly*, or do they disproportionately harm rare classes?** *Why open:* PQ error is measured in aggregate; the per-class distribution is essentially unreported, and this has fairness implications nobody has looked at. *Minimum experiment:* recall@k per class, before and after PQ, on a long-tailed retrieval set. **Very feasible, and genuinely novel as far as I can tell — verify with a literature search first.**

---

## Module 3 — Interview Rapid-Fire Summary

| Concept | The one thing to say |
|---|---|
| Embedding source | penultimate/`[CLS]`/pooled tokens — **never logits** (they project onto the label simplex and discard within-class structure) |
| CE features | separable ≠ metric-optimal; open-set retrieval needs metric learning or SSL |
| Projection head | train the loss on $z$, use $h$ downstream — the head absorbs the invariance the task demands |
| Dim reduction | PCA-whitening to 128-d often *improves* recall and shrinks the index 16×; Matryoshka gives truncatable embeddings |
| Classical retrieval | BoVW = tf-idf + inverted index on visual words; VLAD = residual aggregation; NetVLAD = the differentiable version; GeM > GAP for retrieval |
| Two-stage paradigm | cheap global recall → expensive precise rerank. **Same shape as RAG**; RANSAC is the original reranker |
| Metrics | for normalised vectors L2 ≡ cosine ≡ IP; cosine is not a metric; **MIPS ≠ NN** and breaks pruning guarantees |
| mAP | retrieval mAP (rank list per query) ≠ detection mAP (confidence sweep, per class, at an IoU) |
| Siamese | one network applied twice (shared weights ⇒ common space); contrastive loss margin is in **absolute** distance ⇒ brittle |
| Triplet loss | $[d(a,p) - d(a,n) + \alpha]_+$; relative constraint; **mining is the crux** — $O(N^3)$ triplets, almost all zero-gradient |
| Mining | semi-hard (FaceNet) or batch-hard (Hermans); hardest-negative → collapse via label noise; PK batch sampling is part of the method |
| Currency | face recognition moved **from triplet to ArcFace-family angular-margin softmax** (no mining); triplet survives for unbounded label sets |
| InfoNCE | $(2N{-}1)$-way classification with free labels; $\tau$ trades alignment vs uniformity |
| SimCLR | crop + **colour jitter** is the critical pair (colour histogram is the shortcut); projection head; huge batches |
| MoCo | queue (65k negatives) + momentum encoder ($m{=}0.999$) decouples negatives from batch size; **shuffling BN** stops the BN leak |
| Collapse | negatives, or stop-gradient+predictor (SimSiam), or explicit variance/decorrelation (VICReg/Barlow), or centring+sharpening (DINO) |
| SSL limits | contrastive learns augmentation-*invariant* features ⇒ weaker for dense tasks than masked modelling (MAE) |
| ANN triangle | recall vs latency vs memory — pick two |
| IVF | k-means cells; `nprobe` is the query-time recall dial; boundary problem caps recall |
| PQ | $m$ sub-vectors × 256 centroids ⇒ $m$ bytes/vector (32× compression); **ADC lookup table**, asymmetric beats symmetric; OPQ rotates first |
| HNSW | multi-layer proximity graph (skip-list in metric space); `M`, `efConstruction`, `efSearch`; **memory-bound**, awkward deletions |
| Practical | don't use ANN below ~1M vectors; always plot Recall@k vs QPS; ANN works because real data has low intrinsic dimension |
| Filters | low-cardinality + static → partition; high-churn → filter-aware traversal with over-fetch |

---

*End of Module 3 notes. Drills in `Module-03-Drills.md`.*
