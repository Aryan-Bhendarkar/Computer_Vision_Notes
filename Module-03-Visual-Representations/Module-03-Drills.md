# Module 3 — Drills, Interview Bank & Spaced Repetition

---

## Part A — Rapid-fire

1. Where in a network do you take an image embedding from, and where do you never take it from?
2. Why are cross-entropy features not metric-optimal?
3. What is the projection-head finding, and what is the explanation?
4. Why does PCA-whitening to 128-d often *improve* retrieval?
5. What is Matryoshka representation learning?
6. What is BoVW, and which text-search concepts does it reuse?
7. What does VLAD store that BoVW does not?
8. What makes NetVLAD differentiable?
9. Why is GeM pooling better than GAP for retrieval? What does $p$ converge to?
10. Describe the two-stage retrieval paradigm. Map it onto a RAG pipeline.
11. What is the vision analogue of a cross-encoder reranker?
12. Prove L2 and cosine give the same ranking for normalised vectors.
13. Why is MIPS harder than NN search?
14. Is cosine similarity a metric? What is?
15. Retrieval mAP vs detection mAP — one-line difference.
16. When is Recall@k the wrong metric?
17. What is α-query expansion?
18. What does "Siamese" actually mean, and why shared weights?
19. Write the contrastive loss. What does the margin do and why is it needed?
20. What is the structural weakness of an absolute-distance margin?
21. Write the triplet loss. What makes it a *relative* constraint?
22. Why are triplet embeddings L2-normalised?
23. How many triplets exist for $N$ samples, and what fraction contribute gradient?
24. Define easy / semi-hard / hard negatives.
25. Why does hardest-negative mining cause collapse?
26. What is PK batch sampling and why is it part of the method?
27. What loss family replaced triplet loss for face recognition, and why?
28. Write ArcFace's logit. What's the geometric meaning of the margin?
29. When is triplet loss still the right choice over ArcFace?
30. Write InfoNCE. Reframe it as a classification problem.
31. What does the temperature $\tau$ trade off?
32. What are SimCLR's four ingredients?
33. Why is colour jitter essential in SimCLR? What shortcut does it block?
34. What problem does MoCo's queue solve, and what problem does the momentum encoder solve?
35. Why is $m = 0.999$ and not 0.9?
36. What is shuffling BN and which Module 2 property causes the need for it?
37. What prevents collapse in: SimCLR, BYOL, SimSiam, Barlow Twins, DINO?
38. What is a linear probe? A k-NN probe? Which is more informative for retrieval?
39. What is the structural limitation of contrastive SSL, and which paradigm fixes it?
40. Draw the recall/latency/memory triangle and place Flat, IVF-PQ, HNSW on it.
41. How does IVF work? What is `nprobe`?
42. What is the IVF boundary problem?
43. Explain PQ. Compute the compression for $D{=}768$, $m{=}96$.
44. What is ADC and why is asymmetric better than symmetric?
45. What does OPQ add?
46. Describe HNSW's structure and search. What are `M`, `efConstruction`, `efSearch`?
47. What is HNSW's binding constraint at scale?
48. What is ScaNN's anisotropic quantisation optimising for?
49. When should you *not* use an ANN index?
50. Pre-filter vs post-filter — when does each fail?
51. Why does ANN search work at all, given the curse of dimensionality?
52. How do you tune an ANN index properly? What curve do you plot?

---

## Part B — Whiteboard problems

**B1 — Embedding audit.** You have a 2048-d ResNet-50 penultimate embedding, unnormalised. Walk through every transformation you'd apply before indexing, and justify each: L2-normalise? PCA? whitening? centring? dimension choice? State what each fixes.

**B2 — Triplet mining simulation.** In a PK batch with $P=8$ identities, $K=4$ images each:
(a) How many anchors? How many (anchor, positive) pairs? How many valid triplets?
(b) Under batch-hard, how many loss terms per batch?
(c) Suppose one of the 32 images is mislabelled. Trace what happens to the batch-hard loss and to the gradient.

**B3 — InfoNCE derivation.** Show that InfoNCE with $2N$ views reduces to a softmax cross-entropy over $2N-1$ classes. Then show what happens to the gradient w.r.t. $z_i$ as $\tau \to 0$ and as $\tau \to \infty$, and connect each limit to alignment/uniformity.

**B4 — ANN capacity planning.** 50M vectors, 1024-d float32, p99 < 30 ms, Recall@10 ≥ 0.98, budget = one 256 GB machine.
(a) Raw storage? Feasible as Flat? As HNSW?
(b) Design an index. Show the memory arithmetic.
(c) What rerank stage do you add and why?
(d) How would your answer change if the requirement were Recall@100 ≥ 0.90 instead?

**B5 — PQ by hand.** $D = 8$, $m = 2$, $k^* = 4$ centroids per sub-space. Given a tiny toy dataset, run one iteration of PQ training and encode a vector. Then build the ADC lookup table for a query and compute the approximate distance. (Do it numerically — it makes ADC permanent.)

**B6 — System design.** Design "search by photo" for a 30M-item marketplace where sellers upload photos continuously and items go out of stock constantly. Cover: unit of retrieval, encoder choice and training, index choice, filters, freshness, evaluation, and monitoring.

---

## Part C — Traps

| Trap | Common wrong answer | Correct answer |
|---|---|---|
| "Use the model's output as the embedding." | logits | Logits project onto the label simplex and destroy within-class structure. Use penultimate / pooled tokens. |
| "Cosine is better than L2 for embeddings." | "Yes, always." | Identical ranking for L2-normalised vectors. Only differs for unnormalised ones — and then the question is which your loss trained. |
| "More dimensions = better embedding." | "Yes." | PCA-whitening to 128-d often *raises* recall and shrinks the index 16×. |
| "Triplet loss is the standard for face recognition." | "Yes." | It *was*. ArcFace-family angular-margin softmax replaced it — no mining, stable, geometric margin. |
| "Mine the hardest negatives." | "Yes, hardest is best." | Hardest negatives are dominated by label noise → collapse to a constant embedding at loss $\alpha$. Use semi-hard or batch-hard. |
| "SimCLR needs strong augmentation." | "Yes, more is better." | The specific finding is that **crop + colour jitter** is the critical *pair*; without colour jitter the model solves the task via a colour-histogram shortcut. |
| "Use the projection head output as your feature." | "That's what was trained." | Use $h$, before the head. The head absorbs the augmentation-invariance the downstream task doesn't want. |
| "BYOL works because of the momentum encoder." | "Yes." | SimSiam showed **stop-gradient** is the essential ingredient; momentum helps but isn't required. |
| "FAISS is a vector database." | "Yes." | FAISS is an index *library* — no persistence, no filtering, no replication, no API. A vector DB wraps one. |
| "HNSW is the best index." | "Yes." | HNSW is memory-bound. At 100M+ vectors you need PQ compression or DiskANN. "Best" depends on which corner of the triangle you're constrained on. |
| "We need an ANN index." | at 200k vectors | Below ~1M, exact search (especially on GPU) is fast, simpler, and exact. Adding an index adds a recall bug surface for no gain. |
| "Just post-filter the results." | "Fine." | With a selective filter you may return far fewer than $k$ — or nothing. Selective filters need filter-aware traversal or partitioning. |
| "Contrastive SSL gives the best features for everything." | "Yes." | It learns *augmentation-invariant* features, so it discards exactly what augmentation removes — weaker for dense tasks than masked modelling (MAE). |

---

## Part D — Spaced-repetition cards

```
Q: Where to take an image embedding from
A: Penultimate (post-GAP) layer, ViT [CLS], or mean-pooled patch tokens. NEVER logits — they project onto the label simplex and discard within-class structure.

Q: Why are CE features not metric-optimal?
A: Softmax+CE only requires linear separability of the training classes; it constrains neither intra-class compactness nor inter-class distances. Open-set retrieval needs metric learning or SSL.

Q: Projection-head finding
A: Train the contrastive loss on z = g(h); use h for downstream (~+10% linear probe). The head absorbs the augmentation-invariance the objective demands, so h keeps colour/orientation info the downstream task needs.

Q: GeM pooling
A: f_c = ((1/|X|)Σ x^p)^(1/p), p learnable (converges ~3). p=1 is average, p→∞ is max. Beats GAP for retrieval because retrieval cares about distinctive local evidence, not the average.

Q: Two-stage retrieval
A: Stage 1 = global descriptor + ANN index → top-1000 (cheap, approximate). Stage 2 = local features + RANSAC geometric verification, or a cross-encoder → top-10 (expensive, precise). Same shape as RAG's bi-encoder + reranker.

Q: BoVW
A: SIFT → k-means visual vocabulary → word-count histogram → tf-idf → inverted index → cosine → RANSAC rerank. Literally the text-search stack applied to images.

Q: L2 vs cosine vs IP
A: For L2-normalised vectors, ||a−b||² = 2 − 2cosθ — identical ranking. Unnormalised, they differ: IP rewards large norms, L2 penalises them. MIPS has no triangle inequality, so it weakens index pruning guarantees.

Q: Retrieval mAP vs detection mAP
A: Retrieval AP averages precision over the ranked database list per query. Detection mAP averages precision over a confidence-threshold sweep, per class, at an IoU threshold.

Q: Contrastive loss (Hadsell/Chopra/LeCun)
A: L = Y·D² + (1−Y)·max(0, m − D)². Margin bounds the repulsion; without it, distances explode. Weakness: m is in ABSOLUTE distance units.

Q: Triplet loss
A: L = [ d(a,p)² − d(a,n)² + α ]₊ , embeddings L2-normalised, α ≈ 0.2. A RELATIVE constraint, so no scale degeneracy.

Q: Triplet negative categories
A: hard: d(a,n) < d(a,p) · semi-hard: d(a,p) < d(a,n) < d(a,p)+α · easy: d(a,n) > d(a,p)+α (zero gradient).

Q: Why hardest-negative mining collapses
A: Globally hardest negatives are disproportionately label noise/outliers → huge wrong gradients → f(x)=const, which satisfies the loss trivially at value α.

Q: PK sampling
A: Batches of P identities × K images. Guarantees positives per anchor and P−1 identities of negatives. Batch construction is part of the loss in metric learning.

Q: ArcFace
A: −log[ e^{s·cos(θ_y + m)} / (e^{s·cos(θ_y+m)} + Σ_{j≠y} e^{s·cosθ_j}) ], s=64, m=0.5 rad. Additive ANGULAR margin. Replaced triplet loss: no mining, stable, geometrically uniform margin. Needs an enumerable training identity set.

Q: InfoNCE
A: L = −log[ exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ) ]. A (2N−1)-way softmax classification with free labels.

Q: Temperature τ
A: Small τ (0.05–0.1) → sharp → gradient concentrates on hardest negatives → strong uniformity, sensitive to false negatives. Large τ → soft → tolerant, less discriminative. Trades alignment vs uniformity (Wang & Isola 2020).

Q: SimCLR's four ingredients
A: (1) composed strong augmentation — crop + COLOUR JITTER is the critical pair, (2) non-linear projection head (train on z, use h), (3) large batches (= many negatives), (4) cosine similarity / L2 normalisation.

Q: Why colour jitter in SimCLR
A: Two crops of the same image share a colour histogram, so the network can solve InfoNCE by matching colour statistics — a shortcut. Colour distortion destroys it and forces shape/structure.

Q: MoCo
A: Queue of 65k keys decouples #negatives from batch size. Momentum key encoder θ_k ← mθ_k + (1−m)θ_q with m=0.999 keeps old queue entries consistent. Shuffling BN prevents the BatchNorm batch-statistics leak.

Q: What prevents representation collapse?
A: Negatives/uniformity (SimCLR, MoCo) · predictor + stop-gradient (+EMA) asymmetry (BYOL, SimSiam — stop-grad is the essential part) · explicit variance/decorrelation terms (VICReg, Barlow Twins) · centring + sharpening (DINO).

Q: Contrastive SSL's structural limit
A: It learns augmentation-INVARIANT features, so it discards exactly what augmentation removes (colour, precise position, scale) — weaker for dense tasks than masked modelling (MAE).

Q: ANN trade-off triangle
A: Recall vs latency vs memory — pick two. Flat = recall+exact/slow+huge. HNSW = recall+fast/memory-heavy. IVF-PQ = compact+fast/recall reduced.

Q: IVF
A: k-means into nlist Voronoi cells; search the nprobe nearest cells. Speedup ≈ nlist/nprobe. nlist ≈ √N to 4√N. nprobe is the query-time recall dial. Boundary problem caps recall.

Q: Product Quantisation
A: Split D dims into m sub-vectors, k-means 256 centroids each → m bytes/vector. D=768 float32 = 3072 B → m=96 → 96 B (32× compression). Effective codebook 256^m.

Q: ADC
A: Keep the query un-quantised; precompute an m×256 lookup table T[j][c] = ||q_j − c_{j,c}||². Distance = Σ_j T[j][code_j(x)] — m lookups and adds, no multiplies. Asymmetric beats symmetric by avoiding the query's own quantisation error.

Q: OPQ
A: A learned rotation applied before splitting, balancing variance across sub-spaces and decorrelating them. Typically +1–3% recall, free.

Q: HNSW
A: Multi-layer proximity graph; level ℓ = floor(−ln U(0,1) · mL), so higher layers are exponentially sparser (a skip-list in metric space). Greedy descent from the top, beam search (efSearch) at layer 0. M = neighbours/node, efConstruction = build beam, efSearch = query dial. Memory-bound; deletions are awkward.

Q: When NOT to use an ANN index
A: Below ~1M vectors — GPU flat search is fast and exact, and an index only adds a recall bug surface.

Q: Filtered vector search
A: Low-cardinality + static filter → partition into separate indexes. High-churn or high-cardinality → filter-aware traversal with over-fetch. Naive pre-filter breaks graph connectivity; naive post-filter can return < k results.

Q: Why ANN works despite the curse of dimensionality
A: Real embeddings lie on a low intrinsic-dimensional manifold. In truly uniform high-dim data, all points are near-equidistant and no index helps.

Q: How to tune an ANN index
A: Build exact ground truth for ~1000 held-out queries, then sweep the query-time dial (nprobe/efSearch) and plot Recall@k vs QPS. Set the recall target from the product, then minimise memory subject to it. Re-measure after any embedding change.
```

---

## Part E — Self-assessment rubric

| Skill | 1 | 3 | 5 (top 1%) |
|---|---|---|---|
| Embeddings | "the vector from the model" | knows penultimate vs logits | explains CE-vs-metric mismatch, projection-head finding, whitening gains, Matryoshka |
| Classical retrieval | never heard of BoVW | knows BoVW exists | maps BoVW↔tf-idf↔RAG, explains GeM and NetVLAD, names geometric verification as the original reranker |
| Metrics | knows cosine | knows L2≡cosine when normalised | knows MIPS≠NN and why it breaks pruning; distinguishes retrieval vs detection mAP |
| Metric learning | knows triplet loss formula | knows mining matters | explains collapse mechanism, PK sampling, and why ArcFace displaced triplet |
| Contrastive SSL | "augment and pull together" | knows InfoNCE and SimCLR | explains colour shortcut, projection head, MoCo queue+momentum, shuffling BN, and answers "what prevents collapse" across five families |
| ANN | "use FAISS" | knows IVF and HNSW exist | works the recall/latency/memory triangle numerically, explains ADC, knows HNSW is memory-bound, handles filtered search |
| Systems | describes components | designs a pipeline | plans freshness, monitoring, rebuild strategy, hot/cold index, and shadow-recall alarms |

---

## Part F — Module 3 exit criteria

- [ ] State the projection-head finding *and* its explanation without notes.
- [ ] Derive $\|a-b\|^2 = 2-2\cos\theta$ and state the MIPS caveat.
- [ ] Explain triplet-loss collapse and name the two mining strategies that avoid it.
- [ ] Write InfoNCE and explain $\tau$ via alignment/uniformity.
- [ ] Explain PQ + ADC well enough to implement it (and have implemented it).
- [ ] Produce a Recall@10 vs QPS plot across 5 index types on 1M vectors, annotated with memory.
- [ ] Be able to narrate the vision↔RAG mapping table from memory in 60 seconds.
