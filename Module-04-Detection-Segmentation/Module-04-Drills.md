# Module 4 — Drills, Interview Bank & Spaced Repetition

---

## Part A — Rapid-fire

1. Why is detection structurally different from classification? Answer in one sentence.
2. Name the three sub-problems every detector must solve.
3. Write the R-CNN box delta encoding. Why divide by $w_a$? Why the log?
4. Why is the box-regression loss computed only on positives?
5. What were Viola-Jones' three contributions, and which one survives everywhere?
6. How is HOG related to SIFT?
7. What is DPM's deformation cost, and what modern operator revives the idea?
8. Tell the R-CNN → Fast → Faster story in three sentences, in terms of computation sharing.
9. What is RoI Pooling's quantisation problem? Where exactly does the rounding happen?
10. Why does RoI Align help masks 10× more than boxes?
11. State the classic anchor assignment rule, including the two details people forget.
12. Write smooth L1. Why not L2? Why not L1?
13. What does Cascade R-CNN fix, and what is the mismatch it addresses?
14. Name the four multi-scale architectures and the flaw of each of the first three.
15. Write FPN's merge equation. What does each of the three convs do?
16. Why nearest-neighbour upsampling in FPN, and what is the final $3\times3$ conv for?
17. Write the FPN RoI level-assignment formula and interpret it.
18. Why can FPN share one detection head across all levels?
19. How is FPN related to the Laplacian pyramid?
20. What do PANet and BiFPN each add, and why?
21. What is ViTDet's counterpoint to FPN?
22. Why does YOLOv2 sigmoid the centre prediction?
23. Why does YOLOv2 use IoU distance for anchor k-means?
24. Write FCOS's centre-ness and say what it fixes.
25. What is ATSS's central claim?
26. What does SimOTA formulate assignment as?
27. How do YOLOv10/YOLO26 remove NMS?
28. Which current detectors are AGPL and which are permissive? Why does it matter?
29. Do the focal-loss motivation arithmetic: 100k negatives at $p_t{=}0.99$ vs 10 positives at $p_t{=}0.5$.
30. Write focal loss. What is the down-weighting at $p_t = 0.9$ with $\gamma=2$?
31. Why is $\alpha = 0.25$ (weighting the *rare* class down) not a typo?
32. Write the prior bias initialisation and say what happens without it.
33. Why does Faster R-CNN not need focal loss?
34. What does focal loss do to calibration?
35. What is the 2026 nuance about focal loss vs label assignment?
36. Define IoU. Why is it a bad loss on its own?
37. GIoU, DIoU, CIoU — what does each add?
38. Write the NMS loop. What is its structural failure and why can't a threshold fix it?
39. What does Soft-NMS change, and what does it buy?
40. Describe COCO AP computation in five steps. What is the one-match rule?
41. Why is COCO AP much lower than VOC mAP@0.5?
42. Does the confidence threshold change AP? Does it change your product?
43. Name a domain where mAP is the wrong metric and say what's used instead.
44. Sketch DETR's architecture. What is an object query?
45. Write the bipartite matching objective. Which algorithm solves it, at what cost?
46. Why does one-to-one matching remove the need for NMS?
47. Why is DETR slow to converge? Name two fixes and the mechanism of each.
48. What are deformable attention's *two* benefits?
49. What did DAB-DETR reveal about object queries?
50. Semantic vs instance vs panoptic — and what is the things/stuff distinction?
51. What constraint defines a panoptic output?
52. What did Mask2Former unify, and by reframing the task as what?
53. Why does U-Net concatenate while FPN adds?
54. Why does Mask R-CNN use $K$ sigmoid masks rather than $K+1$ softmax channels?
55. Trace the skip-connection lineage across four architectures.
56. Write the Dice↔IoU relation. Do they rank models differently?
57. Why is Dice-as-a-loss robust to class imbalance? Why still combine with CE?
58. Write PQ. What does each factor mean?
59. What does mIoU fail to measure, and what metric fixes it?
60. What is TIDE and what are its six error types?

---

## Part B — Whiteboard problems

**B1 — Anchor arithmetic.** Input $1024\times1024$, FPN levels $P_3..P_7$ (strides 8–128), 9 anchors per location.
(a) Total anchors.
(b) With 8 objects per image, estimate the positive fraction.
(c) Compute the cross-entropy loss mass ratio between negatives and positives at $p=0.05$ for negatives and $p=0.4$ for positives.
(d) Recompute with focal loss, $\gamma=2$, $\alpha=0.25$. Show the flip.

**B2 — FPN level assignment.** For proposals of size $32\times32$, $100\times50$, $224\times224$, $600\times400$, compute the FPN level. Then explain what breaks if all proposals go to $P_4$.

**B3 — Focal loss derivation.** Derive $\partial \text{FL}/\partial p$ and show that for $\gamma > 0$ the gradient magnitude on well-classified examples decays faster than CE's. Then explain what happens to the gradient of a misclassified positive.

**B4 — AP by hand.** Given 6 detections on 4 GT boxes with specified scores and IoUs, compute AP@0.5 step by step (TP/FP assignment, cumulative P and R, interpolated area). Then recompute after removing NMS so two detections hit the same GT, and quantify the AP loss.

**B5 — Hungarian matching.** Given a $3\times3$ cost matrix, find the optimal assignment by hand and show that the greedy assignment differs. Then explain what this means for a detector that used greedy matching instead.

**B6 — System design.** Design a detection system for a warehouse robot: detect pallets, boxes, people; 640×480 camera at 20 FPS on a Jetson Orin Nano; safety-critical for people. Cover model choice, resolution, label assignment, loss, NMS, thresholds per class, evaluation metric, calibration, and failure handling.

**B7 — Debugging.** Your detector's training loss is decreasing but AP is 0.0 after 10 epochs. List, in order, the eight things you'd check.

---

## Part C — Traps

| Trap | Common wrong answer | Correct answer |
|---|---|---|
| "Why does ResNet-based RetinaNet need focal loss?" | "Because one-stage detectors are less accurate." | Because there is no proposal cascade to filter the ~1000:1 background:foreground ratio; the imbalance must be handled in the loss instead of the architecture. |
| "Anchor-free is better than anchor-based." | "Yes, it's newer." | ATSS showed they perform identically once **label assignment** is controlled. The difference was never the anchors. |
| "FPN improves accuracy because it's deeper." | agreeing | It adds almost no depth or compute. It routes **semantics down** to **high-resolution** levels — a separation of concerns, not extra capacity. |
| "Just use shallow layers for small objects." | "Sure." | Shallow layers have resolution but no semantics — that's exactly why SSD's pyramid underperformed and why FPN exists. |
| "Higher NMS threshold = better recall = better AP." | "Yes." | Recall rises, precision falls, AP is a shallow optimum. And no threshold works for both crowded and sparse regions of the same image. |
| "DETR doesn't need NMS because transformers are smarter." | vague | Because the **Hungarian matching is one-to-one**: every non-matched prediction is explicitly trained to output $\varnothing$, so duplicates are directly penalised. Dense detectors use one-to-many assignment, which *encourages* duplicates. |
| "Deformable attention is just faster attention." | "Yes." | Its linear cost is what makes **multi-scale features affordable**, which is what fixes DETR's small-object AP — and the sampling locality is what gives the 10× convergence speedup. |
| "Increase mAP and the product improves." | "Yes." | AP is threshold-free and averages over IoU and classes. Check AP_small/large, per-class AP, and precision/recall **at your deployed threshold**. |
| "Dice and IoU can disagree about which model is better." | "Yes." | They're monotonically related ($\text{Dice}=2\text{IoU}/(1+\text{IoU})$) — identical ranking. They differ in interpretability and as *losses*, not as rankings. |
| "Dice 0.92 means the segmentation is good." | "Yes." | Dice is interior-dominated. Check Boundary IoU / 95% Hausdorff and the **per-case distribution**. |
| "Use softmax over classes for the mask branch." | "Natural." | Mask R-CNN uses $K$ independent sigmoids so masks don't compete across classes — decoupling segmentation from classification is worth several mask AP. |
| "Random train/val split." | "Standard." | Splits must be by video / patient / site / session. Consecutive frames are near-duplicates and random splits leak. |
| "Our model is 12 GFLOPs so it'll run at X FPS." | "Right." | Pre-processing, post-processing and **NMS** are routinely 30–50% of detector wall-clock and are invisible in FLOPs. Measure end to end. |

---

## Part D — Spaced-repetition cards

```
Q: Detection as a formal problem
A: Set prediction with unknown cardinality. Three sub-problems: (1) where to look — proposals/anchors/queries, (2) label assignment, (3) duplicate removal — NMS or one-to-one matching.

Q: R-CNN box encoding
A: t_x=(x−x_a)/w_a, t_y=(y−y_a)/h_a, t_w=log(w/w_a), t_h=log(h/h_a). Division ⇒ scale invariance; log ⇒ positivity + symmetric penalty for halving/doubling.

Q: RoI Align
A: RoI Pooling quantises twice (box→feature-map coords, then bin boundaries) ⇒ up to ~half-stride (~8 px) misalignment. RoI Align samples 4 points per bin at exact float coords via bilinear interpolation. +~10% relative mask AP, +~1% box AP.

Q: Classic anchor assignment
A: Positive if IoU ≥ 0.7 OR it's the highest-IoU anchor for some GT (guarantees every GT gets a positive). Negative if IoU < 0.3. IGNORED in between. Sample 256/image at ~1:1.

Q: Smooth L1 / Huber
A: 0.5x² for |x|<1, |x|−0.5 otherwise. L2 alone: gradient ∝ x, outliers explode. L1 alone: discontinuous at 0, never anneals. Huber gets both.

Q: FPN merge
A: P_l = conv3×3( conv1×1(C_l) + upsample2×(P_{l+1}) ). 1×1 = channel projection to 256 (enables addition + shared head); nearest-neighbour upsample; final 3×3 = anti-aliasing.

Q: FPN core insight
A: Resolution comes from the bottom-up pathway, semantics from the top-down pathway; lateral connections give every level both. It's a learned Laplacian pyramid run in reverse (compose instead of decompose).

Q: FPN RoI level assignment
A: k = floor(k0 + log2(√(wh)/224)), k0 = 4. A 224² RoI → P4 (stride 16); halving the RoI size moves it one level finer.

Q: Why SSD's pyramid underperforms
A: Shallow feature maps have high resolution but weak semantics (small receptive field, low-level features), so "is this a pedestrian?" is unanswerable there.

Q: FCOS centre-ness
A: √( min(l,r)/max(l,r) · min(t,b)/max(t,b) ). 1 at the centre, 0 at the edge; multiplied into the score at inference to down-rank off-centre low-quality boxes.

Q: ATSS's claim
A: Anchor-based and anchor-free detectors perform identically once label assignment is controlled. Adaptive threshold = mean + std of IoUs of the k nearest candidates per GT.

Q: Focal loss
A: FL(p_t) = −α_t (1−p_t)^γ log(p_t). γ=2: p_t=0.9 ⇒ 100× down-weight; p_t=0.968 ⇒ 1000×; p_t=0.5 ⇒ 4×; p_t=0.1 ⇒ 1.2× (hard examples untouched). It is SOFT OHEM — re-weight, don't discard.

Q: Focal loss motivation arithmetic
A: 100,000 easy negatives × CE(0.99)=0.01 ⇒ 1000, vs 10 positives × CE(0.5)=0.69 ⇒ 6.9. Background contributes ~145× more loss. With γ=2 the balance flips.

Q: Why α = 0.25 in focal loss
A: γ already over-suppresses the easy negatives, so the balance has over-corrected toward positives; α pulls it back. α and γ are coupled — optimal α decreases as γ increases.

Q: Focal loss prior initialisation
A: Init the final classification conv bias to b = −log((1−π)/π) with π=0.01, so σ(b)=0.01. Without it, ~100k anchors at p≈0.5 produce a huge first gradient and RetinaNet DIVERGES.

Q: Focal loss side-effect
A: Poor calibration — under-confident on positives. Needs temperature scaling if scores are consumed as probabilities.

Q: IoU loss family
A: IoU (scale-invariant, but zero gradient with no overlap) → GIoU (enclosing-box penalty ⇒ gradient without overlap) → DIoU (+ normalised centre distance ⇒ faster convergence) → CIoU (+ aspect-ratio term).

Q: NMS failure mode
A: Crowded scenes — two genuinely distinct objects overlapping above the threshold, one gets deleted. No threshold fixes both crowding and duplicates. Soft-NMS decays instead of deleting (+~1 AP); NMS-free architectures remove the problem.

Q: COCO AP computation
A: Sort by confidence → greedily match to unmatched GT at IoU ≥ τ (each GT once; a second hit is an FP) → cumulative P/R → 101-point interpolated area → mean over classes → averaged over IoU 0.5:0.05:0.95. Also AP_small/medium/large at 32² and 96².

Q: DETR loss
A: Hungarian bipartite matching minimising Σ[−p̂(c_i) + L_box] (probability, not log, for commensurability), O(N³) with N=100. Then CE + L1 + GIoU on matched pairs, with the ∅ class down-weighted 10×.

Q: Why DETR needs no NMS
A: One-to-one matching means exactly one prediction is responsible per GT, and every other prediction is trained to output ∅. Duplicates are directly penalised. Dense detectors use one-to-many assignment, which encourages duplicates.

Q: DETR's slow convergence
A: (1) Hungarian matching is unstable early — a query's target flips between steps, giving inconsistent supervision. (2) Cross-attention starts uniform and must learn spatial selectivity from scratch. Fixes: deformable attention (locality prior + linear cost, ~10× faster) and query denoising (DN/DINO-DETR).

Q: Deformable attention's two benefits
A: (1) O(HW·K) instead of O((HW)²) ⇒ MULTI-SCALE features become affordable ⇒ fixes small-object AP. (2) The K learned sampling offsets impose a locality prior ⇒ ~10× faster convergence.

Q: 2026 detector landscape
A: YOLO26 (Jan 2026, NMS-free end-to-end, edge-optimised, AGPL-3.0) · YOLOv12 (attention-centric, Area Attention + R-ELAN) · RF-DETR (DINOv2 backbone, first real-time >60 mAP COCO, Apache-2.0) · RT-DETR · RTMDet (MIT, 300+ FPS). Licence matters commercially.

Q: Panoptic segmentation
A: Every pixel gets exactly one (class, instance-id) label — no overlaps, no gaps. Things (countable) vs stuff (amorphous). PQ = SQ × RQ.

Q: Panoptic Quality
A: PQ = [Σ_TP IoU / |TP|] × [|TP| / (|TP| + ½|FP| + ½|FN|)] = SQ × RQ. Matching at IoU > 0.5 is unique because panoptic segments can't overlap.

Q: U-Net vs FPN merge
A: U-Net CONCATENATES — the decoder must reconstruct exact boundaries and needs both signals intact. FPN ADDS — channel counts must stay equal across levels so one detection head can be shared.

Q: Mask R-CNN mask branch
A: K per-class binary masks with per-pixel SIGMOID + BCE, not a softmax over classes. Decouples "is this pixel part of the object" from "which class", worth several mask AP.

Q: Dice ↔ IoU
A: Dice = 2·IoU/(1+IoU); IoU = Dice/(2−Dice). Monotonically related ⇒ identical model ranking. Dice-as-loss is region-based ⇒ robust to class imbalance; standard practice is CE + Dice.

Q: mIoU's blind spot
A: Interior pixels dominate, so boundary quality is invisible. Use Boundary IoU (IoU within a band around the boundary) and 95% Hausdorff distance. Report per-case distributions, not just the mean.

Q: TIDE
A: Error decomposition of AP loss into six types: classification, localisation, both, duplicate, background FP, missed GT. Run it BEFORE changing the architecture — it turns "improve the model" into a specific action.
```

---

## Part E — Self-assessment rubric

| Skill | 1 | 3 | 5 (top 1%) |
|---|---|---|---|
| Framing | "detection finds boxes" | knows the pipeline stages | frames it as set prediction; names the three sub-problems; explains the box encoding's log |
| Two-stage | knows Faster R-CNN exists | describes RPN + anchors | tells the computation-sharing story, explains RoI Align's two quantisations and the box/mask asymmetry, knows the ignore band |
| FPN | "it's multi-scale" | draws the top-down path | states the separation of concerns, writes the level formula, explains the anti-aliasing conv, connects it to the Laplacian pyramid, cites ViTDet |
| One-stage | knows YOLO | knows anchors vs anchor-free | cites ATSS, explains centre-ness and SimOTA, knows NMS-free dual assignment and the licensing landscape |
| Focal loss | writes the formula | explains the modulating factor | does the loss-mass arithmetic, explains the α inversion, knows the prior bias init, knows the calibration cost and the 2026 nuance |
| Metrics | knows mAP | can compute AP | knows the one-match rule, IoU averaging, that AP is threshold-free, and names a domain-appropriate alternative |
| DETR | "transformers for detection" | describes queries + matching | explains why matching removes NMS, diagnoses slow convergence correctly, knows deformable attention's dual benefit and DAB-DETR's insight |
| Segmentation | knows the three types | knows U-Net and Mask R-CNN | explains concat-vs-add, sigmoid-vs-softmax masks, PQ's factorisation, Boundary IoU, and Mask2Former's unification |
| Practical | can fine-tune a model | evaluates properly | runs TIDE first, splits by the right unit, freezes BN, calibrates, measures end-to-end latency including NMS |

---

## Part F — Module 4 exit criteria

- [ ] Do the focal-loss loss-mass arithmetic on a whiteboard from memory, and state the prior-bias init.
- [ ] Draw FPN, write the merge equation and the level-assignment formula, and explain why (c) fails.
- [ ] Compute COCO AP by hand on a 6-detection example.
- [ ] Explain in 60 seconds why DETR needs no NMS, and why it used to take 500 epochs.
- [ ] State ATSS's claim and what it implies about the anchor debate.
- [ ] Have implemented: IoU + NMS + Soft-NMS + COCO AP (validated against pycocotools), focal loss (validated against torchvision), FPN, the Hungarian matcher, and U-Net.
- [ ] Have fine-tuned one detector end to end with a TIDE analysis and a measured int8 latency number.
