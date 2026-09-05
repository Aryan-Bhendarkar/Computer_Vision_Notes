# Review Round 2 — Coverage & Usability Audits

Two agents reviewed the complete draft: a **CV domain expert** auditing coverage against a full modern body of knowledge, and a **self-taught learner** auditing whether the material can actually be learned unsupervised. Both completed. This records what they found and what changed.

---

## A. Coverage audit — verdict

> *"Roughly **80% complete** and, within its chosen scope, of unusually high quality. Modules 2, 3, 4 and 6 are close to complete for a modern deep-learning-for-vision curriculum; Modules 1 and 5 have structural holes, because both were written as 'classical foundations that lead to the deep learning' rather than as complete subjects."*

**The three biggest holes identified**, all now closed:

| Gap | Why it mattered | Fixed by |
|---|---|---|
| **Point-cloud & 3D deep learning** — PointNet through BEV perception, entirely absent | "cuts off robotics and autonomous-driving roles completely"; also the natural bridge between the geometry and the deep learning the material already teaches | `Module-05-Appendix-3D-Deep-Learning.md` |
| **Classical image processing** — thresholding, morphology, connected components, Hough, histogram ops, classical segmentation | "both the most-used CV in production and the easiest interview warm-up to fail" | `Module-01-Appendix-Image-Processing.md` |
| **Robustness, uncertainty and governance** — adversarial, OOD, epistemic uncertainty, ethics, licensing, biometric regulation | none appeared anywhere; "shipping on a mis-licensed dataset is a real, common, career-affecting mistake" | `Supplement-Robustness-Scale-and-Governance.md` |

**Also added from the CRITICAL/IMPORTANT lists:** pose & keypoint estimation and industrial anomaly detection (`Module-04-Appendix-Pose-and-Anomaly.md`); monocular depth, MVS/ICP/TSDF, and non-RGB sensors (M5 appendix); histogram/CLAHE, Hough, template matching, frequency-domain and the resampling/resize bug (M1 appendix); knowledge distillation, pruning/sparsity, distributed training, long-tail recognition, conformal prediction, and the tooling ecosystem (robustness supplement); VQ-VAE/VQGAN, autoregressive & masked image generation, and video generation (new §6.15b).

Estimated coverage after these additions: **~92–95%**.

## B. Usability audit — verdict

> *"For Modules 1–4 I could genuinely learn this alone, and that is rare. But the material degrades sharply in the second half... I would most likely fail at 6.13/6.14, where the two hardest derivations in diffusion are replaced by 'After deriving the variational bound…' and a section titled 'Classifier-free guidance — derive it' that contains no derivation."*

**Blockers found and fixed:**

| # | Defect | Fix |
|---|---|---|
| 1 | **6.14 CFG: a section titled "derive it" that doesn't derive it** — and whose mastery-check answer restated the formula | Wrote the four-step derivation: score identity → classifier guidance → Bayes to eliminate the classifier → collect terms |
| 2 | **6.13: "After deriving the variational bound…"** hid the hardest step in diffusion | Added the four-move path from the ELBO to $\mathcal{L}_{\text{simple}}$, plus the point that **$\mathcal{L}_\text{simple}$ is a *reweighted*, not exact, ELBO** |
| 3 | **6.13: the closed form asserted, not shown** | Added the two-line Gaussian-merge induction |
| 4 | **5.8: $\mathbf{v}$, $\mathbf{w}$, $g$ used and never defined** — making the Schur build task impossible | Defined them as the camera/point blocks of $-g$ in $\mathcal{H}\Delta = -g$ |
| 5 | **$H$ meant both homography and BA Hessian in the same file**, twenty lines apart in the same drill deck | Renamed the BA Hessian to $\mathcal{H}$ throughout §5.8 and the drills; added a collision table to `NOTATION.md` |
| 6 | **2.4: $\nabla_{aW}L = \frac1a\nabla_W L$ asserted as a "therefore"** — the README's designated "one derivation to own" | Added the chain-rule step, plus the $1/a^2$ effective-LR consequence |
| 7 | **5.5: the $\lambda$ scale-invariance step compressed into a parenthesis** — exactly where the drills demand a live derivation | Written out explicitly |
| 8 | **5.4: $F=[\mathbf{e}']_\times H$ asserted, and the DoF count wrong** ("one-parameter" for a free epipole) | Added the one-line derivation; corrected to a **two-parameter** family |
| 9 | **16 concepts had no mastery check** (5.1, 5.3, 5.4, 5.7, 5.9, 5.10; 6.1, 6.3, 6.6–6.9, 6.11–6.13, 6.15) — a third of the syllabus, in the modules where fatigue is highest | All 16 written. Every concept now has one except the practical/capstone sections, which carry deliverables instead |
| 10 | **The README's "six beats" promise was false** — "Connect it" appeared in Module 1 only | README amended to describe connections as threaded through the prose, which is what they are |

**Structural gaps found and fixed:** no glossary → `GLOSSARY.md`; no formula sheet → `FORMULA-SHEET.md`; no notation table → `NOTATION.md`; no progress tracker → `PROGRESS.md`; no environment/setup guidance and no GPU-or-time estimates on builds → added to the README; the README's schedule contradicted its own warning → replaced with one usable plan carrying the corrected allocation.

**Still outstanding** (deliberately, for now): collapsed answer keys for Drills Parts A and B, and per-task time estimates on all ~50 builds rather than the priority ten. Both are worth doing before or during the web-artifact build, where they are cheap to generate and easy to attach.

## C. What the audits praised — do not change these

Worked derivations rather than asserted results; verified arithmetic; mastery-check answer sketches that are frequently better than the section they test; the traps tables; the "top-1% distinction" sections where they name a specific failure mode rather than a fact; the cross-module through-lines; and the honesty of the README's job-ROI framing.
