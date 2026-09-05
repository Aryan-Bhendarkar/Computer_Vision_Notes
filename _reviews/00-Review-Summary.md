# Review Round 1 — Findings and Actions

Three perspective reviews were commissioned on the complete draft: **instructor** (technical correctness), **job mentor** (hiring value), **self-learner** (usability). The mentor review completed in full. The instructor and self-learner agents were cut short by an API rate limit mid-run; the instructor's highest-value function — verifying every numeric claim — was completed directly instead, deterministically, and is reported below.

---

## 1. Instructor review — arithmetic verification (COMPLETED, deterministic)

Every numeric claim in all six modules was recomputed in Python and checked against the text.

**Verified correct (no change needed):**

| Claim | Checked |
|---|---|
| RANSAC table, all 7 rows ($N = \log(1-p)/\log(1-w^s)$) | homography 72 ✓, 8-pt F 1177 ✓, 7-pt 587 ✓, 5-pt 145 ✓, F@w=0.3 70,188 ✓ |
| Scale-space: octave 2, $s{=}4$, $\sigma_0{=}1.6$, idx 1 | $k{=}1.1892$, $\sigma_{\rm oct}{=}1.9027$, eff $7.611$, radius $10.76$ ✓ |
| Harris $R$ at $\lambda=(4000,3)$, $\kappa{=}0.05$ | $-789{,}200$ ✓ |
| ResNet-50 conv1 | 9,408 params (bias=False, correct), 118 M MACs ✓ |
| $3{\times}3$ 128→256 s2 vs bottleneck | 295,168 vs 61,440 params (4.80×); 231.2 M vs 67.4 M MACs (3.43×) ✓ |
| Depthwise-separable 512→512 @14² | 462 M → 52.3 M, ratio 8.84 ✓ |
| VGG $27C^2$ vs $49C^2$ | 45% fewer ✓ |
| PQ compression 3072 B → 96 B | 32× ✓ |
| Dice(IoU = 0.6) | 0.750 ✓ |
| FPN level for 64×128 | $\sqrt{wh}=90.5$, $\log_2 = -1.307$, → $P_2$ ✓ |
| Focal loss down-weighting at $\gamma{=}2$ | 0.5→4×, 0.9→100×, 0.1→1.2× ✓; prior bias $-4.595$ ✓ |
| Focal loss-mass ratio | 1005 vs 6.93 = **145×** ✓; with $\gamma{=}2$: 0.10 vs 1.73 (flips) ✓ |
| Intrinsics: 24 mm lens, 6 µm pitch | $f_x = 4000$ px ✓ |
| Stereo $\delta Z = Z^2\delta d/(fB)$ | 1.19 cm @2 m, 29.8 cm @10 m, 7.44 m @50 m ✓; $B = 3.214$ m ✓ |
| BA: 500 cams + 80k points | 244,500 unknowns ✓, 478 GB dense ✓, $1.6\times10^5$ speed-up ✓ |
| ViT 224→384 | 196→576 tokens (2.94×), attention 8.64× ✓ |
| Latent diffusion compression | 786,432/16,384 = 48× ✓ |

**Corrected (4 rounding drifts, now fixed in the files):**

| File | Was | Now |
|---|---|---|
| 1.8 RANSAC table, line row | 16 | **17** (exact value is 16.0; the table rounds up) |
| 1.8 mastery (a), $w{=}0.35$, $s{=}8$ | 20,470 | **20,450** |
| 1.8 mastery (b), 7-point | 7,190 | **7,155** |
| 1.8 mastery, 5-point | 876 | **875** |
| 4.3 mastery (a), anchor count | ~28,350 (used $\lceil 1000/16\rceil$) | **~27,900** ($\lfloor 1000/16 \rfloor = 62$) |
| 4.6 down-weighting table, $p_t{=}0.968$ | 1000× | **~1000×** (exact 977×) |

**Verdict: the mathematical content is sound.** No conceptual or derivational errors were found; the only defects were rounding presentation.

---

## 2. Job-market mentor review — the honest verdict (COMPLETED)

> **Reviewer's framing:** *"This is excellent at solving a problem Aryan does not have."*

### The core criticism, accepted

The material assumes he already has the interview. **Getting the interview is the actual bottleneck** — a 3rd-year student (graduating 2028) with LeetCode ~1417 targeting remote roles that filter for 2–5 years of experience. Nothing in 95,000 words produces a clickable artefact, a referral, a rehearsed project narrative, or a coding-screen pass.

Second, subtler risk flagged: material written to make you *sound* impressive invites interviewers to probe exactly what you volunteer. A rehearsed "Gram anchoring preserves dense features at long schedules" that collapses on the first follow-up scores **worse** than "I don't know."

### Market reality check

- **Genuinely job-relevant (~25–30%, and rising in value):** all of **Module 3** (retrieval infrastructure — the load-bearing skill in most GenAI products, and it converts your RAG résumé line into demonstrable competence); **6.5 / 6.10 / 6.11** (CLIP, VLMs, composed pipelines — 6.10 was called *"the most job-relevant section in the document and one of the shortest"*); **2.7 + S.3** (systems skills in a CV costume); **2.8–2.11** (applied ML taught through vision).
- **Academic for this goal (~40%):** all of Module 5, most of Module 1, 4.2, the GAN lineage, most of 6.15. Module 5 is asked in robotics/AV/AR loops — overwhelmingly on-site and hardware-adjacent, not GenAI.
- **What actually gets hired in 2026:** evals above all → serving (vLLM/SGLang, continuous batching, KV-cache economics) → agentic systems that don't fall over → multimodal document pipelines.

> *"The mistake isn't studying CV — it's studying all of CV at uniform depth."*

### The single biggest miscalibration identified

The README elevating **epipolar geometry and the Schur complement** into "the six concepts that carry the most weight." **Fixed** — see §4.

### Top-1% sections that land vs. don't

**Land:** 2.4's BN scale-invariance derivation · 2.7's roofline / FLOPs≠latency · 4.6's loss-mass arithmetic and prior-bias init · 4.7's "AP is threshold-free but your product isn't" · 3.7's pre/post-filter and "don't use ANN below 1M" · 6.10's "connector + what's trained" · S.2's shortcut test.

**Don't (impressive-sounding trivia):** the scale-space uniqueness axiom · the DoG derivation the README called critical · Harris/SIFT/ORB internals · DINOv3 Gram anchoring and parameter-count history (six-month half-life).

### Missing content — the actual questions that will be asked

- **LLM/GenAI:** "Walk me through your chunking strategy and how you validated it." · "Evaluate a RAG pipeline end to end — separate retrieval from generation quality." · "Answers are wrong 15% of the time; is it retrieval or generation?" · "How do you handle a 200-page PDF with tables and figures?" (*highest-frequency real question in 2026*) · "Design an agent — how do you stop it looping?" · "What is prompt injection for a system ingesting user documents?" · "Estimate cost at 1M requests/day." · "Explain the KV cache and continuous batching."
- **ML system design:** the three system-design drills (3.B6, 4.B6, 6.B7) have **no answer sketches and no rubric** — the highest-value practice in the document has no feedback loop.
- **Practical judgement / behavioural — completely absent:** "Walk me through a project in five minutes" (*the most common elimination point, and 100% trainable*) · "Model worked in dev, failed in prod" · "A client project that went badly" (agency co-founder — a gift question).
- **Coding — the gate upstream of everything:** ~1417 will not clear frontier/FAANG-tier screens. *"Nine weeks of CV followed by elimination at problem two is the worst outcome and the most likely one on the current plan."* Also: ML coding rounds are a separate muscle (implement MHA in 25 min; implement a KV cache).
- **CV-specific:** noisy labels (**entirely absent, very commonly asked**) · semi-supervised / pseudo-labelling / active learning · systematic domain shift · "estimate GPU memory to train this" · **OCR / document AI as a task family** (*the most commercially common applied-vision job in 2026*).

### Actions taken

1. **README recalibrated** — the "critical concepts" table now separates *load-bearing for the subject* from *highest job ROI*, and carries an explicit, honest ROI section (§4 below).
2. **`Supplement-Interview-Readiness.md` added** — covering the LLM/GenAI question bank, ML system design **with answer sketches**, behavioural and project-narrative prep, noisy labels / semi-supervised / domain shift, OCR & document AI, evaluation engineering, and the coding-round reality.
3. **Nothing deleted.** The mentor recommended cutting Modules 1 and 5 by ~85–90%. That recommendation is *correct for the hiring goal alone* and *wrong for the stated goal of this program*, which is course mastery plus top-1% depth. The resolution is honest labelling, not deletion: the README now says plainly which sections are academic for a GenAI track, so the choice to study them is made with open eyes rather than by default.

---

## 3. Self-learner usability review — INCOMPLETE

The agent was cut off before producing findings. **Re-run this review before the web-artifact build**, since usability is exactly what the artifact must get right. The specific questions still outstanding:

- Where does a derivation skip a step an unsupervised learner couldn't fill in?
- Is any notation used before it is defined? (Candidate: $[\mathbf{t}]_\times$ in 5.5 — defined, but check the order.)
- Are the mastery checks answerable from the preceding section alone?
- Which build tasks are underspecified (no dataset named, no success criterion)?
- Which sections read smoothly but leave the reader unable to reproduce the reasoning (false-confidence zones)?
- Is a glossary / formula sheet / progress tracker missing?

---

## 4. What changed as a result

| # | Change | Driver |
|---|---|---|
| 1 | 6 arithmetic corrections | instructor verification |
| 2 | README: split "load-bearing" from "job ROI"; added an honest ROI section | mentor — biggest miscalibration |
| 3 | New `Supplement-Interview-Readiness.md` | mentor — missing content |
| 4 | Answer sketches + rubric for the three system-design drills | mentor — no feedback loop |
| 5 | Flagged the currency half-life of version-number content | mentor — six-month half-life |
