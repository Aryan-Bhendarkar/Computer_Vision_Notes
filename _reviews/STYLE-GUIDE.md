# The Self-Study Teaching Standard

The notes were first written in a dense reference voice — correct, compressed, aimed at someone who already knows the field and wants the key points. That voice fails a self-learner, because it **states conclusions instead of teaching the reasoning that produces them**. This file is the standard every concept is being rewritten to.

## The core rule

> **A self-learner has no one to ask "but why?". So every claim must arrive with the reasoning that would let the reader have derived it — or with an explicit admission that we're taking it on trust and where to go for the proof.**

## The seven moves

**1. Open with the question, not the definition.**
Definitions answer a question the reader hasn't asked yet, so they don't stick. Pose the problem first; the definition then lands as the answer.

**2. Derive, don't present.**
A formula that appears fully formed is memorised, not understood. Show where it comes from — even three lines. If a derivation is genuinely out of scope, say so explicitly ("we won't derive this; the intuition is X, the proof is in Y") rather than letting the reader assume they missed something.

**3. Walk worked examples in sentences.**
`$r_1 = 3, j_1 = 1$; $r_2 = 4, j_2 = 2$` is a checkable answer, not an explanation. Say what you're doing and why at each step. The reader should be able to do the *next* example unaided.

**4. Name the confusion.**
The most valuable sentences in self-study material start "You might expect…" or "The thing that trips people up here is…". Anticipate the wrong model the reader is likely holding and correct it explicitly.

**5. Insert a pause.**
One short "stop and predict" question mid-section, before the answer is given. Retrieval mid-explanation is worth more than a check at the end, because a wrong prediction makes the correct one memorable.

**6. Build from what they already know.**
The reader knows deep learning. Anchor new ideas to that: "this is the same move as X, applied to Y." Reuse the curriculum's own through-lines.

**7. Close by asking for it in their own words.**
End the teaching part of each concept with a one-line prompt to restate it without notation. If they can't, they haven't got it.

## What not to change

- **Every technical claim, number, derivation, table and citation stays.** This is a rewrite of *voice*, not of content. Nothing is deleted.
- The 🎯 **Top-1% distinction**, ✅ **Mastery check** and 🔨 **Build + read** blocks stay as they are — they already work.
- Tables stay where they are genuinely reference material (comparisons, decision matrices, invariance audits). Convert a table to prose only when it was being used to *explain* something rather than to *compare* things.
- Keep the density. This is not about making it longer or gentler — it is about making the reasoning visible. A good rewrite often adds 20–40% to a section and removes the need to read it twice.

## Before / after

**Before** — states the recursion and labels an example:

> For a stack of layers with kernel sizes $k_l$ and strides $s_l$:
> $$j_l = j_{l-1}\cdot s_l \qquad r_l = r_{l-1} + (k_l - 1)\cdot j_{l-1}$$
> **Worked:** three $3\times3$ stride-1 convs. $r_1 = 1 + 2 = 3$; $r_2 = 3+2 = 5$; $r_3 = 5+2 = 7$.

**After** — derives it, then walks it:

> **Work out the two-layer case by hand first; the general rule falls out of it.**
>
> One $3\times3$ conv: an output unit is fed by a $3\times3$ patch. Receptive field 3.
>
> Now stack a second $3\times3$ on top. Its output looks at three neighbouring units of layer 1 — and *each of those* looks at 3 input pixels. But those three windows **overlap**: neighbouring layer-1 units are one pixel apart, so their windows are offset by one, not by three. Counting the union: the leftmost covers pixels 1–3, the middle 2–4, the rightmost 3–5. Five pixels, not nine.
>
> That overlap is the whole content of the formula. Each new layer extends the reach by $(k-1)$ — the amount the window sticks out past its centre — **not** by $k$. Hence $r_l = r_{l-1} + (k_l - 1)$.
>
> **Stride is the second ingredient.** If a layer downsamples by 2, then units in the *next* layer are 2 input pixels apart rather than 1, so each step of that next kernel reaches 2 pixels further. We track that multiplier as the **jump** $j_l = j_{l-1}\cdot s_l$ — how far apart, in input pixels, two adjacent units of layer $l$ are. Now the extension per layer is $(k_l - 1)$ *jumps*, not pixels:
> $$r_l = r_{l-1} + (k_l - 1)\cdot j_{l-1}$$
>
> **Pause:** before reading on — three $3\times3$ stride-1 convs. What's the receptive field, and does it match a single $7\times7$?
>
> Each layer adds $(3-1)\times1 = 2$, starting from 1: $1 \to 3 \to 5 \to 7$. **Seven** — exactly one $7\times7$ kernel's reach, with 45% of the parameters and two extra non-linearities. That is the entire argument behind VGG's all-$3\times3$ design (2.5), and you have now derived it rather than been told it.
>
> **In your own words:** why does each layer add $k-1$ rather than $k$?
