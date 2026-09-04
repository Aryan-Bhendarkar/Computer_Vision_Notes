# The Learning System — What We Should Actually Build

*Design note. Read this before we write any code for the web artefact.*

---

## 1. The problem the system has to solve

We now have ~190,000 words across six module files, six drill files, three appendices, and five supplements. That is a genuinely good body of material — and it is also **too large to learn by reading.**

This is the failure mode to design against, and it is specific. A learner opens Module 4, reads 25,000 words over three days, feels the click of understanding at every section, closes the file, and six weeks later cannot derive focal loss. Nothing was wrong with the reading. The problem is that **reading produces recognition, and interviews require retrieval** — and recognition feels exactly like retrieval from the inside, right up until someone asks you a question.

So the system's job is not to present the notes nicely. Obsidian already presents markdown nicely, for free. The system's job is to **force the conversion from recognition to retrieval, and to make the gap visible when it hasn't happened.**

That reframing decides every design question below.

## 2. The key structural asset

The notes are not prose. They are already a **structured dataset**, because we wrote them to a fixed six-beat schema:

```
## <id> <title> <priority-tag>
  intuition
  the math
  Pause:            ← inline retrieval prompt, answer follows
  connections
  🎯 Top-1% distinction
  ✅ Mastery check   ← question + collapsed <details> answer
  🔨 Build + read    ← task + resources
  In your own words: ← restatement prompt
```

Plus six drill files with Part A / Part B / Part C problems, a formula sheet, a glossary and a notation table.

That means a build step can parse every file into a graph of ~70 concepts, each carrying: a priority tag, a body, **1–3 retrieval prompts with answers**, a build task, and links to prerequisites. **We do not need to author flashcards — we already have ~200 of them, embedded, with answers.** That is the whole reason a custom system beats Obsidian here, and it is worth being explicit about, because it is the only real justification for building anything.

## 3. Recommendation: build the study tool, not the reader

I said earlier this was a choice between a **reader** (navigation, search, KaTeX, progress bar) and a **study tool** (spaced repetition, active recall, persistent state). It isn't really a choice.

A reader adds close to nothing over opening the markdown in VS Code or Obsidian — both of which already do search, KaTeX and outline navigation, and neither of which you have to maintain. If we ship a reader, the honest description of what we built is "a worse Obsidian with our content baked in."

**Build the study tool.** The reading surface comes along for free (it's the same content, rendered), but the *reason to open it* is the practice loop.

## 4. Architecture

```
  six module .md ─┐
  six drill .md   ├─→  parse.py  ─→  content.json  ─→  index.html (single file)
  supplements     ┘     (build)        (~2 MB)          KaTeX + app inline
                                                              │
                                                        state persistence
                                                    (review history, progress)
```

**One build script, one output file.** The markdown stays the source of truth — you keep editing notes in your editor, re-run the build, and the app updates. No CMS, no database of content, no drift between two copies. If the app dies, the notes are untouched.

**Why a single self-contained HTML file:** it works offline, it opens on your phone, it survives being emailed, and it can be published as an Artifact with a URL you can return to. State persistence is the one thing a static file can't do alone — for a durable, cross-device review history we should use an Artifact runtime capability rather than `localStorage`, which is per-browser and can vanish.

## 5. The four surfaces

### 5.1 Read
Rendered notes, KaTeX math, module/concept sidebar, full-text search, and — the one thing a plain reader doesn't do — **`Pause:` blocks and mastery-check answers are collapsed by default and must be clicked to reveal.** The material was written to make you predict before you're told; the interface should enforce that rather than undo it by rendering everything at once.

### 5.2 Practice — the core
A spaced-repetition queue built from the extracted prompts. Use **FSRS** rather than SM-2; it's better calibrated and the reference implementation is small enough to inline.

Three card types, because they test different things:

| Type | Source | What it tests |
|---|---|---|
| **Recall** | `Pause:` and ✅ mastery checks | Can you produce the answer? |
| **Derive** | Formula sheet entries | Can you get from premises to the formula on paper? Self-graded, with the derivation revealed after. |
| **Explain** | `In your own words:` prompts | Can you say it without notation? Free-text; graded by you, or optionally by Claude via an artifact capability. |

Grade honestly on the FSRS 1–4 scale. The tool is worthless if you grade generously, and this should be said in the UI.

### 5.3 Track
Not a progress bar — a **readiness map**. A 70-cell grid, one per concept, coloured by *retrieval strength* (from the FSRS state) rather than by "have I read this". The distinction is the entire point: a concept you read three weeks ago and never recalled shows red, not green. The 🔴 critical concepts get their own row so weakness there is impossible to miss.

Plus: build-task completion checkboxes, and a "cold spots" list — the five concepts most overdue, ranked by priority tag × overdue-ness.

### 5.4 Interview mode
Timed rapid-fire drawing from the module-end summaries. 45 seconds per question, no reveal until you commit an answer. This is the surface that will feel worst and be worth the most.

## 6. What to leave out

- **User accounts, multi-user, sharing.** One user. Anything else is procrastination dressed as engineering.
- **A rich editor.** Notes are edited in your editor. The app is read-and-practise only.
- **Analytics dashboards beyond the readiness map.** Metrics you don't act on are decoration.
- **Anything requiring a server.** The moment this needs a backend it stops being something you'll still be using in March.

## 7. Build order

| Phase | Ships | Effort |
|---|---|---|
| 1 | `parse.py` → `content.json`; verify all ~70 concepts and ~200 prompts extract cleanly | ~1 day |
| 2 | Read surface: render, navigate, search, collapsed reveals | ~1 day |
| 3 | Practice surface: FSRS queue, three card types, persistent state | ~2 days |
| 4 | Track surface: readiness map, cold spots | ~half day |
| 5 | Interview mode | ~half day |

Phase 1 is the risky one and everything depends on it — if the parse is unreliable, the whole thing is. Do it first and verify by count before writing a line of UI.

## 8. The honest caveat

There is a real chance the best version of this is: keep the notes in Obsidian, and put the ~200 prompts into Anki, which already implements FSRS properly, syncs to your phone, and is maintained by other people.

The case for building ours anyway is that Anki separates the card from its context — you get the question with no path back to the derivation that produced it — and that the build-task tracking and the readiness map have no Anki equivalent. That case is real but not overwhelming. **If, three weeks after we ship this, you aren't opening it daily, export to Anki and don't feel bad about it.** The notes are the asset; the app is a convenience.
