# Retrieval Lab — build pipeline

The markdown notes are the source of truth. This turns them into the study tool.

```
cd "A:\AI-ML Track\Computer Vision\_app"
python parse.py --root .. --out content.json
python build.py
```

`parse.py` walks the six module files and extracts every concept plus its
retrieval prompts (`Pause:`, `✅ Mastery check` with its `<details>` answer,
`In your own words:`) and its `🔨 Build + read` task. `build.py` inlines the
resulting `content.json` into `template.html` and writes `index.html`.

Current extraction: **70 concepts · 189 cards · 44 build tasks · 7 critical**.

After a rebuild, ask Claude to republish `index.html` to the same artifact URL
so the hosted copy and your review history stay together. `Retrieval-Lab.html`
in this folder is a standalone offline copy — it works by double-clicking, but
its progress lives only in that browser.

## Editing the notes

Keep the six-beat schema intact and the parser keeps working:

- `## <n>.<m> Title` with an optional 🔴 / 🟡 / 🟢 tag — starts a concept
- `**Pause:**` followed by the answer paragraphs — becomes a recall card
- `### ✅ Mastery check` with `<details>…</details>` — becomes a recall card
- `**In your own words:**` — becomes a self-graded explain card
- `### 🔨 Build + read` — becomes a tracked build task

`parse.py` prints per-module counts. If a count drops after an edit, a heading
or marker was changed — that is the check to watch.

## Known gap

26 concepts have no `🔨 Build + read` block: 2.11, 3.8, 4.9, 4.10, 4.13, 5.1–5.4,
5.6, 5.7, 5.9–5.11, 6.1, 6.3, 6.6–6.9, 6.11–6.13, 6.15, 6.15b, 6.16. Some are
practicals or syntheses that carry their own deliverable, but several (5.1–5.4,
5.6, 5.7, 6.1, 6.3, 6.12, 6.13) genuinely should have one.
