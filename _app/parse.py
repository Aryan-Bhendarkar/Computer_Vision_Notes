#!/usr/bin/env python3
"""
Build step for the CV study tool.

Reads the module / drill / supplement markdown and emits content.json:
a flat list of concepts, each with its body and its retrieval prompts.

Usage:  python3 parse.py [--root DIR] [--out content.json]
The markdown is the source of truth. Re-run this after editing notes.
"""
import re, json, argparse, pathlib, sys, hashlib

MODULES = [
    ("Module-01-Classical-CV/Module-01-Notes.md",              1, "Classical CV Foundations"),
    ("Module-02-DL-for-Vision/Module-02-Notes.md",             2, "Deep Learning for Vision"),
    ("Module-03-Visual-Representations/Module-03-Notes.md",    3, "Visual Representations"),
    ("Module-04-Detection-Segmentation/Module-04-Notes.md",    4, "Detection & Segmentation"),
    ("Module-05-Geometry-3D/Module-05-Notes.md",               5, "Geometry & 3D Vision"),
    ("Module-06-Modern-Foundation-Models/Module-06-Notes.md",  6, "Modern & Foundation Models"),
]

# "## 2.4 Batch/Layer Normalisation 🔴"  ->  id, title, priority
H2 = re.compile(r"^##\s+(?P<id>\d+\.\d+[a-z]?)\s+(?P<title>.+?)\s*$")
PRIORITY = {"🔴": "critical", "🟡": "important", "🟢": "enrichment"}

PAUSE   = re.compile(r"^\*\*Pause:?\*\*\s*(?P<q>.*)$", re.I)
IYOW    = re.compile(r"^\*\*In your own words:?\*\*\s*(?P<q>.*)$", re.I)
MASTERY = re.compile(r"^###\s*✅\s*Mastery check", re.I)
TOP1    = re.compile(r"^###\s*🎯", re.I)
BUILD   = re.compile(r"^###\s*🔨", re.I)
DETAILS_OPEN  = re.compile(r"<details>\s*(<summary>.*?</summary>)?", re.I)
DETAILS_CLOSE = re.compile(r"</details>", re.I)

def strip_tags(s):
    s = re.sub(r"</?(b|i|em|strong|code|br)\s*/?>", "", s, flags=re.I)
    return s.strip()

def cid(*parts):
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:10]

def parse_module(path, num, name):
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # locate concept boundaries
    marks = []
    for i, ln in enumerate(lines):
        m = H2.match(ln)
        if m and m.group("id").startswith(f"{num}."):
            marks.append((i, m.group("id"), m.group("title")))
    marks.append((len(lines), None, None))

    concepts = []
    for k in range(len(marks) - 1):
        start, ident, raw_title = marks[k]
        end = marks[k + 1][0]
        body = lines[start + 1:end]

        prio = "standard"
        for glyph, label in PRIORITY.items():
            if glyph in raw_title:
                prio = label
        title = raw_title
        for glyph in PRIORITY:
            title = title.replace(glyph, "")
        title = title.strip()

        cards = []

        # --- Pause cards: prompt line, answer = following block until IYOW/### /--- ---
        j = 0
        while j < len(body):
            m = PAUSE.match(body[j].strip())
            if m:
                q = [m.group("q")] if m.group("q") else []
                j += 1
                # a Pause question may wrap onto continuation lines before a blank line
                while j < len(body) and body[j].strip() and not body[j].startswith("#"):
                    q.append(body[j]); j += 1
                ans = []
                while j < len(body):
                    s = body[j].strip()
                    if s.startswith("###") or s.startswith("## ") or s == "---" \
                       or IYOW.match(s) or PAUSE.match(s):
                        break
                    ans.append(body[j]); j += 1
                qt = strip_tags(" ".join(x.strip() for x in q))
                at = "\n".join(ans).strip()
                if qt and at:
                    cards.append({"type": "recall", "source": "pause",
                                  "q": qt, "a": at})
                continue
            j += 1

        # --- Mastery check ---
        for j, ln in enumerate(body):
            if MASTERY.match(ln.strip()):
                q, a, in_details = [], [], False
                for ln2 in body[j + 1:]:
                    s = ln2.strip()
                    if not in_details and (s.startswith("### ") or s == "---" or s.startswith("## ")):
                        break
                    if DETAILS_OPEN.search(s) and not in_details:
                        in_details = True
                        tail = DETAILS_OPEN.sub("", s).strip()
                        if tail and not DETAILS_CLOSE.search(tail):
                            a.append(tail)
                        continue
                    if in_details and DETAILS_CLOSE.search(s):
                        in_details = False
                        break
                    (a if in_details else q).append(ln2)
                qt = "\n".join(q).strip()
                at = "\n".join(a).strip()
                if qt:
                    cards.append({"type": "recall", "source": "mastery",
                                  "q": qt, "a": at or "_(no answer key — grade yourself against the section.)_"})
                break

        # --- In your own words ---
        for ln in body:
            m = IYOW.match(ln.strip())
            if m and m.group("q"):
                cards.append({"type": "explain", "source": "iyow",
                              "q": strip_tags(m.group("q")), "a": ""})

        # --- build task ---
        build = ""
        for j, ln in enumerate(body):
            if BUILD.match(ln.strip()):
                blk = []
                for ln2 in body[j + 1:]:
                    s = ln2.strip()
                    if s.startswith("### ") or s.startswith("## ") or s == "---":
                        break
                    blk.append(ln2)
                build = "\n".join(blk).strip()
                break

        for c in cards:
            c["id"] = cid(ident, c["source"], c["q"][:60])
            c["concept"] = ident
            c["module"] = num
            c["priority"] = prio

        concepts.append({
            "id": ident, "module": num, "moduleName": name,
            "title": title, "priority": prio,
            "body": "\n".join(body).strip(),
            "build": build, "cards": cards,
        })
    return concepts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="content.json")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    all_concepts, missing = [], []
    for rel, num, name in MODULES:
        p = root / rel
        if not p.exists():
            missing.append(rel); continue
        cs = parse_module(p, num, name)
        all_concepts += cs
        ncards = sum(len(c["cards"]) for c in cs)
        nb = sum(1 for c in cs if c["build"])
        print(f"  M{num}: {len(cs):>3} concepts  {ncards:>3} cards  {nb:>3} build tasks", file=sys.stderr)

    if missing:
        print("MISSING: " + ", ".join(missing), file=sys.stderr)

    # extras: supplements and reference files, carried as reading material
    extras = []
    for rel in ["README.md", "NOTATION.md", "FORMULA-SHEET.md", "GLOSSARY.md",
                "LEARNING-SYSTEM-SPEC.md",
                "Module-07-Supplements/Supplement-Interview-Readiness.md",
                "Module-07-Supplements/Supplement-Beyond-The-Syllabus.md",
                "Module-07-Supplements/Supplement-Research-Frontiers.md",
                "Module-07-Supplements/Supplement-Capstone-Projects.md",
                "Module-07-Supplements/Supplement-Robustness-Scale-and-Governance.md"]:
        p = root / rel
        if p.exists():
            t = p.read_text(encoding="utf-8")
            first = next((l for l in t.split("\n") if l.startswith("# ")), rel)
            extras.append({"id": rel, "title": first.lstrip("# ").strip(), "body": t})

    cards = [c for con in all_concepts for c in con["cards"]]
    data = {
        "generated": True,
        "concepts": all_concepts,
        "extras": extras,
        "stats": {
            "concepts": len(all_concepts),
            "cards": len(cards),
            "recall": sum(1 for c in cards if c["type"] == "recall"),
            "explain": sum(1 for c in cards if c["type"] == "explain"),
            "builds": sum(1 for c in all_concepts if c["build"]),
            "critical": sum(1 for c in all_concepts if c["priority"] == "critical"),
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"\n  TOTAL: {data['stats']}", file=sys.stderr)
    print(f"  wrote {args.out} ({pathlib.Path(args.out).stat().st_size/1e6:.2f} MB)", file=sys.stderr)

if __name__ == "__main__":
    main()
