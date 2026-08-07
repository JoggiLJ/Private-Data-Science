#!/usr/bin/env python3
"""Convert one-line, tab-separated TruthfulQA questions into MMLU-style JSON.

Input line shape:
  Q1: <question>\tA) <opt> <-- correct\tB) <opt>\tC) <opt> ...
The correct option is flagged with a trailing " <-- correct".
"""
import json
import re
import sys

Q_RE = re.compile(r"^Q\d+:\s*(.*)$")
OPT_RE = re.compile(r"^([A-Z])\)\s*(.*)$")
CORRECT_MARK = "<-- correct"


def parse(path):
    items = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split("\t")
            qm = Q_RE.match(parts[0].strip())
            if not qm:
                sys.stderr.write(f"WARN line {lineno}: no Q# prefix, skipped\n")
                continue
            text = qm.group(1).strip()
            options = []
            correct_label = None
            for seg in parts[1:]:
                seg = seg.strip()
                if not seg:
                    continue
                is_correct = CORRECT_MARK in seg
                seg = seg.replace(CORRECT_MARK, "").strip()
                om = OPT_RE.match(seg)
                if not om:
                    sys.stderr.write(f"WARN line {lineno}: bad option '{seg[:40]}'\n")
                    continue
                label, otext = om.group(1), om.group(2).strip()
                options.append({"label": label, "text": otext})
                if is_correct:
                    correct_label = label
            items.append({
                "item_id": len(items) + 1,
                "text": text,
                "options": options,
                "correct_label": correct_label,
                "split": "validation",
            })
    return items


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    items = parse(inp)

    # Validation
    problems = 0
    for it in items:
        if not it["options"]:
            sys.stderr.write(f"item {it['item_id']}: no options\n"); problems += 1
        if it["correct_label"] is None:
            sys.stderr.write(f"item {it['item_id']}: no correct label\n"); problems += 1
        labels = {o["label"] for o in it["options"]}
        if it["correct_label"] and it["correct_label"] not in labels:
            sys.stderr.write(f"item {it['item_id']}: correct label not in options\n"); problems += 1

    doc = {
        "version": "truthfulqa-mc-v1",
        "source": "TruthfulQA (multiple-choice), validation split | dataset truthfulqa/truthful_qa on Hugging Face | exported 2026-07-01.",
        "items": items,
    }
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"Parsed {len(items)} questions; {problems} validation problems.")


if __name__ == "__main__":
    main()
