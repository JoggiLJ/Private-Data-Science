#!/usr/bin/env python3
"""Convert MMLU-Redux .arrow files into the standard MMLU JSON format.

For each subject under MMLU Redux/MMLU Redux original set/<subject>/test/data-00000-of-00001.arrow,
writes MMLU Redux/JSON_set/<Subject_Title>/test/MMLU_Redux_<subject>_test.json,
matching the schema used by MMLU/JSON_set/*/*/MMLU_*.json exactly:

{
  "variant_id": "mmlu_redux_<subject>_test",
  "description": "<Subject> multiple-choice questions from the MMLU-Redux benchmark (test split).",
  "source": "MMLU-Redux (Gema et al., 2024) — <subject> subject, test split. Dataset: edinburgh-dawg/mmlu-redux-2.0 (Hugging Face).",
  "items": [
    {"item_id": 1, "text": "...", "options": [{"label": "A", "text": "..."}, ...],
     "correct_label": "B", "split": "test"}
  ]
}

correct_label is derived from the dataset's original 'answer' index (0-3 -> A-D),
the same field standard MMLU JSON files use, so every item always has exactly one
valid correct_label and the schema stays byte-for-byte compatible with the
standard MMLU JSON files.
"""
import glob
import json
import os

import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "MMLU Redux original set")
OUT_ROOT = os.path.join(ROOT, "JSON_set")

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def read_arrow(path):
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    return table.to_pylist()


def build_json(subject, rows):
    items = []
    for i, row in enumerate(rows, start=1):
        choices = row["choices"]
        options = [{"label": LABELS[j], "text": c} for j, c in enumerate(choices)]
        correct_label = LABELS[row["answer"]]
        items.append(
            {
                "item_id": i,
                "text": row["question"],
                "options": options,
                "correct_label": correct_label,
                "split": "test",
            }
        )

    subject_display = subject.replace("_", " ")
    doc = {
        "variant_id": f"mmlu_redux_{subject}_test",
        "description": f"{subject_display.capitalize()} multiple-choice questions from the MMLU-Redux benchmark (test split).",
        "source": (
            f"MMLU-Redux (Gema et al., 2024) — {subject} subject, test split. "
            "Dataset: edinburgh-dawg/mmlu-redux-2.0 (Hugging Face)."
        ),
        "items": items,
    }
    return doc


def main():
    arrow_files = sorted(
        glob.glob(os.path.join(DATA_ROOT, "*", "test", "data-00000-of-00001.arrow"))
    )
    print(f"Found {len(arrow_files)} arrow files")

    written = []
    for path in arrow_files:
        subject = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows = read_arrow(path)
        doc = build_json(subject, rows)

        subject_title = subject.capitalize()
        out_dir = os.path.join(OUT_ROOT, subject_title, "test")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"MMLU_Redux_{subject}_test.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        written.append((subject, len(rows), out_path))
        print(f"{subject}: {len(rows)} items -> {out_path}")

    print(f"\nDone. Wrote {len(written)} JSON files.")
    return written


if __name__ == "__main__":
    main()
