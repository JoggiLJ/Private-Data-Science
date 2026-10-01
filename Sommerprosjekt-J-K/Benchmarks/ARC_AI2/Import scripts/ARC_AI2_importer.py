import os

from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
HERE = os.path.dirname(os.path.abspath(__file__))

# Which split(s) to export. ARC-Challenge has "train", "validation" and "test".
SPLITS = ["train", "validation", "test"]

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=CACHE_DIR)

for split in SPLITS:
    out_path = os.path.join(HERE, f"ARC_AI2_ARC-Challenge_{split}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split], start=1):
            # ARC stores choices as {"text": [...], "label": [...]}.
            # Labels are usually A/B/C/D but a few items use 1/2/3/4, so we
            # normalise everything to letters and remap the answer key to match.
            orig_labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            label_map = {orig: LETTERS[j] for j, orig in enumerate(orig_labels)}

            f.write(f"Q{i}: {ex['question']}\n")
            for orig, choice in zip(orig_labels, texts):
                f.write(f"  {label_map[orig]}) {choice}\n")

            answer = label_map.get(ex["answerKey"], ex["answerKey"])
            f.write(f"Answer: {answer}\n\n")

    print(f"Wrote {len(ds[split])} questions -> {out_path}")
