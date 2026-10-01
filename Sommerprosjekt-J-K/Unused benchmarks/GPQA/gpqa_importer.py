import os
from datasets import load_dataset

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

sets = ["gpqa_diamond", "gpqa_experts", "gpqa_extended"]

for set_name in sets:
    ds = load_dataset("Idavidrein/gpqa", set_name)
    for split_name, split_ds in ds.items():
        out_path = os.path.join(OUTPUT_DIR, f"{set_name}_{split_name}.csv")
        split_ds.to_csv(out_path)
        print(f"Saved {out_path}")