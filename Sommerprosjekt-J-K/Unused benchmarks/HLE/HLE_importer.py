import os

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ds = load_dataset("cais/hle")
ds.save_to_disk(OUTPUT_DIR)

print(f"Saved dataset to {OUTPUT_DIR}")