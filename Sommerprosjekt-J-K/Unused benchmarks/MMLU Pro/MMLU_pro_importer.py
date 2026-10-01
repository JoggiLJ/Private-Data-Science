import os

from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MMLU Pro")
ds.save_to_disk(output_dir)