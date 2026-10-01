import os

from datasets import load_dataset

ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LogiQA")
ds.save_to_disk(output_dir)