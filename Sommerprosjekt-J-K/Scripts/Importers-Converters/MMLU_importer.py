from datasets import load_dataset

ds = load_dataset("cais/mmlu", "conceptual_physics")

labels = ["A", "B", "C", "D"]

with open("MMLU_conceptual_physics.txt", "w") as f:
    for i, ex in enumerate(ds["test"], start=1):
        f.write(f"Q{i}: {ex['question']}\n")
        for label, choice in zip(labels, ex["choices"]):
            f.write(f"  {label}) {choice}\n")
        f.write(f"Answer: {labels[ex['answer']]}\n\n")
