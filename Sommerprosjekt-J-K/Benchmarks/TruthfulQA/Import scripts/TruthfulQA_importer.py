from datasets import load_dataset

# Load the official TruthfulQA multiple-choice dataset (namespaced repo id)
dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
mc_data = dataset["validation"]

print(f"Loaded {len(mc_data)} questions\n")

output_path = "truthfulqa_questions.txt"
lines = []

for i, ex in enumerate(mc_data):
    question = ex["question"]
    choices = ex["mc1_targets"]["choices"]
    labels = ex["mc1_targets"]["labels"]
    correct_idx = labels.index(1)

    lines.append(f"Q{i + 1}: {question}")
    for j, choice in enumerate(choices):
        marker = " <-- correct" if j == correct_idx else ""
        lines.append(f"    {chr(65 + j)}) {choice}{marker}")
    lines.append("")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved {len(mc_data)} questions to {output_path}")