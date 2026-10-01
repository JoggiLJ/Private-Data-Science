"""
Plots benchmark accuracy for two models as a grouped bar chart.

Expects one folder per benchmark, named "<benchmark>_results", each
containing a results file per model named "<model>_..._results.json"
with an "Accuracy" field that looks like "67/102, 65.69%".
"""

import json
import os
import re
import matplotlib.pyplot as plt

results_dir = "."
models = ["qwen3:4b-instruct", "qwen3:14b"]
colors = ["#2a78d6", "#008300"]

num_words = []
num_tokens = []
num_entropy = []
score_list = []

# One results folder per benchmark, e.g. "MMLU_College_physics_results".
benchmark_folders = sorted(
    f for f in os.listdir(results_dir)
    if f.endswith("_results") and os.path.isdir(f)
)
benchmarks = [folder.replace("_results", "") for folder in benchmark_folders]


def read_accuracy(folder, model):
    """Return a model's accuracy (%) from its results file in `folder`."""
    filename = next(
        f for f in os.listdir(folder)
        if f.startswith(model + "_") and f.endswith("_results.json")
    )
    with open(os.path.join(folder, filename)) as infile:
        data = json.load(infile)
    match = re.search(r"([\d.]+)%", data["Accuracy"])
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        num_words.append(entry["Words"])
        num_tokens.append(entry["Tokens"])
        num_entropy.append(entry["Entropy"])
        score_list.append(1 if entry["Result"] == "Correct" else 0)
    return float(match.group(1)) if match else None


# accuracy_by_model["qwen3:14b"][i] is the accuracy for benchmarks[i], etc.
accuracy_by_model = {
    model: [read_accuracy(folder, model) for folder in benchmark_folders]
    for model in models
}

# --- Grouped bar chart: one x position per benchmark, one bar per model ---
bar_width = 0.8 / len(models)
x = range(len(benchmarks))

fig, ax = plt.subplots()
for i, model in enumerate(models):
    positions = [xi + i * bar_width for xi in x]
    ax.bar(positions, accuracy_by_model[model], width=bar_width,
           label=model, color=colors[i])

# Center the tick labels under each group of bars.
group_centers = [xi + bar_width * (len(models) - 1) / 2 for xi in x]
ax.set_xticks(group_centers)
ax.set_xticklabels(benchmarks, rotation=45, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.legend()
fig.tight_layout()
plt.show()

plt.scatter(num_words, score_list)
plt.grid()
plt.show()

plt.scatter(num_tokens, score_list)
plt.grid()
plt.show()

plt.scatter(num_entropy, score_list)
plt.grid()
plt.show()