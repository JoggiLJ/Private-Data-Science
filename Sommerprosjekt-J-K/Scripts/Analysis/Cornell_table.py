import json
import os

import matplotlib.pyplot as plt
import numpy as np

filenames = [
    "../Cornell_critical_results/qwen3:4b-instruct_cornell_reasoning_copy_results.json",
    "../Cornell_critical_results/qwen3:14b_cornell_reasoning_copy_results.json",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def model_name(file):
    return "qwen3:4b-instruct" if "qwen3:4b-instruct" in os.path.basename(file) else "qwen3:14b"


model_matrices = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)

    items = [item for key, item in data.items() if key != "Accuracy"]

    groups = sorted({item["Group"] for item in items})
    forms = sorted({item["Form"] for item in items})

    correct = {(g, f): 0 for g in groups for f in forms}
    total = {(g, f): 0 for g in groups for f in forms}

    for item in items:
        key = (item["Group"], item["Form"])
        total[key] += 1
        if item["Result"] == "Correct":
            correct[key] += 1

    # Column 0 is the group Total; forms occupy columns 1..len(forms).
    matrix = np.full((len(groups) + 1, len(forms) + 1), np.nan)
    counts = np.zeros((len(groups) + 1, len(forms) + 1), dtype=int)

    for i, g in enumerate(groups):
        for j, f in enumerate(forms):
            if total[(g, f)] > 0:
                matrix[i, j + 1] = correct[(g, f)] / total[(g, f)]
                counts[i, j + 1] = total[(g, f)]

    for i, g in enumerate(groups):
        group_total = sum(total[(g, f)] for f in forms)
        group_correct = sum(correct[(g, f)] for f in forms)
        matrix[i, 0] = group_correct / group_total
        counts[i, 0] = group_total

    for j, f in enumerate(forms):
        form_total = sum(total[(g, f)] for g in groups)
        form_correct = sum(correct[(g, f)] for g in groups)
        matrix[-1, j + 1] = form_correct / form_total
        counts[-1, j + 1] = form_total

    overall_total = sum(total.values())
    overall_correct = sum(correct.values())
    matrix[-1, 0] = overall_correct / overall_total
    counts[-1, 0] = overall_total

    model_matrices[model_name(file)] = {
        "groups": groups,
        "forms": forms,
        "matrix": matrix,
        "counts": counts,
    }

models = sorted(model_matrices.keys())

fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    info = model_matrices[model]
    matrix, groups, forms, counts = info["matrix"], info["groups"], info["forms"], info["counts"]
    n_rows, n_cols = matrix.shape

    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["Total"] + forms)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"Group {g}" for g in groups] + ["Total"])
    ax.set_title(model)

    ax.axhline(n_rows - 1.5, color="black", linewidth=1.2)
    ax.axvline(0.5, color="black", linewidth=1.2)

    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            n = counts[i, j]
            if np.isnan(value):
                continue
            label = f"{int(round(value))}" if n == 1 else f"{value:.0%}\n(n={n})"
            weight = "bold" if (i == n_rows - 1 or j == 0) else "normal"
            text_color = "white" if value > 0.6 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color, fontweight=weight)

    fig.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)

fig.suptitle("Cornell Critical Reasoning — Accuracy by Group and Form")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(SCRIPT_DIR, "cornell_group_form_table.png"), dpi=300)
plt.close(fig)
