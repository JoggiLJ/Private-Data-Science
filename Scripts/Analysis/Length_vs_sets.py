import json
import os

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.stats import linregress

filenames = [
    "../SAT_Math_results/qwen3:4b-instruct_sat-math-mc_copy_results.json",
    "../SAT_Math_results/qwen3:14b_sat-math-mc_copy_results.json",
    "../SAT_rw_results/qwen3:4b-instruct_sat-rw-mc_copy_results.json",
    "../SAT_rw_results/qwen3:14b_sat-rw-mc_copy_results.json",
    "../Cornell_critical_results/qwen3:4b-instruct_cornell_reasoning_copy_results.json",
    "../Cornell_critical_results/qwen3:14b_cornell_reasoning_copy_results.json",
    "../Statistics_inventory_results/qwen3:4b-instruct_Statistics_inventory_copy_results.json",
    "../Statistics_inventory_results/qwen3:14b_Statistics_inventory_copy_results.json",
    "../LSAT_results/qwen3:4b-instruct_lsat_copy_results.json",
    "../LSAT_results/qwen3:14b_lsat_copy_results.json",
    "../TruthfulQA_results/qwen3:4b-instruct_truthfulqa_questions_copy_results.json",
    "../TruthfulQA_results/qwen3:14b_truthfulqa_questions_copy_results.json",
    "../MMLU_Logical_fallacies_results/qwen3:4b-instruct_MMLU_logical_fallacies_test_copy_results.json",
    "../MMLU_Logical_fallacies_results/qwen3:14b_MMLU_logical_fallacies_test_copy_results.json",
    "../MMLU_Formal_logic_results/qwen3:4b-instruct_MMLU_formal_logic_test_copy_results.json",
    "../MMLU_Formal_logic_results/qwen3:14b_MMLU_formal_logic_test_copy_results.json",
    "../MMLU_Us_foreign_policy_results/qwen3:4b-instruct_MMLU_us_foreign_policy_test_copy_results.json",
    "../MMLU_Us_foreign_policy_results/qwen3:14b_MMLU_us_foreign_policy_test_copy_results.json",
    "../MMLU_High_school_physics_results/qwen3:4b-instruct_MMLU_HSphysics_test_copy_results.json",
    "../MMLU_High_school_physics_results/qwen3:14b_MMLU_HSphysics_test_copy_results.json",
    "../MMLU_Conceptual_physics_results/qwen3:4b-instruct_MMLU_conceptual_physics_test_copy_results.json",
    "../MMLU_Conceptual_physics_results/qwen3:14b_MMLU_conceptual_physics_test_copy_results.json",
    "../MMLU_College_physics_results/qwen3:4b-instruct_MMLU_college_physics_test_copy_results.json",
    "../MMLU_College_physics_results/qwen3:14b_MMLU_college_physics_test_copy_results.json",
    "../GMAT_results/qwen3:4b-instruct_gmat_items_results.json",
    "../GMAT_results/qwen3:14b_gmat_items_results.json",
    "../ARC_AI2_results/qwen3:4b-instruct_ARC_AI2_ARC-Challenge_test_results.json",
    "../ARC_AI2_results/qwen3:14b_ARC_AI2_ARC-Challenge_test_results.json",
]

benchmark_data = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)
    accuracy = float(data["Accuracy"].split(",")[-1].strip().rstrip("%"))

    items = [item for key, item in data.items() if key != "Accuracy"]
    words = [float(item["Words"]) for item in items]
    tokens = [float(item["Tokens"]) for item in items]
    entropies = [float(item["Entropy"].split()[0]) for item in items]
    avg_words = sum(words) / len(words)
    avg_tokens = sum(tokens) / len(tokens)
    avg_entropy = sum(entropies) / len(entropies)

    dataset = os.path.basename(os.path.dirname(file)).removesuffix("_results")
    model = "qwen3:4b-instruct" if "qwen3:4b-instruct" in os.path.basename(file) else "qwen3:14b"

    benchmark_data[file] = {
        "dataset": dataset,
        "model": model,
        "avg_words": avg_words,
        "avg_tokens": avg_tokens,
        "avg_entropy": avg_entropy,
        "accuracy": accuracy,
    }

models = sorted({entry["model"] for entry in benchmark_data.values()})
MODEL_COLORS = {"qwen3:14b": "#1F77B4", "qwen3:4b-instruct": "#D62728"}
model_to_color = {name: MODEL_COLORS[name] for name in models}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDED_DATASETS = {"SAT_rw", "LSAT"}


def make_figure(entries, output_name):
    accuracies = [entry["accuracy"] for entry in entries]
    point_colors = [model_to_color[entry["model"]] for entry in entries]
    labels = [entry["dataset"] for entry in entries]

    plot_specs = [
        ([entry["avg_words"] for entry in entries], "Average number of words", "Average Question Length vs. Accuracy"),
        ([entry["avg_tokens"] for entry in entries], "Average number of tokens", "Average Token Count vs. Accuracy"),
        ([entry["avg_entropy"] for entry in entries], "Average entropy (bits/token)", "Average Entropy vs. Accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for ax, (x_values, x_label, title) in zip(axes, plot_specs):
        regression = linregress(x_values, accuracies)
        fit_x = np.array([min(x_values), max(x_values)])
        fit_y = regression.slope * fit_x + regression.intercept

        ax.scatter(x_values, accuracies, c=point_colors, s=25)
        ax.plot(fit_x, fit_y, color="black", linestyle="--", linewidth=1)

        texts = [ax.text(x, y, label, fontsize=7) for x, y, label in zip(x_values, accuracies, labels)]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{title}\n(slope={regression.slope:.3f}, p={regression.pvalue:.3f})", fontsize=10)
        ax.grid()

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=name)
        for name, color in model_to_color.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, 0), fontsize=8)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(SCRIPT_DIR, output_name), dpi=300)
    plt.close(fig)


make_figure(list(benchmark_data.values()), "length_vs_sets.png")
make_figure(
    [entry for entry in benchmark_data.values() if entry["dataset"] not in EXCLUDED_DATASETS],
    "length_vs_sets_excl_outliers.png",
)