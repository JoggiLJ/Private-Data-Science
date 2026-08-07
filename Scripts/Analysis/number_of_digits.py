import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def benchmark_name(filename):
    folder = filename.split("/")[-2]
    return folder.removesuffix("_results").replace("_", " ").replace("-", " ")


def model_name(filename):
    basename = filename.split("/")[-1]
    return basename.split("_")[0]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# model -> benchmark -> {"digit_counts": [...], "scores": [...]}
model_data = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)

    model = model_name(file)
    bench = benchmark_name(file)

    digit_counts = []
    scores = []
    for key, item in data.items():
        if key == "Accuracy":
            continue
        prompt = item["Prompt"]
        digit_counts.append(sum(char.isdigit() for char in prompt))
        scores.append(1 if item["Result"] == "Correct" else 0)

    model_data.setdefault(model, {})[bench] = {
        "digit_counts": digit_counts,
        "scores": scores,
    }

models = sorted(model_data.keys())

for model in models:
    benchmarks = model_data[model]
    bench_names = sorted(benchmarks.keys())

    ncols = 5
    nrows = -(-len(bench_names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    for ax, bench in zip(axes, bench_names):
        entry = benchmarks[bench]
        x = np.array(entry["digit_counts"], dtype=float)
        y = np.array(entry["scores"], dtype=float)

        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.05, 0.05, size=len(y))

        ax.scatter(x, y + jitter, s=15, alpha=0.5, color="#4C72B0")

        if len(set(x)) > 1:
            regression = linregress(x, y)
            fit_x = np.array([x.min(), x.max()])
            fit_y = regression.slope * fit_x + regression.intercept
            ax.plot(fit_x, fit_y, color="black", linestyle="--", linewidth=1)
            subtitle = f"slope={regression.slope:.4f}, p={regression.pvalue:.3f}"
        else:
            subtitle = "insufficient variation"

        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_xlabel("Numbers in prompt")
        ax.set_ylabel("Score")
        ax.set_title(f"{bench}\n{subtitle}", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[len(bench_names):]:
        ax.axis("off")

    fig.suptitle(f"{model}: Score vs. Numbers in Prompt, per Benchmark", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe_model = model.replace(":", "-")
    fig.savefig(os.path.join(SCRIPT_DIR, f"number_of_digits_{safe_model}.png"), dpi=150)
    plt.close(fig)
