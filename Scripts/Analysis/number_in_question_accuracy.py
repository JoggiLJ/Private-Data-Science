import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

MODEL_ORDER = ["qwen3:4b-instruct", "qwen3:14b"]

CONDITION_COLORS = {
    "With #": "#D62728",
    "Without #": "#1F77B4",
}

benchmarks = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)

    total_with_numbers = 0
    total_without_numbers = 0
    cor_with_numbers = 0
    cor_without_numbers = 0

    for item in data:
        if item in ("Accuracy", "Estimated IQ"):
            continue
        prompt = data[item]["Prompt"]
        has_numbers = any(char.isdigit() for char in prompt)
        is_correct = data[item]["Result"] == "Correct"

        if has_numbers:
            total_with_numbers += 1
            if is_correct:
                cor_with_numbers += 1
        else:
            total_without_numbers += 1
            if is_correct:
                cor_without_numbers += 1

    bench = benchmark_name(file)
    benchmarks.setdefault(bench, {})[model_name(file)] = {
        "with_acc": cor_with_numbers / total_with_numbers if total_with_numbers else 0,
        "without_acc": cor_without_numbers / total_without_numbers if total_without_numbers else 0,
        "with_n": total_with_numbers,
        "without_n": total_without_numbers,
    }

# One bar chart per benchmark, with both models grouped side by side; color marks
# whether the question had numbers, so the with/without effect is easy to spot
# while models sit right next to each other for comparison.
ncols = 3
nrows = -(-len(benchmarks) // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows))
axes = axes.flatten()

x = range(len(MODEL_ORDER))
width = 0.35
conditions = [("With #", "with"), ("Without #", "without")]

for ax, (bench, models) in zip(axes, benchmarks.items()):
    for i, (label, key) in enumerate(conditions):
        values = [models[model][f"{key}_acc"] for model in MODEL_ORDER]
        ns = [models[model][f"{key}_n"] for model in MODEL_ORDER]
        offset = (i - 0.5) * width
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, values, width=width, color=CONDITION_COLORS[label], label=label)
        for bar, val, n in zip(bars, values, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.0%}\n(n={n})",
                     ha="center", va="bottom", fontsize=6.5)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(bench, fontsize=10)

for ax in axes[len(benchmarks):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2)
fig.suptitle("Accuracy by Presence of Numbers in Question, per Benchmark", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("number_in_question_accuracy_per_benchmark_model.png", dpi=150)
