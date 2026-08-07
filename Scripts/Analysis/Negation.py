import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

negation_list = [
    "not",
    "no",
    "never",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
    "except",
    "excluding",
    "unless",
    "least",
    "false",
    "incorrect",
    "untrue",
    "lack",
    "lacking",
    "fail",
    "fails",
    "failed",
]

def detect_negation(text):
    words = re.findall(r"[a-z']+", text.lower())
    return sum(1 for word in words if word in negation_list)

negation_correct = 0
negation_total = 0
no_negation_correct = 0
no_negation_total = 0

benchmark_stats = {}
negation_scores = []
correct_flags = []

for file in filenames:
    benchmark = file.split("/")[1].removesuffix("_results")

    with open(file, "r") as infile:
        data = json.load(infile)

    bench_stats = benchmark_stats.setdefault(
        benchmark,
        {"negation_correct": 0, "negation_total": 0, "no_negation_correct": 0, "no_negation_total": 0},
    )

    for item, item_data in data.items():
        if not isinstance(item_data, dict):
            continue
        prompt = item_data["Prompt"]
        question_text = prompt.split("\n\n")[0].strip()
        negation_score = detect_negation(question_text)
        has_negation = negation_score > 0
        is_correct = item_data["Result"] == "Correct"

        negation_scores.append(negation_score)
        correct_flags.append(int(is_correct))

        if has_negation:
            negation_total += 1
            negation_correct += is_correct
            bench_stats["negation_total"] += 1
            bench_stats["negation_correct"] += is_correct
        else:
            no_negation_total += 1
            no_negation_correct += is_correct
            bench_stats["no_negation_total"] += 1
            bench_stats["no_negation_correct"] += is_correct

accuracies = [
    no_negation_correct / no_negation_total,
    negation_correct / negation_total,
]
labels = [
    f"No negation (n={no_negation_total})",
    f"Negation (n={negation_total})",
]

plt.bar(labels, accuracies, color=["#1F77B4", "#D62728"])
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy by Presence of Negation in Question")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.1%}", ha="center")
plt.tight_layout()
plt.savefig("negation_accuracy.png")

# Per-benchmark breakdown
benchmarks = list(benchmark_stats.keys())
no_negation_acc = [
    benchmark_stats[b]["no_negation_correct"] / benchmark_stats[b]["no_negation_total"]
    if benchmark_stats[b]["no_negation_total"] else 0
    for b in benchmarks
]
negation_acc = [
    benchmark_stats[b]["negation_correct"] / benchmark_stats[b]["negation_total"]
    if benchmark_stats[b]["negation_total"] else 0
    for b in benchmarks
]

no_negation_n = [benchmark_stats[b]["no_negation_total"] for b in benchmarks]
negation_n = [benchmark_stats[b]["negation_total"] for b in benchmarks]

x = range(len(benchmarks))
width = 0.4

plt.figure(figsize=(12, 6))
no_negation_bars = plt.bar([i - width / 2 for i in x], no_negation_acc, width=width, label="No negation", color="#1F77B4")
negation_bars = plt.bar([i + width / 2 for i in x], negation_acc, width=width, label="Negation", color="#D62728")
plt.ylabel("Accuracy")
plt.ylim(0, 1.08)
plt.title("Accuracy by Presence of Negation, per Benchmark")
plt.xticks(list(x), benchmarks, rotation=45, ha="right")
plt.legend()
for bar, n in zip(no_negation_bars, no_negation_n):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}",
              ha="center", va="bottom", fontsize=7, rotation=90)
for bar, n in zip(negation_bars, negation_n):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}",
              ha="center", va="bottom", fontsize=7, rotation=90)
plt.tight_layout()
plt.savefig("negation_accuracy_per_benchmark.png")

# Scatterplot of negation-score vs correctness with linear regression
negation_scores = np.array(negation_scores)
correct_flags = np.array(correct_flags)

slope, intercept, r_value, p_value, std_err = stats.linregress(negation_scores, correct_flags)

plt.figure(figsize=(8, 6))
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.05, 0.05, size=correct_flags.shape)
plt.scatter(negation_scores, correct_flags + jitter, alpha=0.15, s=15, color="#1F77B4")

x_line = np.linspace(negation_scores.min(), negation_scores.max(), 100)
plt.plot(x_line, intercept + slope * x_line, color="#D62728", linewidth=2,
          label=f"slope={slope:.4f}, p={p_value:.3g}")

plt.xlabel("Negation score (number of negation words in question)")
plt.ylabel("Correct (1) / Incorrect (0)")
plt.title("Correctness vs. Negation Score")
plt.legend()
plt.tight_layout()
plt.savefig("negation_score_scatter.png")