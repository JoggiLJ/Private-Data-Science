import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter


def extract_letter(answer):
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in "ABCDE":
        return answer
    matches = re.findall(r"\b([A-E])\b", answer)
    return matches[-1] if matches else "Unparsed"

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
    "../IQ-test_results/qwen3:4b-instruct_iq_results.json",
    "../IQ-test_results/qwen3:14b_iq_results.json",
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

SERIES_COLORS = {"Correct options": "#BDBDBD", "qwen3:14b": "#1F77B4", "qwen3:4b-instruct": "#D62728"}

cor_options = {}
model_choices = {}

def model_benchmark_extract(filename):
    folder, basename = filename.split("/")[-2:]
    model = basename.split("_")[0]
    benchmark = folder.removesuffix("_results").replace("_", " ").replace("-", " ")
    return model, benchmark

benchmarks = []
models_by_benchmark = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)
    counts = Counter()
    m_counts = Counter()
    for item in data:
        if item in ("Accuracy", "Estimated IQ"):
            continue
        else:
            counts[data[item]["Correct label"].upper()] += 1
            m_counts[extract_letter(data[item]["Model answer"])] += 1

    model, benchmark = model_benchmark_extract(file)
    cor_options[(model, benchmark)] = counts
    model_choices[(model, benchmark)] = m_counts

    if benchmark not in models_by_benchmark:
        benchmarks.append(benchmark)
        models_by_benchmark[benchmark] = []
    models_by_benchmark[benchmark].append(model)

# IQ items are answered via image/pattern choice, not lettered options, so the
# "IQ test" panel is mostly "Unparsed" and isn't comparable to the MC benchmarks.
benchmarks = [b for b in benchmarks if b != "IQ test"]

ncols = 3
nrows = -(-len(benchmarks) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, benchmark in zip(axes, benchmarks):
    models = models_by_benchmark[benchmark]
    labels = sorted(cor_options[(models[0], benchmark)].keys())

    bars = [("Correct options", [cor_options[(models[0], benchmark)][label] for label in labels])]
    for model in models:
        key = (model, benchmark)
        bars.append((model, [model_choices[key][label] for label in labels]))

    x = np.arange(len(labels))
    width = 0.8 / len(bars)

    for i, (name, values) in enumerate(bars):
        offset = (i - (len(bars) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name, color=SERIES_COLORS.get(name))

    ax.set_xticks(x, labels)
    ax.set_title(benchmark)
    ax.grid(axis="y")
    ax.legend()

for ax in axes[len(benchmarks):]:
    ax.set_visible(False)

fig.tight_layout()

plt.savefig("answer_letter_distribution.png", dpi=150)