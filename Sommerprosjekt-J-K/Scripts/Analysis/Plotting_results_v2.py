import json
import os
import random
import re
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

OPTION_PATTERN = re.compile(r"(?im)^\s*[A-Za-z]\)")

def question_chance(prompt):
    """Chance of a correct guess on this question: 1 / (number of lettered options)."""
    n_options = len(OPTION_PATTERN.findall(prompt))
    return 1 / n_options if n_options else None

filenames = [
    "../MMLU_High_school_physics_results/qwen3:14b_MMLU_HSphysics_test_copy_results.json",
    "../MMLU_College_physics_results/qwen3:14b_MMLU_college_physics_test_copy_results.json",
    "../MMLU_Conceptual_physics_results/qwen3:14b_MMLU_conceptual_physics_test_copy_results.json",
    "../MMLU_Formal_logic_results/qwen3:14b_MMLU_formal_logic_test_copy_results.json",
    "../MMLU_Logical_fallacies_results/qwen3:14b_MMLU_logical_fallacies_test_copy_results.json",
    "../MMLU_Us_foreign_policy_results/qwen3:14b_MMLU_us_foreign_policy_test_copy_results.json",
    "../Statistics_inventory_results/qwen3:14b_Statistics_inventory_copy_results.json",
    "../IQ-test_results/qwen3:14b_iq_results.json",
    "../Cornell_critical_results/qwen3:14b_cornell_reasoning_copy_results.json",
    "../LSAT_results/qwen3:14b_lsat_copy_results.json",
    "../SAT_Math_results/qwen3:14b_sat-math-mc_copy_results.json",
    "../SAT_rw_results/qwen3:14b_sat-rw-mc_copy_results.json",
    "../TruthfulQA_results/qwen3:14b_truthfulqa_questions_copy_results.json",
    "../GMAT_results/qwen3:14b_gmat_items_results.json",
    "../ARC_AI2_results/qwen3:14b_ARC_AI2_ARC-Challenge_test_results.json",
    "../MMLU_High_school_physics_results/qwen3:4b-instruct_MMLU_HSphysics_test_copy_results.json",
    "../MMLU_College_physics_results/qwen3:4b-instruct_MMLU_college_physics_test_copy_results.json",
    "../MMLU_Conceptual_physics_results/qwen3:4b-instruct_MMLU_conceptual_physics_test_copy_results.json",
    "../MMLU_Formal_logic_results/qwen3:4b-instruct_MMLU_formal_logic_test_copy_results.json",
    "../MMLU_Logical_fallacies_results/qwen3:4b-instruct_MMLU_logical_fallacies_test_copy_results.json",
    "../MMLU_Us_foreign_policy_results/qwen3:4b-instruct_MMLU_us_foreign_policy_test_copy_results.json",
    "../Statistics_inventory_results/qwen3:4b-instruct_Statistics_inventory_copy_results.json",
    "../IQ-test_results/qwen3:4b-instruct_iq_results.json",
    "../Cornell_critical_results/qwen3:4b-instruct_cornell_reasoning_copy_results.json",
    "../LSAT_results/qwen3:4b-instruct_lsat_copy_results.json",
    "../SAT_Math_results/qwen3:4b-instruct_sat-math-mc_copy_results.json",
    "../SAT_rw_results/qwen3:4b-instruct_sat-rw-mc_copy_results.json",
    "../TruthfulQA_results/qwen3:4b-instruct_truthfulqa_questions_copy_results.json",
    "../GMAT_results/qwen3:4b-instruct_gmat_items_results.json",
    "../ARC_AI2_results/qwen3:4b-instruct_ARC_AI2_ARC-Challenge_test_results.json",
]

MODELS = ["qwen3:14b", "qwen3:4b-instruct"]  # longest/most specific first
SUFFIX_PATTERNS = ["_copy_results", "_results"]  # checked in order, first match stripped once

def parse_model_and_benchmark(filename):
    stem = os.path.basename(filename)
    stem = stem[:-len(".json")] if stem.endswith(".json") else stem

    model_name = next((m for m in MODELS if stem.startswith(m + "_")), None)
    if model_name is None:
        raise ValueError(f"Unrecognized model prefix in filename: {filename}")
    remainder = stem[len(model_name) + 1:]

    for suffix in SUFFIX_PATTERNS:
        if remainder.endswith(suffix):
            remainder = remainder[: -len(suffix)]
            break

    return model_name, remainder

def plot_ratio_regression(ax, x_values, raw_scores, jittered_scores):
    """Draw the jittered scatter and a linear regression fit directly on the raw 0/1 scores."""
    ax.scatter(x_values, jittered_scores, alpha=0.3, s=15, label="Result (jittered)")

    fit = linregress(x_values, raw_scores) if len(set(x_values)) > 1 else None
    if fit is not None:
        x_line = np.array([min(x_values), max(x_values)])
        ax.plot(x_line, fit.intercept + fit.slope * x_line, color="red", linewidth=2, label="Linear fit")
    return fit


def fit_title_suffix(fit, n):
    """Format n, slope, and p-value for appending to a plot title."""
    if fit is None:
        return f"\nn={n}"
    return f"\nn={n}, slope={fit.slope:.4g}, p={fit.pvalue:.3g}"


LABEL_OVERRIDES = {"HSphysics": "High school physics"}

def clean_benchmark_label(b):
    """Turn a raw benchmark key like 'MMLU_HSphysics_test' into 'MMLU High school physics':
    drop '_test' segments, replace remaining underscores with spaces, and apply
    friendlier wording for known abbreviations."""
    parts = [LABEL_OVERRIDES.get(p, p) for p in b.split("_") if p.lower() != "test"]
    return " ".join(parts)

accuracy = []
benchmark = []
model = []
plot_data = {}
chance_by_benchmark = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)
    m, b = parse_model_and_benchmark(file)
    model.append(m)
    benchmark.append(b)
    accuracy.append(float(data["Accuracy"].split()[1].rstrip("%")))

    words = []
    tokens = []
    entropy = []
    scores = []
    chances = []

    for key, entry in data.items():
        if key in ("Accuracy", "Estimated IQ"):
            continue
        words.append(float(entry["Words"]))
        tokens.append(float(entry["Tokens"]))
        entropy.append(float(entry["Entropy"].split()[0]))
        if entry["Result"] == "Correct":
            scores.append(1)
        else:
            scores.append(0)
        chance = question_chance(entry["Prompt"])
        if chance is not None:
            chances.append(chance)

    jittered_scores = [s + random.uniform(-0.03, 0.03) for s in scores]
    plot_data[(m, b)] = {
        "words": words,
        "tokens": tokens,
        "entropy": entropy,
        "scores": jittered_scores,
        "raw_scores": scores,
    }
    if chances:
        chance_by_benchmark.setdefault(b, []).extend(chances)

    fig, ax = plt.subplots()
    fit = plot_ratio_regression(ax, words, scores, jittered_scores)
    ax.set_xlabel("Words")
    ax.set_ylabel("Score")
    ax.set_title(f"{b} tested on {m}{fit_title_suffix(fit, len(words))}")
    ax.set_yticks([0, 1])
    ax.grid()
    ax.legend(fontsize=8)
    fig.savefig(f"{b}_{m}_words_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fit = plot_ratio_regression(ax, tokens, scores, jittered_scores)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Score")
    ax.set_title(f"{b} tested on {m}{fit_title_suffix(fit, len(tokens))}")
    ax.set_yticks([0, 1])
    ax.grid()
    ax.legend(fontsize=8)
    fig.savefig(f"{b}_{m}_tokens_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fit = plot_ratio_regression(ax, entropy, scores, jittered_scores)
    ax.set_xlabel("Entropy [bits/token]")
    ax.set_ylabel("Score")
    ax.set_title(f"{b} tested on {m}{fit_title_suffix(fit, len(entropy))}")
    ax.set_yticks([0, 1])
    ax.grid()
    ax.legend(fontsize=8)
    fig.savefig(f"{b}_{m}_entropy_scatter.png")
    plt.close(fig)

metrics = [("words", "Words"), ("tokens", "Tokens"), ("entropy", "Entropy [bits/token]")]

for b in sorted(set(benchmark)):
    fig, axes = plt.subplots(len(MODELS), len(metrics), figsize=(5 * len(metrics), 4 * len(MODELS)))
    for row, m in enumerate(MODELS):
        pd = plot_data.get((m, b))
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            fit = None
            n = 0
            if pd is not None:
                fit = plot_ratio_regression(ax, pd[metric_key], pd["raw_scores"], pd["scores"])
                n = len(pd[metric_key])
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Score")
            ax.set_yticks([0, 1])
            ax.set_title(f"{m}: {metric_label}{fit_title_suffix(fit, n)}")
            ax.grid()

    fig.suptitle(b)
    fig.tight_layout()
    fig.savefig(f"{b}_combined_scatter.png")
    plt.close(fig)

benchmarks = sorted(set(benchmark))
accuracy_by_model_benchmark = {(model[j], benchmark[j]): accuracy[j] for j in range(len(accuracy))}
n_by_benchmark = {b: len(plot_data[(m, b)]["raw_scores"]) for (m, b) in plot_data if b in benchmarks}
colors = {"qwen3:14b": "#1F77B4", "qwen3:4b-instruct": "#D62728"}

# Drop "iq" (no fixed-option chance level, not comparable to the MC benchmarks) and
# sort by qwen3:14b accuracy so the strongest benchmarks read left to right.
chart_benchmarks = sorted(
    (b for b in benchmarks if b != "iq"),
    key=lambda b: accuracy_by_model_benchmark[("qwen3:14b", b)],
    reverse=True,
)
benchmarks = chart_benchmarks

bar_width = 0.8 / len(MODELS)
x = range(len(benchmarks))

fig, ax = plt.subplots(figsize=(12, 6))
for i, m in enumerate(MODELS):
    offsets = [xi + i * bar_width for xi in x]
    values = [accuracy_by_model_benchmark[(m, b)] for b in benchmarks]
    bars = ax.bar(offsets, values, width=bar_width, label=m, color=colors.get(m))
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8, rotation=90)

variance_by_model = {
    m: statistics.pvariance(accuracy_by_model_benchmark[(m, b)] for b in benchmarks)
    for m in MODELS
}
variance_subtitle = "  |  ".join(f"{m}: σ²={variance_by_model[m]:.1f}" for m in MODELS)

fixed_chance_labeled = False
variable_chance_labeled = False
for i, b in enumerate(benchmarks):
    chances = chance_by_benchmark.get(b)
    if not chances:
        continue
    chance_pct = 100 * sum(chances) / len(chances)
    varies = len(set(chances)) > 1
    if varies:
        label = None if variable_chance_labeled else "Chance level (varies per question, avg shown)"
        variable_chance_labeled = True
    else:
        label = None if fixed_chance_labeled else "Chance level"
        fixed_chance_labeled = True
    ax.hlines(
        chance_pct,
        i,
        i + len(MODELS) * bar_width,
        linestyle="--",
        color="red" if varies else "black",
        linewidth=1.2,
        label=label,
    )

ax.set_xticks([xi + bar_width * (len(MODELS) - 1) / 2 for xi in x])
ax.set_xticklabels([f"{clean_benchmark_label(b)}\n(n={n_by_benchmark[b]})" for b in benchmarks], rotation=45, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 130)
ax.set_title("Model accuracy by benchmark", y=1.16, fontsize=13)
ax.text(
    0.5, 1.09,
    f"Accuracy variance across benchmarks — {variance_subtitle}",
    transform=ax.transAxes, ha="center", fontsize=9.5, color="gray",
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(handles), frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("Benchmarks_models_results.png")