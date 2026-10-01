import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.stats import gaussian_kde

result_filenames = [
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

sets_filenames = [
    "../SAT_Math_results/sat-math-mc_copy.json",
    "../SAT_Math_results/sat-math-mc_copy.json",
    "../SAT_rw_results/sat-rw-mc_copy.json",
    "../SAT_rw_results/sat-rw-mc_copy.json",
    "../Cornell_critical_results/cornell_reasoning_copy.json",
    "../Cornell_critical_results/cornell_reasoning_copy.json",
    "../Statistics_inventory_results/Statistics_inventory_copy.json",
    "../Statistics_inventory_results/Statistics_inventory_copy.json",
    "../LSAT_results/lsat_copy.json",
    "../LSAT_results/lsat_copy.json",
    "../TruthfulQA_results/truthfulqa_questions_copy.json",
    "../TruthfulQA_results/truthfulqa_questions_copy.json",
    None,  # IQ-test items are image/symbol-based, no lettered option text
    None,
    "../MMLU_Logical_fallacies_results/MMLU_logical_fallacies_test_copy.json",
    "../MMLU_Logical_fallacies_results/MMLU_logical_fallacies_test_copy.json",
    "../MMLU_Formal_logic_results/MMLU_formal_logic_test_copy.json",
    "../MMLU_Formal_logic_results/MMLU_formal_logic_test_copy.json",
    "../MMLU_Us_foreign_policy_results/MMLU_us_foreign_policy_test_copy.json",
    "../MMLU_Us_foreign_policy_results/MMLU_us_foreign_policy_test_copy.json",
    "../MMLU_High_school_physics_results/MMLU_HSphysics_test_copy.json",
    "../MMLU_High_school_physics_results/MMLU_HSphysics_test_copy.json",
    "../MMLU_Conceptual_physics_results/MMLU_conceptual_physics_test_copy.json",
    "../MMLU_Conceptual_physics_results/MMLU_conceptual_physics_test_copy.json",
    "../MMLU_College_physics_results/MMLU_college_physics_test_copy.json",
    "../MMLU_College_physics_results/MMLU_college_physics_test_copy.json",
    "../GMAT_results/gmat_items.json",
    "../GMAT_results/gmat_items.json",
    "../ARC_AI2_results/ARC_AI2_ARC-Challenge_test.json",
    "../ARC_AI2_results/ARC_AI2_ARC-Challenge_test.json",
]


def extract_letter(answer):
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in "ABCDEFGH":
        return answer
    matches = re.findall(r"\b([A-H])\b", answer)
    return matches[-1] if matches else "Unparsed"


def model_benchmark_extract(filename):
    folder, basename = filename.split("/")[-2:]
    model = basename.split("_")[0]
    benchmark = folder.removesuffix("_results").replace("_", " ").replace("-", " ")
    return model, benchmark


# item_id -> {label: option_text} per benchmark, skipping options with no text
# (e.g. image-based questions where the option content isn't in the text field)
options_by_benchmark = {}
set_file_cache = {}

for result_file, set_file in zip(result_filenames, sets_filenames):
    if set_file is None:
        continue
    _, benchmark = model_benchmark_extract(result_file)
    if benchmark in options_by_benchmark:
        continue
    if set_file not in set_file_cache:
        with open(set_file, "r") as infile:
            set_file_cache[set_file] = json.load(infile)
    data = set_file_cache[set_file]
    # Most set files are {"items": [...]}; Statistics_inventory_copy.json is
    # just a bare list of items.
    items = data["items"] if isinstance(data, dict) else data
    item_options = {}
    for item in items:
        labeled = {
            opt["label"].upper(): opt["text"] for opt in item["options"] if opt["text"]
        }
        if labeled:
            item_options[str(item["item_id"])] = labeled
    options_by_benchmark[benchmark] = item_options

# Rank options within each item by length, longest = rank 1. For each benchmark,
# track how often the correct label falls at each rank, and how often each
# model's chosen label falls at each rank.
MAX_RANK = 6

correct_rank_counts = defaultdict(Counter)  # benchmark -> {rank: count}
correct_rank_totals = defaultdict(Counter)  # benchmark -> {rank: n items with >= rank options}
correct_items_seen = defaultdict(set)  # benchmark -> {item keys already counted}

model_rank_counts = defaultdict(Counter)  # (model, benchmark) -> {rank: count}
model_rank_totals = defaultdict(Counter)  # (model, benchmark) -> {rank: n items}

# (model, benchmark) -> list of (correct option length, was model correct)
accuracy_by_correct_length = defaultdict(list)

# benchmark -> lengths of the correct option (deduped per item)
correct_answer_lengths = defaultdict(list)
# (model, benchmark) -> lengths of the option the model chose
model_answer_lengths = defaultdict(list)

benchmarks_order = []
models_by_benchmark = defaultdict(list)

for result_file in result_filenames:
    model, benchmark = model_benchmark_extract(result_file)
    item_options = options_by_benchmark.get(benchmark)
    if item_options is None:
        continue

    with open(result_file, "r") as infile:
        data = json.load(infile)

    for key, entry in data.items():
        if key in ("Accuracy", "Estimated IQ"):
            continue
        options = item_options.get(key)
        if not options:
            continue

        lengths = {label: len(text) for label, text in options.items()}
        ranked_labels = sorted(lengths, key=lambda label: (-lengths[label], label))
        rank_of = {label: rank + 1 for rank, label in enumerate(ranked_labels)}
        n_ranks = min(len(ranked_labels), MAX_RANK)

        correct_label = entry["Correct label"].upper()
        if correct_label in rank_of and key not in correct_items_seen[benchmark]:
            correct_items_seen[benchmark].add(key)
            correct_rank_counts[benchmark][rank_of[correct_label]] += 1
            for r in range(1, n_ranks + 1):
                correct_rank_totals[benchmark][r] += 1
            correct_answer_lengths[benchmark].append(lengths[correct_label])

        model_label = extract_letter(entry["Model answer"])
        if model_label in rank_of:
            model_rank_counts[(model, benchmark)][rank_of[model_label]] += 1
            for r in range(1, n_ranks + 1):
                model_rank_totals[(model, benchmark)][r] += 1
            model_answer_lengths[(model, benchmark)].append(lengths[model_label])

        if correct_label in lengths:
            accuracy_by_correct_length[(model, benchmark)].append(
                (lengths[correct_label], entry["Result"] == "Correct")
            )

    if benchmark not in benchmarks_order:
        benchmarks_order.append(benchmark)
    if model not in models_by_benchmark[benchmark]:
        models_by_benchmark[benchmark].append(model)

RANK_LABELS = ["1st", "2nd", "3rd", "4th", "5th", "6th"]

SERIES_COLORS = {"Correct answer": "#BDBDBD", "qwen3:14b": "#1F77B4", "qwen3:4b-instruct": "#D62728"}

ncols = 3
nrows = -(-len(benchmarks_order) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, benchmark in zip(axes, benchmarks_order):
    max_rank = max(correct_rank_totals[benchmark], default=0)
    ranks = list(range(1, max_rank + 1))
    models = models_by_benchmark[benchmark]

    series = [("Correct answer", [
        correct_rank_counts[benchmark][r] / correct_rank_totals[benchmark][r]
        if correct_rank_totals[benchmark][r] else 0
        for r in ranks
    ])]
    for model in models:
        totals = model_rank_totals[(model, benchmark)]
        counts = model_rank_counts[(model, benchmark)]
        series.append((model, [
            counts[r] / totals[r] if totals[r] else 0 for r in ranks
        ]))

    x = np.arange(len(ranks))
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name, color=SERIES_COLORS.get(name))

    ax.set_xticks(x, [RANK_LABELS[r - 1] for r in ranks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(benchmark)
    ax.grid(axis="y")
    ax.legend(fontsize=8)

for ax in axes[len(benchmarks_order):]:
    ax.set_visible(False)

fig.suptitle("How often the longest / 2nd-longest / ... option is correct vs. chosen")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("option_rank_rates.png", dpi=150)

# --- P(model correct) as a function of the correct option's length ---
# Bin items into equal-sized quantile buckets by correct-option length, since
# raw lengths are too sparse/continuous to read as a probability directly.


def quantile_bins(pairs, nbins):
    pairs = sorted(pairs, key=lambda p: p[0])
    nbins = max(1, min(nbins, len(pairs)))
    xs, ys, ns = [], [], []
    for chunk_idx in np.array_split(np.arange(len(pairs)), nbins):
        if len(chunk_idx) == 0:
            continue
        chunk = [pairs[i] for i in chunk_idx]
        xs.append(np.mean([length for length, _ in chunk]))
        ys.append(np.mean([1 if correct else 0 for _, correct in chunk]))
        ns.append(len(chunk))
    return xs, ys, ns


fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes3 = axes3.flatten()

for ax, benchmark in zip(axes3, benchmarks_order):
    for model in models_by_benchmark[benchmark]:
        pairs = accuracy_by_correct_length[(model, benchmark)]
        if not pairs:
            continue
        nbins = min(6, max(2, len(pairs) // 15))
        xs, ys, ns = quantile_bins(pairs, nbins)
        ax.plot(xs, ys, marker="o", label=model)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Correct option length (characters)")
    ax.set_ylabel("P(model correct)")
    ax.set_title(benchmark)
    ax.grid(True)
    ax.legend(fontsize=8)

for ax in axes3[len(benchmarks_order):]:
    ax.set_visible(False)

fig3.suptitle("Model accuracy vs. length of the correct option (quantile-binned)")
fig3.tight_layout()
fig3.savefig("option_length_vs_accuracy.png", dpi=150)

# --- Density of correct-answer lengths vs. model-chosen-answer lengths ---
def plot_density(ax, values, label):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) == 0:
        return
    kde = gaussian_kde(values)
    xs = np.linspace(values.min(), values.max(), 200)
    ax.plot(xs, kde(xs), label=label)


fig4, axes4 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes4 = axes4.flatten()

for ax, benchmark in zip(axes4, benchmarks_order):
    plot_density(ax, correct_answer_lengths[benchmark], "Correct answer")
    for model in models_by_benchmark[benchmark]:
        plot_density(ax, model_answer_lengths[(model, benchmark)], f"{model} answer")

    ax.set_xlabel("Option length (characters)")
    ax.set_ylabel("Density")
    ax.set_title(benchmark)
    ax.grid(True)
    ax.legend(fontsize=8)

for ax in axes4[len(benchmarks_order):]:
    ax.set_visible(False)

fig4.suptitle("Density of correct-answer length vs. model-chosen-answer length")
fig4.tight_layout()
fig4.savefig("option_length_density.png", dpi=150)
