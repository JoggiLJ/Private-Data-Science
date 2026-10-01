import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OPTION_PATTERN = re.compile(r"^[A-Za-z]\)", re.MULTILINE)

filenames = [
    "../Statistics_inventory_results/qwen3:14b_Statistics_inventory_copy_results.json",
    "../Cornell_critical_results/qwen3:14b_cornell_reasoning_copy_results.json",
    "../SAT_Math_results/qwen3:14b_sat-math-mc_copy_results.json",
    "../SAT_rw_results/qwen3:14b_sat-rw-mc_copy_results.json",
    "../Statistics_inventory_results/qwen3:4b-instruct_Statistics_inventory_copy_results.json",
    "../Cornell_critical_results/qwen3:4b-instruct_cornell_reasoning_copy_results.json",
    "../SAT_Math_results/qwen3:4b-instruct_sat-math-mc_copy_results.json",
    "../SAT_rw_results/qwen3:4b-instruct_sat-rw-mc_copy_results.json",
    "../ARC_AI2_results/qwen3:14b_ARC_AI2_ARC-Challenge_test_results.json",
    "../ARC_AI2_results/qwen3:4b-instruct_ARC_AI2_ARC-Challenge_test_results.json",
]

BASE_KEYS = {"Prompt", "Model answer", "Correct label", "Result", "Words", "Tokens", "Entropy"}
MODELS = ["qwen3:14b", "qwen3:4b-instruct"]
MODEL_COLORS = {"qwen3:14b": "#1F77B4", "qwen3:4b-instruct": "#D62728"}


def parse_filename(file):
    name = file.split("/")[-1]
    if name.endswith("_copy_results.json"):
        name = name[: -len("_copy_results.json")]
    elif name.endswith("_results.json"):
        name = name[: -len("_results.json")]
    for model in MODELS:
        if name.startswith(model + "_"):
            return model, name[len(model) + 1:]
    return "unknown", name


dimension = []
scores = []
# dim_scores[benchmark][field][value][model] -> list of 0/1 scores
dim_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
benchmark_chance = defaultdict(list)

for file in filenames:
    model, benchmark = parse_filename(file)
    with open(file, "r") as infile:
        data = json.load(infile)
    for key, entry in data.items():
        if key == "Accuracy":
            continue
        extra = {k: v for k, v in entry.items() if k not in BASE_KEYS}
        dimension.append(extra)
        score = 1 if entry["Result"] == "Correct" else 0
        scores.append(score)
        for field, value in extra.items():
            dim_scores[benchmark][field][value][model].append(score)
        n_alternatives = len(OPTION_PATTERN.findall(entry["Prompt"]))
        if n_alternatives:
            benchmark_chance[benchmark].append(1 / n_alternatives)

chance_level = {b: sum(vals) / len(vals) for b, vals in benchmark_chance.items()}

width = 0.35
log_lines = []


def log(message):
    print(message)
    log_lines.append(message)


def add_bar_labels(ax, bars, counts, horizontal):
    for bar, n in zip(bars, counts):
        if horizontal:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}", ha="center", fontsize=7)


DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def value_sort_key(field, value):
    if field == "Difficulty":
        return DIFFICULTY_ORDER.get(value, len(DIFFICULTY_ORDER))
    return value


def plot_field(ax, benchmark, field, chance):
    values = sorted(dim_scores[benchmark][field].keys(), key=lambda v: value_sort_key(field, v))
    horizontal = field == "Skill"
    pos = range(len(values))

    for i, model in enumerate(MODELS):
        accuracies = []
        counts = []
        for v in values:
            s = dim_scores[benchmark][field][v][model]
            acc = sum(s) / len(s) if s else 0
            accuracies.append(acc)
            counts.append(len(s))
            log(f"{benchmark} | {field}={v} | {model}: {acc:.0%} (n={len(s)})")
        offset = (i - (len(MODELS) - 1) / 2) * width
        positions = [p + offset for p in pos]
        if horizontal:
            bars = ax.barh(positions, accuracies, width, label=model, color=MODEL_COLORS[model])
        else:
            bars = ax.bar(positions, accuracies, width, label=model, color=MODEL_COLORS[model])
        add_bar_labels(ax, bars, counts, horizontal)

    if horizontal:
        ax.set_yticks(list(pos))
        ax.set_yticklabels([str(v) for v in values])
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="x")
        ax.axvline(chance, color="black", linestyle="--", linewidth=1, label=f"Chance ({chance:.0%})")
    else:
        ax.set_xticks(list(pos))
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y")
        ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"Chance ({chance:.0%})")

    ax.set_title(field)
    ax.legend()


for benchmark in sorted(dim_scores.keys()):
    all_fields = list(dim_scores[benchmark].keys())
    main_fields = [f for f in all_fields if f != "Skill"]

    fig, axes = plt.subplots(len(main_fields), 1, figsize=(10, 4 * len(main_fields)))
    if len(main_fields) == 1:
        axes = [axes]
    fig.suptitle(benchmark)

    for ax, field in zip(axes, main_fields):
        plot_field(ax, benchmark, field, chance_level[benchmark])

    plt.tight_layout()
    plt.savefig(f"{benchmark}.png")
    plt.close(fig)

    if "Skill" in all_fields:
        n_skills = len(dim_scores[benchmark]["Skill"])
        fig, ax = plt.subplots(figsize=(10, 0.4 * n_skills + 2))
        fig.suptitle(f"{benchmark} - Skill")
        plot_field(ax, benchmark, "Skill", chance_level[benchmark])
        plt.tight_layout()
        plt.savefig(f"{benchmark}_skill.png")
        plt.close(fig)

with open("benchmark_results.txt", "w") as results_file:
    results_file.write("\n".join(log_lines) + "\n")