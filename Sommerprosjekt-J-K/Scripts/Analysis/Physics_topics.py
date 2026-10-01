import json
import matplotlib.pyplot as plt
from collections import Counter

filenames = [
    "qwen3:14b_MMLU_HSphysics_test_copy_results.json",
    "qwen3:14b_MMLU_conceptual_physics_test_copy_results.json",
    "qwen3:14b_MMLU_college_physics_test_copy_results.json",
    "qwen3:4b-instruct_MMLU_HSphysics_test_copy_results.json",
    "qwen3:4b-instruct_MMLU_conceptual_physics_test_copy_results.json",
    "qwen3:4b-instruct_MMLU_college_physics_test_copy_results.json",
]

models = ["qwen3:14b", "qwen3:4b-instruct"]

def parse_title(filename):
    core = filename.removesuffix("_test_copy_results.json")
    parts = core.split("_")
    model = parts[0]
    benchmark = " ".join(parts[2:])
    return model, benchmark

topic_counts = {}
benchmark_topic_cor_counts = {}

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)
    counts = Counter()
    cor_counts = Counter()
    for item in data:
        if item == "Accuracy":
            continue
        else:
            counts[data[item]["Topic"]] += 1
        if data[item]["Result"] == "Correct":
            cor_counts[data[item]["Topic"]] += 1

    benchmark_topic_cor_counts[file] = cor_counts
    topic_counts[file] = counts
    print(file, dict(counts))


plt.figure(figsize=(18, 6))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.bar(topic_counts[filenames[i]].keys(),topic_counts[filenames[i]].values())
    model, benchmark = parse_title(filenames[i])
    plt.title(benchmark)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y")

plt.suptitle(f"{model} — Question Count per Topic")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("topic_question_counts.png", dpi=150)

accuracy = []
for i in range(len(filenames)):
    file = filenames[i]
    acc = {topic: benchmark_topic_cor_counts[file][topic] / topic_counts[file][topic]
           for topic in topic_counts[file]}
    accuracy.append(acc)

plt.figure(figsize=(18, 10))
for i in range(len(filenames)):
    file = filenames[i]
    topics = list(accuracy[i].keys())
    percentages = [accuracy[i][topic] * 100 for topic in topics]
    ns = [topic_counts[file][topic] for topic in topics]

    overall_n = sum(topic_counts[file].values())
    overall_pct = sum(benchmark_topic_cor_counts[file].values()) / overall_n * 100
    topics.append("Overall")
    percentages.append(overall_pct)
    ns.append(overall_n)

    colors = ["tab:blue"] * (len(topics) - 1) + ["tab:orange"]

    plt.subplot(2,3,i+1)
    bars = plt.bar(topics, percentages, color=colors)
    for bar, n in zip(bars, ns):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  f"n={n}", ha="center", va="bottom", fontsize=7)
    model, benchmark = parse_title(file)
    plt.title(f"{model} — {benchmark}", fontsize=10, pad=14)
    plt.ylim(0, 112)
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y")

plt.suptitle("Correct Answers per Topic by Model and Benchmark")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("topic_accuracy.png", dpi=150)


plt.figure(figsize=(18, 10))
for i in range(3):
    file_a = filenames[i]          # qwen3:14b
    file_b = filenames[i + 3]      # qwen3:4b-instruct

    topics = list(topic_counts[file_a].keys()) + ["Overall"]

    acc_a = {t: benchmark_topic_cor_counts[file_a][t] / topic_counts[file_a][t] for t in topic_counts[file_a]}
    acc_b = {t: benchmark_topic_cor_counts[file_b][t] / topic_counts[file_b][t] for t in topic_counts[file_b]}
    acc_a["Overall"] = sum(benchmark_topic_cor_counts[file_a].values()) / sum(topic_counts[file_a].values())
    acc_b["Overall"] = sum(benchmark_topic_cor_counts[file_b].values()) / sum(topic_counts[file_b].values())

    absolute_gain = [(acc_a[t] - acc_b[t]) * 100 for t in topics]
    relative_gain = [((acc_a[t] - acc_b[t]) / acc_b[t] * 100) if acc_b[t] > 0 else 0.0 for t in topics]
    ns = [topic_counts[file_a][t] for t in topics[:-1]] + [sum(topic_counts[file_a].values())]

    benchmark = parse_title(file_a)[1]

    plt.subplot(2, 3, i + 1)
    bars = plt.bar(topics, absolute_gain, color=["tab:green" if v >= 0 else "tab:red" for v in absolute_gain])
    for bar, v in zip(bars, absolute_gain):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:+.1f}",
                  ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{benchmark} — Absolute Gain (pp)", fontsize=10)
    plt.ylabel("qwen3:14b − qwen3:4b-instruct (pp)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y")

    plt.subplot(2, 3, i + 4)
    bars = plt.bar(topics, relative_gain, color=["tab:green" if v >= 0 else "tab:red" for v in relative_gain])
    for bar, v, n in zip(bars, relative_gain, ns):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:+.0f}%\nn={n}",
                  ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{benchmark} — Relative Gain (%)", fontsize=10)
    plt.ylabel("Relative change vs qwen3:4b-instruct")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y")

plt.suptitle("qwen3:14b vs qwen3:4b-instruct — Accuracy Gain by Topic")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("model_accuracy_gain.png", dpi=150)