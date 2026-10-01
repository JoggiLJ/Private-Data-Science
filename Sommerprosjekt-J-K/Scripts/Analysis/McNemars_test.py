import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

df = 1

results_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

filenames = [
    "MMLU_High_school_physics_results/qwen3:14b_MMLU_HSphysics_test_copy_results.json",
    "MMLU_College_physics_results/qwen3:14b_MMLU_college_physics_test_copy_results.json",
    "MMLU_Conceptual_physics_results/qwen3:14b_MMLU_conceptual_physics_test_copy_results.json",
    "MMLU_Formal_logic_results/qwen3:14b_MMLU_formal_logic_test_copy_results.json",
    "MMLU_Logical_fallacies_results/qwen3:14b_MMLU_logical_fallacies_test_copy_results.json",
    "MMLU_Us_foreign_policy_results/qwen3:14b_MMLU_us_foreign_policy_test_copy_results.json",
    "Statistics_inventory_results/qwen3:14b_Statistics_inventory_copy_results.json",
    "IQ-test_results/qwen3:14b_iq_results.json",
    "Cornell_critical_results/qwen3:14b_cornell_reasoning_copy_results.json",
    "LSAT_results/qwen3:14b_lsat_copy_results.json",
    "SAT_Math_results/qwen3:14b_sat-math-mc_copy_results.json",
    "SAT_rw_results/qwen3:14b_sat-rw-mc_copy_results.json",
    "TruthfulQA_results/qwen3:14b_truthfulqa_questions_copy_results.json",
    "GMAT_results/qwen3:14b_gmat_items_results.json",
    "ARC_AI2_results/qwen3:14b_ARC_AI2_ARC-Challenge_test_results.json",
    "MMLU_High_school_physics_results/qwen3:4b-instruct_MMLU_HSphysics_test_copy_results.json",
    "MMLU_College_physics_results/qwen3:4b-instruct_MMLU_college_physics_test_copy_results.json",
    "MMLU_Conceptual_physics_results/qwen3:4b-instruct_MMLU_conceptual_physics_test_copy_results.json",
    "MMLU_Formal_logic_results/qwen3:4b-instruct_MMLU_formal_logic_test_copy_results.json",
    "MMLU_Logical_fallacies_results/qwen3:4b-instruct_MMLU_logical_fallacies_test_copy_results.json",
    "MMLU_Us_foreign_policy_results/qwen3:4b-instruct_MMLU_us_foreign_policy_test_copy_results.json",
    "Statistics_inventory_results/qwen3:4b-instruct_Statistics_inventory_copy_results.json",
    "IQ-test_results/qwen3:4b-instruct_iq_results.json",
    "Cornell_critical_results/qwen3:4b-instruct_cornell_reasoning_copy_results.json",
    "LSAT_results/qwen3:4b-instruct_lsat_copy_results.json",
    "SAT_Math_results/qwen3:4b-instruct_sat-math-mc_copy_results.json",
    "SAT_rw_results/qwen3:4b-instruct_sat-rw-mc_copy_results.json",
    "TruthfulQA_results/qwen3:4b-instruct_truthfulqa_questions_copy_results.json",
    "GMAT_results/qwen3:4b-instruct_gmat_items_results.json",
    "ARC_AI2_results/qwen3:4b-instruct_ARC_AI2_ARC-Challenge_test_results.json",
]

model_data = {}

for file in filenames:
    with open(os.path.join(results_dir, file), "r") as infile:
        data = json.load(infile)
    basename = os.path.basename(file)
    model = basename.split("_")[0]
    benchmark = basename.removesuffix("_results.json").removeprefix(model + "_")

    benchmark_scores = []

    for item in data:
        if item in ("Accuracy", "Estimated IQ"):
            continue
        benchmark_scores.append(1 if data[item]["Result"] == "Correct" else 0)
    model_data.setdefault(model, {})[benchmark] = benchmark_scores

benchmarks = list(model_data["qwen3:14b"].keys())

results = {}

for b in benchmarks:
    fourteen_b_scores = model_data["qwen3:14b"][b]
    four_b_scores = model_data["qwen3:4b-instruct"][b]

    both_cor = 0
    fourteen_w = 0
    four_w = 0
    both_w = 0

    for i in range(len(fourteen_b_scores)):
        if fourteen_b_scores[i] == 1 and four_b_scores[i] == 1:
            both_cor += 1
        elif fourteen_b_scores[i] == 1 and four_b_scores[i] == 0:
            four_w += 1
        elif fourteen_b_scores[i] == 0 and four_b_scores[i] == 1:
            fourteen_w += 1
        else:
            both_w += 1

    #Chi
    if four_w + fourteen_w < 25:
        chi_squared = (abs(four_w-fourteen_w)-1)**2/(four_w + fourteen_w)
    else:
        chi_squared = (four_w-fourteen_w)**2/(four_w + fourteen_w)
    p_value = chi2.sf(chi_squared, df)

    #Cohens Kappa
    total = len(fourteen_b_scores)
    p0 = (both_cor + both_w)/(total)
    pe = ((fourteen_w + both_cor)/(total)) * ((four_w + both_cor)/(total)) + ((both_w + fourteen_w)/(total)) * ((both_w + four_w)/(total))
    kappa = (p0-pe)/(1-pe)

    print(f"{b}: Both correct = {both_cor}, 14b wrong = {fourteen_w}, 4b wrong = {four_w}, Both wrong = {both_w}, Chi^2 = {chi_squared:.3f}, p = {p_value:.3f}, Kappa = {kappa:.3f}" + "\n")

    results[b] = {
        "both_cor": both_cor,
        "fourteen_w": fourteen_w,
        "four_w": four_w,
        "both_w": both_w,
        "chi_squared": chi_squared,
        "p_value": p_value,
        "kappa": kappa,
    }

benchmarks_sorted = sorted(benchmarks, key=lambda b: results[b]["chi_squared"], reverse=True)

print("Benchmarks sorted by Chi^2 (highest to lowest):\n")
for b in benchmarks_sorted:
    r = results[b]
    print(f"{b}: Both correct = {r['both_cor']}, 14b wrong = {r['fourteen_w']}, 4b wrong = {r['four_w']}, Both wrong = {r['both_w']}, Chi^2 = {r['chi_squared']:.3f}, p = {r['p_value']:.3f}, Kappa = {r['kappa']:.3f}")

# "iq" has no fixed multiple-choice options, so it isn't comparable to the other
# (MC) benchmarks here; drop it from the figure.
benchmarks_sorted = [b for b in benchmarks_sorted if b != "iq"]

CLEAN_LABELS = {
    "MMLU_college_physics_test_copy": "MMLU College physics",
    "sat-math-mc_copy": "SAT Math",
    "truthfulqa_questions_copy": "TruthfulQA",
    "cornell_reasoning_copy": "Cornell critical",
    "MMLU_conceptual_physics_test_copy": "MMLU Conceptual physics",
    "MMLU_HSphysics_test_copy": "MMLU High school physics",
    "gmat_items": "GMAT",
    "sat-rw-mc_copy": "SAT rw",
    "MMLU_us_foreign_policy_test_copy": "MMLU Us foreign policy",
    "MMLU_formal_logic_test_copy": "MMLU Formal logic",
    "MMLU_logical_fallacies_test_copy": "MMLU Logical fallacies",
    "ARC_AI2_ARC-Challenge_test": "ARC AI2",
    "Statistics_inventory_copy": "Statistics inventory",
    "lsat_copy": "LSAT",
}

fig, axes = plt.subplots(5, 3, figsize=(13, 19))

for ax, b in zip(axes.flat, benchmarks_sorted):
    both_cor = results[b]["both_cor"]
    fourteen_w = results[b]["fourteen_w"]
    four_w = results[b]["four_w"]
    both_w = results[b]["both_w"]
    chi_squared = results[b]["chi_squared"]
    p_value = results[b]["p_value"]
    kappa = results[b]["kappa"]

    data_grid = np.array([[both_cor, four_w], [fourteen_w, both_w]])
    ax.matshow(data_grid, cmap='Blues')
    threshold = data_grid.max() / 2
    for (row, col), value in np.ndenumerate(data_grid):
        text_color = 'white' if value > threshold else 'black'
        ax.text(col, row, str(value), color=text_color,
            fontsize=14, ha='center', va='center', weight='bold')
    ax.set_title(f"{CLEAN_LABELS.get(b, b)}\n" + r"$\chi^2$" + f" = {chi_squared:.3f}, p = {p_value:.3f}, Kappa = {kappa:.3f}", fontsize=9, pad=28)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["4b Correct", "4b Wrong"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["14b Correct", "14b Wrong"], fontsize=8)

for ax in axes.flat[len(benchmarks_sorted):]:
    ax.axis('off')

fig.suptitle(f"qwen3:14b vs qwen3:4b-instruct — McNemar contingency tables", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.9, wspace=0.4)
plt.savefig("McNemars_test.png")