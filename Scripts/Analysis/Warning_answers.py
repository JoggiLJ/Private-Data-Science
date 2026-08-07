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

models = ["qwen3:4b-instruct", "qwen3:14b"]
warning_data = {model: {} for model in models}

for file in filenames:
    model = "qwen3:4b-instruct" if "qwen3:4b-instruct" in file else "qwen3:14b"
    with open(file, "r") as infile:
        data = json.load(infile)
    for item in data:
        if item == "Accuracy" or not isinstance(data[item], dict):
            continue
        else:
            words = data[item]["Words"]
            tokens = data[item]["Tokens"]
            entropy = data[item]["Entropy"].split()[0]
            if len(data[item]["Model answer"]) > 1:
                warning_data[model][(file, item)] = {"Words": words, "Tokens": tokens, "Entropy": entropy}
            else:
                continue

num_words = [[entry["Words"] for entry in warning_data[model].values()] for model in models]
num_tokens = [[entry["Tokens"] for entry in warning_data[model].values()] for model in models]
num_entropy = [[entry["Entropy"] for entry in warning_data[model].values()] for model in models]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(num_words, bins=20, label=models)
axes[0].set_xlabel("Words")
axes[0].set_ylabel("Number of warning questions")

axes[1].hist(num_tokens, bins=20, label=models)
axes[1].set_xlabel("Tokens")

axes[2].hist(num_entropy, bins=20, label=models)
axes[2].set_xlabel("Entropy [bits/token]")

for ax in axes:
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

plt.tight_layout()
plt.savefig("warning_answers_distribution.png", dpi=150)