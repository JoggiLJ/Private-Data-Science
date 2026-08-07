import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def benchmark_name(filename):
    folder = filename.split("/")[-2]
    return folder.removesuffix("_results").replace("_", " ").replace("-", " ")


def model_name(filename):
    basename = filename.split("/")[-1]
    return basename.split("_")[0]

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

error_data = {}
total_questions_with_numbers = 0
total_questions_without_numbers = 0
error_questions_with_numbers = 0
error_questions_without_numbers = 0
correct_questions_with_numbers = 0
correct_questions_without_numbers = 0

has_numbers_flags = []
is_correct_flags = []
digit_counts = []
item_benchmarks = []
item_models = []

for file in filenames:
    with open(file, "r") as infile:
        data = json.load(infile)

    error_questions = []
    for item in data:
        if item in ("Accuracy", "Estimated IQ"):
            continue
        prompt = data[item]["Prompt"]
        model_answer = data[item]["Model answer"]
        has_numbers = any(char.isdigit() for char in prompt)
        is_correct = data[item]["Result"] == "Correct"
        digit_count = sum(char.isdigit() for char in prompt)

        has_numbers_flags.append(int(has_numbers))
        is_correct_flags.append(int(is_correct))
        digit_counts.append(digit_count)
        item_benchmarks.append(benchmark_name(file))
        item_models.append(model_name(file))

        if has_numbers:
            total_questions_with_numbers += 1
            if is_correct:
                correct_questions_with_numbers += 1
        else:
            total_questions_without_numbers += 1
            if is_correct:
                correct_questions_without_numbers += 1

        if len(model_answer) > 1:
            error_questions.append(prompt)
            if has_numbers:
                error_questions_with_numbers += 1
            else:
                error_questions_without_numbers += 1

    error_data[file] = {
        "Number of error questions": len(error_questions),
        "Error questions": error_questions,
    }

with open("error_questions.txt", "w") as outfile:
    for key, value in error_data.items():
        outfile.write(f"{key}:" + "\n")
        for question in value["Error questions"]:
            outfile.write(question + "\n")

accuracy_with_numbers = correct_questions_with_numbers / total_questions_with_numbers
accuracy_without_numbers = correct_questions_without_numbers / total_questions_without_numbers

total_questions = total_questions_with_numbers + total_questions_without_numbers
total_error_questions = error_questions_with_numbers + error_questions_without_numbers

report_lines = []
report_lines.append(f"Total questions with numbers: {total_questions_with_numbers}")
report_lines.append(f"Error questions with numbers: {error_questions_with_numbers}")
report_lines.append(f"Accuracy for questions with numbers: {accuracy_with_numbers:.2%}")
report_lines.append(f"Accuracy for questions without numbers: {accuracy_without_numbers:.2%}")
report_lines.append(f"Share of error questions with numbers: {error_questions_with_numbers/(total_error_questions)*100:.2f}% (n={total_error_questions})")
report_lines.append(f"Share of total questions with numbers: {total_questions_with_numbers/(total_questions)*100:.2f}% (n={total_questions})")

# --- Statistical analysis: does having numbers in the question affect accuracy? ---

contingency_table = np.array(
    [
        [correct_questions_with_numbers, total_questions_with_numbers - correct_questions_with_numbers],
        [correct_questions_without_numbers, total_questions_without_numbers - correct_questions_without_numbers],
    ]
)
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

n1, n2 = total_questions_with_numbers, total_questions_without_numbers
p1, p2 = accuracy_with_numbers, accuracy_without_numbers
p_pool = (correct_questions_with_numbers + correct_questions_without_numbers) / (n1 + n2)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
z_stat = (p1 - p2) / se_pool
z_p = 2 * stats.norm.sf(abs(z_stat))

se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
ci_low, ci_high = (p1 - p2) - 1.96 * se_diff, (p1 - p2) + 1.96 * se_diff

has_numbers_arr = np.array(has_numbers_flags)
is_correct_arr = np.array(is_correct_flags)
digit_count_arr = np.array(digit_counts)

lr_binary = stats.linregress(has_numbers_arr, is_correct_arr)
lr_digits = stats.linregress(digit_count_arr, is_correct_arr)

report_lines.append("")
report_lines.append("--- Effect of numbers in the question on accuracy ---")
report_lines.append(f"Accuracy difference (with - without numbers): {p1 - p2:+.2%}  (95% CI: {ci_low:+.2%} to {ci_high:+.2%})")
report_lines.append(f"Chi-square test of independence: chi2={chi2:.3f}, p={chi2_p:.4g}")
report_lines.append(f"Two-proportion z-test: z={z_stat:.3f}, p={z_p:.4g}")
report_lines.append(
    f"Linear regression (correct ~ has_numbers): slope={lr_binary.slope:+.4f}, "
    f"p={lr_binary.pvalue:.4g}, R^2={lr_binary.rvalue**2:.4f}"
)
report_lines.append(
    f"Linear regression (correct ~ digit_count): slope={lr_digits.slope:+.5f}, "
    f"p={lr_digits.pvalue:.4g}, R^2={lr_digits.rvalue**2:.4f}"
)

# --- Isolate the effect: control for benchmark and model (fixed effects) ---


def ols_fit(X, y):
    # macOS Accelerate BLAS emits spurious matmul RuntimeWarnings here even
    # though the results are numerically correct (verified against manual
    # dot products); suppress just that noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, k = X.shape
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        dof = n - k
        sigma2 = (residuals @ residuals) / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    return beta, se, t_stats, p_values


df = pd.DataFrame(
    {
        "has_numbers": has_numbers_flags,
        "digit_count": digit_counts,
        "is_correct": is_correct_flags,
        "benchmark": item_benchmarks,
        "model": item_models,
    }
)

benchmark_dummies = pd.get_dummies(df["benchmark"], prefix="bench", drop_first=True, dtype=float)
model_dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True, dtype=float)
intercept = pd.Series(1.0, index=df.index, name="intercept")

X_fe_binary = pd.concat([intercept, df["has_numbers"].astype(float), benchmark_dummies, model_dummies], axis=1)
beta_b, se_b, t_b, p_b = ols_fit(X_fe_binary.values, df["is_correct"].values)
idx_b = X_fe_binary.columns.get_loc("has_numbers")

X_fe_digits = pd.concat([intercept, df["digit_count"].astype(float), benchmark_dummies, model_dummies], axis=1)
beta_d, se_d, t_d, p_d = ols_fit(X_fe_digits.values, df["is_correct"].values)
idx_d = X_fe_digits.columns.get_loc("digit_count")

report_lines.append("")
report_lines.append("--- Isolated effect (controlling for benchmark + model fixed effects) ---")
report_lines.append(
    f"correct ~ has_numbers + benchmark + model: coef={beta_b[idx_b]:+.4f}, "
    f"se={se_b[idx_b]:.4f}, t={t_b[idx_b]:.3f}, p={p_b[idx_b]:.4g}"
)
report_lines.append(
    f"correct ~ digit_count + benchmark + model: coef={beta_d[idx_d]:+.5f}, "
    f"se={se_d[idx_d]:.5f}, t={t_d[idx_d]:.3f}, p={p_d[idx_d]:.4g}"
)

report_lines.append("")
report_lines.append("--- Per-benchmark accuracy: with vs without numbers ---")
benchmark_stats = []
for b in pd.unique(df["benchmark"]):
    sub = df[df["benchmark"] == b]
    with_mask = sub["has_numbers"] == 1
    without_mask = ~with_mask
    n_with, n_without = int(with_mask.sum()), int(without_mask.sum())
    if n_with == 0 or n_without == 0:
        report_lines.append(f"{b:30s}  skipped (only one group present, n_with={n_with}, n_without={n_without})")
        benchmark_stats.append(
            {"benchmark": b, "acc_with": None, "acc_without": None, "n_with": n_with, "n_without": n_without}
        )
        continue
    a_b = int(sub.loc[with_mask, "is_correct"].sum())
    c_b = int(sub.loc[without_mask, "is_correct"].sum())
    b_b = n_with - a_b
    d_b = n_without - c_b
    acc_with = a_b / n_with
    acc_without = c_b / n_without
    report_lines.append(
        f"{b:30s}  with={acc_with:5.1%} (n={n_with:4d})  "
        f"without={acc_without:5.1%} (n={n_without:4d})  diff={acc_with - acc_without:+.1%}"
    )
    benchmark_stats.append(
        {
            "benchmark": b,
            "acc_with": acc_with,
            "acc_without": acc_without,
            "n_with": n_with,
            "n_without": n_without,
            "a": a_b,
            "b": b_b,
            "c": c_b,
            "d": d_b,
        }
    )

# --- Benchmark-adjusted 2x2 table (Cochran-Mantel-Haenszel + direct standardization) ---
# Only benchmarks with both groups present carry information about the effect;
# SAT Math / Cornell Critical / IQ test (single-group strata) are dropped, same
# as they would be from a CMH test.

strata = [r for r in benchmark_stats if r["acc_with"] is not None]

cmh_num = 0.0
cmh_var = 0.0
mh_or_num = 0.0
mh_or_den = 0.0
for r in strata:
    a_b, b_b, c_b, d_b = r["a"], r["b"], r["c"], r["d"]
    n_b = a_b + b_b + c_b + d_b
    expected_a = (a_b + b_b) * (a_b + c_b) / n_b
    var_a = ((a_b + b_b) * (c_b + d_b) * (a_b + c_b) * (b_b + d_b)) / (n_b**2 * (n_b - 1))
    cmh_num += a_b - expected_a
    cmh_var += var_a
    mh_or_num += a_b * d_b / n_b
    mh_or_den += b_b * c_b / n_b

cmh_chi2 = cmh_num**2 / cmh_var
cmh_p = stats.chi2.sf(cmh_chi2, df=1)
mh_odds_ratio = mh_or_num / mh_or_den

strata_n = np.array([r["a"] + r["b"] + r["c"] + r["d"] for r in strata], dtype=float)
strata_weights = strata_n / strata_n.sum()
standardized_acc_with = sum(w * r["acc_with"] for w, r in zip(strata_weights, strata))
standardized_acc_without = sum(w * r["acc_without"] for w, r in zip(strata_weights, strata))

n_with_adj = sum(r["a"] + r["b"] for r in strata)
n_without_adj = sum(r["c"] + r["d"] for r in strata)
adj_correct_with = round(n_with_adj * standardized_acc_with)
adj_correct_without = round(n_without_adj * standardized_acc_without)

adjusted_contingency_table = np.array(
    [
        [adj_correct_with, n_with_adj - adj_correct_with],
        [adj_correct_without, n_without_adj - adj_correct_without],
    ]
)

report_lines.append("")
report_lines.append("--- Benchmark-adjusted 2x2 table (direct standardization + CMH test) ---")
report_lines.append(
    f"Strata used: {len(strata)} benchmarks with both groups present "
    f"(dropped {len(benchmark_stats) - len(strata)} single-group benchmarks)"
)
report_lines.append(
    f"Standardized accuracy: with numbers={standardized_acc_with:.2%}, "
    f"without numbers={standardized_acc_without:.2%}, "
    f"diff={standardized_acc_with - standardized_acc_without:+.2%}"
)
report_lines.append(f"Cochran-Mantel-Haenszel test: chi2={cmh_chi2:.3f}, p={cmh_p:.4g}")
report_lines.append(f"Mantel-Haenszel common odds ratio: {mh_odds_ratio:.3f}")

with open("number_effect_analysis.txt", "w") as outfile:
    outfile.write("\n".join(report_lines) + "\n")

benchmarks = list(dict.fromkeys(benchmark_name(file) for file in filenames))
models = list(dict.fromkeys(model_name(file) for file in filenames))

fig1 = plt.figure(1)
x = np.arange(len(benchmarks))
width = 0.8 / len(models)

for i, model in enumerate(models):
    heights = [
        next(
            error_data[file]["Number of error questions"]
            for file in filenames
            if benchmark_name(file) == benchmark and model_name(file) == model
        )
        for benchmark in benchmarks
    ]
    offset = (i - (len(models) - 1) / 2) * width
    plt.bar(x + offset, heights, width, label=model)

plt.xticks(x, benchmarks, rotation=90, ha="center")
plt.ylabel("Number of error questions")
plt.title("Parsing-error questions per benchmark and model")
plt.grid(axis="y")
plt.legend()
plt.tight_layout()

# --- Accuracy comparison plot (with/without numbers), with 95% CI error bars ---

se1 = np.sqrt(p1 * (1 - p1) / n1)
se2 = np.sqrt(p2 * (1 - p2) / n2)

fig2, (ax1, ax2) = plt.subplots(1, 2, num=2, figsize=(10, 5))

bars1 = ax1.bar(
    ["With numbers", "Without numbers"],
    [p1, p2],
    yerr=[1.96 * se1, 1.96 * se2],
    capsize=8,
    color=["tab:blue", "tab:orange"],
)
for bar, n, err in zip(bars1, [n1, n2], [1.96 * se1, 1.96 * se2]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        min(bar.get_height() + err + 0.02, 0.97),
        f"n={n}",
        ha="center",
        fontsize=8,
    )
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1)
ax1.grid(axis="y")
ax1.set_title(f"Accuracy by number presence\n(chi2 p={chi2_p:.3g}, z-test p={z_p:.3g})")

bin_edges = [0, 1, 3, 6, 11, 21, 41, 81, digit_count_arr.max() + 1]
bin_labels = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "81+"]
bin_idx = np.digitize(digit_count_arr, bin_edges[1:-1])
bin_means = [is_correct_arr[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan for i in range(len(bin_labels))]
bin_counts = [int(np.sum(bin_idx == i)) for i in range(len(bin_labels))]

ax2.bar(bin_labels, bin_means, color="tab:green")
for i, (mean, count) in enumerate(zip(bin_means, bin_counts)):
    if not np.isnan(mean):
        ax2.text(i, mean + 0.02, f"n={count}", ha="center", fontsize=7)
ax2.set_xlabel("Digit characters in prompt")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.grid(axis="y")
ax2.tick_params(axis="x", rotation=45)
ax2.set_title(f"Accuracy vs. amount of numeric content\n(linreg slope={lr_digits.slope:+.4f}, p={lr_digits.pvalue:.3g})")

plt.tight_layout()

# --- 2x2 contingency table plot ---

fig3, ax3 = plt.subplots(num=3, figsize=(6, 4.8))
im = ax3.imshow(contingency_table, cmap="Blues")
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["Correct", "Incorrect"])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["With numbers", "Without numbers"])
row_totals = contingency_table.sum(axis=1, keepdims=True)
for i in range(2):
    for j in range(2):
        count = contingency_table[i, j]
        share = count / row_totals[i, 0]
        text_color = "white" if count > contingency_table.max() / 2 else "black"
        ax3.text(j, i, f"{count}\n({share:.1%})", ha="center", va="center", color=text_color, fontsize=11)
ax3.set_title(f"2x2 contingency table (raw, unadjusted)\nchi2={chi2:.2f}, p={chi2_p:.3g}", fontsize=10)
fig3.colorbar(im, ax=ax3, label="Count")
plt.tight_layout()

# --- Benchmark-adjusted 2x2 table plot (direct standardization + CMH test) ---

fig5, ax5 = plt.subplots(num=5, figsize=(6, 4.8))
im5 = ax5.imshow(adjusted_contingency_table, cmap="Blues")
ax5.set_xticks([0, 1])
ax5.set_xticklabels(["Correct", "Incorrect"])
ax5.set_yticks([0, 1])
ax5.set_yticklabels(["With numbers", "Without numbers"])
row_totals_adj = adjusted_contingency_table.sum(axis=1, keepdims=True)
for i in range(2):
    for j in range(2):
        count = adjusted_contingency_table[i, j]
        share = count / row_totals_adj[i, 0]
        text_color = "white" if count > adjusted_contingency_table.max() / 2 else "black"
        ax5.text(j, i, f"{count}\n({share:.1%})", ha="center", va="center", color=text_color, fontsize=11)
ax5.set_title(
    f"2x2 contingency table (adjusted for benchmark)\nCMH chi2={cmh_chi2:.2f}, p={cmh_p:.3g}  "
    f"(n={len(strata)} strata)",
    fontsize=10,
)
fig5.colorbar(im5, ax=ax5, label="Standardized count")
plt.tight_layout()

# --- 2x2 contingency table per benchmark ---

n_cols6 = 4
n_rows6 = -(-len(strata) // n_cols6)
fig6, axes6 = plt.subplots(n_rows6, n_cols6, num=6, figsize=(4 * n_cols6, 3.6 * n_rows6))
axes6_flat = axes6.flatten()

for ax, r in zip(axes6_flat, strata):
    bench_table = np.array([[r["a"], r["b"]], [r["c"], r["d"]]])
    chi2_b, chi2_p_b, _, _ = stats.chi2_contingency(bench_table)
    _, fisher_p = stats.fisher_exact(bench_table)
    ax.imshow(bench_table, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Incorrect"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["With #", "Without #"], fontsize=8)
    row_totals_b = bench_table.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            count = bench_table[i, j]
            share = count / row_totals_b[i, 0]
            text_color = "white" if count > bench_table.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({share:.0%})", ha="center", va="center", color=text_color, fontsize=8)
    ax.set_title(
        f"{r['benchmark']}\nchi2={chi2_b:.2f}, p={chi2_p_b:.3g}\nFisher p={fisher_p:.3g}", fontsize=9
    )

for ax in axes6_flat[len(strata):]:
    ax.axis("off")

fig6.suptitle("2x2 contingency table per benchmark (with vs without numbers)", fontsize=13)
plt.tight_layout()

# --- Per-benchmark accuracy plot (with vs without numbers) ---

plot_rows = [r for r in benchmark_stats if r["acc_with"] is not None]
labels4 = [r["benchmark"] for r in plot_rows]
x4 = np.arange(len(labels4))
width4 = 0.35

acc_with_pct = [r["acc_with"] * 100 for r in plot_rows]
acc_without_pct = [r["acc_without"] * 100 for r in plot_rows]

fig4, ax4 = plt.subplots(num=4, figsize=(10, 5))
bars_with = ax4.bar(x4 - width4 / 2, acc_with_pct, width4, label="With numbers", color="tab:blue")
bars_without = ax4.bar(x4 + width4 / 2, acc_without_pct, width4, label="Without numbers", color="tab:orange")


# The bars in a pair sit flush against each other (no gap), so when one bar is
# shorter, a label placed just above its own top can land on top of the taller
# neighboring bar instead. Only the taller bar's label can safely sit right
# above its own top; the shorter bar's label needs to clear the taller bar too.
for bar_w, bar_wo, r in zip(bars_with, bars_without, plot_rows):
    h_w = bar_w.get_height()
    h_wo = bar_wo.get_height()
    tallest = max(h_w, h_wo)
    y_with = h_w + 1 if h_w >= h_wo else tallest + 4
    y_without = h_wo + 1 if h_wo >= h_w else tallest + 4
    ax4.text(bar_w.get_x() + bar_w.get_width() / 2, y_with, f"n={r['n_with']}", ha="center", fontsize=7)
    ax4.text(bar_wo.get_x() + bar_wo.get_width() / 2, y_without, f"n={r['n_without']}", ha="center", fontsize=7)

ax4.set_xticks(x4)
ax4.set_xticklabels(labels4, rotation=90, ha="center")
ax4.set_ylabel("Accuracy (%)")
ax4.set_ylim(0, 118)
ax4.grid(axis="y")
ax4.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
ax4.set_title("Per-benchmark accuracy: with vs without numbers", pad=28)
plt.tight_layout()

fig1.savefig("error_counts_per_benchmark.png", dpi=150)
fig2.savefig("accuracy_overview.png", dpi=150)
fig3.savefig("contingency_table_raw.png", dpi=150)
fig4.savefig("per_benchmark_accuracy.png", dpi=150)
fig5.savefig("contingency_table_adjusted.png", dpi=150)
fig6.savefig("contingency_table_per_benchmark.png", dpi=150)