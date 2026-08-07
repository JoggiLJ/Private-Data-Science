"""
Benchmark a local Ollama model on the Astronomy Diagnostic Test (ADT) 2.0.

Answer-key caveat
------------------
ADT_astronomy_diagnostic_test.json ships with every "correct_label" set to
null: the official CAER answer key is restricted (available only on request
from deming@astro.umd.edu -- see the source note in that JSON and in
Question sets/03-astronomy-diagnostic-test-ADT.md). ANSWER_KEY below is NOT
that official key. It's derived independently here from the underlying
astronomy/physics fact each item is testing, with the reasoning recorded
inline, so the set can be scored at all. Treat accuracy numbers from this
script as measured against that derived key, not the CAER original.

Two items (18, 19) reference diagrams and ship with zero answer options, and
twelve items (22-33) are demographic/confidence survey questions with no
correct answer to begin with -- all 14 are excluded rather than guessed at,
leaving 19 scoreable content items.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download  (Mac: `brew install ollama`)
    2. Pull the model:   ollama pull qwen3:14b
       (Ollama serves it automatically at http://localhost:11434)

Usage:
    python adt-astronomy-benchmark.py --model qwen3:14b
    python adt-astronomy-benchmark.py --model qwen3:14b --host http://localhost:11434
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR.parent / "JSON_set" / "ADT_astronomy_diagnostic_test.json"

# item_id -> (correct_label, rationale). See the module docstring: this is a
# derived key, not the official restricted CAER one.
ANSWER_KEY = {
    1: ("E", "Overhead noon sun only occurs within the tropics; assumes a non-tropical (typical US) location."),
    2: ("B", "A solar eclipse (Moon covering Sun) can only happen at New Moon."),
    3: ("D", "Moon distance / Earth diameter = 384400/12742 = 30, so a 12in basketball scales to ~30ft."),
    4: ("B", "Galileo: without air resistance, all masses fall at the same rate."),
    5: ("B", "Radio waves and visible light are both EM radiation, same speed in vacuum."),
    6: ("B", "Astronauts float because they and the shuttle are in continuous free fall together, not zero gravity."),
    7: ("D", "Seasons are caused by axial tilt, not orbital distance; a circular orbit wouldn't change them."),
    8: ("A", "The Sun is powered by nuclear fusion: light elements fusing into heavier ones."),
    9: ("A", "After the Sept equinox the sunset point moves south (toward the winter solstice) in the N. hemisphere."),
    10: ("C", "Over a few hours the Sun's position against the background stars barely changes; still near Gemini at sunset."),
    11: ("A", "The Shuttle orbits only ~100-400 km up, negligible next to the Moon's ~384000 km."),
    12: ("B", "The Big Dipper's stars are light-years away; only traveling to another star would visibly change its shape."),
    13: ("C", "Distance order: Moon (~384000 km) < Sun (~150M km) < Pluto (~5.9B km) << stars (light-years)."),
    14: ("D", "Weight depends on Earth's mass (and radius); atmosphere, spin rate, and Sun distance don't set it."),
    15: ("D", "Inverse-square law: doubling distance drops intensity to 1/4, so 4 bulbs restore the original brightness."),
    16: ("E", "Modern cosmology: the universe has no center; it expands uniformly everywhere."),
    17: ("A", "By Wien's law the hottest stars radiate at the shortest wavelength, which is blue."),
    20: ("D", "The Sun's angular size falls 10x at Saturn's distance; a spaghetti strand is ~10x thinner than a thumb, the closest match among the options."),
    21: ("C", "Global warming is driven by CO2 trapping heat (the greenhouse effect), not nitrogen or ozone loss."),
}

EXCLUDED_ITEM_IDS = {18, 19}  # diagram-only items, ship with zero text options

PROMPT_TEMPLATE = """Answer this multiple-choice question. Respond with ONLY the single letter of the correct choice and nothing else.

{text}

{options}

Answer:"""

ENCODING = tiktoken.get_encoding("cl100k_base")


def entropy(text):
    """Shannon entropy of text, in bits/character."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def format_options(options):
    return "\n".join(f"{opt['label']}) {opt['text']}" for opt in options)


def load_items(path):
    data = json.load(open(path, encoding="utf-8"))
    items = []
    for it in data["items"]:
        if it["item_type"] != "content" or it["item_id"] in EXCLUDED_ITEM_IDS:
            continue
        correct_label, rationale = ANSWER_KEY[it["item_id"]]
        text = it["text"]
        items.append({
            "item_id": it["item_id"],
            "text": text,
            "options": it["options"],
            "correct_label": correct_label,
            "rationale": rationale,
            "words": len(text.split()),
            "tokens": len(ENCODING.encode(text)),
            "alternatives": len(it["options"]),
            "entropy": entropy(text),
        })
    items.sort(key=lambda it: it["item_id"])
    return items


def ask_ollama(host, model, prompt, timeout=120):
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "think": False,
              "options": {"temperature": 0.0}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def extract_letter(raw, valid_labels):
    matches = re.findall(r"\b([A-E])\b", raw.strip().upper())
    for m in reversed(matches):
        if m in valid_labels:
            return m
    return matches[-1] if matches else None


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_benchmark(args, items):
    results = []
    t0 = time.time()
    for idx, it in enumerate(items, 1):
        valid_labels = {o["label"] for o in it["options"]}
        prompt = PROMPT_TEMPLATE.format(text=it["text"], options=format_options(it["options"]))
        try:
            raw = ask_ollama(args.host, args.model, prompt)
        except requests.exceptions.RequestException as e:
            elapsed_so_far = time.time() - t0
            eta = elapsed_so_far / idx * (len(items) - idx)
            print(f"  [{it['item_id']:>2}] ERROR calling Ollama: {e}  (ETA {format_duration(eta)})")
            continue
        pred = extract_letter(raw, valid_labels)
        correct = pred == it["correct_label"]
        results.append({
            "item_id": it["item_id"], "predicted": pred, "correct_label": it["correct_label"],
            "correct": correct, "raw_response": raw.strip()[:200],
            "words": it["words"], "tokens": it["tokens"],
            "alternatives": it["alternatives"], "entropy": it["entropy"],
        })
        running_acc = sum(r["correct"] for r in results) / len(results)
        elapsed_so_far = time.time() - t0
        eta = elapsed_so_far / idx * (len(items) - idx)
        print(f"  [{it['item_id']:>2}/{items[-1]['item_id']}] pred={pred or '?':<1} correct={it['correct_label']} "
              f"{'OK' if correct else 'X'}  (running acc {running_acc:.1%}, ETA {format_duration(eta)})")
    return results


# ---------------------------------------------------------------------------
# Plotting helpers (accuracy vs. word count / token count / entropy /
# alternatives), matching the style used for the other item banks in
# Utvalgte set/Python/Plotting/Within sets/Plotting_withing_sets.py.
# ---------------------------------------------------------------------------

BAR_COLOR = "#2a78d6"
TOTAL_BAR_COLOR = "#b0b0b0"
RESULT_COLORS = {"Correct": "#0ca30c", "Incorrect": "#d03b3b"}
SIGNIFICANCE_COLOR = "#7a5cbf"
CI_Z = 1.96


def accuracy_by_even_bins(data, key, n_bins):
    values = [q[key] for q in data]
    lo, hi = min(values), max(values)
    if hi == lo:
        n_bins = 1
    edges = np.linspace(lo, hi, n_bins + 1)

    correct = [0] * n_bins
    total = [0] * n_bins
    for q in data:
        idx = 0 if hi == lo else min(int((q[key] - lo) / (hi - lo) * n_bins), n_bins - 1)
        total[idx] += 1
        correct[idx] += q["Result"] == "Correct"

    decimals = 2 if hi == lo else max(0, -int(np.floor(np.log10((hi - lo) / n_bins))))
    labels = [f"{edges[i]:.{decimals}f}-{edges[i + 1]:.{decimals}f}" for i in range(n_bins)]
    accuracies = [100 * c / t if t else float("nan") for c, t in zip(correct, total)]
    return labels, accuracies, total


def accuracy_by_exact_value(data, key):
    values = sorted(set(q[key] for q in data))
    labels, accuracies, totals = [], [], []
    for v in values:
        subset = [q for q in data if q[key] == v]
        correct = sum(q["Result"] == "Correct" for q in subset)
        labels.append(str(int(v)))
        accuracies.append(100 * correct / len(subset))
        totals.append(len(subset))
    return labels, accuracies, totals


def plot_accuracy_bars(ax, labels, accuracies, totals, xlabel, title):
    x = np.arange(len(labels))
    width = 0.35

    bars = ax.bar(x - width / 2, accuracies, width, color=BAR_COLOR, label="Accuracy")
    for bar, n in zip(bars, totals):
        if n == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"n={n}",
                 ha="center", va="bottom", fontsize=8, color="#666666", rotation=90)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, totals, width, color=TOTAL_BAR_COLOR, label="Total")
    ax2.set_ylabel("Total questions")
    ax2.set_ylim(0, max(totals) * 1.4 if max(totals) else 1)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, pad=28)
    ax.set_ylim(0, 122)
    ax.grid(axis="y", color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")

    bars_h, bars_l = ax.get_legend_handles_labels()
    bars2_h, bars2_l = ax2.get_legend_handles_labels()
    ax.legend(bars_h + bars2_h, bars_l + bars2_l, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)


def plot_scatter_by_result(ax, data, xkey, ykey, xlabel, ylabel, title):
    for result, color in RESULT_COLORS.items():
        subset = [q for q in data if q["Result"] == result]
        counts = Counter((q[xkey], q[ykey]) for q in subset)
        points = list(counts.items())
        xs = [xy[0] for xy, _ in points]
        ys = [xy[1] for xy, _ in points]
        sizes = [14 * np.sqrt(n) for _, n in points]
        ax.scatter(xs, ys, s=sizes, color=color, marker="o", alpha=0.6, edgecolors="none")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)

    result_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none",
               markersize=8, label=f"{result} (n={sum(q['Result'] == result for q in data)})")
        for result, color in RESULT_COLORS.items()
    ]
    ax.legend(handles=result_handles, frameon=False, loc="upper left", title="Result")


def plot_scatter_score(ax, data, xkey, xlabel, title, seed=0):
    xs = [q[xkey] for q in data]
    scores = np.array([1 if q["Result"] == "Correct" else 0 for q in data], dtype=float)
    jitter = np.random.default_rng(seed).uniform(-0.05, 0.05, size=len(scores))

    ax.scatter(xs, scores + jitter, s=14, color=BAR_COLOR, alpha=0.4, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.15, 1.15)
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)


def plot_correct_incorrect_ratio(ax, data, xkey, xlabel, title, n_points=200, min_count=0.5):
    xs_all = np.array([q[xkey] for q in data], dtype=float)
    lo, hi = xs_all.min(), xs_all.max()
    pad = (hi - lo) * 0.05 or 1.0
    grid = np.linspace(lo - pad, hi + pad, n_points)

    n_all = len(xs_all)
    shared_bw = 1.06 * xs_all.std(ddof=1) * n_all ** (-1 / 5) if n_all > 1 else 1.0

    counts = {}
    for result in ("Correct", "Incorrect"):
        xs = np.array([q[xkey] for q in data if q["Result"] == result], dtype=float)
        if len(xs) < 2 or np.ptp(xs) == 0:
            counts[result] = None
            continue
        xs_std = xs.std(ddof=1)
        bw_factor = shared_bw / xs_std if xs_std > 0 else 1.0
        counts[result] = gaussian_kde(xs, bw_method=bw_factor)(grid) * len(xs)

    if counts["Correct"] is None or counts["Incorrect"] is None:
        ax.set_title(title + "\n(not enough data in one class)")
        return

    ratio = np.full_like(grid, np.nan)
    valid = counts["Incorrect"] > min_count
    ratio[valid] = counts["Correct"][valid] / counts["Incorrect"][valid]

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, label="Equal (1:1)")
    ax.plot(grid, ratio, color=BAR_COLOR)
    ax.fill_between(grid, ratio, color=BAR_COLOR, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correct : Incorrect ratio")
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)


def bootstrap_kde_ci(xs, grid, n_boot=150, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(xs)
    boot = np.full((n_boot, grid.size), np.nan)
    for i in range(n_boot):
        sample = xs[rng.integers(0, n, size=n)]
        if np.ptp(sample) == 0:
            continue
        boot[i] = gaussian_kde(sample)(grid)
    lo_pct, hi_pct = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return np.nanpercentile(boot, lo_pct, axis=0), np.nanpercentile(boot, hi_pct, axis=0)


def plot_density_by_result(ax, data, xkey, xlabel, title, n_points=200):
    xs_all = np.array([q[xkey] for q in data], dtype=float)
    lo, hi = xs_all.min(), xs_all.max()
    pad = (hi - lo) * 0.05 or 1.0
    grid = np.linspace(lo - pad, hi + pad, n_points)

    bands = {}
    for result, color in RESULT_COLORS.items():
        xs = np.array([q[xkey] for q in data if q["Result"] == result], dtype=float)
        if len(xs) < 2 or np.ptp(xs) == 0:
            continue
        density = gaussian_kde(xs)(grid)
        ci_lo, ci_hi = bootstrap_kde_ci(xs, grid)
        bands[result] = (ci_lo, ci_hi)
        ax.plot(grid, density, color=color, label=f"{result} (n={len(xs)})")
        ax.fill_between(grid, density, color=color, alpha=0.15)
        ax.fill_between(grid, ci_lo, ci_hi, color=color, alpha=0.12, linewidth=0)

    if len(bands) == 2:
        (lo_a, hi_a), (lo_b, hi_b) = bands.values()
        significant = (lo_a > hi_b) | (lo_b > hi_a)
        runs = np.flatnonzero(np.diff(np.concatenate(([0], significant.astype(int), [0]))))
        for i, (s, e) in enumerate(zip(runs[0::2], runs[1::2])):
            ax.axvspan(grid[s], grid[e - 1], color=SIGNIFICANCE_COLOR, alpha=0.15,
                       zorder=0, label="p < 0.05 (bootstrap)" if i == 0 else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)


def plot_smoothed_accuracy(ax, data, xkey, xlabel, title, n_points=200):
    xs = np.array([q[xkey] for q in data], dtype=float)
    ys = np.array([1.0 if q["Result"] == "Correct" else 0.0 for q in data])
    lo, hi = xs.min(), xs.max()
    pad = (hi - lo) * 0.05 or 1.0
    grid = np.linspace(lo - pad, hi + pad, n_points)

    n = len(xs)
    std = xs.std(ddof=1) if n > 1 else 0.0
    bw = 1.06 * std * n ** (-1 / 5) if std > 0 else 1.0

    weights = np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bw) ** 2)
    weight_sums = weights.sum(axis=1)
    nonzero = weight_sums > 0
    p = np.full_like(weight_sums, np.nan)
    p[nonzero] = (weights[nonzero] * ys).sum(axis=1) / weight_sums[nonzero]
    accuracy = p * 100

    weight_sq_sums = (weights ** 2).sum(axis=1)
    ess = np.full_like(weight_sums, np.nan)
    ess[nonzero] = weight_sums[nonzero] ** 2 / weight_sq_sums[nonzero]

    se = np.sqrt(p * (1 - p) / ess)
    ci_lo = np.clip(accuracy - CI_Z * se * 100, 0, 100)
    ci_hi = np.clip(accuracy + CI_Z * se * 100, 0, 100)

    baseline = ys.mean() * 100
    significant = (ci_lo > baseline) | (ci_hi < baseline)
    significant = np.nan_to_num(significant)
    runs = np.flatnonzero(np.diff(np.concatenate(([0], significant.astype(int), [0]))))
    for i, (s, e) in enumerate(zip(runs[0::2], runs[1::2])):
        ax.axvspan(grid[s], grid[e - 1], color=SIGNIFICANCE_COLOR, alpha=0.15,
                   zorder=0, label="p < 0.05 (differs from overall)" if i == 0 else None)

    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1,
               label=f"Overall accuracy ({baseline:.1f}%)")
    ax.plot(grid, accuracy, color=BAR_COLOR)
    ax.fill_between(grid, ci_lo, ci_hi, color=BAR_COLOR, alpha=0.15)
    ax.plot(xs, np.full_like(xs, -3), "|", color=BAR_COLOR, alpha=0.5, clip_on=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


METRIC_KEYS = {"Words": "words", "Tokens": "tokens", "Alternatives": "alternatives", "Entropy": "entropy"}
METRIC_XLABELS = {
    "Words": "Word count", "Tokens": "Token count",
    "Alternatives": "Number of alternatives", "Entropy": "Entropy [bits/character]",
}
GRID_COLS = 4


def make_plots(args, model, results, out_dir):
    scored = [
        {"Words": r["words"], "Tokens": r["tokens"], "Alternatives": r["alternatives"],
         "Entropy": r["entropy"], "Result": "Correct" if r["correct"] else "Incorrect"}
        for r in results if r["predicted"] is not None
    ]
    if not scored:
        print("No scoreable results (model returned no parseable answers) -- skipping metric plots.")
        return

    overall = sum(q["Result"] == "Correct" for q in scored) / len(scored)

    # Overall accuracy bar, matching the other single-number benchmark charts
    # in this repo (e.g. IQ-test/Benchmark/iq-benchmark.py).
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar([model], [overall], color="#4caf50" if overall >= 0.2 else "#f44336")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="Chance (~20%, mostly 5-option items)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"ADT astronomy accuracy -- {model}\n{overall:.1%} ({sum(q['Result'] == 'Correct' for q in scored)}/{len(scored)}, derived key)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    n_bins = min(5, len({q["Words"] for q in scored}))
    metrics = ["Words", "Tokens", "Alternatives", "Entropy"]
    if len({q["Alternatives"] for q in scored}) <= 1:
        metrics.remove("Alternatives")
    n_metrics = len(metrics)
    combined_rows = []

    # --- accuracy bar chart per metric ---
    fig1, bar_axes = plt.subplots(1, n_metrics, figsize=(6.25 * n_metrics, 6), constrained_layout=True)
    bar_axes = np.atleast_1d(bar_axes)
    bar_row = []
    for ax, metric in zip(bar_axes, metrics):
        if metric == "Alternatives":
            labels, acc, totals = accuracy_by_exact_value(scored, metric)
        else:
            labels, acc, totals = accuracy_by_even_bins(scored, metric, n_bins)
        title = f"Accuracy vs. {metric}"
        plot_accuracy_bars(ax, labels, acc, totals, METRIC_XLABELS[metric], title)
        bar_row.append((plot_accuracy_bars, (labels, acc, totals, METRIC_XLABELS[metric], title), 1))
    fig1.suptitle(f"ADT astronomy -- {model} (n={len(scored)})")
    fig1.savefig(out_dir / "accuracy_bar_charts.png", dpi=150)
    plt.close(fig1)
    combined_rows.append(bar_row)

    # --- words vs tokens scatter ---
    fig2, scatter_ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plot_scatter_by_result(scatter_ax, scored, "Words", "Tokens", "Word count", "Token count",
                            "Every Question: Words vs. Tokens")
    fig2.savefig(out_dir / "words_vs_tokens_scatter.png", dpi=150)
    plt.close(fig2)
    combined_rows.append([(plot_scatter_by_result,
                            (scored, "Words", "Tokens", "Word count", "Token count",
                             "Every Question: Words vs. Tokens"), GRID_COLS)])

    # --- score vs metric scatter ---
    fig3, score_axes = plt.subplots(1, n_metrics, figsize=(6.25 * n_metrics, 6), constrained_layout=True)
    score_axes = np.atleast_1d(score_axes)
    score_row = []
    for ax, metric in zip(score_axes, metrics):
        title = f"Score vs. {metric}"
        plot_scatter_score(ax, scored, metric, METRIC_XLABELS[metric], title)
        score_row.append((plot_scatter_score, (scored, metric, METRIC_XLABELS[metric], title), 1))
    fig3.savefig(out_dir / "score_vs_metric_scatter.png", dpi=150)
    plt.close(fig3)
    combined_rows.append(score_row)

    # --- smoothed correct:incorrect ratio ---
    fig3a, ratio_axes = plt.subplots(1, n_metrics, figsize=(6.25 * n_metrics, 6), constrained_layout=True)
    ratio_axes = np.atleast_1d(ratio_axes)
    ratio_row = []
    for ax, metric in zip(ratio_axes, metrics):
        title = f"Correct:Incorrect Ratio vs. {metric}"
        plot_correct_incorrect_ratio(ax, scored, metric, METRIC_XLABELS[metric], title)
        ratio_row.append((plot_correct_incorrect_ratio, (scored, metric, METRIC_XLABELS[metric], title), 1))
    fig3a.savefig(out_dir / "correct_incorrect_ratio.png", dpi=150)
    plt.close(fig3a)
    combined_rows.append(ratio_row)

    # --- KDE density by result ---
    fig3b, density_axes = plt.subplots(1, n_metrics, figsize=(6.25 * n_metrics, 6), constrained_layout=True)
    density_axes = np.atleast_1d(density_axes)
    density_row = []
    for ax, metric in zip(density_axes, metrics):
        title = f"Score Density vs. {metric}"
        plot_density_by_result(ax, scored, metric, METRIC_XLABELS[metric], title)
        density_row.append((plot_density_by_result, (scored, metric, METRIC_XLABELS[metric], title), 1))
    fig3b.savefig(out_dir / "score_density_by_result.png", dpi=150)
    plt.close(fig3b)
    combined_rows.append(density_row)

    # --- smoothed accuracy curve ---
    fig3c, smoothed_axes = plt.subplots(1, n_metrics, figsize=(6.25 * n_metrics, 6), constrained_layout=True)
    smoothed_axes = np.atleast_1d(smoothed_axes)
    smoothed_row = []
    for ax, metric in zip(smoothed_axes, metrics):
        title = f"Smoothed Accuracy vs. {metric}"
        plot_smoothed_accuracy(ax, scored, metric, METRIC_XLABELS[metric], title)
        smoothed_row.append((plot_smoothed_accuracy, (scored, metric, METRIC_XLABELS[metric], title), 1))
    fig3c.savefig(out_dir / "smoothed_accuracy.png", dpi=150)
    plt.close(fig3c)
    combined_rows.append(smoothed_row)

    # --- everything replayed onto one combined figure ---
    fig_all = plt.figure(figsize=(25, 6 * len(combined_rows)), constrained_layout=True)
    grid = fig_all.add_gridspec(len(combined_rows), GRID_COLS)
    for row_idx, row in enumerate(combined_rows):
        col = 0
        for func, fargs, span in row:
            ax = fig_all.add_subplot(grid[row_idx, col:col + span])
            func(ax, *fargs)
            col += span
    fig_all.suptitle(f"ADT astronomy -- {model} (n={len(scored)})")
    fig_all.savefig(out_dir / "all_plots_combined.png", dpi=150)
    plt.close(fig_all)

    print(f"Saved plots -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark a local Ollama model on the ADT astronomy diagnostic test.")
    ap.add_argument("--model", default="qwen3:14b", help="Ollama model tag")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="Path to ADT_astronomy_diagnostic_test.json")
    args = ap.parse_args()

    items = load_items(args.json)
    print(f"Benchmarking {args.model} on {len(items)} scoreable ADT content items "
          f"(excluded: 2 diagram-only items, 12 demographic items)")
    print("NOTE: scored against a derived answer key, not the official restricted CAER key -- see script docstring.\n")

    results = run_benchmark(args, items)

    tag = args.model.replace(":", "-").replace("/", "-")
    scored_n = sum(1 for r in results if r["predicted"] is not None)
    correct_n = sum(r["correct"] for r in results)

    out_json = SCRIPT_DIR / f"{tag}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "n": len(results), "scored_n": scored_n, "correct_n": correct_n,
            "answer_key_note": "Derived key, not the official restricted CAER key -- see script docstring.",
            "results": results,
        }, f, indent=1)
    print(f"\nSaved raw results -> {out_json}")

    overall = correct_n / scored_n if scored_n else 0
    print(f"Overall accuracy: {overall:.1%} ({correct_n}/{scored_n})")

    make_plots(args, args.model, results, SCRIPT_DIR)


if __name__ == "__main__":
    main()
