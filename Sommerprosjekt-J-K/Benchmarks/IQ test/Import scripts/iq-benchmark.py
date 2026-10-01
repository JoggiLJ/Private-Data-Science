"""
Benchmark a local Ollama model against the Open-Source Psychometrics Project
IQ test (a Raven's-Progressive-Matrices-style visual matrix-reasoning test).

The 25 items are visual 3x3 grids (one panel missing, 8 lettered candidates
A-H). Since the target model is a plain text model, each item is fed in as a
structured text description instead of an image -- using the symbol data
already extracted by psychometrics_IQ_test_converter.py into all_results.json
(suit / placement / position per panel). See that file's "how_to_solve" and
"legend" blocks for the notation this script's prompt is built from.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download  (Mac: `brew install ollama`)
    2. Pull the model:   ollama pull qwen3:14b
       (Ollama serves it automatically at http://localhost:11434)

Usage:
    python ollama-iq-benchmark.py --model qwen3:14b
    python ollama-iq-benchmark.py --model qwen3:14b --host http://localhost:11434

Notes:
    - The 25-item score -> IQ conversion table on the last page of
      intelligence-quotient-test.pdf is only valid for the full 25-item set.
      Running with --n < 25 still reports accuracy but skips the IQ estimate.
    - qwen3 is a hybrid-thinking model; thinking is disabled via the
      Ollama "think": false request field so the response is just the
      answer letter.
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR.parent / "Open-Source Psychometrics Project" / "all_results.json"

# Answer key from the "Scoring" page of intelligence-quotient-test.pdf.
ANSWER_KEY = {
    1: "H", 2: "A", 3: "C", 4: "E", 5: "F", 6: "D", 7: "B", 8: "H", 9: "C", 10: "D",
    11: "A", 12: "H", 13: "E", 14: "A", 15: "F", 16: "H", 17: "A", 18: "G", 19: "F", 20: "B",
    21: "F", 22: "B", 23: "D", 24: "F", 25: "H",
}

# Score -> IQ conversion table from the same page. Ends are open intervals
# (score<=5 -> IQ<=73, score>=22 -> IQ>=139), so score_to_iq() returns both
# a comparison operator and the boundary value rather than a single number.
SCORE_TO_IQ = {
    6: 77, 7: 79, 8: 84, 9: 88, 10: 92, 11: 94, 12: 98, 13: 101, 14: 104,
    15: 108, 16: 111, 17: 114, 18: 119, 19: 123, 20: 125, 21: 132,
}

PROMPT_TEMPLATE = """This is one item from a Raven's-Progressive-Matrices-style visual IQ test, \
encoded as structured symbol data instead of an image.

{reference}

Grid (panel 9 is the one to solve for):
[1] {p1}
[2] {p2}
[3] {p3}
[4] {p4}
[5] {p5}
[6] {p6}
[7] {p7}
[8] {p8}
[9] ?

Candidates:
A) {a}
B) {b}
C) {c}
D) {d}
E) {e}
F) {f}
G) {g}
H) {h}

Respond with ONLY the single letter (A-H) of the candidate that completes the pattern, and nothing else.

Answer:"""


def panel_str(panel):
    if not panel["symbols"]:
        return "empty"
    return ", ".join(f"{s['suit']}@{s['placement']}{s['position']}" for s in panel["symbols"])


def load_items(data):
    items = []
    for filename, result in data["results"].items():
        n = int(re.search(r"(\d+)", filename).group(1))
        qg = result["question_grid"]
        ag = result["answer_grid"]
        items.append({
            "n": n,
            "panels": {i: panel_str(qg[str(i)]) for i in range(1, 9)},
            "choices": {letter: panel_str(ag[letter]) for letter in "ABCDEFGH"},
            "correct_label": ANSWER_KEY[n],
        })
    items.sort(key=lambda it: it["n"])
    return items


def build_reference_text(data):
    """Render all_results.json's how_to_solve + legend blocks (minus the
    batch-style answer_format_requirement, which doesn't apply to this
    script's one-item-at-a-time prompting) as prompt-ready text."""
    hts = data["how_to_solve"]
    leg = data["legend"]
    lines = [
        "How to solve this type of item:",
        hts["overview"],
        "",
        "Grid layout: " + hts["grid_layout"],
        "",
        "Reading a panel: " + hts["reading_a_panel"],
        "",
        "Solving procedure:",
    ]
    lines += [f"  {step}" for step in hts["solving_procedure"]]
    lines += [
        "",
        "Legend:",
        "  question_grid: " + leg["top_level"]["question_grid"],
        "  answer_grid: " + leg["top_level"]["answer_grid"],
        "  count: " + leg["panel_fields"]["count"],
        "  symbols: " + leg["panel_fields"]["symbols"],
        "  suit: " + leg["suit"],
        "  placement:",
        "    vertex: " + leg["placement"]["vertex"],
        "    cell: " + leg["placement"]["cell"],
        "    unknown: " + leg["placement"]["unknown"],
        "  position:",
        "    note: " + leg["position"]["note"],
        "    vertex: " + leg["position"]["vertex"],
        "    cell: " + leg["position"]["cell"],
    ]
    return "\n".join(lines)


def ask_ollama(host, model, prompt, timeout=180):
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "think": False,
              "options": {"temperature": 0.0}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def extract_letter(raw):
    matches = re.findall(r'\b([A-H])\b', raw.strip().upper())
    return matches[-1] if matches else None


def score_to_iq(score):
    if score <= 5:
        return "<=", 73
    if score >= 22:
        return ">=", 139
    return "=", SCORE_TO_IQ[score]


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser(description="Benchmark a local Ollama model on the Open-Source "
                                              "Psychometrics Project IQ test.")
    ap.add_argument("--model", default="qwen3:14b", help="Ollama model tag, e.g. qwen3:14b")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="Path to all_results.json")
    ap.add_argument("--n", type=int, default=0,
                     help="Number of items to run, in order (0 = all 25). Running fewer than "
                          "25 skips the official IQ estimate, which is only valid for the full test.")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    items = load_items(data)
    reference = build_reference_text(data)
    if args.n:
        items = items[:args.n]
    print(f"Benchmarking {args.model} on {len(items)}/25 IQ-test items (chance = 12.5%)")

    results = []
    t0 = time.time()
    for idx, it in enumerate(items, 1):
        prompt = PROMPT_TEMPLATE.format(
            reference=reference,
            p1=it["panels"][1], p2=it["panels"][2], p3=it["panels"][3],
            p4=it["panels"][4], p5=it["panels"][5], p6=it["panels"][6],
            p7=it["panels"][7], p8=it["panels"][8],
            a=it["choices"]["A"], b=it["choices"]["B"], c=it["choices"]["C"], d=it["choices"]["D"],
            e=it["choices"]["E"], f=it["choices"]["F"], g=it["choices"]["G"], h=it["choices"]["H"],
        )
        try:
            raw = ask_ollama(args.host, args.model, prompt)
        except requests.exceptions.RequestException as e:
            elapsed_so_far = time.time() - t0
            eta = elapsed_so_far / idx * (len(items) - idx)
            print(f"  [{it['n']:>2}/25] ERROR calling Ollama: {e}  (ETA {format_duration(eta)})")
            continue
        pred = extract_letter(raw)
        correct = pred is not None and pred.upper() == it["correct_label"].upper()
        results.append({
            "question": it["n"], "predicted": pred, "correct_label": it["correct_label"],
            "correct": correct, "raw_response": raw.strip()[:200],
        })
        running_acc = sum(r["correct"] for r in results) / len(results)
        elapsed_so_far = time.time() - t0
        eta = elapsed_so_far / idx * (len(items) - idx)
        print(f"  [{it['n']:>2}/25] pred={pred or '?':<1} correct={it['correct_label']} "
              f"{'OK' if correct else 'X'}  (running acc {running_acc:.1%}, ETA {format_duration(eta)})")

    elapsed = time.time() - t0
    tag = args.model.replace(":", "-").replace("/", "-")

    score = sum(r["correct"] for r in results)
    n = len(results)

    out_json = SCRIPT_DIR / f"{tag}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n": n, "score": score, "elapsed_sec": elapsed,
                   "results": results}, f, indent=1)
    print(f"\nSaved raw results -> {out_json}")

    overall = score / n if n else 0
    print(f"\nOverall accuracy: {overall:.1%} ({score}/{n})")
    print(f"Raw score: {score}/25" if n == 25 else f"Raw score: {score}/{n} (partial run, not out of the full 25)")

    if n == 25:
        op, iq = score_to_iq(score)
        print(f"Estimated IQ: {op}{iq}  (per the score->IQ table in intelligence-quotient-test.pdf)")
    else:
        print("Skipping IQ estimate: the score->IQ table only applies to the full 25-item test.")

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar([args.model], [overall], color="#4caf50" if overall >= 0.125 else "#f44336")
    ax.axhline(0.125, color="gray", linestyle="--", linewidth=1, label="Chance (12.5%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    title = f"IQ test accuracy -- {args.model}\n{overall:.1%} accuracy"
    if n == 25:
        op, iq = score_to_iq(score)
        title += f" (score {score}/25, IQ {op}{iq})"
    else:
        title += f" (score {score}/{n})"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig_path = SCRIPT_DIR / f"{tag}_accuracy.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart -> {fig_path}")


if __name__ == "__main__":
    main()
