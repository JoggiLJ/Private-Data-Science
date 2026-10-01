import json
import os
import re
import tiktoken
from collections import Counter
import math
import requests
import time

#Metadata
encoding = tiktoken.get_encoding("cl100k_base")

def word_tok_entropy(text):
    #Words
    words = len(text.split())
    #Tokens
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    #Entropy
    counts = Counter(tokens)
    entropy = -sum((c / num_tokens) * math.log2(c / num_tokens) for c in counts.values())
    return words, num_tokens, entropy

#Benchmarking
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_OPTIONS = ["qwen3:4b-instruct", "qwen3:14b"]
print("Choose a model:")
for i, name in enumerate(MODEL_OPTIONS, start=1):
    print(f"{i}) {name}")
choice = input(f"Enter 1 or 2: ")
MODEL = MODEL_OPTIONS[int(choice) - 1]

# Answer key from the "Scoring" page of intelligence-quotient-test.pdf.
# all_results.json only stores the question/answer panels, not the correct
# label, so this table is required to score the test at all.
ANSWER_KEY = {
    1: "H", 2: "A", 3: "C", 4: "E", 5: "F", 6: "D", 7: "B", 8: "H", 9: "C", 10: "D",
    11: "A", 12: "H", 13: "E", 14: "A", 15: "F", 16: "H", 17: "A", 18: "G", 19: "F", 20: "B",
    21: "F", 22: "B", 23: "D", 24: "F", 25: "H",
}

# Score -> IQ conversion table from the same page. Ends are open intervals
# (score<=5 -> IQ<=73, score>=22 -> IQ>=139).
SCORE_TO_IQ = {
    6: 77, 7: 79, 8: 84, 9: 88, 10: 92, 11: 94, 12: 98, 13: 101, 14: 104,
    15: 108, 16: 111, 17: 114, 18: 119, 19: 123, 20: 125, 21: 132,
}

PROMPT_TEMPLATE = """
{text}

{options}

Answer this multiple choice question. Respond with ONLY the single letter of the correct choice and nothing else.

Answer:
"""

def format_options(options: list) -> str:
    return "\n".join(f"{opt.get('label')}) {opt.get('text')}" for opt in options)

def prompt_model(prompt: str, model: str = MODEL, num_predict: int = 5) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
            },
        },
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

def extract_letter(response: str, options: list) -> str:
    valid_labels = {opt.get("label", "").upper() for opt in options if opt.get("label")}
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    for letter in reversed(re.findall(r"\b([A-Za-z])\b", cleaned)):
        if letter.upper() in valid_labels:
            return letter.upper()
    return cleaned

def get_model_answer(prompt: str, options: list, model: str = MODEL) -> str:
    for num_predict in (5, 200):
        raw = prompt_model(prompt, model=model, num_predict=num_predict)
        letter = extract_letter(raw, options)
        if letter.upper() in {opt.get("label", "").upper() for opt in options}:
            return letter
    return letter

def score_to_iq(score: int):
    if score <= 5:
        return "<=", 73
    if score >= 22:
        return ">=", 139
    return "=", SCORE_TO_IQ[score]

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"

def panel_str(panel: dict) -> str:
    if not panel["symbols"]:
        return "empty"
    return ", ".join(f"{s['suit']}@{s['placement']}{s['position']}" for s in panel["symbols"])

def build_reference_text(data: dict) -> str:
    """Render all_results.json's how_to_solve + legend blocks into prompt-ready text."""
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

def load_psychometrics_items(data: dict) -> list:
    """Turn all_results.json's visual grid entries into the standard
    {item_id, text, options, correct_label} shape the rest of this script
    (and the Results/ folder convention) expects."""
    reference = build_reference_text(data)
    items = []
    for filename, result in data["results"].items():
        n = int(re.search(r"(\d+)", filename).group(1))
        qg = result["question_grid"]
        ag = result["answer_grid"]
        grid_lines = "\n".join(f"[{i}] {panel_str(qg[str(i)])}" for i in range(1, 9))
        text = (
            f"{reference}\n\n"
            f"Grid (panel 9 is the one to solve for):\n{grid_lines}\n[9] ?"
        )
        options = [{"label": letter, "text": panel_str(ag[letter])} for letter in "ABCDEFGH"]
        items.append({
            "item_id": filename,
            "text": text,
            "options": options,
            "correct_label": ANSWER_KEY[n],
        })
    items.sort(key=lambda it: it["item_id"])
    return items

#Running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PSYCHOMETRICS_JSON = os.path.join(SCRIPT_DIR, "IQ-test", "Open-Source Psychometrics Project", "all_results.json")

with open(PSYCHOMETRICS_JSON) as infile:
    raw_data = json.load(infile)
items = load_psychometrics_items(raw_data)

total_items = len(items)
completed = 0
start_time = time.time()

file_data = {}
score = 0

for item in items:
    question_text = item["text"]
    question_id = item["item_id"]
    correct_label = item["correct_label"]
    options = item["options"]
    words, num_tokens, entropy = word_tok_entropy(question_text)

    prompt = PROMPT_TEMPLATE.format(text=question_text, options=format_options(options))
    model_answer = get_model_answer(prompt, options)
    is_correct = model_answer.upper() == correct_label.upper()

    if is_correct:
        score += 1

    file_data[question_id] = {
        "Prompt": prompt,
        "Model answer": model_answer,
        "Correct label": correct_label,
        "Result": "Correct" if is_correct else "Incorrect",
        "Words": words,
        "Tokens": num_tokens,
        "Entropy": f"{entropy:.2f} [bits/token]"
    }
    completed += 1
    elapsed = time.time() - start_time
    eta = format_duration((elapsed / completed) * (total_items - completed))
    print(f"[{completed}/{total_items}] {PSYCHOMETRICS_JSON} - elapsed {format_duration(elapsed)}, ETA {eta}")

file_data["Accuracy"] = f"{score}/{total_items}, {score/total_items*100:.2f}%"
if total_items == 25:
    op, iq = score_to_iq(score)
    file_data["Estimated IQ"] = f"{op}{iq}"

output_path = os.path.join(os.path.dirname(PSYCHOMETRICS_JSON), f"{MODEL}_all_results_results.json")
with open(output_path, "w") as outfile:
    json.dump(file_data, outfile, indent=2)

print(f"Done with set: {PSYCHOMETRICS_JSON}")
