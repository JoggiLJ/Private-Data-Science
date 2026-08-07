import json
import os
import glob
import re
import tiktoken
from collections import Counter
import math
import requests
import random
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
SAMPLE_SIZE = 200
RANDOM_SEED = 42

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

def sample_items(data: dict, sample_size: int = SAMPLE_SIZE, seed=RANDOM_SEED) -> dict:
    items = data.get("items", [])
    if len(items) > sample_size:
        rng = random.Random(seed)
        items = rng.sample(items, sample_size)
    return {**data, "items": items}

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"

#Running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
filenames = [f for f in glob.glob(os.path.join(SCRIPT_DIR, "*.json")) if not f.endswith("_results.json")]

question_sets = []
for file in filenames:
    with open(file) as infile:
        data = json.load(infile)
    if isinstance(data, list):
        data = {"items": data}
    if os.path.basename(file).startswith("sat-math-mc"):
        data = {**data, "items": [item for item in data.get("items", []) if item.get("transcribed_equation") is True]}
    question_sets.append((file, sample_items(data)))

total_items = sum(len(data.get("items", [])) for _, data in question_sets)
completed = 0
start_time = time.time()

for file, data in question_sets:
    file_data = {}
    items = data.get("items", [])
    score = 0
    total_in_set = 0
    for item in items:
        question_text = item.get("text", "")
        question_id = item.get("item_id", "")
        correct_label = item.get("correct_label", "")
        options = item.get("options", [])

        prompt = PROMPT_TEMPLATE.format(text=question_text, options=format_options(options))
        words, num_tokens, entropy = word_tok_entropy(prompt)
        model_answer = get_model_answer(prompt, options)
        is_correct = model_answer.upper() == correct_label.upper()

        if is_correct:
            score += 1

        total_in_set += 1

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
        print(f"[{completed}/{total_items}] {file} - elapsed {format_duration(elapsed)}, ETA {eta}")

    file_data["Accuracy"] = f"{score}/{total_in_set}, {score/total_in_set*100:.2f}%"

    basename = os.path.basename(file).removesuffix(".json")
    output_path = os.path.join(SCRIPT_DIR, f"{MODEL}_{basename}_results.json")
    with open(output_path, "w") as outfile:
        json.dump(file_data, outfile, indent=2)

    print(f"Done with set: {file}")