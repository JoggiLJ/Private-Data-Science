import os
import json
import tiktoken

folder = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(folder, "..")

question_sets = {
    "GMAT": os.path.join(results_folder, "GMAT_results", "gmat_items.json"),
    "ARC": os.path.join(results_folder, "ARC_AI2_results", "ARC_AI2_ARC-Challenge_test.json"),
}

encoding = tiktoken.get_encoding("cl100k_base")

for name, json_path in question_sets.items():
    with open(json_path, "r") as infile:
        data = json.load(infile)

    items = data["items"] if isinstance(data, dict) else data

    token_counts = [len(encoding.encode(item.get("text", ""))) for item in items]
    avg_token_count = sum(token_counts) / len(token_counts)

    print(f"Average token count for {name} items ({len(items)} questions): {avg_token_count}")
