# MMLU-Pro

**Source:** `TIGER-Lab/MMLU-Pro` on HuggingFace.

**What it tests:** A harder, cleaned-up successor to MMLU — same general-knowledge spirit across many subjects, but questions were filtered to remove ones solvable without real reasoning, and expanded from 4 to up to 10 answer options per question (to reduce the chance of guessing correctly). Built specifically because the original MMLU had become too easy (near-saturated) for frontier models.

**Format:** Multiple-choice, up to 10 options per question, single split covering ~12k questions.

**What's in this folder:** `MMLU Pro original set/` and `JSON_set/` (converted questions), plus `MMLU_pro_importer.py`.
