# TruthfulQA

**Source:** `truthfulqa/truthful_qa`, "multiple_choice" config (Lin, Hilton & Evans, 2021).

**What it tests:** Whether a model gives truthful answers to questions that humans often answer wrong due to common misconceptions, folklore, or false beliefs (e.g. myths, misquotations) — measuring susceptibility to generating plausible-sounding but false answers ("imitative falsehoods") rather than general knowledge.

**Format:** MC1 multiple-choice variant used here — one correct answer among several options per question, drawn from the validation split (817 questions across 38 categories).

**What's in this folder:** `truthfulqa_questions.json` (converted questions), `metadata.json`, `TruthfulQA_importer.py`, and `2026-07-01-convert-truthfulqa.py`.
