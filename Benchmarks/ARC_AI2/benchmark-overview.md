# ARC (AI2 Reasoning Challenge)

**Source:** `allenai/ai2_arc`, "ARC-Challenge" config (Allen Institute for AI, 2018).

**What it tests:** Grade-school-level (3rd–9th grade) natural science multiple-choice questions. The "Challenge" set (used here, as opposed to the easier "Easy" set) contains only questions that simple retrieval and co-occurrence baselines get wrong, so it's meant to require actual reasoning rather than keyword matching.

**Format:** ~2,590 questions total across train/validation/test splits, 4 (occasionally more) answer options per question, single correct answer.

**What's in this folder:** The three official splits (`train/`, `validation/`, `test/`) exported to text via `ARC_AI2_importer.py`, plus scripts (`apply-tags.py`, `arc-challenge-tag-taxonomy.py`, `tag-classifier.py`) that assign topic/skill tags to each question, and a `cache/` folder from the HuggingFace `datasets` download.
