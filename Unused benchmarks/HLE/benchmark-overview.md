# Humanity's Last Exam (HLE)

**Source:** `cais/hle` on HuggingFace (Center for AI Safety), test split.

**What it tests:** Extremely difficult, expert-level questions spanning many academic fields (math, physics, philosophy, humanities, etc.), designed specifically to be hard for frontier LLMs — intended as a benchmark that won't saturate the way many older benchmarks have.

**Format:** Originally 2,500 items mixing multiple-choice and exact-match/short-answer questions, some with attached images. This folder holds a filtered subset: 513 multiple-choice, no-image items (image-dependent and short-answer items removed, since the local pipeline here works with text-only MC questions).

**What's in this folder:** `HLE_MC.json` (the 513 filtered items with options parsed out of the embedded "Answer Choices:" text), `HLE_importer.py`, `HLE_categories.py` (categorizes items by subject), `category_counts.png` (a chart of category distribution), and `data/`.
