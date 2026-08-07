# GPQA (Graduate-Level Google-Proof Q&A)

**Source:** `Idavidrein/gpqa` on HuggingFace (Rein et al., 2023).

**What it tests:** PhD-level questions in biology, physics, and chemistry, written and validated by domain experts, specifically designed to be very hard to answer even with unrestricted internet/search access ("Google-proof") — intended as a benchmark that stays hard for frontier models even with tool use.

**Format:** Multiple-choice, answer options shuffled and relabeled A–D. Three overlapping subsets of increasing size/looser filtering:
- `Diamond/` — the highest-quality, most rigorously filtered subset (198 items) — the one most commonly reported in papers.
- `Extended/` — a larger, less strictly filtered set (546 items).
- `Experts/` — not question items, but a CSV of the human expert validators themselves (their domain, qualifications, and accuracy on the questions), used as a human-baseline comparison.

**What's in this folder:** `Diamond/`, `Extended/`, `Experts/` (JSON + original CSV exports) and `gpqa_importer.py`.
