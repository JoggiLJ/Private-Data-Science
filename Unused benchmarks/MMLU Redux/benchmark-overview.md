# MMLU-Redux

**Source:** `edinburgh-dawg/mmlu-redux-2.0` on HuggingFace.

**What it tests:** The same subjects/questions as the original MMLU, but re-annotated to fix known errors in MMLU — mislabeled answers, ambiguous questions, and other data-quality issues that were discovered in the original release. Intended as a more trustworthy ground truth for the same content, not a new/harder benchmark.

**Format:** Same 4-option multiple-choice structure as MMLU, organized by subject.

**What's in this folder:** `MMLU Redux original set/` and `JSON_set/` (per-subject converted sets), plus `MMLU_Redux_importer.py` and `MMLU_Redux_converter.py`.
