# SAT

**Source:** College Board SAT Question Bank (official Bluebook export), split into Math and Reading & Writing PDFs, text-extracted via `pdftotext`.

**What it tests:** US college-admissions general aptitude — Math (algebra, problem-solving/data analysis, advanced math, geometry) and Reading & Writing (reading comprehension, grammar/expression).

**Format:** Multiple-choice only (grid-in/student-produced-response math items excluded). Math: 1,324 items — but many render their equations/graphs as images in the source PDF, so only ~130 are fully text-recoverable (flagged `has_image` in the JSON). Reading & Writing: 1,687 items, mostly clean text.

**What's in this folder:**
- `Math/` and `Reading and writing/` — the converted question sets (`sat-math-mc.json`, `sat-reading-writing-mc.json`), source PDFs, and similarity-check scripts (likely for de-duplication/near-duplicate detection).
- `Benchmark/` — `ollama-sat-benchmark.py`, a script for running a local Ollama model against this question bank, plus results/charts from a `llama3.1-8b` run (`both_llama3.1-8b_results.json` and by-difficulty/by-domain charts).
