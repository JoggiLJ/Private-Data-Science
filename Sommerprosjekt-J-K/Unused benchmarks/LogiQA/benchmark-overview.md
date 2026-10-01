# LogiQA

**Source:** LogiQA1 — `lucasmccabe/logiqa` (originally sourced from the Chinese Civil Service Examination logical reasoning section, translated to English). LogiQA2 — the expanded successor dataset released by the same research group.

**What it tests:** Logical reasoning over short passages — deductive, categorical, and conditional reasoning presented as reading-comprehension-style questions, meant to be harder for models that rely on surface pattern matching rather than actual logical inference.

**Format:** LogiQA1: passage + question + 4 options, single correct answer, with train/validation/test splits. LogiQA2 expands this into two tasks — **MRC** (machine reading comprehension, multiple-choice) and **NLI** (natural language inference, i.e. does a statement follow from the passage) — each with its own train/dev/test splits.

**What's in this folder:** `LogiQA1/` (imported set, JSON set, importer script) and `LogiQA2/` (`MRC/` and `NLI/` splits).
