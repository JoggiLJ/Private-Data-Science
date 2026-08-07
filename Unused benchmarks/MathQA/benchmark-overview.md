# MathQA (folder is actually AQuA-RAT data — see note)

**Source:** Despite the folder name, the data's own embedded source note identifies it as **AQuA-RAT** (Algebra Question Answering with Rationales; Ling, Yogatama, Dyer & Blunsom, 2017, arXiv:1705.04146; `google-deepmind/AQuA`), **not** the canonical MathQA dataset (Amini et al., 2019). The split sizes (97,467 train / 254 dev / 254 test) match AQuA-RAT exactly, not MathQA's ~37k. Worth relabeling/flagging if this matters for how it's used downstream.

**What it tests:** Grade-school-to-GRE-level algebra word problems, each with a free-text rationale (worked solution) leading to the answer.

**Format:** 5-option multiple-choice (A–E), with `rationale` (worked-solution text) and `correct` fields.

**What's in this folder:** `raw/` (original train/test/dev, tokenized and untokenized variants), `Tokenized/` and `Untokenized/` (converted structured JSON, with the AQuA-RAT identification noted directly in each file's `description`), and `MathQA.pdf`.
