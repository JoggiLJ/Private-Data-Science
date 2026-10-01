# MedQA

**Source:** `bigbio/med_qa` (originally `jind11/MedQA` on GitHub), US subset.

**What it tests:** USMLE-style clinical medical knowledge — questions drawn from actual US Medical Licensing Examination material, spanning Step 1 (foundational biomedical/basic science, ~2nd year of med school) through Step 2 CK / Step 3 (clinical patient-care reasoning, 4th year and residency). The `difficulty` field records which USMLE step a question was drawn from.

**Format:** Multiple-choice, offered in two variants — official 4-option and 5-option — each with train/dev/test splits (10,178 / 1,272 / 1,273 items). Answer options were shuffled and relabeled A–D/A–E during conversion.

**What's in this folder:** `US_version/4_options_JSON/` and `US_version/5_options_JSON/` (train/dev/test each) and `MedQA.pdf`.
