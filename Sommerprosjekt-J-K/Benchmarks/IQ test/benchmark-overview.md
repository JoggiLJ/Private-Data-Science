# IQ Tests

This folder holds two different IQ-style instruments, not one benchmark:

## Open-Source Psychometrics Project
A public-domain, matrix-reasoning ("Raven's-style") IQ test: each question shows a 3×3 grid of visual panels with one missing, and the answer is the panel that completes the pattern. Sourced as `intelligence-quotient-test.pdf` with question images in `Question screenshots (white gaps) (cleaned)/`. Because the items are visual grids rather than text, `psychometrics_IQ_test_converter.py` uses OpenCV to programmatically segment each puzzle image into its component panels/symbols so items can be scored without a human transcribing each picture. Results of running items through this pipeline are in `all_results.json`.

## Verbal
A text-based ("Verbal IQ") test — `TextIQ.pdf` plus a reference screenshot — testing verbal/language reasoning rather than visual pattern matching.
