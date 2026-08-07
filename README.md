# Sommerprosjekt INTED J&K

Data, model results, and analysis scripts from our summer project benchmarking LLMs (run via Ollama, Qwen3 models) against a set of academic reasoning and knowledge exams.

## Structure

- **Benchmarks/** — Source questions/data for each benchmark used in the report (ARC_AI2, Cornell Critical, GMAT, IQ test, LSAT, MMLU, SAT, Statistics Concept Inventory, TruthfulQA), each with a `benchmark-overview.md` describing the source, format, and what's included, plus import scripts.
- **Model results (JSON)/** — Raw per-model, per-benchmark results (one JSON file per model per benchmark), mirroring the Benchmarks folder structure.
- **Scripts/Analysis/** — Python scripts used to analyze results: answer-distribution histograms, error analysis, McNemar's test, negation effects, answer length vs. token/entropy effects, effect of numbers in questions, topic breakdowns, and plotting.
- **Scripts/Importers-Converters/** — Scripts that convert raw benchmark data into the common format used across the project.
- **Figures/** — Generated figures used in the report, organized by analysis type (Answer distribution, Answering errors, Effect of numbers, General, Length-tokens-entropy, McNemar, Negation, Sub-topics).
- **Plots not in report/** — Supplementary plots generated during analysis but not included in the final report (an earlier figures pass, IQ test analysis, and per-benchmark combined scatterplots).
- **Unused benchmarks/** — Benchmarks explored but ultimately not used in the report (Biological Concepts Instrument, Concept Inventory of Natural Selection, FCI, GPQA, HLE, LogiQA, MathQA, MedQA, MMLU Pro, MMLU Redux, and others).

## Notes

- Large files as `Benchmarks/SAT/Math/SAT_Math.pdf` are tracked with Git LFS — see `.gitattributes`.
