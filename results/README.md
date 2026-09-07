# Results (aggregated)

Final 230-run CUDA matrix (AutoDL instance, RTX 4090, PyTorch 2.5.1): per architecture 40 ERM + 40 GroupDRO + 35 DFR runs, seeds 0-4, N ∈ {0, 50, 100, 250, 500, 1000, 2500, 5000}.

- `main_results.csv` — one row per run: all counterfactual metrics + train/GPU minutes.
- `run_registry.csv` — run status and config/metrics paths (relative to the raw output tree, which is NOT in the repo: ~1.7 GB, regenerable via `run_matrix.py`).
- `final_analysis/` — N80 summaries, threshold sensitivity, paired method differences, breaking-curve figures, and the statistical validation report (统计验证报告.md).

Raw per-run outputs and checkpoints stay local; everything here is derived from them by `analyze.py`.
