# Statistical validation report

- **Source**: the 230-run unified CUDA experiment matrix
- **Date**: 2026-07-12
- **Verification status**: ANALYZED (not VERIFIED, see Reproducibility)
- **Overall confidence**: CAUTION
- **Reason**: the data and design are complete, the effects are large and
  broadly stable across seeds; but there are only five seeds, N80 comes from
  a discrete budget grid, and a small amount of training instability exists.

## Completeness and consistency

- 230 unique runs, 230 with `status=completed`.
- LeNet and ResNet-18 each contribute 40 ERM, 40 GroupDRO and 35 DFR runs.
- No duplicate keys, missing cells, NaNs, wrong rho values or wrong N80 flags.
- Every configuration uses 5,000 images per class, 10 epochs, batch size 256,
  seeds 0-4.
- 80 ERM checkpoints exist; GroupDRO and DFR keep none, per storage policy.
- Training log lengths: ERM 80x10, GroupDRO 80x10, DFR 70x30, all finite.

## Findings

| Finding | Evidence | Interpretation | Confidence |
|---|---|---|---|
| A strong color shortcut exists at N=0 | ERM aligned accuracy about 99%, conflict accuracy about 5-7%, flip rate about 93-95% | High aligned accuracy does not show that the model learned shape | SOLID |
| GroupDRO is stable across architectures | Median N80 is 500 for both models, range 250-500 | Robust to the architecture change in this setting | SOLID |
| DFR shows an architecture interaction | LeNet median N80 250, ResNet-18 1,000 | DFR's sample efficiency depends on the backbone representation | SOLID |
| ERM's threshold varies widely | Median N80 is 1,000 on both models, but the maximum reaches 2,500 | The exact breaking point carries seed variability | CAUTION |
| Threshold sensitivity is broadly stable | The 70/80/90% analysis moves only LeNet-ERM at 90% | The main ordering is not an artifact of one cutoff | SOLID |

No p-values are reported. Five seeds cannot support complex asymptotic
tests, so the main report uses raw seed points, mean +/- SD, median and
range, and paired effect sizes.

## Warnings

| Type | Detail | Affected |
|---|---|---|
| Coarse threshold grid | N80 is the smallest observed budget; no interpolation between neighbouring budgets | every N80 |
| Small seed count | five seeds per cell | SD, range, and the precision of method differences |
| Training instability | ResNet-ERM at N=0/seed 1 and N=100/seed 2 ended at 94.83% and 89.75% final validation accuracy | two runs, retained as-is to avoid selective deletion |
| Group information | GroupDRO and DFR use known conflict/group information | external validity where real bias is unknown |
| Synthetic benchmark | Colored MNIST contains one controlled color shortcut | extrapolation to natural images and multiple shortcuts |

## Fallacy scan

Coverage: 11 of 11 fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Results are broken out by model, method and seed. The overall direction of improvement agrees with most within-group directions, but DFR's architecture interaction must be reported separately. |
| Ecological fallacy | NOTE | The unit of inference stays the model or run; no individual behaviour is inferred from group means. |
| Berkson's paradox | NOTE | The sample is not filtered on model outcomes; balanced MNIST sampling is fixed by design in advance. |
| Collider bias | NOTE | No control variable caused jointly by method and outcome enters the analysis. |
| Base rate neglect | NOTE | Digit classes and color marginals are explicitly fixed; the metrics are not diagnostic screening probabilities. |
| Regression to the mean | NOTE | Budgets are not chosen from extreme model performance; every preset budget ran in full. |
| Survivorship bias | NOTE | 230 of 230 runs completed; the two low-validation runs were not deleted. |
| Look-elsewhere effect | NOTE | Primary metrics, budgets and N80 were fixed before the experiment; 70% and 90% appear only as clearly labelled sensitivity analyses. |
| Garden of forking paths | CAUTION | The epoch count was locked at 10 after pilot work. That decision history must be preserved; checkpoints must not be swapped based on the main results. |
| Correlation vs causation | NOTE | N and color are controlled interventions and the counterfactual test supports a causal reading within this synthetic setting. It does not extrapolate to a general causal mechanism in real models. |
| Reverse causality | NOTE | Counterexample injection happens before training, so outcomes cannot determine the budget. |

## Reproducibility

- **Method**: environment-sensitivity comparison over 116 overlapping local
  MPS and CUDA runs.
- **Verdict**: PARTIALLY_REPRODUCIBLE.
- Cross-environment correlations are about 0.998 for conflict accuracy,
  0.998 for flip rate, 0.996 for counterfactual consistency and 0.982 for
  worst-group accuracy.
- A few GroupDRO worst-group values differ substantially, and the LeNet-ERM
  median N80 moves from 2,500 locally to 1,000 on CUDA.
- The final main analysis therefore uses only the unified 230-run CUDA
  matrix. The local MPS results are kept as pilot data and are never mixed
  into the main tables.
- No second fully independent rerun was performed on the same CUDA runtime,
  which is why this report is marked `ANALYZED` rather than `VERIFIED`.

## Verdict

The experimental record is complete enough to support the paper. The most
reliable conclusions are that the color shortcut effect is very large, that
GroupDRO is stable across the two architectures, and that DFR's
effectiveness is clearly architecture-dependent. Any exact N80 should always
be reported together with its seed range and the discrete-budget caveat, and
never stated as a universal constant.
