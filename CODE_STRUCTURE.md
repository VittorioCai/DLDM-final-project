# Code structure and design constraints

`README.md` covers installation and the commands to reproduce the matrix.
This file documents what each module is responsible for and which
experimental constraints the code enforces.

## Layout

```text
configs/
  base.yaml            locked final parameters
  smoke.yaml           1-epoch pipeline check on a small subset
  pilot.yaml           10-epoch local pilot
src/shortcut_learning/
  data.py              balanced split, colorization, nested counterexamples,
                       DFR reweighting subset
  models.py            LeNet-style network and MNIST-adapted ResNet-18
  training.py          ERM, GroupDRO, DFR-style last-layer refitting
  metrics.py           ten-hue counterfactual metrics
  experiment.py        a single run and its artifacts
  batch.py             the experiment matrix, ordered by dependency
  analysis.py          aggregation, N80 tables, mean +/- SD figures
tests/                 automated tests (see tests/README.md)
train.py               run one configuration
evaluate.py            re-evaluate a saved checkpoint
run_matrix.py          run the matrix, skipping completed runs
analyze.py             aggregate runs into tables and figures
results/               aggregated results of the final 230-run matrix
```

`outputs_*/` directories hold raw per-run artifacts. They are not in the
repository: the final tree is about 1.7 GB and is regenerable from the code
and a seed.

## Constraints the code enforces

- Every training set is fixed at 50,000 images, 5,000 per class.
- The N conflicting examples **replace** aligned ones; the total never grows.
- Budgets are nested within a seed, so a larger budget contains every
  conflict of a smaller one.
- The global color marginal is held constant across budgets, so hue
  frequency cannot itself become a cue.
- The ten hues come from an approximately equal-luminance YIQ palette, so
  brightness cannot become a second shortcut.
- GroupDRO groups are `digit x aligned/conflict`, giving up to 20 groups.
- The DFR reweighting set is N conflict examples plus N digit-matched
  aligned examples.
- The primary evaluation is the ten-hue counterfactual test. No saliency
  method is used as evidence of what the model relies on.

## Per-run output

```text
outputs_final/{model}/{method}/N{budget}/seed{seed}/
  config_resolved.yaml
  train_log.csv
  metrics.json
  checkpoint.pt        retained per storage policy
  environment.txt
```

`run_matrix.py` skips any run whose `metrics.json` already reports
`status=completed`. A crashed run is not retried automatically, so a failure
stays visible instead of being silently overwritten.

## Execution environment for the final matrix

The canonical 230-run matrix ran on an AutoDL instance with an NVIDIA
GeForce RTX 4090, CUDA 12.4, PyTorch 2.5.1, Python 3.12.3. Each architecture
contributed 40 ERM, 40 GroupDRO and 35 DFR runs, all completed. Any Python
3.11 or newer with a recent PyTorch is fine for development; see the
reproducibility notes in `README.md` for what changes across environments.
