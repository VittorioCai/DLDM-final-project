# Right for the Wrong Reasons: How Many Counterexamples Does It Take to Break a Color Shortcut?

Code for our Deep Learning and Decision Making (SS 2026) final project. On Colored MNIST, we measure the minimum number of bias-conflicting training examples (N80) that ERM, GroupDRO, and DFR need before a model stops classifying by color, on LeNet and an MNIST-adapted ResNet-18.

## Repository layout

```
configs/              base.yaml (locked final params), smoke.yaml, pilot.yaml
src/shortcut_learning/  data, models, training (ERM/GroupDRO/DFR), counterfactual metrics, batching, analysis
tests/                37 automated tests (data invariants, training, metrics, CLI)
train.py              run a single configuration
evaluate.py           re-evaluate a saved checkpoint
run_matrix.py         run the full experiment matrix; skips completed runs
analyze.py            aggregate runs into N80 tables and breaking-curve figures
results/              canonical aggregated results of the final 230-run matrix (see results/README.md)
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q        # 37 tests, ~6 s — verifies the pipeline end to end
```

Canonical environment for the final results: an AutoDL cloud instance with Python 3.12.3, PyTorch 2.5.1+cu124, NVIDIA RTX 4090, CUDA 12.4. Any Python ≥3.11 with a recent PyTorch works for development; see the reproducibility notes below. To rerun on a cloud GPU, clone this repo on the instance and use the commands below.

## Data

No dataset needs to be downloaded manually. `torchvision` fetches raw MNIST (LeCun et al., 1998) into `./data/` on first run. All Colored MNIST variants are constructed deterministically from the seed: class-balanced 50,000-image splits, equal-luminance ten-hue palette, and nested counterexample sets that replace (never add) aligned images. Rebuilding a dataset requires only the code and a seed.

## Reproducing the experiments

Smoke check (1 epoch, 100 images/class, a few seconds on CPU):

```bash
python train.py --config configs/smoke.yaml --method erm --model lenet \
  --budget 100 --seed 0 --device cpu
```

Single full run:

```bash
python train.py --config configs/base.yaml --method erm --model lenet \
  --budget 500 --seed 0 --device cuda
```

Full 230-run matrix (~6.1 h total on an RTX 4090; completed runs are skipped on restart):

```bash
python run_matrix.py --config configs/base.yaml \
  --models lenet,resnet18_mnist \
  --methods erm,groupdro,dfr \
  --budgets 0,50,100,250,500,1000,2500,5000 \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --output-root outputs_final
```

Each run writes `config_resolved.yaml`, `train_log.csv`, `metrics.json`, `environment.txt`, and (per storage policy) `checkpoint.pt` under `outputs_final/{model}/{method}/N{budget}/seed{seed}/`. DFR is undefined at N=0 and is skipped there automatically.

Aggregate into the tables and figures used in the paper:

```bash
python analyze.py --output-root outputs_final --destination results_regenerated
```

## Results

`results/` contains the canonical aggregates of the final matrix: per-run metrics (`main_results.csv`), N80 summaries, 70/80/90% threshold sensitivity, paired method differences, breaking-curve figures, and the statistical validation report. Raw per-run outputs (~1.7 GB) are regenerable and therefore not in the repo.

## Reproducibility notes

- The seed (0–4) controls the train split, digit-to-hue assignment, counterexample ordering, model initialization, and minibatch order. Budgets are nested within a seed: every smaller conflict set is a subset of every larger one.
- Hyperparameters were fixed after the pilot phase using aligned validation data only; no test result influenced training or model selection.
- Cross-environment check: local Apple-MPS runs correlate with the CUDA matrix at r ≈ 0.998 on conflict accuracy and flip rate, but exact N80 medians can shift near threshold boundaries. The canonical results are the unified CUDA matrix in `results/`.
