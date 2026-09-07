import json

import pytest
import pandas as pd

from shortcut_learning.analysis import (
    budget_axis,
    calculate_n80,
    collect_metrics,
    criterion_sensitivity,
    curve_summary,
    paired_method_differences,
    threshold_by_seed,
    threshold_sensitivity,
    write_analysis_outputs,
)


def test_collect_metrics_and_calculate_first_passing_budget(tmp_path):
    for budget, passed in [(0, False), (50, True)]:
        run_dir = tmp_path / "lenet" / "erm" / f"N{budget}" / "seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "model": "lenet",
                    "method": "erm",
                    "budget_N": budget,
                    "seed": 0,
                    "n80_pass": passed,
                    "aligned_accuracy": 1.0,
                    "bias_conflict_accuracy": 0.8 if passed else 0.1,
                    "pairwise_flip_rate": 0.1 if passed else 0.9,
                }
            )
        )

    frame = collect_metrics(tmp_path)
    n80 = calculate_n80(frame)

    assert len(frame) == 2
    assert n80.iloc[0]["n80"] == 50


def test_budget_axis_uses_only_observed_nonnegative_budget_labels():
    positions, labels, mapping = budget_axis([0, 50, 100, 500, 5000])

    assert positions == [0, 1, 2, 3, 4]
    assert labels == ["0", "50", "100", "500", "5000"]
    assert mapping[0] == 0
    assert mapping[5000] == 4


def test_curve_summary_reports_mean_and_sample_standard_deviation():
    frame = pd.DataFrame(
        {
            "budget_N": [0, 0],
            "bias_conflict_accuracy": [0.1, 0.3],
            "pairwise_flip_rate": [0.9, 0.7],
        }
    )

    summary = curve_summary(frame).iloc[0]

    assert summary["conflict_mean"] == 0.2
    assert summary["conflict_sd"] == pytest.approx(0.1414213562)
    assert summary["flip_mean"] == 0.8


def test_threshold_by_seed_uses_first_jointly_passing_observed_budget():
    frame = pd.DataFrame(
        {
            "model": ["lenet"] * 6,
            "method": ["erm"] * 6,
            "seed": [0, 0, 0, 1, 1, 1],
            "budget_N": [0, 50, 100, 0, 50, 100],
            "bias_conflict_accuracy": [0.1, 0.85, 0.9, 0.1, 0.75, 0.79],
            "pairwise_flip_rate": [0.9, 0.2, 0.08, 0.9, 0.2, 0.11],
        }
    )

    result = threshold_by_seed(frame, accuracy_threshold=0.8, flip_threshold=0.1)

    assert result.loc[result.seed == 0, "threshold_N"].iloc[0] == 100
    assert pd.isna(result.loc[result.seed == 1, "threshold_N"].iloc[0])


def test_threshold_by_seed_distinguishes_first_and_persistent_passing():
    frame = pd.DataFrame(
        {
            "model": ["resnet18"] * 4,
            "method": ["erm"] * 4,
            "seed": [2] * 4,
            "budget_N": [0, 50, 100, 250],
            "bias_conflict_accuracy": [0.70, 0.90, 0.88, 0.95],
            "pairwise_flip_rate": [0.30, 0.09, 0.12, 0.04],
        }
    )

    result = threshold_by_seed(frame, accuracy_threshold=0.8, flip_threshold=0.1)

    assert result.iloc[0]["threshold_N"] == 50
    assert result.iloc[0]["first_passing_N"] == 50
    assert result.iloc[0]["persistent_passing_N"] == 250


def test_criterion_sensitivity_summarizes_first_and_persistent_definitions():
    frame = pd.DataFrame(
        {
            "model": ["resnet18"] * 6,
            "method": ["erm"] * 6,
            "seed": [0, 0, 0, 1, 1, 1],
            "budget_N": [50, 100, 250, 50, 100, 250],
            "bias_conflict_accuracy": [0.90, 0.88, 0.95, 0.70, 0.90, 0.95],
            "pairwise_flip_rate": [0.09, 0.12, 0.04, 0.30, 0.09, 0.04],
        }
    )

    result = criterion_sensitivity(frame, criteria=((0.8, 0.1),))

    first = result[result["definition"] == "first_passing"].iloc[0]
    persistent = result[result["definition"] == "persistent_passing"].iloc[0]
    assert first["median_N"] == 75
    assert persistent["median_N"] == 175


def test_threshold_sensitivity_summarizes_median_and_range():
    frame = pd.DataFrame(
        {
            "model": ["lenet"] * 6,
            "method": ["erm"] * 6,
            "seed": [0, 0, 0, 1, 1, 1],
            "budget_N": [0, 50, 100, 0, 50, 100],
            "bias_conflict_accuracy": [0.1, 0.75, 0.95, 0.1, 0.85, 0.95],
            "pairwise_flip_rate": [0.9, 0.09, 0.05, 0.9, 0.09, 0.05],
        }
    )

    result = threshold_sensitivity(frame, accuracy_thresholds=(0.7, 0.8, 0.9))
    n80 = result[result.accuracy_threshold == 0.8].iloc[0]

    assert n80["median_N"] == 75
    assert n80["min_N"] == 50
    assert n80["max_N"] == 100
    assert n80["seeds_passing"] == 2


def test_paired_method_differences_pair_on_model_budget_and_seed():
    frame = pd.DataFrame(
        {
            "model": ["lenet"] * 4,
            "method": ["erm", "erm", "groupdro", "groupdro"],
            "budget_N": [500] * 4,
            "seed": [0, 1, 0, 1],
            "aligned_accuracy": [0.99, 0.98, 0.98, 0.97],
            "bias_conflict_accuracy": [0.7, 0.6, 0.8, 0.8],
            "pairwise_flip_rate": [0.3, 0.4, 0.2, 0.2],
            "worst_group_accuracy": [0.4, 0.3, 0.6, 0.5],
        }
    )

    result = paired_method_differences(frame).iloc[0]

    assert result["n_pairs"] == 2
    assert result["conflict_delta_mean"] == pytest.approx(0.15)
    assert result["flip_delta_mean"] == pytest.approx(-0.15)
    assert result["aligned_delta_mean"] == pytest.approx(-0.01)


def test_write_analysis_outputs_creates_final_tables_and_figures(tmp_path):
    output_root = tmp_path / "outputs"
    for model in ("lenet", "resnet18_mnist"):
        for method in ("erm", "groupdro", "dfr"):
            for budget in (50, 100):
                for seed in (0, 1):
                    run_dir = output_root / model / method / f"N{budget}" / f"seed{seed}"
                    run_dir.mkdir(parents=True)
                    passed = budget == 100
                    record = {
                        "status": "completed",
                        "model": model,
                        "method": method,
                        "budget_N": budget,
                        "seed": seed,
                        "n80_pass": passed,
                        "aligned_accuracy": 0.99,
                        "bias_conflict_accuracy": 0.9 if passed else 0.7,
                        "pairwise_flip_rate": 0.08 if passed else 0.3,
                        "counterfactual_consistency": 0.8 if passed else 0.4,
                        "worst_group_accuracy": 0.7 if passed else 0.2,
                    }
                    (run_dir / "metrics.json").write_text(json.dumps(record))

    destination = tmp_path / "analysis"
    paths = write_analysis_outputs(output_root, destination)

    expected = {
        "summary.csv",
        "n80_by_seed.csv",
        "n80_summary.csv",
        "threshold_by_seed.csv",
        "threshold_sensitivity.csv",
        "criterion_sensitivity_by_seed.csv",
        "criterion_sensitivity_summary.csv",
        "paired_method_differences.csv",
        "breaking_curves.png",
        "n80_comparison.png",
    }
    assert expected == {path.name for path in paths}
    assert all(path.stat().st_size > 0 for path in paths)
