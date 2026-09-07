import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CRITERIA = (
    (0.7, 0.1),
    (0.8, 0.05),
    (0.8, 0.1),
    (0.8, 0.15),
    (0.9, 0.1),
)


def budget_axis(budgets) -> tuple[list[int], list[str], dict[int, int]]:
    """Map irregular nonnegative budgets to evenly spaced labeled positions."""
    observed = sorted({int(budget) for budget in budgets})
    if not observed or observed[0] < 0:
        raise ValueError("budgets must contain observed nonnegative values")
    positions = list(range(len(observed)))
    return positions, [str(budget) for budget in observed], dict(zip(observed, positions))


def collect_metrics(output_root: str | Path) -> pd.DataFrame:
    """Collect canonical metrics.json files from an experiment output tree."""
    records: list[dict] = []
    for path in sorted(Path(output_root).rglob("metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["run_dir"] = str(path.parent)
        records.append(record)
    if not records:
        raise ValueError(f"No metrics.json files found under {output_root}")
    return pd.DataFrame(records)


def calculate_n80(frame: pd.DataFrame) -> pd.DataFrame:
    """Find the first observed passing budget for each model, method, and seed."""
    rows: list[dict] = []
    for keys, group in frame.groupby(["model", "method", "seed"], sort=True):
        ordered = group.sort_values("budget_N")
        passing = ordered[ordered["n80_pass"].astype(bool)]
        n80: int | str
        if len(passing):
            n80 = int(passing.iloc[0]["budget_N"])
        else:
            n80 = f">{int(ordered['budget_N'].max())}"
        rows.append(
            {
                "model": keys[0],
                "method": keys[1],
                "seed": int(keys[2]),
                "n80": n80,
            }
        )
    return pd.DataFrame(rows)


def curve_summary(group: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a model-method curve across seeds with sample SD."""
    return (
        group.groupby("budget_N")
        .agg(
            conflict_mean=("bias_conflict_accuracy", "mean"),
            conflict_sd=("bias_conflict_accuracy", "std"),
            flip_mean=("pairwise_flip_rate", "mean"),
            flip_sd=("pairwise_flip_rate", "std"),
        )
        .reset_index()
        .sort_values("budget_N")
    )


def threshold_by_seed(
    frame: pd.DataFrame,
    accuracy_threshold: float = 0.8,
    flip_threshold: float = 0.1,
) -> pd.DataFrame:
    """Find first and persistent observed budgets meeting both criteria per seed."""
    rows: list[dict] = []
    for keys, group in frame.groupby(["model", "method", "seed"], sort=True):
        ordered = group.sort_values("budget_N")
        passing_mask = (
            (ordered["bias_conflict_accuracy"] >= accuracy_threshold)
            & (ordered["pairwise_flip_rate"] <= flip_threshold)
        )
        passing = ordered[passing_mask]
        first_passing = float("nan") if passing.empty else int(passing.iloc[0]["budget_N"])
        persistent_passing = float("nan")
        for index in range(len(ordered)):
            if passing_mask.iloc[index:].all():
                persistent_passing = int(ordered.iloc[index]["budget_N"])
                break
        rows.append(
            {
                "model": keys[0],
                "method": keys[1],
                "seed": int(keys[2]),
                "accuracy_threshold": accuracy_threshold,
                "flip_threshold": flip_threshold,
                "threshold_N": first_passing,
                "first_passing_N": first_passing,
                "persistent_passing_N": persistent_passing,
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    frame: pd.DataFrame,
    accuracy_thresholds: tuple[float, ...] = (0.7, 0.8, 0.9),
    flip_threshold: float = 0.1,
) -> pd.DataFrame:
    """Summarize per-seed breaking thresholds for several accuracy criteria."""
    rows: list[dict] = []
    for accuracy_threshold in accuracy_thresholds:
        thresholds = threshold_by_seed(frame, accuracy_threshold, flip_threshold)
        for keys, group in thresholds.groupby(["model", "method"], sort=True):
            observed = group["threshold_N"].dropna()
            rows.append(
                {
                    "model": keys[0],
                    "method": keys[1],
                    "accuracy_threshold": accuracy_threshold,
                    "flip_threshold": flip_threshold,
                    "median_N": observed.median() if len(observed) else float("nan"),
                    "min_N": observed.min() if len(observed) else float("nan"),
                    "max_N": observed.max() if len(observed) else float("nan"),
                    "mean_N": observed.mean() if len(observed) else float("nan"),
                    "sd_N": observed.std() if len(observed) > 1 else 0.0,
                    "seeds_passing": int(observed.count()),
                    "seeds_total": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def criterion_sensitivity_by_seed(
    frame: pd.DataFrame,
    criteria: tuple[tuple[float, float], ...] = DEFAULT_CRITERIA,
) -> pd.DataFrame:
    """Calculate first and persistent thresholds for prespecified criterion pairs."""
    return pd.concat(
        [
            threshold_by_seed(frame, accuracy_threshold, flip_threshold)
            for accuracy_threshold, flip_threshold in criteria
        ],
        ignore_index=True,
    )


def criterion_sensitivity(
    frame: pd.DataFrame,
    criteria: tuple[tuple[float, float], ...] = DEFAULT_CRITERIA,
) -> pd.DataFrame:
    """Summarize threshold sensitivity under first- and persistent-passing rules."""
    thresholds = criterion_sensitivity_by_seed(frame, criteria)
    rows: list[dict] = []
    definitions = {
        "first_passing": "first_passing_N",
        "persistent_passing": "persistent_passing_N",
    }
    for keys, group in thresholds.groupby(
        ["model", "method", "accuracy_threshold", "flip_threshold"],
        sort=True,
    ):
        for definition, column in definitions.items():
            observed = group[column].dropna()
            rows.append(
                {
                    "model": keys[0],
                    "method": keys[1],
                    "accuracy_threshold": keys[2],
                    "flip_threshold": keys[3],
                    "definition": definition,
                    "median_N": observed.median() if len(observed) else float("nan"),
                    "min_N": observed.min() if len(observed) else float("nan"),
                    "max_N": observed.max() if len(observed) else float("nan"),
                    "mean_N": observed.mean() if len(observed) else float("nan"),
                    "sd_N": observed.std() if len(observed) > 1 else 0.0,
                    "seeds_passing": int(observed.count()),
                    "seeds_total": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def paired_method_differences(
    frame: pd.DataFrame,
    reference_method: str = "erm",
) -> pd.DataFrame:
    """Calculate paired method-minus-ERM differences at matching model, budget, and seed."""
    reference = frame[frame["method"] == reference_method]
    metrics = [
        "aligned_accuracy",
        "bias_conflict_accuracy",
        "pairwise_flip_rate",
        "worst_group_accuracy",
    ]
    rows: list[dict] = []
    for method in sorted(set(frame["method"]) - {reference_method}):
        candidate = frame[frame["method"] == method]
        paired = candidate.merge(
            reference,
            on=["model", "budget_N", "seed"],
            suffixes=("_method", "_reference"),
        )
        for keys, group in paired.groupby(["model", "budget_N"], sort=True):
            deltas = {
                metric: group[f"{metric}_method"] - group[f"{metric}_reference"]
                for metric in metrics
            }
            rows.append(
                {
                    "model": keys[0],
                    "method": method,
                    "reference_method": reference_method,
                    "budget_N": int(keys[1]),
                    "n_pairs": int(len(group)),
                    "aligned_delta_mean": deltas["aligned_accuracy"].mean(),
                    "aligned_delta_sd": deltas["aligned_accuracy"].std(),
                    "conflict_delta_mean": deltas["bias_conflict_accuracy"].mean(),
                    "conflict_delta_sd": deltas["bias_conflict_accuracy"].std(),
                    "flip_delta_mean": deltas["pairwise_flip_rate"].mean(),
                    "flip_delta_sd": deltas["pairwise_flip_rate"].std(),
                    "worst_group_delta_mean": deltas["worst_group_accuracy"].mean(),
                    "worst_group_delta_sd": deltas["worst_group_accuracy"].std(),
                }
            )
    return pd.DataFrame(rows)


def write_analysis_outputs(
    output_root: str | Path,
    destination: str | Path,
) -> tuple[Path, ...]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame = collect_metrics(output_root)
    summary_path = destination / "summary.csv"
    n80_path = destination / "n80_by_seed.csv"
    n80_summary_path = destination / "n80_summary.csv"
    threshold_path = destination / "threshold_by_seed.csv"
    sensitivity_path = destination / "threshold_sensitivity.csv"
    criterion_by_seed_path = destination / "criterion_sensitivity_by_seed.csv"
    criterion_summary_path = destination / "criterion_sensitivity_summary.csv"
    paired_path = destination / "paired_method_differences.csv"
    figure_path = destination / "breaking_curves.png"
    n80_figure_path = destination / "n80_comparison.png"
    frame.to_csv(summary_path, index=False)
    calculate_n80(frame).to_csv(n80_path, index=False)
    thresholds = threshold_by_seed(frame)
    sensitivity = threshold_sensitivity(frame)
    criterion_by_seed = criterion_sensitivity_by_seed(frame)
    criterion_summary = criterion_sensitivity(frame)
    thresholds.to_csv(threshold_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    criterion_by_seed.to_csv(criterion_by_seed_path, index=False)
    criterion_summary.to_csv(criterion_summary_path, index=False)
    sensitivity[sensitivity["accuracy_threshold"] == 0.8].to_csv(
        n80_summary_path,
        index=False,
    )
    paired_method_differences(frame).to_csv(paired_path, index=False)

    models = sorted(frame["model"].unique())
    figure, axes = plt.subplots(
        len(models),
        2,
        figsize=(12, 4.2 * len(models)),
        constrained_layout=True,
        squeeze=False,
    )
    positions, budget_labels, position_by_budget = budget_axis(frame["budget_N"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_order = [method for method in ("erm", "groupdro", "dfr") if method in set(frame["method"])]
    display_names = {"lenet": "LeNet", "resnet18_mnist": "ResNet-18"}
    for model_index, model in enumerate(models):
        model_frame = frame[frame["model"] == model]
        for method_index, method in enumerate(method_order):
            group = model_frame[model_frame["method"] == method]
            if group.empty:
                continue
            aggregated = curve_summary(group)
            x_values = [position_by_budget[int(value)] for value in aggregated["budget_N"]]
            color = colors[method_index % len(colors)]
            raw_x = [
                position_by_budget[int(budget)] + (int(seed) - 2) * 0.025
                for budget, seed in zip(group["budget_N"], group["seed"])
            ]
            axes[model_index, 0].scatter(
                raw_x,
                group["bias_conflict_accuracy"],
                color=color,
                alpha=0.25,
                s=16,
            )
            axes[model_index, 1].scatter(
                raw_x,
                group["pairwise_flip_rate"],
                color=color,
                alpha=0.25,
                s=16,
            )
            axes[model_index, 0].errorbar(
                x_values,
                aggregated["conflict_mean"],
                yerr=aggregated["conflict_sd"].fillna(0),
                color=color,
                marker="o",
                capsize=3,
                label={"erm": "ERM", "groupdro": "GroupDRO", "dfr": "DFR-style"}.get(method, method),
            )
            axes[model_index, 1].errorbar(
                x_values,
                aggregated["flip_mean"],
                yerr=aggregated["flip_sd"].fillna(0),
                color=color,
                marker="o",
                capsize=3,
                label={"erm": "ERM", "groupdro": "GroupDRO", "dfr": "DFR-style"}.get(method, method),
            )
        left = axes[model_index, 0]
        right = axes[model_index, 1]
        left.axhline(0.8, color="black", linestyle="--", linewidth=1)
        right.axhline(0.1, color="black", linestyle="--", linewidth=1)
        left.set(
            title=f"{display_names.get(model, model)}: bias-conflict accuracy",
            xlabel="Budget N (categorical grid)",
            ylabel="Accuracy",
        )
        right.set(
            title=f"{display_names.get(model, model)}: counterfactual flip rate",
            xlabel="Budget N (categorical grid)",
            ylabel="Flip rate",
        )
        for axis in (left, right):
            axis.set_xticks(positions, budget_labels, rotation=30)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
        right.legend(fontsize=8)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    n80_summary = sensitivity[sensitivity["accuracy_threshold"] == 0.8].copy()
    n80_summary["label"] = n80_summary.apply(
        lambda row: f"{display_names.get(row['model'], row['model'])}\n{row['method'].upper() if row['method'] != 'groupdro' else 'GroupDRO'}",
        axis=1,
    )
    n80_figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = list(range(len(n80_summary)))
    medians = n80_summary["median_N"].to_numpy()
    lower = medians - n80_summary["min_N"].to_numpy()
    upper = n80_summary["max_N"].to_numpy() - medians
    axis.errorbar(
        x,
        medians,
        yerr=[lower, upper],
        fmt="o",
        capsize=5,
        color="#24527a",
        markersize=7,
    )
    for position, median in zip(x, medians):
        axis.annotate(f"{median:g}", (position, median), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set_xticks(x, n80_summary["label"])
    axis.set_ylabel("N80 (median; min–max across 5 seeds)")
    axis.set_title("Counterexamples required to break the color shortcut")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(bottom=0)
    n80_figure.savefig(n80_figure_path, dpi=180)
    plt.close(n80_figure)
    return (
        summary_path,
        n80_path,
        n80_summary_path,
        threshold_path,
        sensitivity_path,
        criterion_by_seed_path,
        criterion_summary_path,
        paired_path,
        figure_path,
        n80_figure_path,
    )
