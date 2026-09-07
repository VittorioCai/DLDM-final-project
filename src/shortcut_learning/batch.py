from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    model: str
    method: str
    budget: int
    seed: int


def build_run_matrix(
    models: list[str],
    methods: list[str],
    budgets: list[int],
    seeds: list[int],
) -> list[RunSpec]:
    """Expand and dependency-order a requested experiment matrix."""
    method_priority = ("erm", "groupdro", "dfr")
    requested = set(methods)
    runs: list[RunSpec] = []
    for model in models:
        for method in method_priority:
            if method not in requested:
                continue
            for budget in budgets:
                if method == "dfr" and budget == 0:
                    continue
                for seed in seeds:
                    runs.append(RunSpec(model, method, int(budget), int(seed)))
    return runs


def completed_run_path(output_root: str | Path, run: RunSpec) -> Path:
    return (
        Path(output_root)
        / run.model
        / run.method
        / f"N{run.budget}"
        / f"seed{run.seed}"
        / "metrics.json"
    )

