#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shortcut_learning.batch import build_run_matrix, completed_run_path
from shortcut_learning.experiment import run_experiment


def comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in comma_strings(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dependency-ordered experiment matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", default="lenet,resnet18_mnist")
    parser.add_argument("--methods", default="erm,groupdro,dfr")
    parser.add_argument("--budgets", default="0,50,100,250,500,1000,2500,5000")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["experiment"]["output_root"])
    runs = build_run_matrix(
        comma_strings(args.models),
        comma_strings(args.methods),
        comma_ints(args.budgets),
        comma_ints(args.seeds),
    )
    for index, run in enumerate(runs, start=1):
        metrics_path = completed_run_path(output_root, run)
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print(f"[{index}/{len(runs)}] SKIP completed {run}", flush=True)
                continue
        print(f"[{index}/{len(runs)}] START {run}", flush=True)
        run_dir = run_experiment(
            config=config,
            model_name=run.model,
            method=run.method,
            budget=run.budget,
            seed=run.seed,
            device_requested=args.device,
            output_root=output_root,
        )
        print(f"[{index}/{len(runs)}] DONE {run_dir}", flush=True)


if __name__ == "__main__":
    main()

