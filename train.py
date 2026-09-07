#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shortcut_learning.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Colored MNIST shortcut run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["erm", "groupdro", "dfr"], required=True)
    parser.add_argument("--model", choices=["lenet", "resnet18_mnist"], required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = run_experiment(
        config=config,
        model_name=args.model,
        method=args.method,
        budget=args.budget,
        seed=args.seed,
        device_requested=args.device,
        output_root=args.output_root,
    )
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
