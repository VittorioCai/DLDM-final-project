from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import json
import platform
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import yaml

from .data import (
    ColoredMNIST,
    ColoredMetadata,
    CounterfactualMNIST,
    build_balanced_split,
    build_colored_metadata,
    select_dfr_indices,
)
from .metrics import counterfactual_metrics
from .models import build_model
from .training import (
    evaluate_loader,
    fit_dfr_head,
    train_epoch_erm,
    train_epoch_groupdro,
)
from .utils import choose_device, set_seed


@dataclass(frozen=True)
class MNISTTensors:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def should_keep_checkpoint(config: dict, method: str) -> bool:
    """Return whether a method's checkpoint should be retained after its run."""
    methods = config.get("storage", {}).get("keep_checkpoint_methods")
    return True if methods is None else method in methods


def load_mnist(root: str | Path, download: bool = True) -> MNISTTensors:
    train = MNIST(root=str(root), train=True, download=download)
    test = MNIST(root=str(root), train=False, download=download)
    return MNISTTensors(
        train_images=train.data,
        train_labels=train.targets,
        test_images=test.data,
        test_labels=test.targets,
    )


def _write_yaml(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("Training log cannot be empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _build_datasets(
    mnist: MNISTTensors,
    train_per_class: int,
    budget: int,
    seed: int,
) -> tuple[ColoredMNIST, ColoredMNIST, CounterfactualMNIST, ColoredMetadata]:
    train_indices, validation_indices = build_balanced_split(
        mnist.train_labels,
        train_per_class=train_per_class,
        seed=seed,
    )
    train_labels = mnist.train_labels[train_indices]
    train_metadata = build_colored_metadata(train_labels, budget=budget, seed=seed)
    train_dataset = ColoredMNIST(
        mnist.train_images[train_indices],
        train_labels,
        train_metadata,
    )

    validation_labels = mnist.train_labels[validation_indices]
    validation_metadata = ColoredMetadata(
        hues=train_metadata.digit_to_hue[validation_labels],
        is_conflict=torch.zeros(len(validation_labels), dtype=torch.bool),
        digit_to_hue=train_metadata.digit_to_hue,
    )
    validation_dataset = ColoredMNIST(
        mnist.train_images[validation_indices],
        validation_labels,
        validation_metadata,
    )
    counterfactual_dataset = CounterfactualMNIST(
        mnist.test_images,
        mnist.test_labels,
    )
    return train_dataset, validation_dataset, counterfactual_dataset, train_metadata


@torch.no_grad()
def infer_counterfactual_predictions(
    model,
    loader,
    num_shapes: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    predictions = torch.empty((num_shapes, 10), dtype=torch.long)
    for images, _, hues, shape_ids in loader:
        logits = model(images.to(device))
        predictions[shape_ids.long(), hues.long()] = logits.argmax(dim=1).cpu()
    return predictions


def run_experiment(
    config: dict,
    model_name: str,
    method: str,
    budget: int,
    seed: int,
    device_requested: str = "auto",
    output_root: str | Path | None = None,
    mnist: MNISTTensors | None = None,
) -> Path:
    if method not in {"erm", "groupdro", "dfr"}:
        raise ValueError(f"Unknown method: {method}")
    if method == "dfr" and budget == 0:
        raise ValueError("DFR is undefined for budget 0")

    set_seed(seed)
    device = choose_device(device_requested)
    output_root = Path(
        output_root
        if output_root is not None
        else config.get("experiment", {}).get("output_root", "./outputs")
    )
    run_dir = output_root / model_name / method / f"N{budget}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = deepcopy(config)
    resolved["run"] = {
        "model": model_name,
        "method": method,
        "budget_N": budget,
        "seed": seed,
        "device": device.type,
    }
    _write_yaml(run_dir / "config_resolved.yaml", resolved)

    data_config = config["data"]
    training_config = config["training"]
    if mnist is None:
        mnist = load_mnist(data_config.get("root", "./data"), download=True)
    train_dataset, validation_dataset, cf_dataset, metadata = _build_datasets(
        mnist,
        train_per_class=int(data_config["train_per_class"]),
        budget=budget,
        seed=seed,
    )

    batch_size = int(training_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    cf_loader = DataLoader(
        cf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(model_name).to(device)
    epochs = int(training_config["epochs"])
    learning_rate = float(training_config["learning_rate"][model_name])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    history: list[dict[str, float | int]] = []
    training_started = time.perf_counter()

    if method == "dfr":
        erm_checkpoint = (
            output_root / model_name / "erm" / f"N{budget}" / f"seed{seed}" / "checkpoint.pt"
        )
        if not erm_checkpoint.exists():
            raise FileNotFoundError(f"Required ERM checkpoint not found: {erm_checkpoint}")
        saved = torch.load(erm_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        dfr_indices = select_dfr_indices(
            train_dataset.labels,
            metadata.is_conflict,
            seed=seed,
        )
        dfr_loader = DataLoader(
            Subset(train_dataset, dfr_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        dfr_config = config.get("dfr", {})
        dfr_history = fit_dfr_head(
            model,
            dfr_loader,
            device=device,
            epochs=int(dfr_config.get("epochs", epochs)),
            learning_rate=float(dfr_config.get("learning_rate", learning_rate)),
            weight_decay=weight_decay,
        )
        for epoch, result in enumerate(dfr_history, start=1):
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": result["loss"],
                    "train_accuracy": result["accuracy"],
                }
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        group_weights = torch.zeros(20)
        groups = train_dataset.labels * 2 + metadata.is_conflict.long()
        active_groups = torch.unique(groups)
        group_weights[active_groups] = 1.0 / len(active_groups)
        for epoch in range(1, epochs + 1):
            if method == "erm":
                train_result = train_epoch_erm(model, train_loader, optimizer, device)
            else:
                train_result, group_weights = train_epoch_groupdro(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    group_weights=group_weights,
                    step_size=float(config["groupdro"]["step_size"]),
                )
            validation_result = evaluate_loader(model, validation_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "train_accuracy": train_result["accuracy"],
                    "validation_loss": validation_result["loss"],
                    "validation_accuracy": validation_result["accuracy"],
                }
            )

    train_seconds = time.perf_counter() - training_started
    _write_log(run_dir / "train_log.csv", history)
    if should_keep_checkpoint(config, method):
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": model_name,
                "method": method,
                "budget_N": budget,
                "seed": seed,
                "digit_to_hue": metadata.digit_to_hue,
            },
            run_dir / "checkpoint.pt",
        )

    evaluation_started = time.perf_counter()
    predictions = infer_counterfactual_predictions(
        model,
        cf_loader,
        num_shapes=len(mnist.test_labels),
        device=device,
    )
    metrics = counterfactual_metrics(
        predictions,
        mnist.test_labels,
        metadata.digit_to_hue,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    thresholds = config.get("evaluation", {})
    metrics.update(
        {
            "status": "completed",
            "model": model_name,
            "method": method,
            "budget_N": budget,
            "rho": 1 - budget / len(train_dataset),
            "seed": seed,
            "device": device.type,
            "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "n80_pass": (
                metrics["pairwise_flip_rate"]
                <= float(thresholds.get("flip_rate_threshold", 0.10))
                and metrics["bias_conflict_accuracy"]
                >= float(thresholds.get("bias_conflict_accuracy_threshold", 0.80))
            ),
        }
    )
    _write_json(run_dir / "metrics.json", metrics)
    environment = "\n".join(
        [
            f"python={sys.version.split()[0]}",
            f"torch={torch.__version__}",
            f"platform={platform.platform()}",
            f"device={device.type}",
        ]
    )
    (run_dir / "environment.txt").write_text(environment + "\n", encoding="utf-8")
    return run_dir


def evaluate_saved_run(
    run_dir: str | Path,
    mnist: MNISTTensors | None = None,
    device_requested: str = "auto",
) -> dict:
    """Recompute counterfactual metrics from a saved checkpoint without retraining."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    run = config["run"]
    device = choose_device(device_requested)
    if mnist is None:
        mnist = load_mnist(config["data"].get("root", "./data"), download=True)
    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    model = build_model(run["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = CounterfactualMNIST(mnist.test_images, mnist.test_labels)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    predictions = infer_counterfactual_predictions(
        model,
        loader,
        num_shapes=len(mnist.test_labels),
        device=device,
    )
    metrics = counterfactual_metrics(
        predictions,
        mnist.test_labels,
        checkpoint["digit_to_hue"],
    )
    metrics.update(
        {
            "status": "recomputed",
            "model": run["model"],
            "method": run["method"],
            "budget_N": run["budget_N"],
            "seed": run["seed"],
            "device": device.type,
        }
    )
    _write_json(run_dir / "metrics_recomputed.json", metrics)
    return metrics
