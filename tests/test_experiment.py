import json

import torch

from shortcut_learning.experiment import (
    MNISTTensors,
    evaluate_saved_run,
    run_experiment,
    should_keep_checkpoint,
)


def test_one_run_writes_reproducible_artifacts(tmp_path):
    torch.manual_seed(0)
    train_labels = torch.arange(10).repeat_interleave(4)
    test_labels = torch.arange(10).repeat_interleave(2)
    mnist = MNISTTensors(
        train_images=torch.randint(0, 256, (40, 28, 28), dtype=torch.uint8),
        train_labels=train_labels,
        test_images=torch.randint(0, 256, (20, 28, 28), dtype=torch.uint8),
        test_labels=test_labels,
    )
    config = {
        "data": {"train_per_class": 3, "num_classes": 10},
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "learning_rate": {"lenet": 0.001, "resnet18_mnist": 0.001},
        },
        "groupdro": {"step_size": 0.01},
        "dfr": {"epochs": 2, "learning_rate": 0.01},
        "evaluation": {
            "flip_rate_threshold": 0.10,
            "bias_conflict_accuracy_threshold": 0.80,
        },
    }

    run_dir = run_experiment(
        config=config,
        model_name="lenet",
        method="erm",
        budget=0,
        seed=0,
        device_requested="cpu",
        output_root=tmp_path,
        mnist=mnist,
    )

    expected = {
        "config_resolved.yaml",
        "train_log.csv",
        "checkpoint.pt",
        "metrics.json",
        "environment.txt",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["model"] == "lenet"
    assert metrics["method"] == "erm"
    assert metrics["budget_N"] == 0
    assert 0 <= metrics["bias_conflict_accuracy"] <= 1

    recomputed = evaluate_saved_run(run_dir, mnist=mnist, device_requested="cpu")
    assert (run_dir / "metrics_recomputed.json").exists()
    assert recomputed["bias_conflict_accuracy"] == metrics["bias_conflict_accuracy"]


def test_checkpoint_policy_defaults_to_all_and_can_keep_only_erm():
    assert should_keep_checkpoint({}, "groupdro")
    config = {"storage": {"keep_checkpoint_methods": ["erm"]}}
    assert should_keep_checkpoint(config, "erm")
    assert not should_keep_checkpoint(config, "groupdro")
    assert not should_keep_checkpoint(config, "dfr")
