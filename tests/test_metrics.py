import pytest
import torch

from shortcut_learning.metrics import counterfactual_metrics


def test_counterfactual_metrics_match_hand_calculation():
    labels = torch.tensor([0, 1])
    digit_to_hue = torch.arange(10)
    predictions = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )

    metrics = counterfactual_metrics(predictions, labels, digit_to_hue)

    assert metrics["aligned_accuracy"] == pytest.approx(1.0)
    assert metrics["bias_conflict_accuracy"] == pytest.approx(13 / 18)
    assert metrics["pairwise_flip_rate"] == pytest.approx(5 / 18)
    assert metrics["counterfactual_consistency"] == pytest.approx(0.5)
    assert metrics["worst_group_accuracy"] == pytest.approx(0.0)


def test_counterfactual_metrics_reject_wrong_number_of_hues():
    with pytest.raises(ValueError, match="10 hues"):
        counterfactual_metrics(
            torch.zeros((3, 9), dtype=torch.long),
            torch.tensor([0, 1, 2]),
            torch.arange(10),
        )
