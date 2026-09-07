import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from shortcut_learning.data import build_colored_metadata, select_dfr_indices
from shortcut_learning.models import build_model
from shortcut_learning.training import fit_dfr_head


def test_dfr_indices_contain_equal_conflict_and_digit_matched_aligned_examples():
    labels = torch.arange(10).repeat_interleave(20)
    metadata = build_colored_metadata(labels, budget=20, seed=3)

    selected = select_dfr_indices(labels, metadata.is_conflict, seed=11)
    selected_conflicts = metadata.is_conflict[selected]

    assert len(selected) == 40
    assert int(selected_conflicts.sum()) == 20
    assert torch.bincount(labels[selected], minlength=10).tolist() == [4] * 10


def test_dfr_indices_reject_zero_conflicts():
    labels = torch.arange(10).repeat_interleave(4)
    with pytest.raises(ValueError, match="at least one conflict"):
        select_dfr_indices(labels, torch.zeros(40, dtype=torch.bool), seed=0)


def test_dfr_head_fit_keeps_feature_parameters_unchanged():
    torch.manual_seed(2)
    model = build_model("lenet")
    inputs = torch.randn(20, 3, 28, 28)
    labels = torch.arange(10).repeat(2)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=10)
    feature_before = [parameter.detach().clone() for parameter in model.feature_parameters()]

    history = fit_dfr_head(
        model,
        loader,
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    feature_after = list(model.feature_parameters())
    assert len(history) == 2
    assert all(torch.equal(before, after) for before, after in zip(feature_before, feature_after))
    assert all(not parameter.requires_grad for parameter in feature_after)
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
