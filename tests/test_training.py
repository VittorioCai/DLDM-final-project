import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shortcut_learning.training import (
    evaluate_loader,
    train_epoch_erm,
    train_epoch_groupdro,
)


def test_erm_training_reduces_loss_on_a_separable_dataset():
    torch.manual_seed(0)
    inputs = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    first = train_epoch_erm(model, loader, optimizer, torch.device("cpu"))["loss"]
    for _ in range(14):
        final = train_epoch_erm(model, loader, optimizer, torch.device("cpu"))["loss"]

    assert final < first


class IdentityLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return inputs + self.offset * 0


def test_groupdro_increases_weight_for_the_higher_loss_group():
    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    labels = torch.tensor([0, 1])
    groups = torch.tensor([0, 1])
    ids = torch.arange(2)
    loader = DataLoader(TensorDataset(logits, labels, groups, ids), batch_size=2)
    model = IdentityLogits()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    _, updated = train_epoch_groupdro(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        group_weights=torch.tensor([0.5, 0.5]),
        step_size=0.1,
    )

    assert updated[1] > updated[0]


def test_evaluate_loader_does_not_create_gradients():
    inputs = torch.randn(6, 2)
    labels = torch.randint(0, 2, (6,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)
    model = nn.Linear(2, 2)

    result = evaluate_loader(model, loader, torch.device("cpu"))

    assert set(result) == {"loss", "accuracy"}
    assert all(parameter.grad is None for parameter in model.parameters())
