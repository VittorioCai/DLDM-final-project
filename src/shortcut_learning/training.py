from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _inputs_labels_groups(batch: Iterable[torch.Tensor]):
    values = tuple(batch)
    if len(values) < 2:
        raise ValueError("A batch must contain at least inputs and labels")
    inputs, labels = values[:2]
    groups = values[2] if len(values) >= 3 else None
    return inputs, labels, groups


def train_epoch_erm(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        inputs, labels, _ = _inputs_labels_groups(batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach()) * count
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += count
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_epoch_groupdro(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    group_weights: torch.Tensor,
    step_size: float,
) -> tuple[dict[str, float], torch.Tensor]:
    model.train()
    weights = group_weights.to(device).clone()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss(reduction="none")
    for batch in loader:
        inputs, labels, groups = _inputs_labels_groups(batch)
        if groups is None:
            raise ValueError("GroupDRO batches must include group labels")
        inputs = inputs.to(device)
        labels = labels.to(device)
        groups = groups.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        sample_losses = criterion(logits, labels)
        active_groups = torch.unique(groups)
        group_losses = torch.stack(
            [sample_losses[groups == group].mean() for group in active_groups]
        )
        with torch.no_grad():
            weights[active_groups] *= torch.exp(step_size * group_losses.detach())
            weights /= weights.sum()
        active_weights = weights[active_groups]
        robust_loss = (active_weights * group_losses).sum() / active_weights.sum()
        robust_loss.backward()
        optimizer.step()

        count = len(labels)
        total_loss += float(sample_losses.detach().mean()) * count
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += count
    return (
        {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        },
        weights.detach().cpu(),
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        inputs, labels, _ = _inputs_labels_groups(batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        count = len(labels)
        total_loss += float(loss) * count
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += count
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fit_dfr_head(
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    """Freeze a trained feature extractor and fit a new linear head on balanced data."""
    model.to(device)
    model.eval()
    for parameter in model.feature_parameters():
        parameter.requires_grad_(False)

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, _ = _inputs_labels_groups(batch)
            feature_batches.append(model.extract_features(inputs.to(device)).cpu())
            label_batches.append(labels.cpu())

    feature_dataset = TensorDataset(
        torch.cat(feature_batches),
        torch.cat(label_batches),
    )
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=min(loader.batch_size or 256, len(feature_dataset)),
        shuffle=True,
    )
    model.reset_classifier()
    model.classifier.to(device)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        model.classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for features, labels in feature_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            count = len(labels)
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += count
        history.append(
            {
                "loss": total_loss / total_examples,
                "accuracy": total_correct / total_examples,
            }
        )
    return history
