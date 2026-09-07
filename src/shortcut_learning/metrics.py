import torch


def counterfactual_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    digit_to_hue: torch.Tensor,
) -> dict[str, float | list[list[float | None]]]:
    """Calculate causal color-swap metrics from shape-by-hue predictions."""
    predictions = torch.as_tensor(predictions, dtype=torch.long)
    labels = torch.as_tensor(labels, dtype=torch.long)
    digit_to_hue = torch.as_tensor(digit_to_hue, dtype=torch.long)
    if predictions.ndim != 2 or predictions.shape[1] != 10:
        raise ValueError("predictions must have shape [num_shapes, 10 hues]")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must contain the same number of shapes")

    shape_ids = torch.arange(len(labels))
    signature_hues = digit_to_hue[labels]
    correct = predictions == labels[:, None]
    aligned_correct = correct[shape_ids, signature_hues]

    conflict_mask = torch.ones_like(correct, dtype=torch.bool)
    conflict_mask[shape_ids, signature_hues] = False
    reference_predictions = predictions[shape_ids, signature_hues]
    changed = predictions != reference_predictions[:, None]

    group_matrix: list[list[float | None]] = []
    observed_group_accuracies: list[float] = []
    for digit in range(10):
        digit_mask = labels == digit
        row: list[float | None] = []
        for hue in range(10):
            if digit_mask.any():
                accuracy = float(correct[digit_mask, hue].float().mean())
                row.append(accuracy)
                observed_group_accuracies.append(accuracy)
            else:
                row.append(None)
        group_matrix.append(row)

    return {
        "aligned_accuracy": float(aligned_correct.float().mean()),
        "bias_conflict_accuracy": float(correct[conflict_mask].float().mean()),
        "pairwise_flip_rate": float(changed[conflict_mask].float().mean()),
        "counterfactual_consistency": float(correct.all(dim=1).float().mean()),
        "worst_group_accuracy": min(observed_group_accuracies),
        "digit_hue_accuracy": group_matrix,
    }
