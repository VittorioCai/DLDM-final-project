import torch

from shortcut_learning.data import (
    ColoredMNIST,
    CounterfactualMNIST,
    build_balanced_split,
    build_colored_metadata,
    colorize,
    equal_luma_palette,
)


def balanced_targets(per_class: int) -> torch.Tensor:
    return torch.arange(10).repeat_interleave(per_class)


def test_balanced_split_has_fixed_per_class_counts_and_is_reproducible():
    targets = balanced_targets(8)
    train_a, val_a = build_balanced_split(targets, train_per_class=5, seed=3)
    train_b, val_b = build_balanced_split(targets, train_per_class=5, seed=3)

    assert len(train_a) == 50
    assert len(val_a) == 30
    assert torch.equal(train_a, train_b)
    assert torch.equal(val_a, val_b)
    assert torch.bincount(targets[train_a], minlength=10).tolist() == [5] * 10
    assert set(train_a.tolist()).isdisjoint(val_a.tolist())


def test_metadata_has_exact_balanced_conflicts_and_constant_hue_marginal():
    labels = balanced_targets(20)
    base = build_colored_metadata(labels, budget=0, seed=4)
    changed = build_colored_metadata(labels, budget=50, seed=4)

    assert len(changed.hues) == 200
    assert int(changed.is_conflict.sum()) == 50
    conflict_labels = labels[changed.is_conflict]
    assert torch.bincount(conflict_labels, minlength=10).tolist() == [5] * 10
    assert torch.equal(
        torch.bincount(base.hues, minlength=10),
        torch.bincount(changed.hues, minlength=10),
    )
    assert torch.equal(base.digit_to_hue, changed.digit_to_hue)


def test_conflict_budgets_are_nested_for_the_same_seed():
    labels = balanced_targets(30)
    small = build_colored_metadata(labels, budget=20, seed=9)
    large = build_colored_metadata(labels, budget=80, seed=9)

    small_ids = set(torch.where(small.is_conflict)[0].tolist())
    large_ids = set(torch.where(large.is_conflict)[0].tolist())
    assert small_ids < large_ids


def test_metadata_is_seed_reproducible_but_changes_across_seeds():
    labels = balanced_targets(30)
    first = build_colored_metadata(labels, budget=50, seed=1)
    repeat = build_colored_metadata(labels, budget=50, seed=1)
    other = build_colored_metadata(labels, budget=50, seed=2)

    assert torch.equal(first.hues, repeat.hues)
    assert torch.equal(first.is_conflict, repeat.is_conflict)
    assert not (
        torch.equal(first.hues, other.hues)
        and torch.equal(first.is_conflict, other.is_conflict)
    )


def test_equal_luma_palette_and_colorize_preserve_shape_mask():
    palette = equal_luma_palette()
    luma = palette @ torch.tensor([0.299, 0.587, 0.114])
    assert palette.shape == (10, 3)
    assert torch.all((palette >= 0) & (palette <= 1))
    torch.testing.assert_close(luma, torch.full_like(luma, luma.mean()), atol=2e-3, rtol=0)

    gray = torch.zeros(28, 28)
    gray[7:21, 10:18] = 1.0
    red = colorize(gray, 0, palette)
    blue = colorize(gray, 5, palette)

    assert red.shape == (3, 28, 28)
    assert torch.equal(red.sum(0) > 0, gray > 0)
    assert torch.equal(blue.sum(0) > 0, gray > 0)


def test_colored_dataset_returns_group_metadata():
    images = torch.randint(0, 256, (20, 28, 28), dtype=torch.uint8)
    labels = torch.arange(10).repeat_interleave(2)
    metadata = build_colored_metadata(labels, budget=10, seed=5)
    dataset = ColoredMNIST(images, labels, metadata)

    image, label, group, sample_id = dataset[0]
    assert image.shape == (3, 28, 28)
    assert label == int(labels[0])
    assert group in range(20)
    assert sample_id == 0


def test_counterfactual_dataset_renders_every_hue_once():
    images = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8)
    labels = torch.tensor([2, 4, 7])
    dataset = CounterfactualMNIST(images, labels)

    assert len(dataset) == 30
    seen = [dataset[index][2] for index in range(10)]
    shape_ids = [dataset[index][3] for index in range(10)]
    assert seen == list(range(10))
    assert shape_ids == [0] * 10
