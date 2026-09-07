from dataclasses import dataclass
import math

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ColoredMetadata:
    hues: torch.Tensor
    is_conflict: torch.Tensor
    digit_to_hue: torch.Tensor


def equal_luma_palette(
    num_hues: int = 10,
    luma: float = 0.5,
    chroma: float = 0.12,
) -> torch.Tensor:
    """Create RGB colors with nearly identical NTSC luma using YIQ coordinates."""
    angles = 2 * math.pi * torch.arange(num_hues, dtype=torch.float32) / num_hues
    i_component = chroma * torch.cos(angles)
    q_component = chroma * torch.sin(angles)
    red = luma + 0.956 * i_component + 0.621 * q_component
    green = luma - 0.272 * i_component - 0.647 * q_component
    blue = luma - 1.106 * i_component + 1.703 * q_component
    palette = torch.stack((red, green, blue), dim=1)
    if torch.any((palette < 0) | (palette > 1)):
        raise ValueError("Palette parameters produce out-of-gamut RGB values")
    return palette


def build_balanced_split(
    targets: torch.Tensor,
    train_per_class: int,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic class-balanced train indices and all remaining indices."""
    targets = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for digit in range(num_classes):
        class_indices = torch.where(targets == digit)[0]
        if len(class_indices) < train_per_class:
            raise ValueError(
                f"Class {digit} has {len(class_indices)} samples; {train_per_class} required"
            )
        order = torch.randperm(len(class_indices), generator=generator)
        shuffled = class_indices[order]
        train_parts.append(shuffled[:train_per_class])
        validation_parts.append(shuffled[train_per_class:])
    return torch.cat(train_parts), torch.cat(validation_parts)


def build_colored_metadata(
    labels: torch.Tensor,
    budget: int,
    seed: int,
    num_classes: int = 10,
) -> ColoredMetadata:
    """Assign signature hues and a deterministic nested set of color conflicts."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    if budget < 0 or budget % num_classes:
        raise ValueError(f"budget must be a non-negative multiple of {num_classes}")
    counts = torch.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or not torch.all(counts == counts[0]):
        raise ValueError("labels must contain the same number of samples per class")
    conflicts_per_digit = budget // num_classes
    if conflicts_per_digit > int(counts[0]):
        raise ValueError("budget exceeds the available samples")

    generator = torch.Generator().manual_seed(seed)
    digit_to_hue = torch.randperm(num_classes, generator=generator)
    hues = digit_to_hue[labels].clone()
    is_conflict = torch.zeros(len(labels), dtype=torch.bool)

    for digit in range(num_classes):
        class_positions = torch.where(labels == digit)[0]
        order = torch.randperm(len(class_positions), generator=generator)
        selected = class_positions[order[:conflicts_per_digit]]
        is_conflict[selected] = True
        for rank, position in enumerate(selected):
            offset = 1 + rank % (num_classes - 1)
            alternative_digit = (digit + offset) % num_classes
            hues[position] = digit_to_hue[alternative_digit]

    return ColoredMetadata(
        hues=hues,
        is_conflict=is_conflict,
        digit_to_hue=digit_to_hue,
    )


def colorize(
    grayscale: torch.Tensor,
    hue: int,
    palette: torch.Tensor | None = None,
) -> torch.Tensor:
    """Color a grayscale digit while leaving its spatial mask unchanged."""
    palette = equal_luma_palette() if palette is None else palette
    image = torch.as_tensor(grayscale)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {tuple(image.shape)}")
    image = image.to(torch.float32)
    if image.max() > 1:
        image = image / 255.0
    return palette[int(hue)].to(image.dtype)[:, None, None] * image[None, :, :]


class ColoredMNIST(Dataset):
    """Tensor-backed Colored MNIST training or validation dataset."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata: ColoredMetadata,
        palette: torch.Tensor | None = None,
    ) -> None:
        if not (len(images) == len(labels) == len(metadata.hues)):
            raise ValueError("images, labels, and metadata must have the same length")
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.metadata = metadata
        self.palette = equal_luma_palette() if palette is None else palette

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        label = int(self.labels[index])
        conflict = int(self.metadata.is_conflict[index])
        group = label * 2 + conflict
        image = colorize(self.images[index], int(self.metadata.hues[index]), self.palette)
        return image, label, group, int(index)


class CounterfactualMNIST(Dataset):
    """Render every base digit shape in each available hue."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        palette: torch.Tensor | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.palette = equal_luma_palette() if palette is None else palette
        self.num_hues = len(self.palette)

    def __len__(self) -> int:
        return len(self.labels) * self.num_hues

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        shape_index, hue = divmod(int(index), self.num_hues)
        image = colorize(self.images[shape_index], hue, self.palette)
        return image, int(self.labels[shape_index]), hue, shape_index


def select_dfr_indices(
    labels: torch.Tensor,
    is_conflict: torch.Tensor,
    seed: int,
    num_classes: int = 10,
) -> torch.Tensor:
    """Select every conflict and an equal digit-matched sample of aligned examples."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    is_conflict = torch.as_tensor(is_conflict, dtype=torch.bool)
    if len(labels) != len(is_conflict):
        raise ValueError("labels and is_conflict must have the same length")
    if not is_conflict.any():
        raise ValueError("DFR requires at least one conflict example")
    generator = torch.Generator().manual_seed(seed)
    selected_parts: list[torch.Tensor] = []
    for digit in range(num_classes):
        conflict_indices = torch.where((labels == digit) & is_conflict)[0]
        aligned_indices = torch.where((labels == digit) & ~is_conflict)[0]
        required = len(conflict_indices)
        if required == 0:
            raise ValueError(f"Digit {digit} has no conflict examples")
        if len(aligned_indices) < required:
            raise ValueError(f"Digit {digit} has too few aligned examples")
        order = torch.randperm(len(aligned_indices), generator=generator)
        selected_parts.extend((conflict_indices, aligned_indices[order[:required]]))
    return torch.cat(selected_parts)
