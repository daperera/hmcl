from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AmbiguousMNIST(Dataset):
    """MNIST wrapper that hides part of the digit to create ambiguous inputs."""

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        mask: str = "bottom",
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        if mask not in {"top", "bottom", "left", "right"}:
            raise ValueError("mask must be one of 'top', 'bottom', 'left', or 'right'.")
        self.mask = mask
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.base[index]
        image = image.clone()
        _, height, width = image.shape
        if self.mask == "bottom":
            image[:, height // 2 :, :] = 0.0
        elif self.mask == "top":
            image[:, : height // 2, :] = 0.0
        elif self.mask == "left":
            image[:, :, : width // 2] = 0.0
        else:
            image[:, :, width // 2 :] = 0.0
        if self.noise_std > 0:
            image = (image + torch.randn_like(image) * self.noise_std).clamp(0.0, 1.0)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
