import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import os


def get_flowers102_dataloader(
    root: str,
    batch_size: int = 32,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 2,
):
    # Different transforms for training vs validation
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Random crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(
                    p=0.1
                ),  # Some flowers have natural rotations
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2
                ),
                transforms.RandomRotation(30),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1)
                ),  # Small shifts
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random masking
            ]
        )
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    dataset = datasets.Flowers102(
        root=root, split=split, transform=transform, download=True
    )

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )  # Faster data transfer to GPU

    return dataloader
