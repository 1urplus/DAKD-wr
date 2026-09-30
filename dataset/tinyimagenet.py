"""Tiny-ImageNet data loading and the four teacher augmentation domains."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


def get_teacher_transform(aug_group):
    """Return the Tiny-ImageNet version of the existing A0--A3 domains."""
    common = [
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if aug_group == "A0":
        # Existing DCKD A0: spatial augmentation plus random erasing.
        operations = common + [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5),
        ]
    elif aug_group == "A1":
        operations = common + [
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    elif aug_group == "A2":
        operations = common + [
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.ToTensor(),
        ]
    elif aug_group == "A3":
        operations = common + [
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"unknown augmentation group: {aug_group}")

    operations.append(transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD))
    return transforms.Compose(operations)


class TinyImageNetValidation(Dataset):
    """Read the original flat Tiny-ImageNet validation directory."""

    def __init__(self, root, class_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        annotations_path = self.root / "val" / "val_annotations.txt"
        image_dir = self.root / "val" / "images"

        samples = []
        with annotations_path.open(encoding="utf-8") as annotations:
            for line in annotations:
                filename, wnid, *_ = line.rstrip().split("\t")
                samples.append((image_dir / filename, class_to_idx[wnid]))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_tinyimagenet_dataloaders(
        root,
        aug_group=None,
        batch_size=64,
        num_workers=8,
        generator=None,
        worker_init_fn=None,
        train_transform=None):
    root = Path(root).expanduser().resolve()
    required_paths = [
        root / "train",
        root / "val" / "images",
        root / "val" / "val_annotations.txt",
        root / "wnids.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Tiny-ImageNet is incomplete; missing: " + ", ".join(missing))

    if train_transform is None:
        if aug_group is None:
            raise ValueError("aug_group is required when train_transform is not provided")
        train_transform = get_teacher_transform(aug_group)

    train_set = datasets.ImageFolder(root / "train", transform=train_transform)
    if len(train_set.classes) != 200:
        raise ValueError(f"expected 200 training classes, found {len(train_set.classes)}")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
    ])
    val_set = TinyImageNetValidation(
        root,
        class_to_idx=train_set.class_to_idx,
        transform=eval_transform,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader
