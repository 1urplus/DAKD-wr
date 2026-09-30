"""RAF-DB aligned-face train/test loading and A0--A3 teacher domains."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


RAFDB_MEAN = (0.485, 0.456, 0.406)
RAFDB_STD = (0.229, 0.224, 0.225)
RAFDB_SIZE = 96


def get_teacher_transform(aug_group):
    common = [
        transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
        transforms.RandomCrop(RAFDB_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
    ]
    if aug_group == "A0":
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
    operations.append(transforms.Normalize(RAFDB_MEAN, RAFDB_STD))
    return transforms.Compose(operations)


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
    ])


def get_default_train_transform():
    return transforms.Compose([
        transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
        transforms.RandomCrop(RAFDB_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
    ])


def get_rafdb_dataloaders(
        root,
        aug_group=None,
        batch_size=64,
        num_workers=8,
        generator=None,
        worker_init_fn=None,
        train_transform=None):
    root = Path(root).expanduser().resolve()
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(
            f"RAF-DB needs train and test directories under {root}"
        )
    if train_transform is None:
        if aug_group is None:
            train_transform = get_default_train_transform()
        else:
            train_transform = get_teacher_transform(aug_group)
    train_set = datasets.ImageFolder(train_root, transform=train_transform)
    test_set = datasets.ImageFolder(test_root, transform=get_eval_transform())
    if train_set.class_to_idx != test_set.class_to_idx:
        raise ValueError("RAF-DB train/test class directories differ")
    if len(train_set.classes) != 7:
        raise ValueError(f"expected 7 RAF-DB classes, got {len(train_set.classes)}")
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
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader
