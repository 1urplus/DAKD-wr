"""DTD data loading with the official train/validation/test splits."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DTD_MEAN = (0.485, 0.456, 0.406)
DTD_STD = (0.229, 0.224, 0.225)


def get_teacher_transform(aug_group):
    common = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
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
    operations.append(transforms.Normalize(DTD_MEAN, DTD_STD))
    return transforms.Compose(operations)


def get_default_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DTD_MEAN, DTD_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DTD_MEAN, DTD_STD),
    ])


class DTDOfficialSplit(Dataset):
    def __init__(self, root, split_name, split=1, transform=None):
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        image_root = self.root / "images"
        label_path = self.root / "labels" / f"{split_name}{split}.txt"
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"unknown DTD split name: {split_name}")
        if not image_root.is_dir() or not label_path.is_file():
            raise FileNotFoundError(
                f"DTD is incomplete: expected {image_root} and {label_path}"
            )

        classes = sorted(path.name for path in image_root.iterdir() if path.is_dir())
        if len(classes) != 47:
            raise ValueError(f"expected 47 DTD classes, found {len(classes)}")
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.samples = []
        with label_path.open(encoding="utf-8") as split_file:
            for line in split_file:
                relative_path = line.strip()
                if not relative_path:
                    continue
                class_name = relative_path.split("/", 1)[0]
                self.samples.append(
                    (image_root / relative_path, self.class_to_idx[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_dtd_dataloaders(
        root,
        split=1,
        aug_group=None,
        batch_size=64,
        num_workers=8,
        generator=None,
        worker_init_fn=None,
        train_transform=None):
    if split < 1 or split > 10:
        raise ValueError(f"DTD split must be in [1, 10], got {split}")
    if train_transform is None:
        train_transform = (
            get_teacher_transform(aug_group)
            if aug_group is not None
            else get_default_train_transform()
        )
    eval_transform = get_eval_transform()
    train_set = DTDOfficialSplit(root, "train", split, train_transform)
    val_set = DTDOfficialSplit(root, "val", split, eval_transform)
    test_set = DTDOfficialSplit(root, "test", split, eval_transform)

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
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
