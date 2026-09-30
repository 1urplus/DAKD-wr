import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

__all__ = ["get_cub200_dataloader"]

class CUBDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform

        # 读取文件
        images_file = os.path.join(root, "images.txt")
        labels_file = os.path.join(root, "image_class_labels.txt")
        split_file = os.path.join(root, "train_test_split.txt")

        # 加载数据
        with open(images_file) as f:
            self.id2img = {int(x.split()[0]): x.split()[1] for x in f.readlines()}
        with open(labels_file) as f:
            self.id2label = {int(x.split()[0]): int(x.split()[1]) - 1 for x in f.readlines()}  
            # 减1，类别从 0-199
        with open(split_file) as f:
            self.id2split = {int(x.split()[0]): int(x.split()[1]) for x in f.readlines()}

        # 筛选 train/test
        self.samples = []
        for img_id, rel_path in self.id2img.items():
            is_train = self.id2split[img_id] == 1
            if is_train == train:
                label = self.id2label[img_id]
                img_path = os.path.join(root, "images", rel_path)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 数据增强 ----------
def get_cub_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_cub_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------- dataloader ----------
def get_cub200_dataloaders(root, batch_size, num_workers, train_transform):
    train_dataset = CUBDataset(root, train=True, transform=train_transform)
    test_dataset = CUBDataset(root, train=False, transform=get_cub_test_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def get_cub200_dataloaders_ori(root, batch_size, num_workers):
    train_dataset = CUBDataset(root, train=True, transform=get_cub_train_transform())
    test_dataset = CUBDataset(root, train=False, transform=get_cub_test_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader