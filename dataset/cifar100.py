from __future__ import print_function

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = os.environ.get(
        'DCKD_DATA_ROOT',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
    )

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CIFAR100BackCompat(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

class CIFAR100Instance(CIFAR100BackCompat):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(root, batch_size=128, num_workers=8, train_transform=None):
    #train_transform = transforms.Compose([
    #    transforms.RandomCrop(32, padding=4),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    #])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root=root,
                                download=False,
                                train=True,
                                transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root,
                                 download=False,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def get_cifar100_train_val_test_dataloaders(
        root,
        batch_size=128,
        num_workers=8,
        train_transform=None,
        val_size=0,
        split_seed=42):
    """Build the CIFAR-100 train loader and untouched test loader.

    With ``val_size=0``, all 50,000 training examples are used for training and
    the returned validation loader is ``None``. A positive ``val_size`` keeps
    the previous deterministic held-out-validation behavior for compatibility.
    """
    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    train_aug_set = datasets.CIFAR100(
        root=root,
        download=False,
        train=True,
        transform=train_transform,
    )
    if not 0 <= val_size < len(train_aug_set):
        raise ValueError(
            f"val_size must be between 0 and {len(train_aug_set) - 1}, "
            f"got {val_size}"
        )

    if val_size == 0:
        train_loader = DataLoader(
            train_aug_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = None
    else:
        train_clean_set = datasets.CIFAR100(
            root=root,
            download=False,
            train=True,
            transform=clean_transform,
        )
        split_generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(
            len(train_aug_set), generator=split_generator
        ).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_loader = DataLoader(
            Subset(train_aug_set, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(train_clean_set, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_set = datasets.CIFAR100(
        root=root,
        download=False,
        train=False,
        transform=clean_transform,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader

def get_cifar100_dataloaders_ori(root, batch_size=128, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root=root,
                                download=False,
                                train=True,
                                transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root,
                                 download=False,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader

def get_cifar100_dataloaders_clean(root, batch_size=128, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root=root,
                                download=False,
                                train=True,
                                transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    return train_loader

class CIFAR100InstanceSample(CIFAR100BackCompat):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets
       
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
