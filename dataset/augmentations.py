# dataset/augmentations.py
import random
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
def get_augmentation_group(name):
    """
    Return train transform for a given augmentation group
    """
    if name == 'A0':  # CutMix / MixUp / Erasing
        # CutMix/MixUp 通常在 training loop 里做
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif name == 'A1':  # Weak / Clean
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    elif name == 'A2':  # Photometric
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    elif name == 'A3':  # Blur / Noise / JPEG (示意)
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])


# ===== curriculum policy =====这个是断点式
def get_curriculum_aug_name1(epoch, total_epochs):
    p = epoch / total_epochs

    if p < 0.25:
        return 'A0'
    elif p < 0.5:
        return 'A1'
    elif p < 0.75:
        return 'A2'
    else:
        return 'A3'
    
AUG_ORDER = ['A0', 'A1', 'A2', 'A3']
def get_curriculum_aug_name2(epoch, total_epochs, temperature=0.5):
    p = epoch / total_epochs
    positions = np.linspace(0, 1, len(AUG_ORDER))
    # 当前 curriculum 中心
    center = p
    # 高斯权重（连续）
    weights = np.exp(- (positions - center) ** 2 / temperature)
    weights = weights / weights.sum()
    return random.choices(AUG_ORDER, weights=weights, k=1)[0]
import torch
import torch.nn.functional as F
import random
# 断点式衰减：在特定epoch突然增加增强强度
def get_step_rate(epoch, breakpoints=[60, 120, 180, 240]):
    """
    断点式衰减函数
    breakpoints: 增强强度发生变化的epoch节点
    """
    if epoch < breakpoints[0]:
        return 0.2
    elif epoch < breakpoints[1]:
        return 0.3
    elif epoch < breakpoints[2]:
        return 0.5
    elif epoch < breakpoints[3]:
        return 0.75
    else:
        return 1.0


def apply_augmentation(inputs, strengths,epoch):
    batch_size = inputs.size(0)
    _, _, H, W = inputs.shape
    rate = get_step_rate(epoch)
    strengths = torch.clamp(strengths, 0.0, 1.0)
    inputs = torch.clamp(inputs, 0.0, 1.0)
    
    # ============ 1. 亮度调整 ============
    brightness = 1.0 + strengths[:, 0].view(-1, 1, 1, 1) * 0.2 *rate
    inputs = inputs * brightness
    
    # ============ 2. 对比度调整 ============
    mean = inputs.mean(dim=(2, 3), keepdim=True)
    contrast = 1.0 + strengths[:, 1].view(-1, 1, 1, 1) * 0.2 *rate
    inputs = (inputs - mean) * contrast + mean

    # ============ 3. 噪声 ============
    noise_level = strengths[:, 2].view(-1, 1, 1, 1) * 0.05 *rate
    noise = torch.randn_like(inputs) * noise_level
    inputs = inputs + noise
    # ============ 3. mask（关闭） ============
    mask_ratio = strengths[:, 3].view(-1, 1, 1, 1) * 0.5 *rate
    
    for i in range(batch_size):
        ratio = mask_ratio[i].item()
        total_pixels = H * W
        num_mask_pixels = int(total_pixels * ratio)
        mask = torch.ones(H, W, device=inputs.device)
        flat_indices = torch.randperm(total_pixels, device=inputs.device)[:num_mask_pixels]
        mask.view(-1)[flat_indices] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)
        inputs[i] = inputs[i] * mask
    
    inputs = torch.clamp(inputs, 0.0, 1.0)
    
    return inputs
# ===== curriculum wrapper =====
class CurriculumAugment:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, img):
        aug_name = get_curriculum_aug_name1(self.epoch, self.total_epochs)
        transform = get_augmentation_group(aug_name)
        print(aug_name)
        return transform(img)
