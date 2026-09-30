from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t, unreduce=False):
        return [
            self.similarity_loss(f_s, f_t, unreduce=unreduce)
            for f_s, f_t in zip(g_s, g_t)
        ]

    def similarity_loss(self, f_s, f_t, unreduce=False):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        # Each row represents one sample's contribution to the pairwise
        # similarity loss. Its mean is exactly the original scalar SP loss.
        per_sample_loss = (G_diff * G_diff).sum(dim=1) / bsz
        if unreduce:
            return per_sample_loss
        return per_sample_loss.mean()
