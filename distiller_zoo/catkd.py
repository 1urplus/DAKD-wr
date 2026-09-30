import torch
import torch.nn as nn
import torch.nn.functional as F


# distiller_zoo.py 中添加优化版 CATKDLoss

class CATKDLoss(nn.Module):
    """
    Optimized Class-Aware Transfer Knowledge Distillation Loss
    """
    def __init__(self, opt=None, T=4.0, alpha=1.0, beta=1.0, gamma=1.0):
        super(CATKDLoss, self).__init__()
        
        if opt is not None:
            self.T = getattr(opt, 'catkd_T', 4.0)
            self.alpha = getattr(opt, 'catkd_alpha', 1.0)
            self.beta = getattr(opt, 'catkd_beta', 1.0)
            self.gamma = getattr(opt, 'catkd_gamma', 1.0)
        else:
            self.T = T
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
    
    def _compute_similarity_matrix(self, features):
        """计算归一化后的相似度矩阵"""
        # features: [B, C]
        features = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features, features.t())  # [B, B]
        return sim / self.T
    
    def _compute_class_mask(self, labels, num_classes):
        """生成类别掩码矩阵 [B, num_classes]"""
        B = labels.size(0)
        mask = torch.zeros(B, num_classes, device=labels.device)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        return mask
    
    def forward(self, s_features, t_features, s_logits=None, t_logits=None, labels=None):
        """
        Args:
            s_features: [B, C, H, W] or [B, C]
            t_features: [B, C, H, W] or [B, C]
            s_logits: [B, num_classes] (optional)
            t_logits: [B, num_classes] (optional)
            labels: [B]
        """
        if labels is None:
            # 如果没有标签，退化为简单的特征MSE + logit KL
            if s_features.dim() > 2:
                s_f = F.adaptive_avg_pool2d(s_features, (1, 1)).view(s_features.size(0), -1)
            else:
                s_f = s_features
            
            if t_features.dim() > 2:
                t_f = F.adaptive_avg_pool2d(t_features, (1, 1)).view(t_features.size(0), -1)
            else:
                t_f = t_features
            
            feat_loss = F.mse_loss(s_f, t_f.detach())
            
            if s_logits is not None and t_logits is not None:
                logit_loss = F.kl_div(
                    F.log_softmax(s_logits / self.T, dim=1),
                    F.softmax(t_logits.detach() / self.T, dim=1),
                    reduction='batchmean'
                ) * (self.T ** 2)
                return feat_loss + logit_loss
            return feat_loss
        
        # 特征处理：全局平均池化
        if s_features.dim() > 2:
            s_features = F.adaptive_avg_pool2d(s_features, (1, 1))
            s_features = s_features.view(s_features.size(0), -1)
        if t_features.dim() > 2:
            t_features = F.adaptive_avg_pool2d(t_features, (1, 1))
            t_features = t_features.view(t_features.size(0), -1)
        
        B = s_features.size(0)
        num_classes = s_logits.size(1) if s_logits is not None else labels.max().item() + 1
        
        # 计算相似度矩阵
        s_sim = self._compute_similarity_matrix(s_features)  # [B, B]
        t_sim = self._compute_similarity_matrix(t_features).detach()  # [B, B]
        
        # 同类样本掩码。后面的类内 KL 使用一次矩阵计算和 scatter
        # 聚合，避免为每个类别分别发起大量微小 GPU kernel。
        intra_mask = labels[:, None].eq(labels[None, :])
        
        # Inter-class loss: 类间关系
        inter_loss = F.kl_div(
            F.log_softmax(s_sim, dim=1),
            F.softmax(t_sim, dim=1),
            reduction='batchmean'
        )
        
        # Intra-class loss: 对每一行只在同类列上做 softmax，再按类别
        # 等权平均。该式与逐类别 F.kl_div(reduction='batchmean') 等价。
        class_counts = torch.bincount(labels, minlength=num_classes)
        valid_class_mask = class_counts > 1
        if valid_class_mask.any():
            neg_inf = torch.tensor(float('-inf'), device=s_sim.device, dtype=s_sim.dtype)
            s_log_prob = s_sim.masked_fill(~intra_mask, neg_inf)
            t_log_prob = t_sim.masked_fill(~intra_mask, neg_inf)
            s_log_prob = s_log_prob - torch.logsumexp(s_log_prob, dim=1, keepdim=True)
            t_log_prob = t_log_prob - torch.logsumexp(t_log_prob, dim=1, keepdim=True)

            zeros = torch.zeros_like(s_log_prob)
            s_log_prob = torch.where(intra_mask, s_log_prob, zeros)
            t_log_prob = torch.where(intra_mask, t_log_prob, zeros)
            t_prob = torch.where(intra_mask, t_log_prob.exp(), zeros)
            row_kl = (t_prob * (t_log_prob - s_log_prob)).sum(dim=1)

            class_kl_sum = torch.zeros(
                num_classes, device=s_sim.device, dtype=s_sim.dtype
            ).scatter_add_(0, labels, row_kl)
            intra_loss = (
                class_kl_sum[valid_class_mask]
                / class_counts[valid_class_mask].to(s_sim.dtype)
            ).mean()
        else:
            intra_loss = torch.tensor(0.0, device=s_features.device)
        
        # 特征损失
        feature_loss = self.alpha * inter_loss + self.beta * intra_loss
        
        # 添加 logit-based KD loss
        if s_logits is not None and t_logits is not None:
            logit_loss = F.kl_div(
                F.log_softmax(s_logits / self.T, dim=1),
                F.softmax(t_logits.detach() / self.T, dim=1),
                reduction='batchmean'
            ) * (self.T ** 2)
            total_loss = feature_loss + self.gamma * logit_loss
        else:
            total_loss = feature_loss
        
        return total_loss
