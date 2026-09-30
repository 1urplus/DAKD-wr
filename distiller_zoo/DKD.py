from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class DKDLoss(nn.Module):
    def __init__(self, alpha, beta, T):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = T

    def forward(self, logits_student, logits_teacher, target, unreduce=False):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.T, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.T, dim=1)

        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)

        log_pred_student_tckd = torch.log(pred_student_tckd.clamp(min=1e-8))
        tckd_loss = F.kl_div(log_pred_student_tckd,pred_teacher_tckd,reduction="none").sum(1) * (self.T ** 2)

        logits_s_nckd = logits_student / self.T - 1000.0 * gt_mask
        logits_t_nckd = logits_teacher / self.T - 1000.0 * gt_mask

        log_pred_student_nckd = F.log_softmax(logits_s_nckd, dim=1)
        pred_teacher_nckd = F.softmax(logits_t_nckd, dim=1)

        nckd_loss = F.kl_div(
            log_pred_student_nckd,
            pred_teacher_nckd,
            reduction="none"
        ).sum(1) * (self.T ** 2)

        total = self.alpha * tckd_loss + self.beta * nckd_loss

        if unreduce:
            return total
        else:
            return total.mean()

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt