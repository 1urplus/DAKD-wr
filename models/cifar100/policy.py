import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

__all__ = ['Policy', 'PolicyTrans','PolicyAug']

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, output_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.head(input)
        return output 

    
class PolicyTrans(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False):
        super(PolicyTrans, self).__init__()
        self.teacher_num = teacher_num
        
        self.sim_trans = nn.ModuleList([])
        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        self.steam = nn.Sequential(
                nn.Linear(all_input_size, 128, bias=False),
                nn.ReLU())
        self.logit_head = nn.Linear(128, teacher_num, bias=True)
        self.feature_head = nn.Linear(128, teacher_num, bias=True)
                
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic = dynamic
        if dynamic:
            self.logit_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state

        if self.teacher_num > 1:
            weight_loss_t = (1. - F.softmax(t_ces, dim=1)) / (self.teacher_num - 1)
            weight_loss_t_s_logit_div = F.softmax(t_s_logit_div, dim=1)
            weight_loss_t_s_feat_div = F.softmax(t_s_feat_div, dim=1)
        else:
            # 单教师时，所有权重恒为 1，避免除零
            weight_loss_t = torch.ones_like(t_ces)
            weight_loss_t_s_logit_div = torch.ones_like(t_s_logit_div)
            weight_loss_t_s_feat_div = torch.ones_like(t_s_feat_div)

        all_teacher_infos = torch.cat(teacher_infos, dim=1)
        out1 = self.steam(all_teacher_infos)
        logit_weights = self.softmax(self.logit_head(out1))
        feature_weights = self.softmax(self.feature_head(out1))

        if self.dynamic:
            l_f = F.softmax(self.logit_weight_factor, dim=0)
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            all_logit_weights = (l_f[0]*logit_weights + l_f[1]*weight_loss_t + l_f[2]*weight_loss_t_s_logit_div)
            all_feature_weights = (f_f[0] * feature_weights + f_f[1] * weight_loss_t + f_f[2] * weight_loss_t_s_feat_div)
        else:
            all_logit_weights = (logit_weights + weight_loss_t + weight_loss_t_s_logit_div)/ 3.
            all_feature_weights = (feature_weights + weight_loss_t + weight_loss_t_s_feat_div) / 3.

        return all_logit_weights,  all_feature_weights
class PolicyAug(nn.Module):
    def __init__(self, input_size, aug_num, teacher_num=4,dynamic=True):
        super(PolicyAug, self).__init__()
        
        self.aug_num = aug_num
        self.dynamic = dynamic
        self.teacher_num = teacher_num

        self.input_size = 2 + 2 * teacher_num 
        
        print(f"PolicyAug input_size: {self.input_size} (calculated: 2 + 2*{teacher_num})")
        print(f"Expected dimensions: t_consensus(1) + s_entropy(1) + t_s_logit_div({teacher_num}) + t_s_feat_div({teacher_num}) = {self.input_size}")
        
        # Epoch-level contextual policy. The four Beta distributions keep the
        # augmentation controls in [0, 1] while still permitting exploration.
        self.stem = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.alpha_head = nn.Linear(128, aug_num, bias=True)
        self.beta_head = nn.Linear(128, aug_num, bias=True)

    def get_dist(self, agent_state):
        t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

        combined_state = torch.cat([
            t_consensus, 
            s_entropy, 
            t_s_logit_div, 
            t_s_feat_div
        ], dim=1)

        latent = self.stem(combined_state)
        alpha = F.softplus(self.alpha_head(latent)) + 1.0
        beta = F.softplus(self.beta_head(latent)) + 1.0
        return Beta(alpha, beta)

    def forward(self, agent_state):
        """Return the deterministic mean action for evaluation/debugging."""
        dist = self.get_dist(agent_state)
        return dist.mean

    def sample_action(self, agent_state):
        """Sample bounded augmentation controls and return policy statistics."""
        dist = self.get_dist(agent_state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4,dynamic=True):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num

#         self.input_size = 2 + 2 * teacher_num 
        
#         # 主干网络 - 使用正确的输入维度
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),  # 输入维度改为 self.input_size
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # 增强头
#         self.aug_head = nn.Linear(128, aug_num, bias=True)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state
#             #h_feedback = torch.mean(F.softmax(t_s_logit_div, dim=1), dim=1, keepdim=True) # (batch, 1)
#             #heuristic_aug = (1.0 - h_feedback).repeat(1, self.aug_num) 
#             # 直接关闭 heuristic 引导
#             #heuristic_aug = torch.zeros_like(predicted_aug)
#         combined_state = torch.cat([t_consensus, s_entropy, t_s_logit_div, t_s_feat_div], dim=1)

#         latent = self.stem(combined_state)
#         predicted_aug = self.sigmoid(self.aug_head(latent))
#         #final_aug = (predicted_aug + heuristic_aug) / 2.0
#         final_aug = predicted_aug
#         return final_aug


# #diff1
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4, dynamic=True):
#         super(PolicyAug, self).__init__()

#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num
#         self.input_size = 2 + 2 * teacher_num

#         print(f"PolicyAug input_size: {self.input_size}")

#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )

#         # 4 个独立 head
#         self.crop_head = nn.Linear(128, 1)
#         self.flip_head = nn.Linear(128, 1)
#         self.color_head = nn.Linear(128, 1)
#         self.rotate_head = nn.Linear(128, 1)

#         if dynamic:
#             # 每个增强维度各自有一个 mixing 权重，而不是全局两个权重
#             self.weight_factor = nn.Parameter(torch.zeros(aug_num, 2))

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

#         combined_state = torch.cat([
#             t_consensus,
#             s_entropy,
#             t_s_logit_div,
#             t_s_feat_div
#         ], dim=1)

#         latent = self.stem(combined_state)

#         crop = torch.sigmoid(self.crop_head(latent))
#         flip = torch.sigmoid(self.flip_head(latent))
#         color = torch.sigmoid(self.color_head(latent))
#         rotate = torch.sigmoid(self.rotate_head(latent))

#         predicted_aug = torch.cat([crop, flip, color, rotate], dim=1)

#         # 维度特异 heuristic
#         crop_h = torch.clamp(1.0 - t_consensus, 0, 1)
#         flip_h = torch.clamp(torch.sigmoid(-s_entropy), 0, 1)
#         color_h = torch.clamp(torch.mean(torch.softmax(t_s_logit_div, dim=1), dim=1, keepdim=True), 0, 1)
#         rotate_h = torch.clamp(1.0 - torch.mean(t_s_feat_div, dim=1, keepdim=True), 0, 1)

#         heuristic_aug = torch.cat([crop_h, flip_h, color_h, rotate_h], dim=1)

#         if self.dynamic:
#             # [4,2] -> [4,2]
#             w = F.softmax(self.weight_factor, dim=1)
#             # 每个维度单独混合
#             final_aug = (
#                 w[:, 0].unsqueeze(0) * predicted_aug +
#                 w[:, 1].unsqueeze(0) * heuristic_aug
#             )
#         else:
#             final_aug = 0.5 * predicted_aug + 0.5 * heuristic_aug

#         return final_aug

# #diff3
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4,dynamic=False):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.teacher_num = teacher_num
#         self.input_size = 2 + 2 * teacher_num 
        
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # 每个 augmentation 一个独立 head
#         self.aug_heads = nn.ModuleList([
#             nn.Linear(128, 1, bias=True) for _ in range(aug_num)
#         ])
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

#         combined_state = torch.cat([
#             t_consensus, 
#             s_entropy, 
#             t_s_logit_div, 
#             t_s_feat_div
#         ], dim=1)
        
#         latent = self.stem(combined_state)

#         aug_list = []
#         for head in self.aug_heads:
#             aug_i = self.sigmoid(head(latent))   # [B,1]
#             aug_list.append(aug_i)

#         final_aug = torch.cat(aug_list, dim=1)  # [B, aug_num]
#         return final_aug

#diff5
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4, dynamic=True, temperature=1.0):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num
#         self.temperature = temperature

#         self.input_size = 2 + 2 * teacher_num 
        
#         # 主干网络
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # 输出每个 augmentation 的竞争 logits
#         self.aug_head = nn.Linear(128, aug_num, bias=True)

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

#         combined_state = torch.cat([
#             t_consensus,
#             s_entropy,
#             t_s_logit_div,
#             t_s_feat_div
#         ], dim=1)

#         latent = self.stem(combined_state)

#         # [B, aug_num]
#         aug_logits = self.aug_head(latent)

#         # softmax 形成竞争关系
#         final_aug = F.softmax(aug_logits / self.temperature, dim=1)

#         return final_aug

#diff6
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4, dynamic=True, temperature=1.0):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num
#         self.temperature = temperature

#         self.input_size = 2 + 2 * teacher_num 
        
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # augmentation 相对竞争
#         self.aug_head = nn.Linear(128, aug_num, bias=True)
        
#         # 全局增强强度
#         self.gate_head = nn.Linear(128, 1, bias=True)

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

#         combined_state = torch.cat([
#             t_consensus,
#             s_entropy,
#             t_s_logit_div,
#             t_s_feat_div
#         ], dim=1)

#         latent = self.stem(combined_state)

#         # augmentation 间竞争
#         aug_logits = self.aug_head(latent)
#         aug_dist = F.softmax(aug_logits / self.temperature, dim=1)   # [B, aug_num]

#         # 总体增强强度
#         global_gate = torch.sigmoid(self.gate_head(latent))          # [B, 1]

#         # 最终动作
#         final_aug = global_gate * aug_dist                           # [B, aug_num]

#         return final_aug
    
# diff7
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num, teacher_num=4, dynamic=True, temperature=1.0):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num
#         self.temperature = temperature
#         self.input_size = 2 + 2 * teacher_num 
        
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         self.score_head = nn.Linear(128, aug_num)
#         self.comp_head = nn.Linear(128, aug_num)

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state

#         combined_state = torch.cat([
#             t_consensus,
#             s_entropy,
#             t_s_logit_div,
#             t_s_feat_div
#         ], dim=1)

#         latent = self.stem(combined_state)

#         independent_score = torch.sigmoid(self.score_head(latent))              # [B, aug_num]
#         competitive_prior = F.softmax(self.comp_head(latent) / self.temperature, dim=1)

#         final_aug = independent_score * competitive_prior

#         return final_aug
    
#diff8
# class PolicyAug(nn.Module):
#     def __init__(self, input_size, aug_num=4, teacher_num=4, dynamic=True):
#         super(PolicyAug, self).__init__()
        
#         self.aug_num = aug_num
#         self.dynamic = dynamic
#         self.teacher_num = teacher_num
#         self.input_size = 2 + 2 * teacher_num 
        
#         # 主干特征提取（共享）
#         self.stem = nn.Sequential(
#             nn.Linear(self.input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
        
#         # 每个aug一个独立的小网络，保证输出独立
#         # self.aug_heads = nn.ModuleList([
#         #     nn.Sequential(
#         #         nn.Linear(128, 32),
#         #         nn.ReLU(),
#         #         nn.Linear(32, 1)  # 每个头只输出1个值
#         #     ) for _ in range(aug_num)
#         # ])
#         self.aug_heads = nn.ModuleList([
#             nn.Linear(128, 1) for _ in range(aug_num)
#         ])
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, agent_state):
#         t_consensus, s_entropy, t_s_logit_div, t_s_feat_div = agent_state
        
#         # 拼接状态
#         combined_state = torch.cat([t_consensus, s_entropy, t_s_logit_div, t_s_feat_div], dim=1)

#         # 共享特征
#         latent = self.stem(combined_state)
        
#         # 独立预测每个增强强度
#         aug_outputs = []
#         for head in self.aug_heads:
#             out = self.sigmoid(head(latent))
#             aug_outputs.append(out)
        
#         # 拼接成 [batch, 4]
#         final_aug = torch.cat(aug_outputs, dim=1)
#         return final_aug
