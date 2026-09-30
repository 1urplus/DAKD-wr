from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss
import torchvision.transforms as transforms

def train_aug_agent(context, action, advantage, aug_agent, optimizer,
                    entropy_coef=1e-3):
    """Update the epoch-level augmentation policy with REINFORCE.

    ``action`` must be the action that was actually used to construct the
    current epoch's training loader. ``advantage`` is computed only after that
    epoch finishes, which avoids regressing the policy to its own unused
    current prediction.
    """
    aug_agent.train()
    dist = aug_agent.get_dist(context)
    log_prob = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    advantage_tensor = torch.as_tensor(
        advantage,
        dtype=log_prob.dtype,
        device=log_prob.device,
    )
    policy_loss = -(advantage_tensor.detach() * log_prob).mean()
    entropy_bonus = entropy_coef * entropy.mean()
    loss = policy_loss - entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(aug_agent.parameters(), 5.0)
    optimizer.step()

    return {
        'loss': float(loss.item()),
        'policy_loss': float(policy_loss.item()),
        'entropy': float(entropy.mean().item()),
        'grad_norm': float(grad_norm),
    }


def aggregate_aug_states(epoch_states):
    """Aggregate detached sample feedback into one epoch-level context."""
    if not epoch_states:
        return None

    aggregated = []
    for component_idx in range(4):
        component = torch.cat(
            [state[component_idx] for state in epoch_states], dim=0
        )
        aggregated.append(component.mean(dim=0, keepdim=True).detach())
    return tuple(aggregated)

def train_agent(agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')

    for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad() 
        
        action_label = torch.ones_like(agent_pred[0]).detach()
        criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
        logits_loss = criterion_bce(agent_pred[0], action_label)
        feature_loss = criterion_bce(agent_pred[1], action_label)
        logits_loss = (logits_loss.mean(dim=1) * rewards).mean()
        feature_loss = (feature_loss.mean(dim=1) * rewards).mean()
        loss = logits_loss + feature_loss
        loss.backward()
        agent_optimizer.step()
        agent_loss.update(loss.item(), actions.size(0))

def get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div, args):
    trans_student_embeddings = []
    for idx in range(len(trans_student_features)):
        trans_student_embedding = F.adaptive_avg_pool2d(trans_student_features[idx], (1,1))
        trans_student_embedding = trans_student_embedding.view(trans_student_embedding.size(0), -1)
        trans_student_embeddings.append(trans_student_embedding)

    teacher_infos = []
    t_ces = []
    t_s_feat_div = []
    t_s_logit_div = []
    for idx in range(len(teacher_embeddings)):            
        feat_cos_sim = F.cosine_similarity(trans_student_embeddings[idx], teacher_embeddings[idx]).unsqueeze(-1)
        t_s_feat_div.append(feat_cos_sim)
        
        # 使用 KL 散度作为统一的代理指标
        logit_div = DistillKL(4)(logits, teacher_logits[idx], unreduce=True)

        logit_kl = logit_div.unsqueeze(-1)
        t_s_logit_div.append(logit_kl)
        teachers_ce = F.cross_entropy(teacher_logits[idx], targets, reduction='none').unsqueeze(-1)
        t_ces.append(teachers_ce)
        
        teacher_info = torch.cat([feat_cos_sim, logit_kl, teachers_ce, teacher_embeddings[idx], teacher_logits[idx]], dim=1).detach()
        teacher_infos.append(teacher_info)
    
    t_ces = torch.cat(t_ces, dim=1).detach()
    t_s_logit_div = torch.cat(t_s_logit_div, dim=1).detach()
    t_s_feat_div = torch.cat(t_s_feat_div, dim=1).detach()
    return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div

def get_aug_agent_state(t_s_logit_div, t_s_feat_div, logits, teacher_logits):

    with torch.no_grad():        
        all_t_probs = torch.stack([F.softmax(tl, dim=1) for tl in teacher_logits])  
        avg_t_prob = all_t_probs.mean(dim=0)  
        h_avg = -torch.sum(avg_t_prob * torch.log(avg_t_prob + 1e-10), dim=1, keepdim=True)  
        teacher_entropies = -torch.sum(all_t_probs * torch.log(all_t_probs + 1e-10), dim=2)  
        avg_h = torch.mean(teacher_entropies, dim=0, keepdim=True).transpose(0, 1)  
        t_consensus = h_avg - avg_h  
        s_entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1, keepdim=True)  

        aug_agent_state = (t_consensus, s_entropy, t_s_logit_div, t_s_feat_div)
        
        return aug_agent_state
   
def train(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, agent, aug_agent, feat_trans, teacher_models, agent_optimizer,
          aug_optimizer, model_ema=None):
    
    global _global_aug_actions
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_div = AverageMeter('train_loss_div', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)

    kd_weight = args.kd_weight
    feat_weight = args.feat_weight
    if args.loss_anneal_start >= 0 and epoch >= args.loss_anneal_start:
        anneal_span = max(1, args.epochs - 1 - args.loss_anneal_start)
        anneal_progress = min(
            1.0, (epoch - args.loss_anneal_start) / anneal_span
        )
        kd_weight_end = (
            args.kd_weight if args.kd_weight_end is None
            else args.kd_weight_end
        )
        feat_weight_end = (
            args.feat_weight if args.feat_weight_end is None
            else args.feat_weight_end
        )
        kd_weight = (
            args.kd_weight
            + anneal_progress * (kd_weight_end - args.kd_weight)
        )
        feat_weight = (
            args.feat_weight
            + anneal_progress * (feat_weight_end - args.feat_weight)
        )

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    agent.eval()
    aug_agent.eval()
    
    # 为每个epoch收集数据用于更新智能体
    epoch_aug_agent_states = []
    epoch_agent_states = []
    epoch_agent_rewards = []
    epoch_logits_agent_actions = []
    epoch_feature_agent_actions = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break
        batch_start = time.time()
        inputs = inputs.to(device, non_blocking=True)
        
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        features, logits = model(inputs, is_feat=True) 
        trans_student_features = feat_trans(features[-2])
        
        # 获取教师输出
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        with torch.no_grad():
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                teacher_features.append(t_features[-2].detach())
                teacher_logits.append(t_logits.detach())
                teacher_embeddings.append(t_features[-1].detach())

        agent_state = get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div, args)
        aug_agent_state = get_aug_agent_state(agent_state[2], agent_state[3], logits, teacher_logits)

        # 收集当前epoch的数据
        epoch_agent_states.append(agent_state)
        epoch_aug_agent_states.append(aug_agent_state)

        with torch.no_grad():
            logits_actions, feature_actions = agent(agent_state)
            
        # 第一个epoch使用全1初始化
        if epoch == 0 :
            initial_action = (
                1.0 / len(teacher_models)
                if args.normalize_first_epoch_actions else 1.0
            )
            logits_actions = torch.full_like(logits_actions, initial_action).cuda(args.gpu)
            feature_actions = torch.full_like(feature_actions, initial_action).cuda(args.gpu)
        
        # 保存detached的版本
        logits_actions_detached = logits_actions.detach()
        feature_actions_detached = feature_actions.detach()
        epoch_logits_agent_actions.append(logits_actions_detached)
        epoch_feature_agent_actions.append(feature_actions_detached)

        if args.rank == 0 and batch_idx % 800 == 0:
            args.logger.info('actions:{}'.format(str(logits_actions_detached[0])))
        
        # ============ 计算损失 ============
        loss_cls = criterion_ce(logits, targets)
        
        loss_div = torch.tensor(0.).cuda(args.gpu)
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        teacher_num = len(teacher_models)
        if args.distiller == 'catkd':
            # CATKD
            for idx in range(len(teacher_models)):
                catkd_loss = criterion_div(
                    s_features=trans_student_features[idx],
                    t_features=teacher_features[idx],
                    s_logits=logits,
                    t_logits=teacher_logits[idx],
                    labels=targets
                )
                #weight = feature_actions_detached[:, idx].mean()
                #loss_div += weight * catkd_loss
                loss_div += (feature_actions_detached[:, idx] * catkd_loss).mean()
                
                
        elif args.distiller == 'rkd':
            for idx in range(teacher_num):
                # rkd_loss = criterion_div(trans_student_features[idx], teacher_features[idx])
                # weight = feature_actions_detached[:, idx].mean()
                # loss_feat += weight * rkd_loss
                rkd_loss = criterion_div(trans_student_features[idx], teacher_features[idx])
                weight = feature_actions_detached[:, idx].mean()
                loss_feat += weight * rkd_loss
                
            for idx in range(teacher_num):
                sample_kd = DistillKL(4)(logits, teacher_logits[idx].detach(), unreduce=True)
                loss_div += (logits_actions_detached[:, idx] * sample_kd).mean()

        elif args.distiller == 'sp':
            # SP compares the native student/teacher representation geometry;
            # no channel-alignment transform is required. Use row-wise losses
            # so teacher actions genuinely weight individual samples.
            student_features = [features[-2]] * teacher_num
            sp_losses = criterion_div(
                student_features, teacher_features, unreduce=True
            )
            for idx, sp_loss in enumerate(sp_losses):
                loss_feat += (feature_actions_detached[:, idx] * sp_loss).mean()
            
            for idx in range(teacher_num):
                sample_kd = DistillKL(4)(logits, teacher_logits[idx].detach(), unreduce=True)
                loss_div += (logits_actions_detached[:, idx] * sample_kd).mean()

        elif args.distiller == 'dkd':
            for idx in range(teacher_num):
                sample_kd = criterion_div(logits, teacher_logits[idx].detach(), targets, unreduce=True)
                loss_div += (logits_actions_detached[:, idx] * sample_kd).mean()
            
            feat_kd_func = FeatureMSELoss() if args.feat_kd == 'mse' else FeatureKLLoss(args.kd_T)
            for idx in range(teacher_num):
                loss_feat += (feature_actions_detached[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
                
        elif args.distiller == 'kl':
            for idx in range(teacher_num):
                sample_kd = criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)
                loss_div += (logits_actions_detached[:, idx] * sample_kd).mean()
            
            feat_kd_func = FeatureMSELoss() if args.feat_kd == 'mse' else FeatureKLLoss(args.kd_T)
            for idx in range(teacher_num):
                loss_feat += (feature_actions_detached[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
        
        effective_feat_weight = (
            args.sp_weight if args.distiller == 'sp' else feat_weight
        )
        loss_feat = effective_feat_weight * loss_feat
        loss = args.ce_weight * loss_cls + kd_weight * loss_div + loss_feat
        loss.backward()
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        # ============ 计算奖励 ============
        with torch.no_grad():
            sample_ce_loss = F.cross_entropy(logits, targets, reduction='none')
            sample_kd_loss = torch.zeros_like(sample_ce_loss)
            
            # KL 散度作为 KD 损失代理
            for idx in range(len(teacher_models)):
                sample_kd = DistillKL(4)(logits, teacher_logits[idx], unreduce=True)
                sample_kd_loss += logits_actions_detached[:, idx] * sample_kd
            
            # SP uses its true row-wise similarity loss as the reward proxy;
            # the other methods retain the existing per-sample MSE proxy.
            sample_feat_loss = torch.zeros_like(sample_ce_loss)
            if args.distiller == 'sp':
                sp_reward_losses = criterion_div(
                    [features[-2]] * teacher_num,
                    teacher_features,
                    unreduce=True,
                )
                for idx, per_sample_sp in enumerate(sp_reward_losses):
                    sample_feat_loss += (
                        feature_actions_detached[:, idx] * per_sample_sp
                    )
            else:
                for idx in range(len(teacher_models)):
                    s_feat_flat = trans_student_features[idx].view(trans_student_features[idx].size(0), -1)
                    t_feat_flat = teacher_features[idx].view(teacher_features[idx].size(0), -1)
                    per_sample_mse = F.mse_loss(s_feat_flat, t_feat_flat, reduction='none').mean(dim=1)  # [B]
                    sample_feat_loss += feature_actions_detached[:, idx] * per_sample_mse
            
            reward = -(
                args.ce_weight * sample_ce_loss
                + kd_weight * sample_kd_loss
                + effective_feat_weight * sample_feat_loss
            )
            rewards_mean = reward.mean()
            rewards_std = reward.std()
            normalized_reward = (reward - rewards_mean) / (rewards_std + 1e-6)
            normalized_reward = torch.tanh(normalized_reward)

            epoch_agent_rewards.append(normalized_reward.detach())
        
        # 更新统计
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_div.update(loss_div.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        top1, top5 = correct_num(logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)
        
        if args.rank == 0 and batch_idx % 800 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Time: {time.time()-batch_start:.2f}s')
    
    epoch_time = time.time() - start_time
    if len(epoch_agent_states) > 0:
        train_agent(epoch_agent_states, epoch_agent_rewards, epoch_logits_agent_actions, agent, agent_optimizer)
    epoch_context = aggregate_aug_states(epoch_aug_agent_states)

    top1_acc = top1_num / total * 100.
    top5_acc = top5_num / total * 100.

    if args.rank == 0:
        log_msg = (
            f'Epoch: {epoch} | '
            f'Time: {epoch_time:.2f}s | '
            f'LR: {lr:.5f} | '
            f'Loss: {train_loss.avg:.4f} | '
            f'CLS: {train_loss_cls.avg:.4f} | '
            f'Div: {train_loss_div.avg:.4f} | '
            f'Feat: {train_loss_feat.avg:.4f} | '
            f'Top-1 Acc: {top1_acc:.2f}% | '
            f'Top-5 Acc: {top5_acc:.2f}%'
        )
        print(log_msg)

    return epoch_context, float(train_loss.avg), float(top1_acc)


def test(epoch, net, device, val_loader, criterion_ce, args, verbose=True,
         return_loss=False, split_name='Test'):
    test_loss_cls = AverageMeter('Loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break
            batch_start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            features, logits = net(inputs, is_feat=True)
            loss_cls = criterion_ce(logits, targets)

            test_loss_cls.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            
            if args.rank == 0 and verbose and batch_idx % 200 == 0:
                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}s'.format(
                    epoch, batch_idx, len(val_loader), time.time()-batch_start_time))
            
        class_acc1 = round((top1_num/total*100.).item(), 4)
        class_acc5 = round((top5_num/total*100.).item(), 4)

        if args.rank == 0 and verbose:
            args.logger.info('{} epoch:{}\t loss:{:.5f}\n{} top-1 accuracy: {}\n{} top-5 accuracy: {}'
                        .format(split_name, epoch, test_loss_cls.avg,
                                split_name, str(class_acc1),
                                split_name, str(class_acc5)))
    if return_loss:
        return class_acc1, float(test_loss_cls.avg)
    return class_acc1
