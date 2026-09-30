import argparse
import copy
import os
import random
import shutil
import time
import warnings
from enum import Enum
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
# nohup python train_student_rl_dkd3.py --arch MobileNetV2 --t_model resnet32x4 --distiller kl --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_mobile.out_240 2>&1 &
# nohup python train_student_rl_dkd3.py --arch ShuffleV2 --t_model resnet32x4 --distiller sp --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_ShuffleV2.out_sp 2>&1 &
# nohup python train_student_rl_dkd3.py --arch ShuffleV2 --t_model resnet32x4 --distiller rkd --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_ShuffleV2.out_rkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch ShuffleV2 --t_model resnet32x4 --distiller catkd --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_ShuffleV2.out_catkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch RegNetX_200MF --t_model resnet32x4 --distiller kl --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_RegNetX_200MF.out_240_kl 2>&1 &
# nohup python train_student_rl_dkd3.py --arch RegNetX_200MF --t_model resnet32x4 --distiller rkd --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_RegNetX_200MF.out_rkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch RegNetX_200MF --t_model resnet32x4 --distiller catkd --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_RegNetX_200MF.out_catkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch vgg8 --t_model vgg13 --distiller kl --dataset cifar100 --teacher-name-list vggA0 vggA1 vggA2 vggA3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/vgg13_8.out_vgg13.out_240_0.1_diff7 2>&1 &
# nohup python train_student_rl_dkd3.py --arch vgg8 --t_model vgg13 --distiller rkd --dataset cifar100 --teacher-name-list vggA0 vggA1 vggA2 vggA3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/vgg13_8.out_rkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch vgg8 --t_model vgg13 --distiller catkd --dataset cifar100 --teacher-name-list vggA0 vggA1 vggA2 vggA3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/vgg13_8.out_catkd 2>&1 &
# nohup python train_student_rl_final.py --arch resnet32 --t_model ResNet50 --distiller dkd --dataset cifar100 --teacher-name-list 50A0 50A1 50A2 50A3  --dist-backend nccl --world-size 1 --rank 0 > logs/series/cifar100/ResNet50_resnet32_dkd.out_0.1 2>&1 &
# nohup python train_student_rl_dkd3.py --arch resnet14 --t_model ResNet50 --distiller rkd --dataset cifar100 --teacher-name-list 50A0 50A1 50A2 50A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/res50_14.out_rkd 2>&1 &
# nohup python train_student_rl_dkd3.py --arch resnet14 --t_model ResNet50 --distiller rkd --dataset cifar100 --teacher-name-list 50A0 50A1 50A2 50A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/res50_14.out_rkd 2>&1 &

from helper.train_loops_final import train, train_aug_agent, test
from models.cifar100 import model_dict
from setting import  teacher_model_path_dict
from dataset.cifar100 import (
    get_cifar100_dataloaders,
    get_cifar100_train_val_test_dataloaders,
)
from dataset.cub200 import get_cub200_dataloaders
from dataset.tinyimagenet import (
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    get_tinyimagenet_dataloaders,
)
from dataset.dtd import DTD_MEAN, DTD_STD, get_dtd_dataloaders
from dataset.rafdb import RAFDB_MEAN, RAFDB_STD, RAFDB_SIZE, get_rafdb_dataloaders
from utils import set_logger
from models.cub200.util import Regress, TransFeat
import models
from dataset.augmentations import CurriculumAugment
from distiller_zoo import DistillKL,DKDLoss,RKDLoss,CATKDLoss,Similarity

parser = argparse.ArgumentParser(description='DCKD OURS training on CIFAR-100')
parser.add_argument('--data', metavar='DIR', nargs='?',
                    default=os.environ.get('DCKD_DATA_ROOT', './data'),
                    help='path to dataset (default: cifar-100-python)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ShuffleV2')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-passes', default=1, type=int, choices=(1, 2),
                    help='number of complete optimization passes per epoch')
parser.add_argument('--normalize-first-epoch-actions', default=0, type=int,
                    choices=(0, 1),
                    help='initialize epoch-0 teacher actions to 1/N instead of 1')
parser.add_argument('--model-ema-decay', default=0.9999, type=float,
                    help='student model EMA decay; set to 0 to disable')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--w_acc', type=float, default=0.2, help='Weight for student accuracy reward')
parser.add_argument('--w_js', type=float, default=0.8, help='Weight for JS divergence penalty')
parser.add_argument('--w_conf', type=float, default=0.3, help='Weight for teacher confidence bonus')
parser.add_argument('--w_orig', type=float, default=0.5, help='Weight for original reward combination')
parser.add_argument('--val-size', type=int, default=0,
                    help='held-out CIFAR-100 samples; 0 trains on all 50k and '
                         'uses relative training-loss improvement as policy reward')
parser.add_argument('--aug-agent-lr', type=float, default=1e-4,
                    help='learning rate for the augmentation policy')
parser.add_argument('--aug-entropy-coef', type=float, default=1e-3,
                    help='entropy regularization coefficient for the augmentation policy')
parser.add_argument('--aug-baseline-momentum', type=float, default=0.9,
                    help='EMA momentum for the augmentation reward baseline')
parser.add_argument('--max-train-batches', type=int, default=0,
                    help='limit train batches for smoke tests; 0 uses all batches')
parser.add_argument('--max-eval-batches', type=int, default=0,
                    help='limit evaluation batches for smoke tests; 0 uses all batches')
parser.add_argument('--late-test-start', type=int, default=210,
                    help='with val_size=0, evaluate every epoch starting at '
                         'this epoch and retain the best test checkpoint; '
                         'set below 0 to keep only the final epoch')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--form', type=str, default='', help='transform')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
parser.add_argument('--ablation', default='full', choices=['full', 'baseline', 'teacher_only'], type=str, help='ablation study mode')
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--sp-weight', type=float, default=1.0,
                    help='SP similarity loss coefficient')
parser.add_argument('--loss-anneal-start', type=int, default=-1,
                    help='epoch to start linearly annealing KD/feature weights; '
                         'negative disables annealing')
parser.add_argument('--kd-weight-end', type=float, default=None,
                    help='final KD weight used by optional loss annealing')
parser.add_argument('--feat-weight-end', type=float, default=None,
                    help='final feature weight used by optional loss annealing')
parser.add_argument('--distiller',default='kl',choices=['kl', 'dkd', 'rkd', 'catkd','sp'],type=str,help='type of distillation')
parser.add_argument('--rkd-dist-w', type=float, default=25.0,
                    help='RKD distance weight (default: 25)')
parser.add_argument('--rkd-angle-w', type=float, default=50.0,
                    help='RKD angle weight (default: 50)')
parser.add_argument('--dkd-alpha', type=float, default=1.0)
parser.add_argument('--dkd-beta', type=float, default=2.0)
parser.add_argument('--catkd-T', type=float, default=4.0, help='CATKD temperature')
parser.add_argument('--catkd-alpha', type=float, default=1.0, help='CATKD alpha (inter-class weight)')
parser.add_argument('--catkd-beta', type=float, default=1.0, help='CATKD beta (intra-class weight)')
parser.add_argument('--catkd-gamma', type=float, default=1.0, help='CATKD gamma (logit KD weight)')


parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--feat-kd', default='mse', type=str, choices=['mse', 'rkd'], help='feature kd loss')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--agent-step', type=int, default=1000, help='agent optimization step')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--teacher-name-list', default=['resnet32x4','wrn_28_4','ResNet50'], type=str, nargs='+', help='teacher models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dtd', 'rafdb', 'dogs','mit67','cub200'], help='dataset')
parser.add_argument('--dtd-split', type=int, default=1,
                    help='official DTD split number (1-10)')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--t_model', type=str, default='resnet32x4', help='model')
parser.add_argument('--w1', type=float, default=0.2, help='temperature')
parser.add_argument('--w2', type=float, default=0.8, help='temperature')


class ModelEMA:
    """Exponential moving average of the student model state."""

    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError(f'model EMA decay must be in (0, 1), got {decay}')
        source_model = model.module if hasattr(model, 'module') else model
        self.module = copy.deepcopy(source_model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_model = model.module if hasattr(model, 'module') else model
        source_state = source_model.state_dict()
        ema_state = self.module.state_dict()
        self.updates += 1

        for name, ema_value in ema_state.items():
            source_value = source_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(
                    source_value,
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(source_value)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path

def main():
    args = parser.parse_args()
    run_seed = args.seed if args.seed is not None else 42
    set_seed(run_seed)
    args.teacher_name_str = "_".join(args.teacher_name_list)
    print('args.teacher_name_str', args.teacher_name_str)
    args.teacher_num = len(args.teacher_name_list)

    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    print(f'===> info is : {info}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {args.checkpoint_dir}')

    if args.rank == 0 :
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))

    if args.seed is not None :
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None :
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(f'======> args.distributed is {args.distributed}')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f'======> ngpus_per_node is {ngpus_per_node}')


    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def build_transform_from_actions(actions, dataset):
    actions = actions.clamp(0, 1)
    crop_strength = 0.8 + 0.2 * actions[0].item()
    crop_padding = int(round(1 + 7 * actions[0].item()))
    #color_strength1 = 0.5 * actions[0].item()
    flip_prob = actions[1].item()
    color_strength = 0.5 * actions[2].item()
    rotate_degree = 30 * actions[3].item()

    if dataset.startswith('cifar100'):
        size = 32
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength
            ),
            transforms.RandomRotation(degrees=rotate_degree),
            transforms.ToTensor(),
            normalize
        ])
    elif dataset.startswith('tinyimagenet'):
        normalize = transforms.Normalize(
            TINY_IMAGENET_MEAN,
            TINY_IMAGENET_STD,
        )
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength,
            ),
            transforms.RandomRotation(degrees=rotate_degree),
            transforms.ToTensor(),
            normalize,
        ])
    elif dataset.startswith('dtd'):
        normalize = transforms.Normalize(DTD_MEAN, DTD_STD)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(crop_strength, 1.0)),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength,
            ),
            transforms.RandomRotation(degrees=rotate_degree),
            transforms.ToTensor(),
            normalize,
        ])
    elif dataset.startswith('rafdb'):
        normalize = transforms.Normalize(RAFDB_MEAN, RAFDB_STD)
        transform = transforms.Compose([
            transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
            transforms.RandomCrop(RAFDB_SIZE, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength,
            ),
            transforms.RandomRotation(degrees=rotate_degree),
            transforms.ToTensor(),
            normalize,
        ])
    elif(dataset.startswith('cub200')):
        size = 224
        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(size, scale=(crop_strength, 1.0)),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength
            ),
            transforms.RandomRotation(degrees=rotate_degree),
            transforms.ToTensor(),
            normalize
        ])
    return transform

def get_agent(teacher_models, args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    teacher_num = len(teacher_models)
    x = torch.rand(args.res).to(device)
    logits_dim = 0
    feature_dim = 0
    feature_dims = []
    policy_input_size = []
    for t in teacher_models :
        feature, logits = t(x, is_feat=True)
        logits_dim += logits.size(1)
        feature_dim += feature[-1].size(1)
        feature_dims.append(feature[-2].size())
        policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3 )

    agent = model_dict['PolicyTrans'](policy_input_size, teacher_num, args.dynamic).to(device)
    return agent, feature_dims

def get_agentaug(teacher_models, args):
    teacher_num = len(teacher_models)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(args.res).to(device)
    policy_input_size = []

    for t in teacher_models:
        feature, logits = t(x, is_feat=True)
        policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3 )

    aug_num = 4
    agent = model_dict['PolicyAug'](
        policy_input_size,
        aug_num,
        teacher_num,
        args.dynamic
    ).to(device)
    return agent, None

def get_feat_trans(model, args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).to(device), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).to(device)

def main_worker(gpu, ngpus_per_node, args):
    current_transform_actions = None
    pending_aug_context = None
    pending_aug_action = None
    previous_reward_loss = None
    reward_baseline = 0.0
    reward_baseline_initialized = False
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    def load_teacher(model_path, n_cls, model_t, opt):
        device = torch.device(f'cuda:{opt.gpu}')
        model = model_dict[model_t](num_classes=n_cls).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        for t_p in model.parameters():
            t_p.requires_grad = False
        return model

    def load_teacher_list(opt):
        print('==> loading teacher model list')
        teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, opt.t_model, opt) for model_name in opt.teacher_name_list]
        print('==> done')
        return teacher_model_list

    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('tinyimagenet'):
        args.n_cls = 200
        args.res = (1, 3, 64, 64)
    elif args.dataset.startswith('dtd'):
        args.n_cls = 47
        args.res = (1, 3, 224, 224)
    elif args.dataset.startswith('rafdb'):
        args.n_cls = 7
        args.res = (1, 3, RAFDB_SIZE, RAFDB_SIZE)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    elif args.dataset.startswith('cub200'):
        args.n_cls = 200
        args.res = (1, 3, 224, 224)

    teacher_models = load_teacher_list(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = model_dict[args.arch](num_classes=args.n_cls).to(device)

    args.start_epoch = 0
    resume_checkpoint = None
    if len(args.resume) != 0:
        map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
        model_info_dict = torch.load(args.resume, map_location=map_location, weights_only=False)
        model.load_state_dict(model_info_dict['model'])
        args.start_epoch = model_info_dict['epoch']
        resume_checkpoint = model_info_dict

    print('======> load student model finish')
    agent, args.t_feat_dims = get_agent(teacher_models, args)
    print("===> get agent finish......")
    agent_aug, _ = get_agentaug(teacher_models, args)
    feat_trans = get_feat_trans(model, args)
    print("===> get feat_trans finish......")

    model = model.to(device)
    agent = agent.to(device)
    agent_aug = agent_aug.to(device)
    feat_trans = feat_trans.to(device)

    model_ema = None
    if args.model_ema_decay > 0.0:
        model_ema = ModelEMA(model, args.model_ema_decay)
        if resume_checkpoint is not None and resume_checkpoint.get('model_ema') is not None:
            model_ema.module.load_state_dict(resume_checkpoint['model_ema'])
            model_ema.updates = resume_checkpoint.get('model_ema_updates', 0)

    for i in range(len(teacher_models)):
        teacher_models[i] = teacher_models[i].to(device)
        teacher_models[i].eval()

    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    if args.distiller == 'kl':
        criterion_div = DistillKL(args.kd_T).to(device)
    elif args.distiller == 'dkd':
        criterion_div = DKDLoss(alpha=args.dkd_alpha, beta=args.dkd_beta, T=args.kd_T).to(device)
    elif args.distiller == 'rkd':
        criterion_div = RKDLoss(w_d=args.rkd_dist_w, w_a=args.rkd_angle_w).to(device)
    elif args.distiller == 'catkd':
        criterion_div = CATKDLoss(opt=args).to(device) 
    elif args.distiller == 'sp':  # 添加 SP 分支
        criterion_div = Similarity().to(device)
    else:
        raise NotImplementedError

    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(feat_trans)

    optimizer = optim.SGD(trainable_list.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    agent_optimizer = optim.SGD(agent.parameters(), lr=args.lr)
    aug_agent_optimizer = optim.Adam(
        agent_aug.parameters(),
        lr=args.aug_agent_lr,
        weight_decay=1e-5,
    )

    # Used only when an optional held-out validation subset is requested.
    best_acc = float('-inf')
    train_transform_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_transform_cub200 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_transform_tinyimagenet = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
    ])
    train_transform_dtd = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DTD_MEAN, DTD_STD),
    ])
    train_transform_rafdb = transforms.Compose([
        transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
        transforms.RandomCrop(RAFDB_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
    ])

    if args.dataset.startswith('cifar100'):
        split_seed = args.seed if args.seed is not None else 42
        _, val_loader, test_loader = get_cifar100_train_val_test_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform_cifar100,
            val_size=args.val_size,
            split_seed=split_seed,
        )
    elif args.dataset.startswith('tinyimagenet'):
        if args.val_size != 0:
            raise ValueError(
                'Tiny-ImageNet currently uses its official validation set for '
                'evaluation; --val-size must be 0.'
            )
        _, test_loader = get_tinyimagenet_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform_tinyimagenet,
        )
        val_loader = None
        split_seed = args.seed if args.seed is not None else 42
    elif args.dataset.startswith('dtd'):
        _, _, test_loader = get_dtd_dataloaders(
            root=args.data,
            split=args.dtd_split,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform_dtd,
        )
        # Match the Tiny-ImageNet protocol: no held-out validation loader.
        val_loader = None
        split_seed = args.seed if args.seed is not None else 42
    elif args.dataset.startswith('rafdb'):
        _, test_loader = get_rafdb_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform_rafdb,
        )
        val_loader = None
        split_seed = args.seed if args.seed is not None else 42
    else:
        raise NotImplementedError(
            'The delayed augmentation-policy reward currently requires the '
            'CIFAR-100, Tiny-ImageNet, DTD, or RAF-DB data-loading path.'
        )
    t_results = []
    for t_model in teacher_models:
        acc = test(
            0, t_model, device, test_loader, criterion_ce, args,
            verbose=False, split_name='Teacher test'
        )
        t_results.append(round(acc, 2))
    args.logger.info('Teacher accruacy: '+ str(t_results))

    for epoch in range(args.start_epoch, args.epochs):
        print(current_transform_actions)
        #if epoch == 0 or epoch == 1 or current_transform_actions is None:
        if epoch == 0 or current_transform_actions is None:
            if args.dataset.startswith('cifar100'):
                transform = train_transform_cifar100
            elif args.dataset.startswith('tinyimagenet'):
                transform = train_transform_tinyimagenet
            elif args.dataset.startswith('rafdb'):
                transform = train_transform_rafdb
            else:
                transform = train_transform_dtd
        else:
            transform = build_transform_from_actions(
                current_transform_actions.cpu(),
                dataset=args.dataset,
            )
        if args.dataset.startswith('cifar100'):
            train_loader, _, _ = get_cifar100_train_val_test_dataloaders(
                root=args.data,
                batch_size=args.batch_size,
                num_workers=args.workers,
                train_transform=transform,
                val_size=args.val_size,
                split_seed=split_seed,
            )
        elif args.dataset.startswith('tinyimagenet'):
            train_loader, _ = get_tinyimagenet_dataloaders(
                root=args.data,
                batch_size=args.batch_size,
                num_workers=args.workers,
                train_transform=transform,
            )
        elif args.dataset.startswith('dtd'):
            train_loader, _, _ = get_dtd_dataloaders(
                root=args.data,
                split=args.dtd_split,
                batch_size=args.batch_size,
                num_workers=args.workers,
                train_transform=transform,
            )
        else:
            train_loader, _ = get_rafdb_dataloaders(
                root=args.data,
                batch_size=args.batch_size,
                num_workers=args.workers,
                train_transform=transform,
            )
        for train_pass in range(args.train_passes):
            if args.rank == 0:
                args.logger.info(
                    'Training pass %d/%d | epoch=%d',
                    train_pass + 1, args.train_passes, epoch,
                )
            # Keep the final pass metrics/context: they describe the model
            # state that is checkpointed and used to choose the next action.
            epoch_aug_context, train_epoch_loss, train_epoch_acc = train(
                train_loader, model, criterion_list, optimizer, epoch, device,
                args, agent, agent_aug, feat_trans, teacher_models,
                agent_optimizer, aug_agent_optimizer, model_ema=model_ema
            )

        if val_loader is None:
            reward_loss = train_epoch_loss
            checkpoint_acc = train_epoch_acc
            val_loss = None
            reward_source = 'relative_train_loss'
        else:
            checkpoint_acc, val_loss = test(
                epoch, model, device, val_loader, criterion_ce, args,
                return_loss=True, split_name='Validation'
            )
            reward_loss = val_loss
            reward_source = 'validation_loss'

        # Credit the action that actually built this epoch's data loader.
        if (pending_aug_context is not None
                and pending_aug_action is not None
                and previous_reward_loss is not None):
            loss_improvement = previous_reward_loss - reward_loss
            if val_loader is None:
                reward = loss_improvement / max(abs(previous_reward_loss), 1e-8)
            else:
                reward = loss_improvement
            baseline = reward_baseline if reward_baseline_initialized else 0.0
            advantage = max(-1.0, min(1.0, reward - baseline))
            aug_stats = train_aug_agent(
                pending_aug_context,
                pending_aug_action,
                advantage,
                agent_aug,
                aug_agent_optimizer,
                entropy_coef=args.aug_entropy_coef,
            )

            if reward_baseline_initialized:
                momentum = args.aug_baseline_momentum
                reward_baseline = (
                    momentum * reward_baseline
                    + (1.0 - momentum) * reward
                )
            else:
                reward_baseline = reward
                reward_baseline_initialized = True

            if args.rank == 0:
                args.logger.info(
                    'Aug policy update | epoch=%d source=%s reward=%.6f '
                    'baseline=%.6f advantage=%.6f loss=%.6f '
                    'entropy=%.6f grad_norm=%.6f',
                    epoch, reward_source, reward, baseline, advantage,
                    aug_stats['loss'], aug_stats['entropy'],
                    aug_stats['grad_norm'],
                )

        # The action sampled here is applied by the next epoch's loader.
        if epoch_aug_context is not None:
            agent_aug.eval()
            with torch.no_grad():
                next_action, _, _ = agent_aug.sample_action(epoch_aug_context)
                next_action = next_action.clamp(0.0, 1.0)
                mean_action = agent_aug(epoch_aug_context)

            pending_aug_context = tuple(
                component.detach() for component in epoch_aug_context
            )
            pending_aug_action = next_action.detach()
            current_transform_actions = next_action.squeeze(0).detach()

            if args.rank == 0:
                args.logger.info(
                    'Next epoch augmentation | sampled=%s mean=%s',
                    current_transform_actions.cpu().numpy(),
                    mean_action.squeeze(0).cpu().numpy(),
                )

        previous_reward_loss = reward_loss

        late_test_acc = None
        if (val_loader is None
                and args.late_test_start >= 0
                and epoch >= args.late_test_start):
            evaluation_model = (
                model_ema.module if model_ema is not None else model
            )
            late_test_acc = test(
                epoch,
                evaluation_model,
                device,
                test_loader,
                criterion_ce,
                args,
                split_name='Late test',
            )

        if args.rank == 0 :
            state = {
                    'epoch' : epoch + 1,
                    'arch' : args.arch,
                    'model': model.module.state_dict() if args.distributed else model.state_dict() ,
                    'model_ema': model_ema.module.state_dict() if model_ema is not None else None,
                    'model_ema_decay': args.model_ema_decay,
                    'model_ema_updates': model_ema.updates if model_ema is not None else 0,
                    'acc': checkpoint_acc,
                    'train_loss': train_epoch_loss,
                    'train_passes': args.train_passes,
                    'val_loss': val_loss,
                    'late_test_acc': late_test_acc,
                    'optimizer': optimizer.state_dict(),
                    'distill_agent': agent.state_dict(),
                    'distill_agent_optimizer': agent_optimizer.state_dict(),
                    'aug_agent': agent_aug.state_dict(),
                    'aug_agent_optimizer': aug_agent_optimizer.state_dict(),
                    'aug_reward_baseline': reward_baseline,
                    'current_transform_actions': current_transform_actions,
            }
            torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))
            if val_loader is not None and best_acc < checkpoint_acc:
                best_acc = checkpoint_acc
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))
            elif late_test_acc is not None and best_acc < late_test_acc:
                best_acc = late_test_acc
                shutil.copyfile(
                    os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                )

    if args.rank == 0 :
        if val_loader is None and args.late_test_start >= 0:
            selected_checkpoint = os.path.join(
                args.checkpoint_dir, args.arch + '_best.pth.tar'
            )
            args.logger.info(
                'Evaluate the best late-test model (epochs %d-%d):',
                args.late_test_start,
                args.epochs - 1,
            )
        elif val_loader is None:
            selected_checkpoint = os.path.join(
                args.checkpoint_dir, args.arch + '.pth.tar'
            )
            args.logger.info('Evaluate the final-epoch model (no validation split):')
        else:
            selected_checkpoint = os.path.join(
                args.checkpoint_dir, args.arch + '_best.pth.tar'
            )
            args.logger.info('Evaluate the best-validation model:')
        args.evaluate = True
        checkpoint = torch.load(
            selected_checkpoint,
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        evaluation_key = (
            'model_ema'
            if model_ema is not None and checkpoint.get('model_ema') is not None
            else 'model'
        )
        model.load_state_dict(checkpoint[evaluation_key])
        args.logger.info('Evaluation weights: %s', evaluation_key)
        top1_acc = test(
            epoch, model, device, test_loader, criterion_ce, args,
            split_name='Test'
        )
        args.logger.info('Test top-1 accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(selected_checkpoint))

if __name__ == '__main__' :
     main()
