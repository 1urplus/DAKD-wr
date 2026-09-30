import argparse
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
import torchvision.transforms as transforms
import numpy as np
from helper.train_loops_avg import train_avg, test
from torch.utils.tensorboard import SummaryWriter
#from models.cub200 import model_dict
from models.cifar100 import model_dict
from setting import  teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders_ori
from dataset.cub200 import get_cub200_dataloaders_ori
from dataset.tinyimagenet import (
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    get_tinyimagenet_dataloaders,
)
from dataset.dtd import DTD_MEAN, DTD_STD, get_dtd_dataloaders
from dataset.rafdb import RAFDB_MEAN, RAFDB_STD, RAFDB_SIZE, get_rafdb_dataloaders
from utils import set_logger
from models.cub200.util import Regress, TransFeat
from distiller_zoo import DistillKL,DKDLoss,RKDLoss,CATKDLoss,Similarity
# nohup python train_student_avg.py --gpu 1 --arch MobileNetV2 --distiller kl --t_model resnet32x4 --dataset cifar100 --teacher-name-list A0 A1 A2 A3  --dist-backend nccl --world-size 1 --rank 0 > logs/cifar100/resnet32x4_mobilev2_avg_kl.out 2>&1 &

parser = argparse.ArgumentParser(description='DCKD AVG training')
parser.add_argument('--data', metavar='DIR', nargs='?',
                    default=os.environ.get('DCKD_DATA_ROOT', './data'),
                    help='path to dataset (default: cifar-100-python)')
parser.add_argument('--distiller',default='kl',choices=['kl', 'dkd', 'rkd', 'catkd','sp'],type=str,help='type of distillation')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ShuffleV2')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--late-test-start', type=int, default=210,
                    help='without a validation split, evaluate the test set '
                         'from this epoch and retain the best checkpoint; '
                         'set below 0 to evaluate only the final epoch')
parser.add_argument('--dkd-warmup', type=int, default=0,
                    help='linearly warm up the DKD loss for this many epochs; '
                         '0 keeps the previous behavior')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')  # 32*2
parser.add_argument('--t_model', type=str, default='resnet32x4', help='model')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=400, type=int,
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')
parser.add_argument('--sp-weight', type=float, default=1.0,
                    help='SP similarity loss coefficient')
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
#[150,180,210][300,360,420]
parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dtd', 'rafdb', 'dogs', 'cub200', 'mit67'], help='dataset')
parser.add_argument('--dtd-split', type=int, default=1,
                    help='official DTD split number (1-10)')
parser.add_argument('--trial', type=str, default='1', help='trial id')

def set_seed(seed):
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python标准库的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # 设置CUDA的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True  # 确保结果可复现
    torch.backends.cudnn.benchmark = False  # 禁用自动寻找最优算法

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

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed # True
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


def get_feat_trans(model, args):
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(args.gpu), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).cuda(args.gpu)
        
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        
    def load_teacher(model_path, n_cls, model_t, opt=None):
        device = torch.device(f'cuda:{opt.gpu}') 
        
        model = model_dict[model_t](num_classes=n_cls).to(device)
        
        # These legacy teacher checkpoints were produced locally before
        # PyTorch 2.6 changed torch.load's default to weights_only=True.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint) # 有些存档可能直接就是 state_dict
            
        model.eval()
        for t_p in model.parameters():
            t_p.requires_grad = False
        return model


    def load_teacher_list(opt):
        print('==> loading teacher model list')
        teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, opt.t_model, opt) for model_name in opt.teacher_name_list]
        print('==> done')
        return teacher_model_list

    def get_teacher_feat_dims(teacher_models, args):
        x = torch.rand(args.res).cuda(args.gpu)
        feature_dims = []
        for t in teacher_models :
            feature, logits = t(x, is_feat=True)
            feature_dims.append(feature[-2].size())
        return feature_dims


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
    else:
        raise NotImplementedError(
            'AVG training currently supports CIFAR-100, Tiny-ImageNet, DTD, and RAF-DB.'
        )
            
    teacher_models = load_teacher_list(args)
    
    ##### load student model #####
    model = model_dict[args.arch](num_classes=args.n_cls).cuda(args.gpu)
    
    args.start_epoch = 0
    if len(args.resume) != 0:
        map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
        model_info_dict = torch.load(
            args.resume, map_location=map_location, weights_only=False
        )
        model.load_state_dict(model_info_dict['model'])
        args.start_epoch = model_info_dict['epoch']
        
    print('======> load student model finish')
    args.t_feat_dims = get_teacher_feat_dims(teacher_models, args)

    feat_trans = get_feat_trans(model, args)
    print("===> get feat_trans finish......")
        
    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu]) 
                for i, t in enumerate(teacher_models):
                    teacher_models[i] = torch.nn.parallel.DistributedDataParallel(t, device_ids=[args.gpu])
                
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda')
          
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

    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)

    ################### load data ###################
    if args.dataset.startswith('cifar100'):
        train_loader, val_loader = get_cifar100_dataloaders_ori(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        test_loader = val_loader
    elif args.dataset.startswith('tinyimagenet'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ])
        train_loader, test_loader = get_tinyimagenet_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform,
        )
        val_loader = None
    elif args.dataset.startswith('dtd'):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DTD_MEAN, DTD_STD),
        ])
        train_loader, _, test_loader = get_dtd_dataloaders(
            root=args.data,
            split=args.dtd_split,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform,
        )
        val_loader = None
    else:
        train_transform = transforms.Compose([
            transforms.Resize((RAFDB_SIZE, RAFDB_SIZE)),
            transforms.RandomCrop(RAFDB_SIZE, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
        ])
        train_loader, test_loader = get_rafdb_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            train_transform=train_transform,
        )
        val_loader = None
    
    ################### train model ###################
    best_acc = float('-inf')
    
    t_results = []
    for t_model in teacher_models:
        acc = test(0, t_model, device, test_loader, criterion_ce, args, verbose=False)
        t_results.append(round(acc, 2))
    args.logger.info('Teacher accuracy: '+ str(t_results))

    for epoch in range(args.start_epoch, args.epochs) :
        train_avg(train_loader, model, criterion_list, optimizer, epoch, device, args, feat_trans, teacher_models)
        checkpoint_acc = None
        if val_loader is not None:
            checkpoint_acc = test(epoch, model, device, val_loader, criterion_ce, args)
        elif args.late_test_start >= 0 and epoch >= args.late_test_start:
            checkpoint_acc = test(epoch, model, device, test_loader, criterion_ce, args)

        if args.rank == 0 :
            state = {
                    'epoch' : epoch + 1,
                    'arch' : args.arch, 
                    'model': model.module.state_dict() if args.distributed else model.state_dict() ,
                    'acc': checkpoint_acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
            }

            torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))

            if checkpoint_acc is not None and best_acc < checkpoint_acc:
                best_acc = checkpoint_acc
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

    
    if args.rank == 0 :
        if val_loader is None and args.late_test_start < 0:
            selected_checkpoint = os.path.join(
                args.checkpoint_dir, args.arch + '.pth.tar'
            )
            args.logger.info('Evaluate the final-epoch model (no validation split):')
        else:
            selected_checkpoint = os.path.join(
                args.checkpoint_dir, args.arch + '_best.pth.tar'
            )
            if val_loader is None:
                args.logger.info(
                    'Evaluate the best late-test model (epochs %d-%d):',
                    args.late_test_start,
                    args.epochs - 1,
                )
            else:
                args.logger.info('Evaluate the best-validation model:')
        args.evaluate = True
        checkpoint = torch.load(
            selected_checkpoint,
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        model.load_state_dict(checkpoint['model'])
        top1_acc = test(epoch, model, device, test_loader, criterion_ce, args)
        args.logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(selected_checkpoint))

if __name__ == '__main__' :
     main()
