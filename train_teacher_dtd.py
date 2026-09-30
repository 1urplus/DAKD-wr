"""Train one A0--A3 DTD teacher on an official DTD split."""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dtd import get_dtd_dataloaders
from models.cifar100 import model_dict


def parse_args():
    parser = argparse.ArgumentParser("DTD teacher training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="resnet32x4", choices=sorted(model_dict))
    parser.add_argument("--aug-group", required=True, choices=("A0", "A1", "A2", "A3"))
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-decay-epochs", type=int, nargs="+", default=(150, 180, 210))
    parser.add_argument("--lr-decay-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="./checkpoint")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_logger(path):
    logger = logging.getLogger(f"dtd_teacher_{path.parent.name}_{path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def accuracy_counts(logits, targets):
    predictions = logits.topk(5, dim=1).indices
    matches = predictions.eq(targets.unsqueeze(1))
    return matches[:, :1].sum().item(), matches.sum().item()


def set_learning_rate(optimizer, epoch, args):
    steps = sum(epoch > milestone for milestone in args.lr_decay_epochs)
    lr = args.learning_rate * (args.lr_decay_rate ** steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_epoch(loader, model, criterion, optimizer, device, epoch, args, logger):
    model.train()
    loss_sum = top1 = top5 = count = 0.0
    start = time.time()
    for batch_index, (images, targets) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        c1, c5 = accuracy_counts(logits, targets)
        loss_sum += loss.item() * batch_size
        top1 += c1
        top5 += c5
        count += batch_size
        if batch_index % args.print_freq == 0:
            logger.info("[%s] epoch=%d batch=%d/%d loss=%.4f acc1=%.2f",
                        args.aug_group, epoch, batch_index, len(loader),
                        loss_sum / count, 100.0 * top1 / count)
    return {"loss": loss_sum / count, "acc1": 100.0 * top1 / count,
            "acc5": 100.0 * top5 / count, "seconds": time.time() - start}


@torch.no_grad()
def evaluate(loader, model, criterion, device, max_batches=0):
    model.eval()
    loss_sum = top1 = top5 = count = 0.0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        c1, c5 = accuracy_counts(logits, targets)
        loss_sum += loss.item() * batch_size
        top1 += c1
        top5 += c5
        count += batch_size
    return {"loss": loss_sum / count, "acc1": 100.0 * top1 / count,
            "acc5": 100.0 * top5 / count}


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DTD teacher training")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    output_dir = Path(args.checkpoint_dir) / "teachers" / "models" / f"{args.model}_dtd"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir / f"train_{args.aug_group}.log")
    logger.info("args=%s", args)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader, _, test_loader = get_dtd_dataloaders(
        args.data, split=args.split, aug_group=args.aug_group,
        batch_size=args.batch_size, num_workers=args.workers,
        generator=generator, worker_init_fn=seed_worker,
    )
    logger.info("dataset train=%d test=%d classes=47 (no validation split)",
                len(train_loader.dataset), len(test_loader.dataset))

    model = model_dict[args.model](num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    start_epoch, best_acc = 1, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", best_acc)

    latest_path = output_dir / f"{args.model}_{args.aug_group}_last.pth"
    best_path = output_dir / f"{args.model}_{args.aug_group}_best.pth"
    metrics_path = output_dir / f"test_best_metrics_{args.aug_group}.json"
    for epoch in range(start_epoch, args.epochs + 1):
        lr = set_learning_rate(optimizer, epoch, args)
        train_metrics = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, args, logger
        )
        test_metrics = evaluate(test_loader, model, criterion, device, args.max_eval_batches)
        logger.info("[%s] epoch=%d lr=%.6f time=%.1fs train_loss=%.4f "
                    "train_acc1=%.3f test_loss=%.4f test_acc1=%.3f test_acc5=%.3f",
                    args.aug_group, epoch, lr, train_metrics["seconds"],
                    train_metrics["loss"], train_metrics["acc1"], test_metrics["loss"],
                    test_metrics["acc1"], test_metrics["acc5"])
        state = {"epoch": epoch, "model_name": args.model, "dataset": "dtd",
                 "split": args.split, "aug_group": args.aug_group,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best_acc": max(best_acc, test_metrics["acc1"]), "args": vars(args)}
        torch.save(state, latest_path)
        if test_metrics["acc1"] > best_acc:
            best_acc = test_metrics["acc1"]
            torch.save(state, best_path)
            metrics_path.write_text(json.dumps(
                {"epoch": epoch, "aug_group": args.aug_group,
                 "test": test_metrics}, indent=2
            ) + "\n", encoding="utf-8")
            logger.info("[%s] saved best: test_acc1=%.3f",
                        args.aug_group, best_acc)


if __name__ == "__main__":
    main()
