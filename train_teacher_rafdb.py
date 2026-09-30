"""Train an A0--A3 RAF-DB teacher without a validation split."""

import argparse
import fcntl
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset.rafdb import RAFDB_SIZE, get_rafdb_dataloaders
from models.cifar100 import model_dict
from train_teacher_dtd import (
    evaluate,
    make_logger,
    seed_everything,
    seed_worker,
    set_learning_rate,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser("RAF-DB teacher training")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, choices=sorted(model_dict))
    parser.add_argument("--aug-group", required=True, choices=("A0", "A1", "A2", "A3"))
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
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="./checkpoint")
    parser.add_argument("--resume", default="")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a matching final checkpoint exists")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RAF-DB teacher training")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    output_dir = Path(args.checkpoint_dir) / "teachers" / "models" / f"{args.model}_rafdb"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir / f"train_{args.aug_group}.log")
    logger.info("args=%s", args)
    # Two RAF-DB lanes may reach the same queued teacher. Serialize by model
    # and augmentation group, then skip work that the other lane completed.
    lock_handle = (output_dir / f"train_{args.aug_group}.lock").open("a")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    final_path = output_dir / f"{args.model}_{args.aug_group}_final.pth"
    if final_path.exists() and not args.force:
        completed = torch.load(final_path, map_location="cpu", weights_only=False)
        if (completed.get("epoch", 0) >= args.epochs
                and completed.get("model_name") == args.model
                and completed.get("aug_group") == args.aug_group
                and completed.get("dataset") == "rafdb"):
            logger.info("skipping completed checkpoint: %s", final_path)
            return
        raise RuntimeError(
            f"existing final checkpoint is incompatible: {final_path}; "
            "inspect it before using --force"
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader, test_loader = get_rafdb_dataloaders(
        args.data,
        args.aug_group,
        batch_size=args.batch_size,
        num_workers=args.workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    logger.info("dataset train=%d test=%d classes=7 input=%dx%d no validation",
                len(train_loader.dataset), len(test_loader.dataset),
                RAFDB_SIZE, RAFDB_SIZE)

    model = model_dict[args.model](num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    latest_path = output_dir / f"{args.model}_{args.aug_group}_last.pth"
    metrics_path = output_dir / f"test_final_metrics_{args.aug_group}.json"
    for epoch in range(start_epoch, args.epochs + 1):
        lr = set_learning_rate(optimizer, epoch, args)
        train_metrics = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, args, logger
        )
        logger.info("[%s] epoch=%d lr=%.6f time=%.1fs train_loss=%.4f "
                    "train_acc1=%.3f train_acc5=%.3f",
                    args.aug_group, epoch, lr, train_metrics["seconds"],
                    train_metrics["loss"], train_metrics["acc1"], train_metrics["acc5"])
        state = {
            "epoch": epoch,
            "model_name": args.model,
            "dataset": "rafdb",
            "aug_group": args.aug_group,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(state, latest_path)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_metrics = evaluate(
                test_loader, model, criterion, device, args.max_eval_batches
            )
            logger.info("[%s] epoch=%d test_loss=%.4f test_acc1=%.3f test_acc5=%.3f",
                        args.aug_group, epoch, test_metrics["loss"],
                        test_metrics["acc1"], test_metrics["acc5"])
        if epoch == args.epochs:
            torch.save(state, final_path)
            metrics_path.write_text(
                json.dumps({"epoch": epoch, "aug_group": args.aug_group,
                            "test": test_metrics}, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("[%s] saved final checkpoint: %s", args.aug_group, final_path)


if __name__ == "__main__":
    main()
