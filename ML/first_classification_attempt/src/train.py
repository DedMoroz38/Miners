"""Fine-tune a pretrained backbone on the merged 3-class ore dataset.

Example:
    python src/train.py --model efficientnet_v2_s --epochs 30 --img-size 256
    python src/train.py --smoke        # 1-epoch sanity run, no pretrained download
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common import (CLASSES, RUNS_DIR, get_device, set_seed)
from data import (class_counts, class_weights, get_or_make_splits, make_loader,
                  make_splits)
from model import SUPPORTED, build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_pred += logits.argmax(1).cpu().tolist()
        all_true += labels.cpu().tolist()
    acc = sum(int(p == t) for p, t in zip(all_pred, all_true)) / max(1, n)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / max(1, n), acc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_v2_s", choices=SUPPORTED)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience")
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap samples/loader (debug)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run: 1 epoch, tiny subset, no pretrained download")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.limit, args.no_pretrained = 1, 24, True
        args.num_workers, args.batch_size = 0, 8

    set_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"Device: {device} | model: {args.model} | img: {args.img_size}")

    splits = (make_splits(seed=args.seed) if args.rebuild_splits
              else get_or_make_splits(seed=args.seed))
    train_recs, val_recs = splits["train"], splits["val"]
    print(f"Train {len(train_recs)} | Val {len(val_recs)} | Test {len(splits['test'])}")
    print("Train class counts (ordinary/hard/talc):", class_counts(train_recs))

    train_loader = make_loader(train_recs, args.img_size, True,
                               args.batch_size, args.num_workers, args.limit)
    val_loader = make_loader(val_recs, args.img_size, False,
                             args.batch_size, args.num_workers, args.limit)

    model = build_model(args.model, len(CLASSES),
                        pretrained=not args.no_pretrained).to(device)
    weights = class_weights(train_recs).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(RUNS_DIR, f"best_{args.model}.pt")
    best_f1, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f} "
              f"({time.time()-t0:.0f}s)")

        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            torch.save({
                "model": args.model,
                "img_size": args.img_size,
                "classes": CLASSES,
                "state_dict": model.state_dict(),
                "val_macro_f1": va_f1,
                "val_acc": va_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"    ↳ saved best (val macro-F1 {va_f1:.3f}) -> {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val F1 gain for {args.patience} epochs).")
                break

    with open(os.path.join(RUNS_DIR, f"train_summary_{args.model}.json"), "w") as f:
        json.dump({"best_val_macro_f1": best_f1, "checkpoint": ckpt_path}, f, indent=2)
    print(f"Done. Best val macro-F1: {best_f1:.3f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
