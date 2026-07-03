"""Fine-tune EfficientNetV2-S with a stronger augmentation recipe.

This is the augmentation experiment. Vs. the baseline it adds, all toggleable:
  * --aug strong        : TrivialAugmentWide + flips + RandomErasing
  * --sampler weighted  : class-balancing WeightedRandomSampler (oversamples talc)
  * --mix mixup_cutmix  : per-batch MixUp/CutMix with soft-label loss
  * --class-weights auto: turn OFF inverse-freq loss weights when the sampler is
                          already balancing classes (avoid double-correcting).

Example:
    python src/train.py                       # full recipe, 40 epochs
    python src/train.py --aug basic --sampler none --mix none  # ~= baseline
    python src/train.py --smoke               # 1-epoch sanity run
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import CLASSES, RUNS_DIR, get_device, set_seed
from data import (build_mixups, class_counts, class_weights,
                  get_or_make_splits, make_loader, make_splits)
from model import SUPPORTED, build_model


def soft_cross_entropy(logits, target, weight=None):
    """Weighted cross-entropy for soft (probability) targets from MixUp/CutMix.

    Hand-rolled so behaviour is identical across torch versions. Matches the
    weighted-CE convention: normalize each sample by the class-weight mass of its
    (soft) target. With weight=None this is the plain soft cross-entropy.
    """
    logp = F.log_softmax(logits, dim=1)
    if weight is None:
        return -(target * logp).sum(1).mean()
    w = weight.unsqueeze(0)                        # (1, C)
    num = -(w * target * logp).sum(1)             # (B,)
    den = (w * target).sum(1).clamp_min(1e-8)     # (B,)
    return (num / den).mean()


def run_epoch(model, loader, device, criterion, optimizer=None,
              mixups=None, mix_prob=0.0, loss_weight=None, num_classes=3):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    mixup, cutmix = mixups if mixups else (None, None)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # Metrics are always scored against the ORIGINAL hard labels.
        true_labels = labels
        with torch.set_grad_enabled(train):
            use_mix = (train and (mixup or cutmix) and random.random() < mix_prob
                       and imgs.size(0) > 1)
            if use_mix:
                op = random.choice([m for m in (mixup, cutmix) if m is not None])
                mix_imgs, soft_target = op(imgs, labels)
                logits = model(mix_imgs)
                loss = soft_cross_entropy(logits, soft_target, loss_weight)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_pred += logits.argmax(1).cpu().tolist()
        all_true += true_labels.cpu().tolist()
    acc = sum(int(p == t) for p, t in zip(all_pred, all_true)) / max(1, n)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / max(1, n), acc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_v2_s", choices=SUPPORTED)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience")
    # --- augmentation knobs ---
    ap.add_argument("--aug", default="strong", choices=["basic", "strong"])
    ap.add_argument("--sampler", default="weighted", choices=["none", "weighted"])
    ap.add_argument("--mix", default="mixup_cutmix", choices=["none", "mixup_cutmix"])
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--mix-prob", type=float, default=0.5,
                    help="fraction of train batches that get MixUp/CutMix")
    ap.add_argument("--erase-prob", type=float, default=0.25)
    ap.add_argument("--class-weights", default="auto", choices=["auto", "on", "off"],
                    help="inverse-freq loss weights; auto=off when sampler=weighted")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    # --- misc ---
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

    splits = (make_splits(seed=args.seed) if args.rebuild_splits
              else get_or_make_splits(seed=args.seed))
    train_recs, val_recs = splits["train"], splits["val"]

    use_sampler = args.sampler == "weighted"
    use_mix = args.mix == "mixup_cutmix"
    # Decide loss class-weights: never stack them on top of a balancing sampler.
    if args.class_weights == "on":
        use_loss_weights = True
    elif args.class_weights == "off":
        use_loss_weights = False
    else:  # auto
        use_loss_weights = not use_sampler

    print(f"Device: {device} | model: {args.model} | img: {args.img_size}")
    print(f"Aug: {args.aug} | sampler: {args.sampler} | mix: {args.mix} "
          f"(p={args.mix_prob}) | loss class-weights: {use_loss_weights}")
    print(f"Train {len(train_recs)} | Val {len(val_recs)} | Test {len(splits['test'])}")
    print("Train class counts (ordinary/hard/talc):", class_counts(train_recs))

    train_loader = make_loader(
        train_recs, args.img_size, True, args.batch_size, args.num_workers,
        limit=args.limit, policy=args.aug, use_sampler=use_sampler, seed=args.seed,
        erase_prob=args.erase_prob, drop_last=True)  # drop_last: stable BN + MixUp
    val_loader = make_loader(
        val_recs, args.img_size, False, args.batch_size, args.num_workers,
        limit=args.limit, policy=args.aug)

    model = build_model(args.model, len(CLASSES),
                        pretrained=not args.no_pretrained).to(device)

    loss_weight = class_weights(train_recs).to(device) if use_loss_weights else None
    criterion = nn.CrossEntropyLoss(weight=loss_weight,
                                    label_smoothing=args.label_smoothing)
    mixups = build_mixups(len(CLASSES), args.mixup_alpha, args.cutmix_alpha) if use_mix else None
    mix_prob = args.mix_prob if use_mix else 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(RUNS_DIR, f"best_{args.model}.pt")
    best_f1, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, device, criterion, optimizer,
            mixups=mixups, mix_prob=mix_prob, loss_weight=loss_weight,
            num_classes=len(CLASSES))
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
                "aug": args.aug,
                "sampler": args.sampler,
                "mix": args.mix,
            }, ckpt_path)
            print(f"    ↳ saved best (val macro-F1 {va_f1:.3f}) -> {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val F1 gain for {args.patience} epochs).")
                break

    with open(os.path.join(RUNS_DIR, f"train_summary_{args.model}.json"), "w") as f:
        json.dump({
            "best_val_macro_f1": best_f1,
            "checkpoint": ckpt_path,
            "config": {
                "aug": args.aug, "sampler": args.sampler, "mix": args.mix,
                "mix_prob": args.mix_prob, "loss_class_weights": use_loss_weights,
                "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            },
        }, f, indent=2)
    print(f"Done. Best val macro-F1: {best_f1:.3f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
