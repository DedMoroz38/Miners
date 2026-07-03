"""Evaluate a trained checkpoint on the held-out test set.

Example:
    python src/evaluate.py --ckpt runs/best_efficientnet_v2_s.pt
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from common import CLASS_RU, CLASSES, RUNS_DIR, get_device
from data import load_splits, make_loader
from model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        preds += logits.argmax(1).cpu().tolist()
        trues += labels.tolist()
    return np.array(trues), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    img_size = ckpt.get("img_size", 256)
    print(f"Device: {device} | model: {ckpt['model']} | img: {img_size} | "
          f"trained val macro-F1: {ckpt.get('val_macro_f1', float('nan')):.3f}")

    model = build_model(ckpt["model"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    records = load_splits()[args.split]
    loader = make_loader(records, img_size, False, args.batch_size, args.num_workers)
    y_true, y_pred = predict(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    target_names = [CLASS_RU.get(c, c) for c in classes]
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=target_names, zero_division=0, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print(f"\n=== {args.split.upper()} set: {len(y_true)} images ===")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")
    print("\nPer-class report:\n" + report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("         " + "  ".join(f"{c:>9}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>9} " + "  ".join(f"{v:>9d}" for v in cm[i]))

    out = os.path.join(RUNS_DIR, f"eval_{args.split}_{ckpt['model']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split, "n": int(len(y_true)),
            "accuracy": float(acc), "macro_f1": float(macro_f1),
            "classes": classes, "confusion_matrix": cm.tolist(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics -> {out}")


if __name__ == "__main__":
    main()
