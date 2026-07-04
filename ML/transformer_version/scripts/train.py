"""Headless training entrypoint (works on CUDA/CPU).

Reads talc labels from the YOLO-seg export (constants.YOLO_SEG_DIR), builds the
5-class labels, and trains the SegFormer phase segmenter. Runs a small CPU-
friendly subset by default; pass --full on a GPU for the real run.

    PYTHONPATH=src python scripts/train.py            # quick CPU subset
    PYTHONPATH=src python scripts/train.py --full     # full run (GPU)
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from orenet import constants as C
from orenet.cache import build_cache
from orenet.data_index import build_index, talc_records
from orenet.dataset import PatchDataset
from orenet.engine import TrainConfig, train
from orenet.metrics import evaluate
from orenet.model import SegmenterConfig, build_segmenter


def _build_model(n_classes: int):
    """Try a pretrained SegFormer-B0; fall back to offline tiny init on failure."""
    for cfg in (
        SegmenterConfig(checkpoint="nvidia/segformer-b0-finetuned-ade-512-512",
                        pretrained=True, n_classes=n_classes),
        SegmenterConfig(tiny=True, pretrained=False, n_classes=n_classes),
    ):
        try:
            model = build_segmenter(cfg)
            print(f"model: {'tiny' if cfg.tiny else cfg.checkpoint}")
            return model
        except Exception as exc:  # noqa: BLE001 - network/offline fallback
            print(f"model load failed ({cfg.checkpoint or 'tiny'}): {exc}; falling back")
    raise RuntimeError("could not build any model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full run (GPU)")
    ap.add_argument("--rest", type=int, default=40, help="non-talc images (subset mode)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop = 512 if args.full else 384
    batch = 8 if args.full else 2
    steps = 250 if args.full else 8
    epochs = 12 if args.full else 2
    cache = pathlib.Path("data/derived/cache" if args.full else "data/derived/subset")

    records = build_index()
    talc = talc_records(records)
    rest = [r for r in records if r.talc_label is None]
    print(f"records={len(records)} | talc labels found={len(talc)}")
    if not talc:
        raise SystemExit("No talc labels. Run first_labling_attempt/export_to_yoloseg.py first.")

    talc_tr = [r for r in talc if r.talc_split == "train"] or talc[:-2]
    talc_ev = [r for r in talc if r.talc_split == "val"] or talc[-2:]
    if not args.full:
        rest = rest[: args.rest]
    train_recs = talc_tr + rest[: len(rest) * 8 // 10]
    eval_recs = talc_ev + rest[len(rest) * 8 // 10:]

    train_items = build_cache(train_recs, cache / "train")
    eval_items = build_cache(eval_recs, cache / "eval")
    print(f"cached train={len(train_items)} eval={len(eval_items)} "
          f"(talc in eval={sum(it.has_talc for it in eval_items)})")

    train_ds = PatchDataset(train_items, crop=crop, length=steps * batch, augment=True)
    val_ds = PatchDataset(eval_items, crop=crop, length=max(4, batch * 2),
                          augment=False, copy_paste_p=0.0)
    nw = 4 if args.full else 0
    train_loader = DataLoader(train_ds, batch_size=batch, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=nw)

    model = _build_model(C.NUM_CLASSES)
    print(f"device={device} | params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    cfg = TrainConfig(epochs=epochs, steps_per_epoch=steps, amp=args.full)
    history = train(model, train_loader, val_loader, device, cfg)

    print("loss trend:", [round(x, 4) for x in history["loss"]])
    res = evaluate(model, eval_items, device, C.NUM_CLASSES)
    print(f"talc MAE={res.talc_mae*100:.1f}% (n={res.talc_n}) | "
          f"intergrowth acc={res.intergrowth_acc*100:.0f}% (n={res.intergrowth_n})")
    out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "segformer_phases.pt")
    print("saved outputs/segformer_phases.pt")


if __name__ == "__main__":
    main()
