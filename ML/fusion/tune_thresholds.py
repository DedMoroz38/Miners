"""Measure, not guess, the two fusion thresholds.

1. --seg-conf : run YOLO-seg val on the 8-image val split and read the mask-F1
   vs confidence curve (peak + the near-optimal band).
2. --p-thresh (tau) : run the classifier (with TTA) on ITS val split and sweep
   tau: for each candidate, the coverage (share of images with P >= tau) and the
   accuracy among them. Recommend the smallest tau whose empirical accuracy
   >= --target-acc. NB: softmax on a small dataset is overconfident, so an
   empirical tau beats the hand-picked 0.8.

The val split (~15% of ~1180 images) is held out from classifier training, so
the sweep is honest; it IS the model-selection split, so treat the number as a
good estimate rather than gospel.

Run inside an env with ultralytics + torchvision (e.g. first_labling_attempt/.venv):
    python tune_thresholds.py                 # both parts
    python tune_thresholds.py --skip-cls      # seg part only (no classifier ckpt yet)
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(FUSION_DIR)
AUG_SRC = os.path.join(ML_DIR, "first_augmentation_attempt", "src")
CLS_DIR = os.path.join(ML_DIR, "first_classification_attempt")
DEFAULT_CLS_CKPT = os.path.join(ML_DIR, "first_augmentation_attempt", "runs",
                                "best_efficientnet_v2_s.pt")
FALLBACK_CLS_CKPT = os.path.join(CLS_DIR, "runs", "best_efficientnet_v2_s.pt")
SEG_WEIGHTS = os.path.join(ML_DIR, "first_segmentation_attempt", "runs",
                           "talc_seg_v2", "weights", "best.pt")
SEG_DATA = os.path.join(ML_DIR, "first_segmentation_attempt", "data.yaml")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_device():
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sweep_seg_conf(device):
    from ultralytics import YOLO
    print(f"=== Segmentation confidence sweep ({SEG_WEIGHTS}) ===")
    m = YOLO(SEG_WEIGHTS).val(data=SEG_DATA, imgsz=640, device=str(device),
                              plots=False, verbose=False,
                              project=os.path.join(FUSION_DIR, "runs"),
                              name="seg_val", exist_ok=True)
    f1 = np.asarray(m.seg.f1_curve).mean(0)   # mean over classes (only 'talc')
    px = np.asarray(m.seg.px)
    i = int(f1.argmax())
    band = px[f1 >= 0.95 * f1[i]]
    print(f"mask F1 peak {f1[i]:.3f} at conf {px[i]:.3f}; "
          f">=95% of peak for conf in [{band.min():.3f}, {band.max():.3f}]")
    print(f"mask mAP50 {m.seg.map50:.3f}")
    return float(px[i])


def d4_views(x):
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        yield r
        yield torch.flip(r, dims=(-1,))


@torch.no_grad()
def sweep_tau(ckpt_path, device, target_acc, tta=True):
    if not os.path.exists(ckpt_path):
        if os.path.exists(FALLBACK_CLS_CKPT):
            print(f"NOTE: {ckpt_path} missing, sweeping the BASELINE ckpt instead")
            ckpt_path = FALLBACK_CLS_CKPT
        else:
            print("No classifier checkpoint found; skipping tau sweep.")
            return None
    print(f"\n=== Classifier tau sweep ({ckpt_path}, TTA={tta}) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_mod = load_module(os.path.join(AUG_SRC, "model.py"), "tune_cls_model")
    model = model_mod.build_model(ckpt["model"], len(ckpt["classes"]),
                                  pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt.get("img_size", 256)
    tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    with open(os.path.join(CLS_DIR, "splits.json"), encoding="utf-8") as f:
        records = json.load(f)["val"]
    data_root = os.path.join(CLS_DIR, "data")

    ps, correct = [], []
    for rel, label in records:
        img = Image.open(os.path.join(data_root, rel)).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        if tta:
            probs = F.softmax(model(torch.cat(list(d4_views(x)))), dim=1).mean(0)
        else:
            probs = F.softmax(model(x), dim=1)[0]
        ps.append(float(probs.max()))
        correct.append(int(probs.argmax()) == label)
    ps, correct = np.array(ps), np.array(correct)
    print(f"val: {len(ps)} images, overall accuracy {correct.mean():.3f}")

    print(f"{'tau':>5} {'coverage':>9} {'acc@trusted':>12}")
    recommended = None
    for tau in np.arange(0.50, 1.00, 0.05):
        trusted = ps >= tau
        cov = trusted.mean()
        acc = correct[trusted].mean() if trusted.any() else float("nan")
        mark = ""
        if recommended is None and trusted.any() and acc >= target_acc:
            recommended, mark = float(tau), "  <- recommended"
        print(f"{tau:5.2f} {cov:9.1%} {acc:12.3f}{mark}")
    if recommended is None:
        print(f"No tau reaches {target_acc:.0%} accuracy among trusted predictions; "
              f"consider a higher --p-thresh or retraining.")
    else:
        print(f"\nRecommended --p-thresh {recommended:.2f} "
              f"(smallest tau with trusted-accuracy >= {target_acc:.0%})")
    return recommended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-ckpt", default=DEFAULT_CLS_CKPT)
    ap.add_argument("--target-acc", type=float, default=0.95,
                    help="required accuracy among trusted (P>=tau) predictions")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--skip-cls", action="store_true")
    args = ap.parse_args()

    device = get_device()
    if not args.skip_seg:
        # ultralytics val is happiest on cpu for an 8-image split
        sweep_seg_conf("cpu")
    if not args.skip_cls:
        sweep_tau(args.cls_ckpt, device, args.target_acc, tta=not args.no_tta)


if __name__ == "__main__":
    main()
