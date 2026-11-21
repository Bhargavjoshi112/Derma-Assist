"""Train a skin-condition classifier.

Quick test (a few minutes):
    python train.py --data path/to/dataset --max-per-class 60 --epochs 2
Stronger run (recommended on a Kaggle GPU):
    python train.py --data path/to/dataset --arch efficientnet_b2 --img-size 288 \
        --max-per-class 6000 --epochs 25 --patience 6
Several datasets at once (same folder name = same class):
    python train.py --data path/one path/two ...
"""
import argparse
import copy
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac: fall back to CPU if a GPU op is missing

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms as T

import data_utils as du
from common import (ARCHS, MEAN, STD, SkinDataset, build_model, eval_transform, pick_device,
                    predict_loader, save_checkpoint)
from metrics import compute_metrics, report_text

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_train_transform(size, aug):
    if aug == "basic":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
            T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), T.Normalize(MEAN, STD)])
    # "strong": imitates real-life variety (lighting, blur, angle, partial views)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.TrivialAugmentWide(),                                   # random rotation/colour/contrast/shear...
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),    # slightly blurry phone photos
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15))])             # hides small patches -> less reliance on one spot


def run_epoch(model, loader, device, loss_weights, train, optimizer=None, smoothing=0.1):
    model.train(train)
    total_loss, preds, labels = 0.0, [], []
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = F.cross_entropy(out, y, weight=loss_weights, label_smoothing=smoothing)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
        if train and step % 20 == 0:
            print(f"    batch {step}/{len(loader)}", flush=True)
    return total_loss / len(loader.dataset), preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more folders with one sub-folder per class")
    ap.add_argument("--out", default="models", help="where to save the trained model")
    ap.add_argument("--arch", default="efficientnet_b0", choices=list(ARCHS))
    ap.add_argument("--aug", default="strong", choices=["basic", "strong"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=0, help="stop early after this many epochs without improvement (0 = never)")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="epochs training only the last layer")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=None, help="cap images per class")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)   # show print() output immediately (also in notebooks)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device}   model: {args.arch}   image size: {args.img_size}   augmentation: {args.aug}")

    # ---- 1. Find, clean and split the data -------------------------------------------------
    items = []
    for folder in args.data:
        found = du.find_images(folder)
        print(f"{folder}: {len(found)} images")
        items += found
    if not items:
        sys.exit("No images found. Check the --data path(s).")
    if args.max_per_class:   # only scan what we need (x2 leaves room for broken/duplicate files)
        items = du.cap_per_class(items, args.max_per_class * 2, args.seed)
        print(f"Checking {len(items)} of them for broken files and duplicates...")
    else:
        print("Checking for broken files and duplicates (can take a while for big datasets)...")
    items, stats = du.clean_items(items)
    print(f"  removed: {stats}")
    items = du.cap_per_class(items, args.max_per_class, args.seed)
    classes = sorted({lab for _, lab in items})
    if len(classes) < 2:
        sys.exit(f"Need at least 2 classes, found {classes}. Point --data at the folder that CONTAINS the class folders.")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    split = du.stratified_split(items, seed=args.seed)
    print(f"\n{len(classes)} classes. Images per class (train/val/test):")
    for c in classes:
        n = [sum(1 for _, l in split[s] if l == c) for s in ("train", "val", "test")]
        print(f"  {c[:50]:<52}{n[0]:>6}{n[1]:>6}{n[2]:>6}")
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(split, f, indent=1)

    # ---- 2. Data loaders -------------------------------------------------------------------
    size = args.img_size
    workers = 0 if sys.platform == "darwin" else 2
    def loader(name, tf, shuffle):
        ds = SkinDataset(split[name], class_to_idx, tf)
        return DataLoader(ds, args.batch_size, shuffle=shuffle, num_workers=workers,
                          pin_memory=(device.type == "cuda"))
    train_dl = loader("train", make_train_transform(size, args.aug), True)
    val_dl = loader("val", eval_transform(size), False)
    test_dl = loader("test", eval_transform(size), False)

    # ---- 3. Model, class weights (rarer classes count more) --------------------------------
    model = build_model(len(classes), pretrained=True, arch=args.arch).to(device)
    counts = np.array([sum(1 for _, l in split["train"] if l == c) for c in classes], dtype=float)
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)

    # ---- 4. Train --------------------------------------------------------------------------
    ckpt_path = os.path.join(args.out, "skin_model.pt")
    best_f1, best_state, history, stale = -1.0, None, [], 0
    warmup = min(args.warmup_epochs, args.epochs)
    optimizer = scheduler = None
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and warmup > 0:      # stage 1: only the new last layer learns
            for p in model.features.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head)
            print("\nStage 1: training only the last layer")
        if epoch == warmup + 1:            # stage 2: the whole network adapts gently
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup))
            print("\nStage 2: fine-tuning the whole network")

        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, train_dl, device, weights, True, optimizer)
        va_loss, va_p, va_y = run_epoch(model, val_dl, device, weights, False)
        if scheduler and epoch > warmup:
            scheduler.step()
        tr_m, va_m = compute_metrics(tr_y, tr_p, len(classes)), compute_metrics(va_y, va_p, len(classes))
        print(f"Epoch {epoch}/{args.epochs}  train loss {tr_loss:.3f} acc {tr_m['accuracy']:.3f} | "
              f"val loss {va_loss:.3f} acc {va_m['accuracy']:.3f} macro-F1 {va_m['macro_f1']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_m["accuracy"], "val_acc": va_m["accuracy"], "val_macro_f1": va_m["macro_f1"]})
        if va_m["macro_f1"] > best_f1:     # keep the best model, save right away
            best_f1, best_state, stale = va_m["macro_f1"], copy.deepcopy(model.state_dict()), 0
            save_checkpoint(ckpt_path, model, classes, size, {"val_macro_f1": best_f1, "epoch": epoch}, args.arch)
            print("  -> new best model saved")
        else:
            stale += 1
            if args.patience and epoch > warmup and stale >= args.patience:
                print(f"No improvement for {args.patience} epochs - stopping early.")
                break

    # ---- 5. Final exam on the test set (touched only once, at the very end) ----------------
    model.load_state_dict(best_state)
    p0, y0 = predict_loader(model, test_dl, device, tta=False)
    p1, y1 = predict_loader(model, test_dl, device, tta=True)
    m0, m1 = compute_metrics(y0, p0, len(classes)), compute_metrics(y1, p1, len(classes))
    text = (f"Without mirror-averaging: accuracy {m0['accuracy'] * 100:.1f}%  macro-F1 {m0['macro_f1']:.3f}\n"
            f"With mirror-averaging (what the app uses):\n" + report_text(classes, m1))
    print("\n===== TEST SET RESULTS (images the model never saw or chose with) =====\n" + text)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"arch": args.arch, "img_size": size, "test_accuracy": m1["accuracy"],
                   "test_macro_f1": m1["macro_f1"], "history": history}, f, indent=1)
    print(f"\nDone. Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
