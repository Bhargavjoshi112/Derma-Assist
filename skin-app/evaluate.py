"""Measure a trained model on YOUR OWN photos (the honest real-world test).

Make a folder with one sub-folder per condition, named like the model's classes, e.g.
    my_photos/Eczema/a.jpg      my_photos/warts/b.jpg      my_photos/psoriasis/c.jpg
(Short names work: "warts", "psoriasis", "tinea" ... as long as they match only one class.)
Run:  python evaluate.py --data my_photos
"""
import argparse
import os
import sys

from torch.utils.data import DataLoader

import data_utils as du
from common import SkinDataset, eval_transform, load_checkpoint, pick_device, predict_loader
from metrics import compute_metrics, report_text

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
ap.add_argument("--model", default=os.path.join("models", "skin_model.pt"))
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--no-tta", action="store_true", help="turn off mirror-image averaging")
args = ap.parse_args()

device = pick_device()
model, ckpt = load_checkpoint(args.model, device)
classes = ckpt["classes"]
class_to_idx = {c: i for i, c in enumerate(classes)}

items = du.find_images(args.data)
resolved = [(p, du.resolve_class(lab, classes)) for p, lab in items]
known = [(p, c) for p, c in resolved if c is not None]
unknown = sorted({lab for (_, lab), (_, c) in zip(items, resolved) if c is None})
if unknown:
    print("Skipped folders the model does not know:", unknown)
    print("Model classes are:", classes)
if not known:
    sys.exit("No images in folders that match the model's class names.")

dl = DataLoader(SkinDataset(known, class_to_idx, eval_transform(ckpt["img_size"])), args.batch_size)
preds, labels = predict_loader(model, dl, device, tta=not args.no_tta)
print(report_text(classes, compute_metrics(labels, preds, len(classes))))
