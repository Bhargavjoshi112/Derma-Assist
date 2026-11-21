"""Run the model on every image in a folder (no labels needed) and print a table.

Handy for testing pictures you downloaded from the web:
    python predict_folder.py --data ~/Desktop/test_images
"""
import argparse
import os

from PIL import Image, ImageOps

import data_utils as du
from common import eval_transform, load_checkpoint, predict_topk

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder containing images")
ap.add_argument("--model", default=os.path.join("models", "skin_model.pt"))
ap.add_argument("--no-tta", action="store_true", help="turn off mirror-image averaging")
args = ap.parse_args()

model, ckpt = load_checkpoint(args.model)
classes, tfm = ckpt["classes"], eval_transform(ckpt["img_size"])

paths = [p for p, _ in du.find_images(args.data)]
if not paths:
    raise SystemExit(f"No images found in {args.data}")

for path in paths:
    try:
        img = ImageOps.exif_transpose(Image.open(path))
        top = predict_topk(model, img, tfm, classes, k=3, tta=not args.no_tta)
    except Exception as err:
        print(f"{os.path.basename(path)}: could not read ({err})")
        continue
    flag = "" if top[0][1] >= 0.5 else "   (not sure)"
    print(f"\n{os.path.basename(path)}{flag}")
    for label, prob in top:
        print(f"   {prob * 100:5.1f}%  {label}")
