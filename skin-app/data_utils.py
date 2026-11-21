"""Data helpers (no PyTorch needed here): find images, clean them, split them.

The folder name of each image is its label, e.g.
    data/Eczema/img1.jpg   ->  label "Eczema"
"""
import hashlib
import os
import random
import re
from collections import defaultdict

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_label(name):
    """'1. Eczema 1677' -> 'Eczema', '2. Melanoma 15.75k' -> 'Melanoma'."""
    name = re.sub(r"^\s*\d+\s*[.\-)]\s*", "", name)          # leading "1. "
    name = re.sub(r"[\s\-_]+[\d.,]+\s*[kK]?\s*$", "", name)   # trailing " - 1.25k"
    return name.strip(" -_") or name


def find_images(root):
    """Walk the folder tree; every image's label is its parent folder's name."""
    items = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        label = clean_label(os.path.basename(dirpath.rstrip("/\\")))
        for f in files:
            if f.startswith("."):
                continue
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                items.append((os.path.join(dirpath, f), label))
    return sorted(items)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_items(items, progress_every=2000):
    """Drop unreadable files and exact duplicates.

    Why duplicates matter: if the same photo is in both the training set and the
    test set, the model is being tested on something it has already memorised,
    and the accuracy looks better than it really is.
    A photo that appears under two DIFFERENT labels is dropped completely.
    """
    stats = {"unreadable": 0, "duplicates": 0, "conflicting": 0}
    by_hash = defaultdict(list)
    for n, (path, label) in enumerate(items, 1):
        if progress_every and n % progress_every == 0:
            print(f"    checked {n}/{len(items)} files...", flush=True)
        try:
            with Image.open(path) as im:
                im.verify()
            by_hash[_md5(path)].append((path, label))
        except Exception:
            stats["unreadable"] += 1

    kept = []
    for group in by_hash.values():
        if len({lab for _, lab in group}) > 1:
            stats["conflicting"] += len(group)
            continue
        kept.append(group[0])
        stats["duplicates"] += len(group) - 1
    return sorted(kept), stats


def cap_per_class(items, max_per_class, seed=42):
    """Keep at most `max_per_class` images per class (reduces imbalance, speeds training)."""
    if not max_per_class:
        return items
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[it[1]].append(it)
    out = []
    for label in sorted(groups):
        g = groups[label]
        rng.shuffle(g)
        out.extend(g[:max_per_class])
    return sorted(out)


def stratified_split(items, val_frac=0.15, test_frac=0.15, seed=42):
    """Split into train/val/test, keeping each class's share the same in each part."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[it[1]].append(it)
    split = {"train": [], "val": [], "test": []}
    for label in sorted(groups):
        g = groups[label]
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        if n - n_test - n_val < 1:
            raise ValueError(f"Class '{label}' has only {n} usable images - too few.")
        split["test"] += g[:n_test]
        split["val"] += g[n_test:n_test + n_val]
        split["train"] += g[n_test + n_val:]
    return split


def resolve_class(name, classes):
    """Match a folder name to one of the model's classes, forgivingly.

    'eczema' -> 'Eczema';  'psoriasis' -> 'Psoriasis pictures Lichen Planus and ...'
    Returns the class name, or None if nothing (or more than one thing) matches.
    """
    n = clean_label(name).strip().lower()
    low = {c.lower(): c for c in classes}
    if n in low:
        return low[n]
    starts = [c for c in classes if c.lower().startswith(n)]
    if len(starts) == 1:
        return starts[0]
    inside = [c for c in classes if n in c.lower()]
    return inside[0] if len(inside) == 1 else None
