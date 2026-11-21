"""Evaluation maths (numpy only): accuracy, precision/recall/F1, confusion matrix."""
import numpy as np


def compute_metrics(y_true, y_pred, n):
    """Accuracy, per-class precision/recall/F1 and the confusion matrix."""
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    present = cm.sum(axis=1) > 0                      # average only over classes that have test images
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "precision": precision, "recall": recall, "f1": f1, "cm": cm}


def report_text(classes, m):
    lines = [f"Accuracy: {m['accuracy'] * 100:.1f}%   Macro-F1: {m['macro_f1']:.3f}", "",
             f"{'class':<45}{'precision':>10}{'recall':>8}{'f1':>8}{'n':>6}"]
    for i, c in enumerate(classes):
        lines.append(f"{c[:44]:<45}{m['precision'][i]:>10.2f}{m['recall'][i]:>8.2f}"
                     f"{m['f1'][i]:>8.2f}{int(m['cm'][i].sum()):>6}")
    lines += ["", "Confusion matrix (rows = true class, columns = predicted class):"]
    for i, c in enumerate(classes):
        lines.append(f"{i:>2} {c[:30]:<31}" + " ".join(f"{v:>4}" for v in m["cm"][i]))
    return "\n".join(lines)
