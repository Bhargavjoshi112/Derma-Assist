"""Model code shared by train.py, evaluate.py and app.py, so they can never disagree."""
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms as T

# ImageNet statistics: the pretrained networks expect pixels scaled like this.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# name -> (torchvision builder, name of its pretrained-weights enum)
ARCHS = {
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),   # small, fast
    "efficientnet_b2": (models.efficientnet_b2, "EfficientNet_B2_Weights"),   # bigger, use --img-size 288
    "efficientnet_b3": (models.efficientnet_b3, "EfficientNet_B3_Weights"),   # bigger still, --img-size 300
    "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),         # modern, strong on textures
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")            # NVIDIA GPU (Kaggle)
    if torch.backends.mps.is_available():
        return torch.device("mps")             # Apple Silicon GPU (M1/M2/M3...)
    return torch.device("cpu")


def build_model(num_classes, pretrained=True, arch="efficientnet_b0"):
    """A pretrained network with its last layer swapped for `num_classes` outputs."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {', '.join(ARCHS)}")
    builder, weights_name = ARCHS[arch]
    weights = getattr(models, weights_name).IMAGENET1K_V1 if pretrained else None
    model = builder(weights=weights)
    idx = 1 if arch.startswith("efficientnet") else 2      # position of the final Linear layer
    model.classifier[idx] = nn.Linear(model.classifier[idx].in_features, num_classes)
    return model


def eval_transform(img_size):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class SkinDataset(Dataset):
    def __init__(self, items, class_to_idx, tfm):
        self.items, self.class_to_idx, self.tfm = items, class_to_idx, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return self.tfm(img), self.class_to_idx[label]


def save_checkpoint(path, model, classes, img_size, extra=None, arch="efficientnet_b0"):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "classes": list(classes), "img_size": int(img_size),
                "arch": arch, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)  # safe: tensors only
    arch = ckpt.get("arch", "efficientnet_b0")                       # older files have no 'arch'
    model = build_model(len(ckpt["classes"]), pretrained=False, arch=arch)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _probs(model, x, tta):
    """Probabilities for a batch. TTA = also look at the mirrored photo and average both."""
    p = torch.softmax(model(x), dim=1)
    if tta:
        p = (p + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
    return p


@torch.no_grad()
def predict_topk(model, img, tfm, classes, k=3, device="cpu", tta=True):
    """Return the k most likely classes as [(label, probability), ...]."""
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    probs = _probs(model, x, tta)[0].cpu()
    top = torch.topk(probs, min(k, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


@torch.no_grad()
def predict_loader(model, loader, device, tta=False):
    """Predicted and true class indices for a whole DataLoader."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds += _probs(model, x.to(device), tta).argmax(1).cpu().tolist()
        labels += y.tolist()
    return preds, labels
