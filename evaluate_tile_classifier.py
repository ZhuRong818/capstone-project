import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class TinyTileNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, n_classes)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.head(z)


def canonical_label(raw: str) -> str | None:
    n = str(raw).lower()
    if n in ("human", "agent", "player"):
        return "agent"
    if "gem" in n or "core" in n:
        return "gem"
    if "coin" in n:
        return "coin"
    if "key" in n:
        return "key"
    if "boots" in n or "speed" in n:
        return "boots"
    if "ghost" in n or "phasing" in n:
        return "ghost"
    if "shield" in n:
        return "shield"
    if "box" in n:
        return "box"
    if "exit" in n:
        return "exit"
    if "opened" in n or ("door" in n and "open" in n):
        return "door_open"
    if "locked" in n or ("door" in n and "lock" in n):
        return "door_locked"
    if "wall" in n:
        return "wall"
    if "lava" in n:
        return "lava"
    if "floor" in n or "ground" in n or "path" in n:
        return "floor"
    return None


def collect_samples(asset_root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(asset_root):
        label = canonical_label(os.path.basename(dirpath))
        if label is None:
            continue
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            out.append((os.path.join(dirpath, f), label))
    return out


def load_image(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size), Image.NEAREST)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/tile_model.pt")
    parser.add_argument("--assets", default="data/assets")
    parser.add_argument("--max_per_class", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    labels = [str(x) for x in ckpt.get("labels", [])]
    image_size = int(ckpt.get("image_size", 32))
    state_dict = ckpt.get("state_dict")
    if not labels or not isinstance(state_dict, dict):
        raise RuntimeError("Invalid model checkpoint.")

    model = TinyTileNet(len(labels))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    samples = collect_samples(args.assets)
    if not samples:
        raise RuntimeError(f"No samples found in {args.assets}")
    label_to_idx: Dict[str, int] = {l: i for i, l in enumerate(labels)}

    per_class_seen: Dict[str, int] = {}
    filtered: List[Tuple[str, str]] = []
    for p, l in samples:
        if l not in label_to_idx:
            continue
        if args.max_per_class > 0:
            n = per_class_seen.get(l, 0)
            if n >= args.max_per_class:
                continue
            per_class_seen[l] = n + 1
        filtered.append((p, l))

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    with torch.no_grad():
        for p, gt_label in filtered:
            x = load_image(p, image_size)
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            gt_idx = label_to_idx[gt_label]
            cm[gt_idx, pred_idx] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = (correct / max(1, total)) * 100.0
    print(f"samples={total} acc={acc:.2f}%")

    print("\nPer-class metrics:")
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        support = int(cm[i, :].sum())
        print(f"{label:12s} precision={precision:.3f} recall={recall:.3f} support={support}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("labels:", json.dumps(labels))
    print(cm.tolist())


if __name__ == "__main__":
    main()
