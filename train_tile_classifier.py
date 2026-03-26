import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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


def collect_samples(asset_root: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    samples: List[Tuple[str, str]] = []
    labels = set()
    for dirpath, _, files in os.walk(asset_root):
        folder = os.path.basename(dirpath)
        label = canonical_label(folder)
        if label is None:
            continue
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(dirpath, f)
            samples.append((p, label))
            labels.add(label)
    return samples, sorted(labels)


def collect_samples_multi(asset_roots: List[str]) -> Tuple[List[Tuple[str, str]], List[str], Dict[str, int]]:
    samples: List[Tuple[str, str]] = []
    labels = set()
    seen_paths = set()
    per_root_counts: Dict[str, int] = {}
    for root in asset_roots:
        root_abs = os.path.abspath(root)
        root_samples, root_labels = collect_samples(root_abs)
        kept = 0
        for path, label in root_samples:
            norm = os.path.abspath(path)
            if norm in seen_paths:
                continue
            seen_paths.add(norm)
            samples.append((norm, label))
            labels.add(label)
            kept += 1
        per_root_counts[root_abs] = kept
        labels.update(root_labels)
    return samples, sorted(labels), per_root_counts


@dataclass
class TrainConfig:
    asset_root: str = "data/assets"
    image_size: int = 32
    batch_size: int = 64
    epochs: int = 24
    lr: float = 1e-3
    seed: int = 42
    val_ratio: float = 0.2
    focal_gamma: float = 0.0
    label_smoothing: float = 0.02
    min_class_weight: float = 0.8
    max_class_weight: float = 2.0
    sampler_power: float = 0.35
    focus_boost: float = 1.35
    focus_labels: Tuple[str, ...] = ("boots", "key", "gem", "door_locked", "shield", "ghost")
    out_model: str = "data/tile_model.pt"
    out_labels: str = "data/tile_labels.json"


class TileDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], image_size: int, train: bool):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.train:
            return image
        # Mild appearance jitter for renderer robustness.
        if random.random() < 0.7:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.7:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.2))
        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.2))
        return image

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size), Image.NEAREST)
            im = self._augment(im)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
        return x, torch.tensor(y, dtype=torch.long)


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


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float, label_smoothing: float):
        super().__init__()
        self.gamma = max(0.0, float(gamma))
        smooth = max(0.0, min(0.2, float(label_smoothing)))
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            label_smoothing=smooth,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        focal_factor = (1.0 - pt) ** self.gamma
        return torch.mean(focal_factor * ce)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return (correct / max(1, total)) * 100.0


def stratified_split(
    indexed: List[Tuple[str, int]], n_classes: int, val_ratio: float, seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(n_classes)}
    for sample in indexed:
        by_class[sample[1]].append(sample)

    train: List[Tuple[str, int]] = []
    val: List[Tuple[str, int]] = []
    for cls in range(n_classes):
        bucket = by_class.get(cls, [])
        if not bucket:
            continue
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_ratio)))
        # Keep at least one sample in train when possible.
        if len(bucket) > 1:
            n_val = min(n_val, len(bucket) - 1)
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", default="data/assets")
    parser.add_argument(
        "--asset_roots",
        default="",
        help="Comma-separated asset roots. If set, overrides --asset_root.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out_model", default=None)
    parser.add_argument("--out_labels", default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.out_model:
        cfg.out_model = str(args.out_model)
    if args.out_labels:
        cfg.out_labels = str(args.out_labels)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    raw_roots = [p.strip() for p in str(args.asset_roots).split(",") if p.strip()]
    if raw_roots:
        asset_roots = [os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p for p in raw_roots]
    else:
        cfg.asset_root = str(args.asset_root)
        asset_roots = [os.path.join(os.getcwd(), cfg.asset_root)]

    samples, labels, per_root_counts = collect_samples_multi(asset_roots)
    if not samples:
        raise RuntimeError(f"No samples found in {asset_roots}")
    print("asset roots:")
    for root, count in per_root_counts.items():
        print(f"  {root}: {count} samples")
    print(f"total unique samples: {len(samples)}")

    label_to_idx: Dict[str, int] = {k: i for i, k in enumerate(labels)}
    indexed = [(p, label_to_idx[l]) for p, l in samples]
    train_samples, val_samples = stratified_split(indexed, len(labels), cfg.val_ratio, cfg.seed)
    if not val_samples:
        val_samples = train_samples[-max(1, len(train_samples) // 10) :]
    if not train_samples:
        raise RuntimeError("No training samples after split.")

    class_counts = np.zeros(len(labels), dtype=np.float32)
    for _, y in train_samples:
        class_counts[y] += 1.0
    class_counts = np.maximum(class_counts, 1.0)
    # Balance classes without letting one rare class dominate all minibatches.
    class_weights_np = class_counts.sum() / (len(labels) * class_counts)
    class_weights_np = np.clip(class_weights_np, cfg.min_class_weight, cfg.max_class_weight)
    # Extra boost for mechanics-critical labels that still underperform.
    for name in cfg.focus_labels:
        idx = label_to_idx.get(name)
        if idx is None:
            continue
        class_weights_np[idx] = min(cfg.max_class_weight, class_weights_np[idx] * cfg.focus_boost)
    sample_weights = [float(class_weights_np[y] ** cfg.sampler_power) for _, y in train_samples]
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_samples),
        replacement=True,
    )

    train_ds = TileDataset(train_samples, cfg.image_size, train=True)
    val_ds = TileDataset(val_samples, cfg.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTileNet(len(labels)).to(device)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    criterion = WeightedFocalLoss(
        class_weights=class_weights,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:02d} loss={running_loss/max(1,len(train_loader)):.4f} val_acc={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")

    os.makedirs(os.path.dirname(cfg.out_model), exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "image_size": cfg.image_size,
            "labels": labels,
            "focal_gamma": cfg.focal_gamma,
            "label_smoothing": cfg.label_smoothing,
        },
        cfg.out_model,
    )
    with open(cfg.out_labels, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "image_size": cfg.image_size, "best_val_acc": best_acc}, f, indent=2)
    print(f"saved model to {cfg.out_model}")
    print(f"saved labels to {cfg.out_labels}")


if __name__ == "__main__":
    main()
