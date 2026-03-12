import argparse
import os
import random
import shutil
from typing import Dict, List

from PIL import Image, ImageEnhance


DEFAULT_WEAK_LABELS = {"agent", "door_locked", "door_open", "shield", "boots", "key", "gem", "ghost"}


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


def collect_by_label(asset_root: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for dirpath, _, files in os.walk(asset_root):
        label = canonical_label(os.path.basename(dirpath))
        if label is None:
            continue
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            out.setdefault(label, []).append(os.path.join(dirpath, f))
    return out


def aug_one(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.9:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.9:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.7:
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.75, 1.3))
    if rng.random() < 0.35:
        im = ImageEnhance.Sharpness(im).enhance(rng.uniform(0.7, 1.3))
    return im


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/assets")
    parser.add_argument("--dst", default="data/assets_augmented")
    parser.add_argument("--target_per_weak_class", type=int, default=160)
    parser.add_argument("--weak_labels", default="agent,door_locked,door_open,shield,boots,key,gem,ghost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    weak_labels = {x.strip() for x in str(args.weak_labels).split(",") if x.strip()}
    if not weak_labels:
        weak_labels = set(DEFAULT_WEAK_LABELS)

    rng = random.Random(args.seed)
    by_label = collect_by_label(args.src)
    if not by_label:
        raise RuntimeError(f"No assets found in {args.src}")

    if os.path.isdir(args.dst):
        shutil.rmtree(args.dst)
    ensure_dir(args.dst)

    # First copy originals into dst grouped by canonical label.
    for label, paths in by_label.items():
        out_dir = os.path.join(args.dst, label)
        ensure_dir(out_dir)
        for i, p in enumerate(paths):
            ext = os.path.splitext(p)[1].lower() or ".png"
            out_path = os.path.join(out_dir, f"orig_{i:04d}{ext}")
            shutil.copy2(p, out_path)

    # Then upsample weak classes to target count via augmentation.
    for label in sorted(weak_labels):
        paths = by_label.get(label, [])
        if not paths:
            print(f"[warn] no source samples for weak class: {label}")
            continue
        out_dir = os.path.join(args.dst, label)
        cur = len([x for x in os.listdir(out_dir) if x.lower().endswith((".png", ".jpg", ".jpeg"))])
        to_add = max(0, args.target_per_weak_class - cur)
        for i in range(to_add):
            src = rng.choice(paths)
            with Image.open(src) as im:
                rgb = im.convert("RGB")
                aug = aug_one(rgb, rng)
                out_path = os.path.join(out_dir, f"aug_{i:05d}.png")
                aug.save(out_path)
        print(f"{label}: original={len(paths)} final={cur + to_add}")

    print(f"augmented assets written to {args.dst}")


if __name__ == "__main__":
    main()
