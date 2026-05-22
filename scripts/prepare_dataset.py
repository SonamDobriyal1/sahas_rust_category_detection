from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("mild-corrosion", "moderate-corrosion", "severe-corrosion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLOv8-OBB dataset splits and data.yaml."
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root.")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Moving is the default.",
    )
    parser.add_argument(
        "--clip-labels",
        action="store_true",
        help="Clip OBB coordinates in label files to the 0..1 range.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned split counts without changing files.",
    )
    return parser.parse_args()


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: dict[str, tuple[Path | None, Path | None]] = {}

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if images_dir.exists():
            for image_path in images_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                    image, label = pairs.get(image_path.stem, (None, None))
                    pairs[image_path.stem] = (image_path, label)

        if labels_dir.exists():
            for label_path in labels_dir.glob("*.txt"):
                image, label = pairs.get(label_path.stem, (None, None))
                pairs[label_path.stem] = (image, label_path)

    missing_images = sorted(stem for stem, (image, _) in pairs.items() if image is None)
    missing_labels = sorted(stem for stem, (_, label) in pairs.items() if label is None)

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise SystemExit(f"Labels without images: {len(missing_images)}. Examples: {preview}")
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise SystemExit(f"Images without labels: {len(missing_labels)}. Examples: {preview}")

    return [(image, label) for image, label in pairs.values() if image and label]


def split_pairs(
    pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    train_count = round(len(shuffled) * train_ratio)
    val_count = round(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def recreate_dirs(root: Path) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = root / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_path.unlink()


def transfer_pairs(
    target_root: Path, splits: dict[str, list[tuple[Path, Path]]], copy_files: bool
) -> None:
    transfer = shutil.copy2 if copy_files else shutil.move

    for split, pairs in splits.items():
        (target_root / split / "images").mkdir(parents=True, exist_ok=True)
        (target_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for image_path, label_path in pairs:
            target_image = target_root / split / "images" / image_path.name
            target_label = target_root / split / "labels" / label_path.name

            if image_path.resolve() != target_image.resolve():
                transfer(str(image_path), str(target_image))
            if label_path.resolve() != target_label.resolve():
                transfer(str(label_path), str(target_label))


def replace_split_dirs(root: Path, staged_root: Path) -> None:
    for split in SPLITS:
        split_dir = root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        shutil.move(str(staged_root / split), str(split_dir))

    if staged_root.exists():
        shutil.rmtree(staged_root)


def clip_labels(root: Path) -> int:
    changed_rows = 0

    for label_path in root.glob("*/labels/*.txt"):
        new_lines: list[str] = []

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9:
                new_lines.append(line)
                continue

            class_id = parts[0]
            coords = [float(value) for value in parts[1:9]]
            clipped = [min(1.0, max(0.0, value)) for value in coords]
            if clipped != coords:
                changed_rows += 1

            new_lines.append(" ".join([class_id, *[f"{value:.12g}" for value in clipped]]))

        label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

    return changed_rows


def write_data_yaml(root: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CLASS_NAMES)}
names:
{names}
"""
    (root / "data.yaml").write_text(content)


def main() -> None:
    args = parse_args()
    root = args.root
    total_ratio = args.train + args.val + args.test

    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit("--train, --val, and --test ratios must add up to 1.0")

    pairs = collect_pairs(root)
    splits = split_pairs(pairs, args.train, args.val, args.seed)

    print(f"Found {len(pairs)} image/label pairs")
    for split in SPLITS:
        print(f"{split}: {len(splits[split])}")

    if args.dry_run:
        return

    staged_root = root / ".prepared_splits_tmp"
    if staged_root.exists():
        shutil.rmtree(staged_root)

    transfer_pairs(staged_root, splits, copy_files=args.copy)
    replace_split_dirs(root, staged_root)

    if args.clip_labels:
        changed_rows = clip_labels(root)
        print(f"Clipped coordinates in {changed_rows} label rows")

    write_data_yaml(root)
    print(f"Updated {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
