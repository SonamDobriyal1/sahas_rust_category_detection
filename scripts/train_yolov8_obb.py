from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 OBB model for corrosion severity detection."
    )
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--model", default="yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: 0 for GPU, cpu for CPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/obb")
    parser.add_argument("--name", default="rust_corrosion_yolov8_obb")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Run validation after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    project_path = (ROOT / args.project).resolve()

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_path),
        "name": args.name,
        "patience": args.patience,
        "task": "obb",
        "resume": args.resume,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    best_weights = project_path / args.name / "weights" / "best.pt"

    print("\nTraining complete.")
    print(f"Best weights should be here: {best_weights}")

    if args.validate:
        trained_model = YOLO(str(best_weights if best_weights.exists() else args.model))
        metrics = trained_model.val(data=str(data_path), task="obb", imgsz=args.imgsz)
        print(metrics)

    return results


if __name__ == "__main__":
    main()
