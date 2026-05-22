from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a newer YOLO11 OBB model on the corrosion severity dataset."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "data.yaml")
    parser.add_argument(
        "--model",
        default="yolo11s-obb.pt",
        help="Use yolo11n-obb.pt for faster training, yolo11m/l/x-obb.pt for higher capacity.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU id, for example 0. Use cpu only if needed.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "obb_yolo11")
    parser.add_argument("--name", default="rust_corrosion_yolo11s_obb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--export",
        choices=("onnx", "engine", "openvino"),
        default=None,
        help="Optional export format after training. engine is TensorRT and requires NVIDIA setup.",
    )
    return parser.parse_args()


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc

    return YOLO


def print_gpu_status(device: str) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed, so CUDA status could not be checked.")
        return

    if device == "cpu":
        print("Training requested on CPU.")
        return

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU was requested, but PyTorch cannot see CUDA.\n"
            "Install a CUDA-enabled PyTorch build or run with --device cpu."
        )

    gpu_count = torch.cuda.device_count()
    print(f"CUDA available: {gpu_count} GPU(s)")
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {idx}: {props.name} ({memory_gb:.1f} GB)")


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    project_path = args.project.resolve()

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    YOLO = require_ultralytics()
    print_gpu_status(args.device)

    model = YOLO(args.model)
    train_results = model.train(
        data=str(data_path),
        task="obb",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(project_path),
        name=args.name,
        resume=args.resume,
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        plots=True,
    )

    best_weights = project_path / args.name / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best weights: {best_weights}")

    if args.validate:
        trained = YOLO(str(best_weights))
        metrics = trained.val(data=str(data_path), task="obb", imgsz=args.imgsz, device=args.device)
        print(metrics)

    if args.export:
        trained = YOLO(str(best_weights))
        exported = trained.export(format=args.export, imgsz=args.imgsz, device=args.device)
        print(f"Exported model: {exported}")

    return train_results


if __name__ == "__main__":
    main()
