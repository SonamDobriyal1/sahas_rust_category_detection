from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, url_for


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "web_app" / "static" / "uploads"
RESULT_DIR = ROOT / "web_app" / "static" / "results"
DEFAULT_WEIGHTS = ROOT / "runs" / "obb" / "rust_corrosion_yolov8_obb" / "weights" / "best.pt"
WEIGHTS_PATH = Path(os.environ.get("YOLO_WEIGHTS", DEFAULT_WEIGHTS)).resolve()

CLASS_NAMES = {
    0: "mild-corrosion",
    1: "moderate-corrosion",
    2: "severe-corrosion",
}
SEVERITY_RANK = {
    "mild-corrosion": 1,
    "moderate-corrosion": 2,
    "severe-corrosion": 3,
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
model = None


def load_model():
    global model
    if model is not None:
        return model

    weights_path = find_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. Train first or set YOLO_WEIGHTS."
        )

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(weights_path))
    return model


def find_weights_path() -> Path:
    if WEIGHTS_PATH.exists():
        return WEIGHTS_PATH

    candidates = sorted(
        (ROOT / "runs").glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    return WEIGHTS_PATH


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def summarize_detections(result, confidence_threshold: float) -> dict:
    detections = []
    counts = {name: 0 for name in CLASS_NAMES.values()}

    obb = getattr(result, "obb", None)
    if obb is None or obb.cls is None:
        return {
            "overall": "No corrosion detected",
            "detections": [],
            "counts": counts,
            "max_confidence": 0.0,
        }

    classes = obb.cls.cpu().numpy().astype(int).tolist()
    confidences = obb.conf.cpu().numpy().tolist()

    for class_id, confidence in zip(classes, confidences):
        if confidence < confidence_threshold:
            continue

        class_name = CLASS_NAMES.get(class_id, f"class-{class_id}")
        counts[class_name] = counts.get(class_name, 0) + 1
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(confidence), 4),
            }
        )

    if not detections:
        overall = "No corrosion detected"
        max_confidence = 0.0
    else:
        strongest = max(
            detections,
            key=lambda item: (
                SEVERITY_RANK.get(item["class_name"], 0),
                item["confidence"],
            ),
        )
        overall = strongest["class_name"]
        max_confidence = max(item["confidence"] for item in detections)

    return {
        "overall": overall,
        "detections": detections,
        "counts": counts,
        "max_confidence": round(max_confidence, 4),
    }


@app.get("/")
def index():
    return render_template("index.html", weights_path=find_weights_path())


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file."}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "" or not allowed_file(uploaded.filename):
        return jsonify({"error": "Use a JPG, PNG, BMP, or WEBP image."}), 400

    confidence = float(request.form.get("confidence", 0.25))
    confidence = min(1.0, max(0.01, confidence))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    extension = Path(uploaded.filename).suffix.lower()
    image_name = f"{uuid.uuid4().hex}{extension}"
    upload_path = UPLOAD_DIR / image_name
    result_path = RESULT_DIR / image_name
    uploaded.save(upload_path)

    detector = load_model()
    results = detector.predict(str(upload_path), task="obb", conf=confidence, verbose=False)
    result = results[0]
    summary = summarize_detections(result, confidence)

    annotated = result.plot()
    cv2.imwrite(str(result_path), annotated)

    return jsonify(
        {
            **summary,
            "image_url": url_for("static", filename=f"uploads/{image_name}"),
            "result_url": url_for("static", filename=f"results/{image_name}"),
            "weights_path": str(find_weights_path()),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
