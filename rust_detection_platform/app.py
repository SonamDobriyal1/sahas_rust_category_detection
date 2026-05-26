from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "static" / "uploads"
RESULT_DIR = APP_DIR / "static" / "results"
DEFAULT_WEIGHTS = APP_DIR / "model" / "rust_yolo11_best.pt"
WEIGHTS_PATH = Path(os.environ.get("YOLO_WEIGHTS", DEFAULT_WEIGHTS)).resolve()

CLASS_NAMES = {
    0: "mild-corrosion",
    1: "moderate-corrosion",
    2: "severe-corrosion",
}

DISPLAY_NAMES = {
    "mild-corrosion": "Mild corrosion",
    "moderate-corrosion": "Moderate corrosion",
    "severe-corrosion": "Severe corrosion",
}

SEVERITY_RANK = {
    "mild-corrosion": 1,
    "moderate-corrosion": 2,
    "severe-corrosion": 3,
}

RECOMMENDATIONS = {
    "No corrosion detected": "No visible rust category was found above the selected confidence threshold.",
    "mild-corrosion": "Monitor the area, clean the surface, and schedule preventive coating before spread accelerates.",
    "moderate-corrosion": "Plan maintenance soon. Remove loose rust and inspect the surrounding structure for hidden spread.",
    "severe-corrosion": "Prioritize inspection and repair. Severe corrosion can compromise load-bearing strength.",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
model = None


def find_weights_path() -> Path:
    if WEIGHTS_PATH.exists():
        return WEIGHTS_PATH

    repo_root = APP_DIR.parent
    candidates = sorted(
        (repo_root / "runs").glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    return WEIGHTS_PATH


def load_model():
    global model
    if model is not None:
        return model

    weights_path = find_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights were not found at {weights_path}. Copy YOLO11 best.pt or set YOLO_WEIGHTS."
        )

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    model = YOLO(str(weights_path))
    return model


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def normalize_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"class-{class_id}")


def summarize_detections(result, confidence_threshold: float) -> dict:
    counts = {name: 0 for name in CLASS_NAMES.values()}
    detections = []

    obb = getattr(result, "obb", None)
    if obb is None or obb.cls is None or obb.conf is None:
        return empty_summary(counts)

    classes = obb.cls.cpu().numpy().astype(int).tolist()
    confidences = obb.conf.cpu().numpy().tolist()

    for class_id, confidence in zip(classes, confidences):
        if confidence < confidence_threshold:
            continue

        class_name = normalize_name(class_id)
        confidence_value = round(float(confidence), 4)
        counts[class_name] = counts.get(class_name, 0) + 1
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "label": DISPLAY_NAMES.get(class_name, class_name),
                "confidence": confidence_value,
            }
        )

    if not detections:
        return empty_summary(counts)

    overall_detection = max(
        detections,
        key=lambda item: (
            SEVERITY_RANK.get(item["class_name"], 0),
            item["confidence"],
        ),
    )
    overall = overall_detection["class_name"]
    max_confidence = max(item["confidence"] for item in detections)

    return {
        "overall": DISPLAY_NAMES.get(overall, overall),
        "overall_key": overall,
        "recommendation": RECOMMENDATIONS.get(overall, ""),
        "detections": detections,
        "counts": counts,
        "max_confidence": round(max_confidence, 4),
        "total": len(detections),
    }


def empty_summary(counts: dict[str, int]) -> dict:
    return {
        "overall": "No corrosion detected",
        "overall_key": "none",
        "recommendation": RECOMMENDATIONS["No corrosion detected"],
        "detections": [],
        "counts": counts,
        "max_confidence": 0.0,
        "total": 0,
    }


@app.get("/")
def home():
    return render_template("home.html", active_page="home")


@app.get("/detection")
def detection():
    return render_template(
        "detection.html",
        active_page="detection",
        weights_name=find_weights_path().name,
    )


@app.get("/about")
def about():
    return render_template("about.html", active_page="about")


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file."}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "" or not allowed_file(uploaded.filename):
        return jsonify({"error": "Use a JPG, PNG, BMP, or WEBP image."}), 400

    try:
        confidence = float(request.form.get("confidence", 0.25))
    except ValueError:
        confidence = 0.25
    confidence = min(0.95, max(0.01, confidence))

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

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    annotated = result.plot()
    cv2.imwrite(str(result_path), annotated)

    return jsonify(
        {
            **summary,
            "image_url": url_for("static", filename=f"uploads/{image_name}"),
            "result_url": url_for("static", filename=f"results/{image_name}"),
            "weights_name": find_weights_path().name,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
