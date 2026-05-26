# Rust Category Detection Platform

Flask web platform for YOLO11 oriented bounding box rust category detection.

## Run

```powershell
cd rust_detection_platform
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5001`.

## Model

The deployed model is copied from:

`runs/obb_yolo11/rust_corrosion_yolo11s_obb/weights/best.pt`

to:

`rust_detection_platform/model/rust_yolo11_best.pt`

Set `YOLO_WEIGHTS` to use another weights file.
