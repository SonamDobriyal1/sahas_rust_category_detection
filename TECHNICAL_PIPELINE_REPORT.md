# Rust Category Detection Technical Pipeline Report

## 1. Project Objective

This project trains oriented object detection models to detect and categorize rust/corrosion severity in images. The task is not simple image classification; it is object detection with oriented bounding boxes (OBB). Each rust region is localized by a rotated box/polygon and assigned one of three severity categories:

- `mild-corrosion`
- `moderate-corrosion`
- `severe-corrosion`

The project compares two Ultralytics OBB models:

- YOLOv8 OBB: `yolov8n-obb.pt`
- YOLO11 OBB: `yolo11s-obb.pt`

The final trained weights are stored in:

- YOLOv8 best model: `runs/obb/runs/obb/rust_corrosion_yolov8_obb/weights/best.pt`
- YOLO11 best model: `runs/obb_yolo11/rust_corrosion_yolo11s_obb/weights/best.pt`

## 2. Repository Structure

Important project files and folders:

```text
sahas_rust_category_detection/
  data/
    data.yaml
    README.roboflow.txt
    train/images, train/labels
    valid/images, valid/labels
    test/images, test/labels
  scripts/
    prepare_dataset.py
    train_yolov8_obb.py
    train_yolo11_obb_gpu.py
  runs/
    obb/runs/obb/rust_corrosion_yolov8_obb/
    obb_yolo11/rust_corrosion_yolo11s_obb/
  web_app/
  requirements.txt
  yolov8n-obb.pt
  yolo11s-obb.pt
```

## 3. Dataset

The dataset was exported from Roboflow on May 13, 2026. According to `data/README.roboflow.txt`, it contains 2,073 images and was exported in YOLOv8 Oriented Object Detection format. No Roboflow-side preprocessing or augmentation was applied during export.

The active dataset configuration is in `data/data.yaml`:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: mild-corrosion
  1: moderate-corrosion
  2: severe-corrosion
```

Current image/label split:

| Split | Images | Label files |
|---|---:|---:|
| Train | 1,658 | 1,658 |
| Validation | 207 | 207 |
| Test | 208 | 208 |
| Total | 2,073 | 2,073 |

Current object-instance distribution:

| Split | Mild | Moderate | Severe | Total objects |
|---|---:|---:|---:|---:|
| Train | 7,969 | 5,707 | 6,808 | 20,484 |
| Validation | 927 | 694 | 840 | 2,461 |
| Test | 1,029 | 729 | 773 | 2,531 |

The classes are reasonably represented, but they are not perfectly balanced. `moderate-corrosion` has fewer annotations than `mild-corrosion` and `severe-corrosion`, which can make moderate examples harder for the model to learn consistently, especially when the visual boundary between mild/moderate/severe is subjective.

## 4. Dataset Preparation Pipeline

Dataset preparation is handled by `scripts/prepare_dataset.py`.

The script performs the following steps:

1. Collect image/label pairs from `data/train`, `data/valid`, and `data/test`.
2. Verify that every image has a matching `.txt` label file and every label has a matching image.
3. Shuffle the complete dataset using a fixed random seed.
4. Split the data into train/validation/test sets using the default ratio:
   - Train: 80 percent
   - Validation: 10 percent
   - Test: 10 percent
5. Recreate the split folders.
6. Move or copy files into the correct split folders.
7. Optionally clip OBB label coordinates to the normalized `0..1` range.
8. Rewrite `data/data.yaml`.

The default split seed is `42`, which makes the split reproducible when the same input files are used.

Typical command:

```powershell
python scripts/prepare_dataset.py --root data --train 0.8 --val 0.1 --test 0.1 --seed 42 --clip-labels
```

The label format is YOLO OBB format. Each annotation row contains a class id followed by eight normalized coordinates representing the four corners of the oriented box:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

This format is important for rust detection because corrosion often appears on angled metal surfaces, curved panels, or irregular visual regions where a horizontal bounding box would include too much background.

## 5. Environment and Dependencies

The project dependencies are listed in `requirements.txt`:

```text
ultralytics>=8.0.0
flask>=3.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
```

Training is performed using Ultralytics' `YOLO` Python API. The training scripts both load a pretrained OBB checkpoint and fine-tune it on the rust category dataset.

Although `scripts/train_yolo11_obb_gpu.py` is designed to check CUDA availability and train on GPU by default, the saved `args.yaml` for both completed runs shows `device: cpu`. This means the recorded experiments were executed on CPU, which explains the long training times.

## 6. YOLOv8 OBB Training Pipeline

YOLOv8 training is implemented in `scripts/train_yolov8_obb.py`.

Default command:

```powershell
python scripts/train_yolov8_obb.py --validate
```

The script does the following:

1. Resolves the dataset YAML path.
2. Loads the pretrained model `yolov8n-obb.pt`.
3. Calls `model.train(...)` with OBB task mode.
4. Saves training artifacts under the configured project/name directory.
5. Optionally validates the best checkpoint after training when `--validate` is passed.

Recorded YOLOv8 run configuration from `args.yaml`:

| Parameter | Value |
|---|---|
| Model | `yolov8n-obb.pt` |
| Task | `obb` |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 8 |
| Device | CPU |
| Workers | 4 |
| Optimizer | auto |
| Pretrained | true |
| Patience | 25 |
| Seed | 0 |
| Deterministic | true |
| AMP | true |
| Cosine LR | false |
| Close mosaic | 10 |
| Output dir | `runs/obb/runs/obb/rust_corrosion_yolov8_obb` |

Main YOLOv8 training outputs:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `BoxF1_curve.png`
- train batch visualizations
- validation label/prediction visualizations

## 7. YOLO11 OBB Training Pipeline

YOLO11 training is implemented in `scripts/train_yolo11_obb_gpu.py`.

Default command:

```powershell
python scripts/train_yolo11_obb_gpu.py --validate
```

The script does the following:

1. Resolves the dataset YAML path.
2. Checks whether CUDA is available when GPU training is requested.
3. Loads the pretrained `yolo11s-obb.pt` model.
4. Calls `model.train(...)` with OBB task mode and a stronger training configuration.
5. Saves outputs under `runs/obb_yolo11/rust_corrosion_yolo11s_obb`.
6. Optionally validates the best checkpoint.
7. Optionally exports the model to `onnx`, `engine`, or `openvino`.

Recorded YOLO11 run configuration from `args.yaml`:

| Parameter | Value |
|---|---|
| Model | `yolo11s-obb.pt` |
| Task | `obb` |
| Epochs | 100 |
| Image size | 768 |
| Batch size | 8 |
| Device | CPU |
| Workers | 8 |
| Optimizer | auto |
| Pretrained | true |
| Patience | 35 |
| Seed | 0 |
| Deterministic | true |
| AMP | true |
| Cosine LR | true |
| Close mosaic | 10 |
| Output dir | `runs/obb_yolo11/rust_corrosion_yolo11s_obb` |

Main YOLO11 training outputs:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `BoxF1_curve.png`
- train batch visualizations
- validation label/prediction visualizations

## 8. Augmentation Pipeline

No preprocessing or augmentation was applied during Roboflow export. Augmentation was applied inside Ultralytics during training.

The recorded augmentation settings were the same for both YOLOv8 and YOLO11:

| Augmentation | Value | Purpose |
|---|---:|---|
| `hsv_h` | 0.015 | Slight hue variation for lighting/color robustness |
| `hsv_s` | 0.7 | Saturation variation; useful because rust color intensity changes widely |
| `hsv_v` | 0.4 | Brightness/value variation for different exposure conditions |
| `translate` | 0.1 | Position shifts to make detection less location-dependent |
| `scale` | 0.5 | Scale variation to learn small and large corrosion regions |
| `fliplr` | 0.5 | Horizontal flips for left/right invariance |
| `mosaic` | 1.0 | Combines images to expose more object/context combinations |
| `close_mosaic` | 10 | Disables mosaic in the final 10 epochs for more natural final fine-tuning |
| `auto_augment` | `randaugment` | Applies randomized image transformations |
| `erasing` | 0.4 | Random erasing to improve robustness to occlusion/noise |
| `mixup` | 0.0 | Disabled |
| `cutmix` | 0.0 | Disabled |
| `copy_paste` | 0.0 | Disabled |
| `degrees` | 0.0 | No explicit random rotation beyond OBB learning |
| `shear` | 0.0 | Disabled |
| `perspective` | 0.0 | Disabled |
| `flipud` | 0.0 | Vertical flips disabled |

These augmentations are suitable for rust detection because corrosion appearance changes with lighting, distance, surface type, camera angle, and object size. However, aggressive color augmentation can also make mild, moderate, and severe categories harder to separate if severity is strongly tied to color intensity. This is one possible contributor to validation fluctuations.

## 9. Results

Best validation results by `metrics/mAP50-95(B)`:

| Model | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-OBB | 99 | 0.71359 | 0.59229 | 0.67310 | 0.47089 |
| YOLO11s-OBB | 90 | 0.79555 | 0.69486 | 0.77063 | 0.57868 |

Final epoch results:

| Model | Final epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-OBB | 100 | 0.70215 | 0.59133 | 0.66928 | 0.46807 |
| YOLO11s-OBB | 100 | 0.79881 | 0.69363 | 0.76648 | 0.57447 |

Best validation losses:

| Model | Epoch | Val box loss | Val cls loss | Val DFL loss | Val angle loss |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-OBB | 99 | 1.07373 | 1.14517 | 1.41263 | 0.03802 |
| YOLO11s-OBB | 90 | 0.90567 | 0.89077 | 1.36041 | 0.03075 |

YOLO11 achieved higher precision, recall, mAP50, and mAP50-95. It also achieved lower validation box, classification, DFL, and angle losses at its best epoch. This indicates that YOLO11 was not only better at deciding whether rust was present, but also better at localizing the oriented boxes and assigning the correct severity class.

## 10. Metric Interpretation

Precision measures how many predicted rust detections were correct. Higher precision means fewer false positives.

Recall measures how many ground-truth rust regions were found. Higher recall means fewer missed corrosion regions.

mAP50 measures average precision at IoU 0.50. This is a more lenient localization metric.

mAP50-95 averages performance across IoU thresholds from 0.50 to 0.95. This is stricter and is more sensitive to box quality. For OBB tasks, mAP50-95 is especially important because rotated-box angle and corner placement affect IoU heavily.

In this project:

- YOLOv8 best mAP50-95: 0.47089
- YOLO11 best mAP50-95: 0.57868
- Absolute improvement: 0.10779
- Relative improvement: approximately 22.9 percent over YOLOv8

YOLO11 also improved mAP50 from 0.67310 to 0.77063, which suggests better general detection quality, not only stricter high-IoU localization.

## 11. Why YOLO11 Performed Better

Several factors likely contributed to the improvement.

First, YOLO11s is a larger and newer model than YOLOv8n. The YOLOv8 run used the nano OBB checkpoint, while the YOLO11 run used the small OBB checkpoint. The YOLO11 model therefore had more representational capacity to learn subtle visual differences between mild, moderate, and severe corrosion.

Second, YOLO11 was trained at a larger image size: 768 instead of 640. Rust regions can contain fine texture, irregular boundaries, and small color/edge differences. Increasing input resolution preserves more detail, which can improve both classification and rotated-box localization.

Third, YOLO11 used cosine learning-rate scheduling (`cos_lr: true`) while YOLOv8 did not. Cosine scheduling usually gives smoother late-stage optimization. This can help the model settle into a better validation optimum, especially for detection tasks where localization and classification losses must be balanced.

Fourth, YOLO11 used more workers in the recorded config: 8 versus 4. This mainly affects data loading throughput, not final accuracy directly, but smoother input loading can make longer training more practical.

Fifth, YOLO11 had a longer patience setting: 35 versus 25. Both completed 100 epochs in the recorded runs, but the larger patience setting gives more room for validation metrics to improve after temporary plateaus.

Finally, the lower YOLO11 validation losses show that the improvement was broad. It was not only a confidence-threshold effect. The model produced better boxes, better class predictions, and better angle estimates.

## 12. Why mAP or Precision Can Change Between Runs

Even when the dataset is unchanged, detection metrics can change for several reasons.

Model architecture and capacity are the largest factors here. YOLOv8n-OBB is a lightweight model, while YOLO11s-OBB is a stronger model. A small model may underfit complex rust patterns, especially when severity categories are visually close.

Input resolution also matters. At 640 pixels, small or thin corrosion regions may lose detail after resizing. At 768 pixels, the model receives more spatial information, which helps OBB corner placement and severity classification.

Augmentation can affect precision and recall differently. Mosaic, HSV shifts, random scaling, and erasing increase robustness, but they can also create training samples that are visually different from validation images. If augmentations are too strong for severity classification, the model may learn generalized rust presence well but struggle with exact severity boundaries.

Class ambiguity is another important reason. Rust severity labels are inherently subjective. Mild and moderate corrosion may look similar, and severe corrosion may overlap visually with heavy shadows, dirt, or paint damage. This can lower precision if the model predicts corrosion in visually similar non-target regions, and lower recall if the model avoids uncertain detections.

OBB annotation quality strongly affects mAP50-95. Rotated IoU is sensitive to corner placement and angle. A prediction can look visually acceptable but receive lower high-IoU scores if the predicted box is slightly shifted, too wide, too narrow, or rotated incorrectly.

Dataset imbalance can influence class-level behavior. The training split contains fewer moderate examples than mild or severe examples, so moderate corrosion may be less stable. If many errors happen between neighboring severity classes, overall mAP drops even when rust localization is good.

CPU training does not inherently reduce accuracy, but it makes experimentation slower. Longer wall-clock time can limit how many hyperparameter trials are run, which may prevent finding the best configuration.

## 13. End-to-End Training Process Used

The process used in this project can be summarized as:

1. Export the rust dataset from Roboflow in YOLOv8 OBB format.
2. Place the dataset under the `data/` directory.
3. Verify the three classes in `data/data.yaml`.
4. Prepare or confirm train/validation/test splits.
5. Train YOLOv8 OBB using `scripts/train_yolov8_obb.py`.
6. Inspect YOLOv8 results in `runs/obb/runs/obb/rust_corrosion_yolov8_obb/`.
7. Train YOLO11 OBB using `scripts/train_yolo11_obb_gpu.py`.
8. Inspect YOLO11 results in `runs/obb_yolo11/rust_corrosion_yolo11s_obb/`.
9. Compare `results.csv`, `results.png`, PR curves, F1 curves, and confusion matrices.
10. Select `weights/best.pt` from the better-performing run for inference/deployment.

## 14. Recommended Next Steps

The best current model is YOLO11s-OBB based on validation metrics.

Recommended improvements:

- Run final evaluation on the held-out test split using the YOLO11 best checkpoint.
- Compare per-class precision/recall to identify whether mild, moderate, or severe corrosion is weakest.
- Review false positives and false negatives from `val_batch*_pred.jpg`.
- Check whether moderate corrosion needs additional images or relabeling consistency work.
- Try a GPU run with the intended 150-epoch YOLO11 default from the script.
- Try `yolo11m-obb.pt` if GPU memory allows.
- Experiment with less aggressive HSV augmentation if severity classification appears color-sensitive.
- Consider class-balanced sampling or targeted data collection for underperforming classes.

## 15. Conclusion

Both models successfully learned the rust category OBB detection task, but YOLO11s-OBB produced the stronger result. Its best validation mAP50-95 was 0.57868 compared with 0.47089 for YOLOv8n-OBB, and it improved precision from 0.71359 to 0.79555 and recall from 0.59229 to 0.69486.

The improvement is most likely due to the combination of a newer/larger YOLO11s architecture, higher input resolution, cosine learning-rate scheduling, and better localization/classification capacity. For deployment, the YOLO11 `best.pt` checkpoint is the preferred model from the completed experiments.
