# Cloud Training

VisiMove training is cloud-first. Do not use the local Windows desktop for large model training, large dataset downloads, or checkpoint generation.

Use local PC for:

- app development
- webcam runtime testing
- small smoke tests
- ONNX inference validation

Use Colab or Kaggle for:

- blink model training
- optional gaze fine-tuning
- optional YOLO detector training
- checkpoint export

## Storage Plan

- Google Drive: raw datasets, private working copies, checkpoints, training logs.
- Kaggle Datasets: reproducible dataset inputs when licenses allow.
- Hugging Face: final model weights and ONNX exports when licenses allow.
- GitHub: code only.

Do not commit datasets or model weights to GitHub.

## Dataset Plan

Blink:

1. RT-BENE first.
2. MRL Eye for quick testing only.

Gaze:

1. MPIIFaceGaze first.
2. PARKS-Gaze or EVE later only if needed and license-compatible.

YOLO:

- Optional only.
- Use only if MediaPipe/OpenCV face and eye detection are unstable.
- Labels: `face`, `left_eye`, `right_eye`.

## Licensing Warning

Do not upload restricted raw datasets or restricted derived weights publicly unless the dataset license allows it. Check license, citation, redistribution, and commercial-use terms before placing anything on Hugging Face, Kaggle, GitHub, or public Google Drive links.

## Colab Drive Mount

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/visimove'
DATASET_ROOT = f'{PROJECT_ROOT}/datasets'
CHECKPOINT_ROOT = f'{PROJECT_ROOT}/checkpoints'
EXPORT_ROOT = f'{PROJECT_ROOT}/exports'
```

## Kaggle Dataset Path

```python
DATASET_ROOT = '/kaggle/input/visimove-blink-dataset'
OUTPUT_DIR = '/kaggle/working/visimove-checkpoints'
```

## Blink Training

Prepare data in an ImageFolder layout:

```text
rt-bene-eyes/
├── train/
│   ├── open/
│   └── closed/
└── val/
    ├── open/
    └── closed/
```

Run in Colab:

```powershell
python scripts/train_blink_model_colab.py `
  --dataset-root /content/drive/MyDrive/visimove/datasets/rt-bene-eyes `
  --output-dir /content/drive/MyDrive/visimove/checkpoints/blink `
  --epochs 10 `
  --run
```

The script trains a small blink CNN for open/closed eye classification. It is intentionally simple so it can be replaced later by a stronger architecture.

## Export Blink Model To ONNX

Run in Colab:

```powershell
python scripts/export_onnx.py `
  --checkpoint /content/drive/MyDrive/visimove/checkpoints/blink/blink_cnn_best.pt `
  --output /content/drive/MyDrive/visimove/exports/blink_cnn.onnx `
  --run
```

Copy the approved ONNX file to the local app:

```text
models/blink/blink_cnn.onnx
```

Then select it in VisiMove config:

```yaml
blink:
  backend: onnx
  model_path: models/blink/blink_cnn.onnx
  input_width: 64
  input_height: 64
  normalize_mean: 0.5
  normalize_std: 0.5
```

The ONNX file is ignored by Git.

## Optional Gaze Fine-Tuning

Start with MPIIFaceGaze. Use external backends as isolated research code:

```text
external/gazefollower/
external/mobilegaze/
```

Expected outputs:

```text
models/gaze/gazefollower/
models/gaze/mobilegaze/*.onnx
models/gaze/mobilegaze/*.pt
```

Config examples:

```yaml
gaze:
  gaze_backend: gazefollower
  model_path: models/gaze/gazefollower/model.pth
```

```yaml
gaze:
  gaze_backend: mobilegaze
  model_path: models/gaze/mobilegaze/model.onnx
```

Do not fine-tune gaze locally. Keep raw MPIIFaceGaze/PARKS-Gaze/EVE data in Google Drive or Kaggle only.

## Optional YOLO Training

YOLO is not the first detector. Use it only if MediaPipe/OpenCV detection is unstable.

YOLO must only detect:

- `0 face`
- `1 left_eye`
- `2 right_eye`

Do not use YOLO as the gaze-estimation model.

## Roboflow Annotation

Roboflow can be used to annotate and export a YOLO dataset when the source dataset license allows it.

Recommended project setup:

- project type: Object Detection
- classes: `face`, `left_eye`, `right_eye`
- export format: YOLOv8/YOLO11 compatible
- split: train/val/test
- keep raw restricted datasets private

After export, store the dataset in Google Drive or Kaggle:

```text
/content/drive/MyDrive/visimove/datasets/yolo-face-eye/
```

Prepare YOLO data:

```text
yolo-face-eye/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml`:

```yaml
path: /content/drive/MyDrive/visimove/datasets/yolo-face-eye
train: images/train
val: images/val
names:
  0: face
  1: left_eye
  2: right_eye
```

Run in Colab:

```powershell
python scripts/train_yolo_detector_colab.py `
  --data-yaml /content/drive/MyDrive/visimove/datasets/yolo-face-eye/data.yaml `
  --output-dir /content/drive/MyDrive/visimove/checkpoints/yolo `
  --model yolov8n.pt `
  --run
```

Kaggle path example:

```powershell
python scripts/train_yolo_detector_colab.py `
  --data-yaml /kaggle/input/visimove-yolo-face-eye/data.yaml `
  --output-dir /kaggle/working/visimove-yolo-runs `
  --model yolo11n.pt `
  --run
```

Use `yolo11n.pt` first. Try `yolo11s.pt` only if accuracy is not enough and latency remains acceptable.

## Export YOLO ONNX

Export ONNX only after metrics are acceptable. Keep YOLO optional and isolated.

In Colab after training:

```python
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/visimove/checkpoints/yolo/visimove_yolo_face_eye/weights/best.pt')
model.export(format='onnx', imgsz=640)
```

Store the ONNX export in Google Drive:

```text
/content/drive/MyDrive/visimove/exports/yolo_face_eye.onnx
```

For final local runtime testing, copy it to:

```text
models/detection/yolo_face_eye.onnx
```

The file is ignored by Git.

VisiMove config:

```yaml
detection:
  detector_backend: yolo
  yolo_model_path: models/detection/yolo_face_eye.onnx
  yolo_confidence_threshold: 0.4
  yolo_image_size: 640
  yolo_device: cpu
  yolo_detect_every_n_frames: 3
```

If YOLO dependencies or weights are missing, VisiMove falls back to MediaPipe/OpenCV.

## YOLO Storage

Google Drive:

```text
visimove/datasets/yolo-face-eye/
visimove/checkpoints/yolo/
visimove/exports/yolo_face_eye.onnx
```

Hugging Face, only when licenses allow:

- upload final `.onnx`
- include class map `0 face`, `1 left_eye`, `2 right_eye`
- include training dataset/license notes
- include validation metrics
- do not upload restricted raw datasets

## Hugging Face Release Plan

When licenses allow:

1. Upload final `.onnx` files.
2. Include model card with dataset/license notes.
3. Include preprocessing details.
4. Include evaluation metrics and intended accessibility use.
5. Do not upload raw restricted datasets.

## Local Safety

Local `.gitignore` excludes:

- `models/**/*.pt`
- `models/**/*.pth`
- `models/**/*.onnx`
- `models/**/*.ckpt`
- `models/**/*.safetensors`
- images/videos/parquet/csv under `data/`
- external weights, checkpoints, runs, logs, and datasets

Use GitHub for code only.
