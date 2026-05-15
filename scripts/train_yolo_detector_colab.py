from __future__ import annotations

"""
Colab/Kaggle template for optional YOLO detector training.

YOLO is optional for VisiMove and should only be trained if MediaPipe/OpenCV
face/eye detection is unstable. Do not train locally on the desktop.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud template for optional VisiMove YOLO training.")
    parser.add_argument("--data-yaml", default="/content/drive/MyDrive/visimove/datasets/yolo/data.yaml")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/visimove/checkpoints/yolo")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--run", action="store_true", help="Actually start training in a cloud runtime.")
    args = parser.parse_args()

    if not args.run:
        print_guidance(args)
        return

    train(args)


def print_guidance(args: argparse.Namespace) -> None:
    print("YOLO detector training is optional and cloud-only.")
    print("Use labels: face, left_eye, right_eye.")
    print("Google Drive mount example:")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")
    print("\nRun in Colab/Kaggle:")
    print(
        "python scripts/train_yolo_detector_colab.py "
        f"--data-yaml {args.data_yaml} --output-dir {args.output_dir} --run"
    )


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics in the cloud runtime before training.") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.image_size,
        project=args.output_dir,
        name="visimove_yolo_face_eye",
        exist_ok=True,
    )
    print("Export ONNX in the cloud after validating metrics:")
    print("model.export(format='onnx')")


if __name__ == "__main__":
    main()
