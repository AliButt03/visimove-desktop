from __future__ import annotations

"""
Small ONNX export helper.

Run this in Colab/Kaggle for PyTorch checkpoints. Local use is only for small
smoke tests and should not involve large checkpoints or datasets.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a small VisiMove PyTorch model to ONNX.")
    parser.add_argument("--checkpoint", required=False, help="Path to a PyTorch checkpoint.")
    parser.add_argument("--output", default="models/blink/blink_cnn.onnx")
    parser.add_argument("--model-type", choices=["blink_cnn"], default="blink_cnn")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--run", action="store_true", help="Actually export. Otherwise print guidance.")
    args = parser.parse_args()

    if not args.run:
        print_guidance(args)
        return

    export_blink_cnn(args)


def print_guidance(args: argparse.Namespace) -> None:
    print("Use this in Colab/Kaggle after training. Do not commit exported ONNX files.")
    print("Example:")
    print(
        "python scripts/export_onnx.py "
        "--checkpoint /content/drive/MyDrive/visimove/checkpoints/blink/blink_cnn_best.pt "
        "--output /content/drive/MyDrive/visimove/exports/blink_cnn.onnx --run"
    )
    print("\nCopy final approved ONNX into local:")
    print("models/blink/blink_cnn.onnx")
    print("\nVisiMove config:")
    print("blink:")
    print("  backend: onnx")
    print("  model_path: models/blink/blink_cnn.onnx")


def export_blink_cnn(args: argparse.Namespace) -> None:
    try:
        import torch
        from train_blink_model_colab import create_small_blink_cnn
    except ImportError as exc:
        raise SystemExit("Install torch in the cloud runtime before export.") from exc

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required with --run.")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", args.image_size))
    model = create_small_blink_cnn()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dummy_input = torch.randn(1, 1, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["eye"],
        output_names=["closed_logits"],
        dynamic_axes={"eye": {0: "batch"}, "closed_logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
