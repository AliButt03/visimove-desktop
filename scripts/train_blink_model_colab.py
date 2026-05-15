from __future__ import annotations

"""
Colab-first blink training template.

Run this on Google Colab or Kaggle, not on the local desktop. It trains a small
CNN for open/closed eye classification when PyTorch and a prepared dataset are
available. Local execution without --run prints setup guidance only.
"""

import argparse
from pathlib import Path


COLAB_DRIVE_EXAMPLE = r"""
from google.colab import drive
drive.mount('/content/drive')

DATASET_ROOT = '/content/drive/MyDrive/visimove/datasets/rt-bene-eyes'
OUTPUT_DIR = '/content/drive/MyDrive/visimove/checkpoints/blink'
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud template for VisiMove blink model training.")
    parser.add_argument("--dataset-root", default="/content/drive/MyDrive/visimove/datasets/rt-bene-eyes")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/visimove/checkpoints/blink")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--run", action="store_true", help="Actually start training in a cloud runtime.")
    args = parser.parse_args()

    if not args.run:
        print_guidance(args)
        return

    train(args)


def print_guidance(args: argparse.Namespace) -> None:
    print("VisiMove blink training is cloud-first. Do not download large datasets locally.")
    print("\nColab Drive mount example:")
    print(COLAB_DRIVE_EXAMPLE.strip())
    print("\nRecommended dataset order:")
    print("1. RT-BENE first")
    print("2. MRL Eye for quick testing only")
    print("\nRun in Colab/Kaggle:")
    print(
        "python scripts/train_blink_model_colab.py "
        f"--dataset-root {args.dataset_root} --output-dir {args.output_dir} --run"
    )


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("Install torch and torchvision in the cloud runtime before training.") from exc

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_data = datasets.ImageFolder(dataset_root / "train", transform=transform)
    val_data = datasets.ImageFolder(dataset_root / "val", transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_small_blink_cnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        accuracy = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_data.classes,
                    "image_size": args.image_size,
                    "val_accuracy": accuracy,
                },
                output_dir / "blink_cnn_best.pt",
            )


def create_small_blink_cnn():
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2),
    )


def evaluate(model, loader, device: str) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / max(1, total)


if __name__ == "__main__":
    main()
