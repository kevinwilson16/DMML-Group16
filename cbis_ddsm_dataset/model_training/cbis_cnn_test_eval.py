import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU, with debug info."""
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("num devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device 0:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_dataloaders(data_dir: str, batch_size: int = 16):
    """
    Build a DataLoader for the test dataset located at:
        data_dir / "test" / <class_name> / *.png
    """
    test_dir = os.path.join(data_dir, "test")

    test_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_dataset, test_loader


# ---------------------------------------------------------------------------
# Model builders & model selection
# ---------------------------------------------------------------------------


def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    """ResNet-18 head used in earlier experiments."""
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def build_densenet121(num_classes: int, device: torch.device) -> nn.Module:
    """DenseNet-121 head used in the latest training script."""
    model = models.densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


AVAILABLE_MODELS: Dict[str, Dict[str, object]] = {
    "resnet": {
        "name": "ResNet-18 CNN",
        "builder": build_resnet18,
        "weights_file": "cbis_cnn_resnet18_best.pth",
    },
    "densenet": {
        "name": "DenseNet-121 CNN",
        "builder": build_densenet121,
        "weights_file": "cbis_cnn_densenet121_best.pth",
    },
}


def select_model(models_dir: Path) -> Tuple[str, str, Path, Callable[[int, torch.device], nn.Module]]:
    """
    Let the user choose which CNN model to evaluate.
    Returns (key, human_name, weights_path, builder_fn).
    """
    print("\n" + "=" * 70)
    print("AVAILABLE CNN MODELS")
    print("=" * 70)
    for key, cfg in AVAILABLE_MODELS.items():
        path = models_dir / cfg["weights_file"]
        status = "✓ Found" if path.exists() else "✗ Missing"
        print(f"  [{key}] {cfg['name']:<25} -> {path.name} ({status})")
    print("=" * 70 + "\n")

    valid_keys = "/".join(AVAILABLE_MODELS.keys())
    while True:
        choice = input(f"Select model to evaluate ({valid_keys}): ").strip().lower()
        if choice in AVAILABLE_MODELS:
            cfg = AVAILABLE_MODELS[choice]
            weights_path = models_dir / cfg["weights_file"]
            if not weights_path.exists():
                print(f"\n❌ Weights file not found: {weights_path}")
                print("   Train this model first or choose another key.\n")
                continue
            builder = cfg["builder"]
            return choice, cfg["name"], weights_path, builder  # type: ignore[return-value]
        else:
            print(f"Invalid choice. Please enter one of: {valid_keys}.")


def evaluate_overall(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on the full test set and print overall metrics.
    Returns (accuracy, precision, recall, f1).
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    print("\n=== Test Performance (Full Test Set) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return acc, precision, recall, f1


def predict_indices(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    indices: List[int],
    device: torch.device,
    class_names: List[str],
):
    """Predict on specific dataset indices and print per-sample results (with file names)."""
    model.eval()
    n_samples = len(dataset)

    print("\n" + "=" * 70)
    print(f"CNN PREDICTIONS FOR {len(indices)} TEST IMAGES")
    print("=" * 70)
    print(
        f"\n{'Idx':<6} {'ClassIdx':<9} {'TrueLabel':<12} {'PredLabel':<12} "
        f"{'ProbTrue':<10} {'ProbPred':<10} {'Filename'}"
    )
    print("-" * 70)

    with torch.no_grad():
        for idx in indices:
            if idx < 0 or idx >= n_samples:
                print(f"{idx:<6} INVALID INDEX (0-{n_samples-1})")
                continue

            # Get image and metadata
            img, label = dataset[idx]
            # ImageFolder stores (path, class_idx) in .samples
            img_path, _ = dataset.samples[idx]
            filename = os.path.basename(img_path)

            img = img.unsqueeze(0).to(device)
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())

            true_label_name = class_names[label]
            pred_label_name = class_names[pred]

            prob_true = probs[label]
            prob_pred = probs[pred]

            status = "✓" if pred == label else "✗"

            print(
                f"{idx:<6} {label:<9} {true_label_name:<12} {pred_label_name:<12} "
                f"{prob_true*100:>7.2f}%   {prob_pred*100:>7.2f}% {status}  {filename}"
            )

    print("=" * 70 + "\n")


def interactive_mode(
    model: nn.Module,
    test_dataset: datasets.ImageFolder,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
):
    """Interactive CLI similar to model_testing_unified for cross-checking."""
    n_samples = len(test_dataset)

    while True:
        print("\n" + "=" * 70)
        print("CBIS-DDSM CNN TESTING - INTERACTIVE MODE")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Evaluate full test dataset (overall metrics)")
        print("  2. Test a specific image by index")
        print("  3. Test multiple indices (comma-separated)")
        print("  4. Test random images")
        print("  5. Exit")
        print("\n" + "-" * 70)

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            evaluate_overall(model, test_loader, device, class_names)

        elif choice == "2":
            try:
                idx = int(input(f"\nEnter image index (0-{n_samples-1}): "))
                predict_indices(model, test_dataset, [idx], device, class_names)
            except ValueError:
                print("Please enter a valid integer index.")

        elif choice == "3":
            rows_input = input(
                f"\nEnter indices separated by commas (e.g., 0,5,10; 0-{n_samples-1}): "
            )
            try:
                indices = [int(x.strip()) for x in rows_input.split(",") if x.strip()]
                predict_indices(model, test_dataset, indices, device, class_names)
            except ValueError:
                print("Please enter valid integer indices separated by commas.")

        elif choice == "4":
            try:
                n = int(
                    input(
                        f"\nHow many random test images to check? (1-{n_samples}): "
                    )
                )
                n = max(1, min(n, n_samples))
                rand_indices = np.random.choice(n_samples, n, replace=False).tolist()
                predict_indices(model, test_dataset, rand_indices, device, class_names)
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "5":
            print("\nExiting CNN interactive testing.\n")
            break

        else:
            print("Invalid choice! Please enter 1-5.")


def main():
    # Match the path used in the notebook
    data_dir = r"C:\Users\gaura\Desktop\CBIS-DDSM_processed\CBIS-DDSM_processed"

    # Models directory (same as notebook / training script)
    cbis_root = Path(__file__).resolve().parent.parent
    models_dir = cbis_root / "models"

    device = get_device()
    print(f"Loading data from: {data_dir}")

    # Let the user pick which CNN to evaluate
    model_key, model_name, model_path, builder = select_model(models_dir)
    print(f"Selected model: {model_name}")
    print(f"Loading model from: {model_path}")

    test_dataset, test_loader = build_dataloaders(str(data_dir), batch_size=16)
    class_names = test_dataset.classes
    print("Test classes:", class_names)
    print("Test samples:", len(test_dataset))

    # Build selected architecture
    model = builder(len(class_names), device)

    # We saved only state_dict, so weights_only=True is safe
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Start interactive mode (includes full-test evaluation option)
    interactive_mode(model, test_dataset, test_loader, device, class_names)


if __name__ == "__main__":
    main()


