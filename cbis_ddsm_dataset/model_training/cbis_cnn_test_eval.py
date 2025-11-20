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
    roc_curve,
    auc,
)

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


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


def build_resnet50(num_classes: int, device: torch.device) -> nn.Module:
    """ResNet-50 head for improved accuracy."""
    model = models.resnet50(pretrained=False)
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
    "resnet18": {
        "name": "ResNet-18 CNN",
        "builder": build_resnet18,
        "weights_file": "cbis_cnn_resnet18_best.pth",
    },
    "resnet50": {
        "name": "ResNet-50 CNN",
        "builder": build_resnet50,
        "weights_file": "cbis_cnn_resnet50_best.pth",
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
    output_dir: Path = None,
    model_name: str = "Model",
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on the full test set and print overall metrics.
    Returns (accuracy, precision, recall, f1).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    print("\n=== Test Performance (Full Test Set) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Generate visualizations if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "="*70)
        print("GENERATING EVALUATION VISUALIZATIONS")
        print("="*70)
        
        # 1. Confusion Matrix
        plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png", model_name)
        
        # 2. ROC Curve
        plot_roc_curve(all_labels, all_probs[:, 1], output_dir / "roc_curve.png", model_name)
        
        # 3. Metrics Summary
        plot_metrics_summary(acc, precision, recall, f1, output_dir / "metrics_summary.png", model_name)
        
        # 4. Classification Report Visualization
        plot_classification_report(all_labels, all_preds, class_names, 
                                   output_dir / "classification_report.png", model_name)
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
        print("="*70)

    return acc, precision, recall, f1


def predict_indices(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    indices: List[int],
    device: torch.device,
    class_names: List[str],
    output_dir: Path = None,
    model_name: str = "Model",
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

    # Store results for visualization
    results = []

    with torch.no_grad():
        for idx in indices:
            if idx < 0 or idx >= n_samples:
                print(f"{idx:<6} INVALID INDEX (0-{n_samples-1})")
                continue

            # Get image and metadata
            img_tensor, label = dataset[idx]
            # ImageFolder stores (path, class_idx) in .samples
            img_path, _ = dataset.samples[idx]
            filename = os.path.basename(img_path)

            # Load original image for visualization
            from PIL import Image
            img_pil = Image.open(img_path).convert('RGB')

            img_batch = img_tensor.unsqueeze(0).to(device)
            outputs = model(img_batch)
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

            results.append({
                'image': img_pil,
                'true_label': label,
                'pred_label': pred,
                'true_name': true_label_name,
                'pred_name': pred_label_name,
                'confidence': prob_pred,
                'correct': pred == label
            })

    print("=" * 70 + "\n")

    # Generate visualization if output directory provided
    if output_dir and results:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_sample_predictions(results, class_names, 
                               output_dir / "sample_predictions.png", model_name)
        print(f"✓ Sample predictions visualization saved to: {output_dir / 'sample_predictions.png'}")


def interactive_mode(
    model: nn.Module,
    test_dataset: datasets.ImageFolder,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path,
    model_name: str,
):
    """Interactive CLI similar to model_testing_unified for cross-checking."""
    n_samples = len(test_dataset)

    while True:
        print("\n" + "=" * 70)
        print("CBIS-DDSM CNN TESTING - INTERACTIVE MODE")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Evaluate full test dataset (overall metrics + visualizations)")
        print("  2. Test a specific image by index")
        print("  3. Test multiple indices (comma-separated)")
        print("  4. Test random images (with visualizations)")
        print("  5. Exit")
        print("\n" + "-" * 70)

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            evaluate_overall(model, test_loader, device, class_names, output_dir, model_name)

        elif choice == "2":
            try:
                idx = int(input(f"\nEnter image index (0-{n_samples-1}): "))
                predict_indices(model, test_dataset, [idx], device, class_names, output_dir, model_name)
            except ValueError:
                print("Please enter a valid integer index.")

        elif choice == "3":
            rows_input = input(
                f"\nEnter indices separated by commas (e.g., 0,5,10; 0-{n_samples-1}): "
            )
            try:
                indices = [int(x.strip()) for x in rows_input.split(",") if x.strip()]
                predict_indices(model, test_dataset, indices, device, class_names, output_dir, model_name)
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
                predict_indices(model, test_dataset, rand_indices, device, class_names, output_dir, model_name)
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "5":
            print("\nExiting CNN interactive testing.\n")
            break

        else:
            print("Invalid choice! Please enter 1-5.")


def plot_confusion_matrix(cm, class_names, save_path, model_name):
    """Plot enhanced confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12, rotation=0)
    
    # Add statistics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    stats_text = f"Accuracy: {accuracy:.3f}\n"
    stats_text += f"Sensitivity: {sensitivity:.3f}\n"
    stats_text += f"Specificity: {specificity:.3f}"
    
    ax.text(1.5, 0.5, stats_text, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            transform=ax.transData)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path.name}")
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path, model_name):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: {save_path.name}")
    plt.close()


def plot_metrics_summary(acc, precision, recall, f1, save_path, model_name):
    """Plot metrics summary bar chart"""
    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), 
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
                   edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance Metrics - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics summary saved: {save_path.name}")
    plt.close()


def plot_classification_report(y_true, y_pred, class_names, save_path, model_name):
    """Plot classification report as heatmap"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    
    # Create data matrix
    data = np.array([precision, recall, f1]).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=class_names,
                cbar_kws={'label': 'Score'},
                linewidths=2, linecolor='black',
                ax=ax)
    
    ax.set_title(f'Classification Report - {model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add support counts
    for i, sup in enumerate(support):
        ax.text(3.2, i + 0.5, f'n={sup}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Classification report saved: {save_path.name}")
    plt.close()


def plot_sample_predictions(results, class_names, save_path, model_name):
    """Plot sample predictions with images"""
    n_samples = min(len(results), 16)
    rows = 4
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    fig.suptitle(f'Sample Predictions - {model_name}', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            result = results[idx]
            img = result['image']
            
            # Display image
            ax.imshow(img)
            
            # Color: green if correct, red if wrong
            color = 'green' if result['correct'] else 'red'
            
            # Title
            title = f"True: {result['true_name']}\n"
            title += f"Pred: {result['pred_name']}\n"
            title += f"Conf: {result['confidence']:.3f}"
            
            ax.set_title(title, fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round', facecolor='white', 
                                 edgecolor=color, linewidth=2))
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Sample predictions saved: {save_path.name}")
    plt.close()


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

    # Create output directory for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"cnn_eval_results_{model_key}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

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
    interactive_mode(model, test_dataset, test_loader, device, class_names, 
                    output_dir, model_name)


if __name__ == "__main__":
    main()


