
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ===========================
# Configuration
# ===========================

DATA_DIR = r"C:\Users\gaura\Desktop\CBIS-DDSM_processed\CBIS-DDSM_processed"
TEST_DIR = os.path.join(DATA_DIR, "test")

CBIS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = CBIS_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cbis_mlp_best.pth"

OUTPUT_DIR = Path(__file__).resolve().parent / "mlp_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 128
INPUT_DIM = 1 * IMG_SIZE * IMG_SIZE

print("="*80)
print("CBIS-DDSM MLP MODEL EVALUATION")
print("="*80)
print(f"Model Path: {MODEL_PATH}")
print(f"Test Data: {TEST_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print("="*80)


# ===========================
# Model Definition
# ===========================

class MLPImageClassifier(nn.Module):
    """MLP Image Classifier - same architecture as training"""
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] → flatten
        x = x.view(x.size(0), -1)
        return self.net(x)


# ===========================
# Data Loading
# ===========================

def load_test_data():
    """Load test dataset"""
    print("\n[1/6] Loading test data...")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {test_dataset.classes}")
    print(f"✓ Class distribution:")
    
    # Count samples per class
    class_counts = {}
    for _, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        print(f"    {class_name}: {count}")
    
    return test_dataset, test_loader


# ===========================
# Model Loading
# ===========================

def load_model(device):
    """Load trained MLP model"""
    print("\n[2/6] Loading model...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    model = MLPImageClassifier(input_dim=INPUT_DIM, num_classes=2)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully from: {MODEL_PATH}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    return model


# ===========================
# Prediction & Metrics
# ===========================

def evaluate_model(model, test_loader, device, class_names):
    """Run evaluation and collect predictions"""
    print("\n[3/6] Running model evaluation...")
    
    all_labels = []
    all_preds = []
    all_probs = []  # For ROC curve
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples...", end='\r')
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"\n✓ Evaluation complete on {len(all_labels)} samples")
    
    return all_labels, all_preds, all_probs


# ===========================
# Metrics Calculation
# ===========================

def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate comprehensive metrics"""
    print("\n[4/6] Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC
    try:
        if len(np.unique(y_true)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            roc_auc = None
    except:
        roc_auc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class
    }
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC:           {roc_auc:.4f}")
    print("="*80)
    
    print("\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision_per_class[i]:<12.4f} "
              f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} "
              f"{support_per_class[i]:<10}")
    print("-"*80)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return metrics


# ===========================
# Visualization
# ===========================

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - MLP Model', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(len(class_names)/2, -0.15, f'Overall Accuracy: {accuracy:.2%}',
             ha='center', va='top', fontsize=12, fontweight='bold',
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_probs, class_names, save_path):
    """Plot ROC curve for binary classification"""
    if len(class_names) != 2:
        print("⚠ ROC curve only for binary classification")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - MLP Model', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: {save_path}")
    plt.close()


def plot_metrics_summary(metrics, class_names, save_path):
    """Plot bar chart of metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall metrics
    ax1 = axes[0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]
    
    if metrics['roc_auc']:
        metric_names.append('ROC-AUC')
        metric_values.append(metrics['roc_auc'])
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax1.bar(metric_names, metric_values, color=colors[:len(metric_names)],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.75, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (75%)')
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Per-class metrics
    ax2 = axes[1]
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax2.bar(x - width, metrics['precision_per_class'], width,
                    label='Precision', color='#2E86AB', edgecolor='black', alpha=0.8)
    bars2 = ax2.bar(x, metrics['recall_per_class'], width,
                    label='Recall', color='#A23B72', edgecolor='black', alpha=0.8)
    bars3 = ax2.bar(x + width, metrics['f1_per_class'], width,
                    label='F1-Score', color='#F18F01', edgecolor='black', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics summary saved: {save_path}")
    plt.close()


# ===========================
# Save Results
# ===========================

def save_results(metrics, class_names):
    """Save evaluation results to file"""
    print("\n[5/6] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"evaluation_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CBIS-DDSM MLP MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test Data: {TEST_DIR}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        if metrics['roc_auc']:
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write("\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {metrics['precision_per_class'][i]:<12.4f} "
                   f"{metrics['recall_per_class'][i]:<12.4f} "
                   f"{metrics['f1_per_class'][i]:<12.4f} "
                   f"{metrics['support_per_class'][i]:<10}\n")
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
    
    print(f"✓ Results saved: {results_file}")


# ===========================
# Main Function
# ===========================

def main():
    """Main evaluation pipeline"""
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load test data
        test_dataset, test_loader = load_test_data()
        class_names = test_dataset.classes
        
        # Load model
        model = load_model(device)
        
        # Evaluate
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, class_names)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)
        
        # Generate visualizations
        print("\n[6/6] Generating visualizations...")
        
        # Confusion matrix
        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
        
        # ROC curve
        if len(class_names) == 2:
            roc_path = OUTPUT_DIR / "roc_curve.png"
            plot_roc_curve(y_true, y_probs, class_names, roc_path)
        
        # Metrics summary
        summary_path = OUTPUT_DIR / "metrics_summary.png"
        plot_metrics_summary(metrics, class_names, summary_path)
        
        # Save results
        save_results(metrics, class_names)
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE!")
        print("="*80)
        print(f"Results saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - confusion_matrix.png")
        print(f"  - roc_curve.png (if binary)")
        print(f"  - metrics_summary.png")
        print(f"  - evaluation_results_[timestamp].txt")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

