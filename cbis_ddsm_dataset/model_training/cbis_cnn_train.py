import os
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ==== Paths and hyperparameters (mirrors the notebook) ====

DATA_DIR = r"C:\Users\gaura\Desktop\CBIS-DDSM_processed\CBIS-DDSM_processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")  # acts as validation / test set

CBIS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = CBIS_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU, with debug prints."""
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("num devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device 0:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_dataloaders(batch_size: int = BATCH_SIZE):
    """Create train and validation dataloaders using the same transforms as the notebook."""
    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    print("Train classes:", train_dataset.classes)
    print("Val classes  :", val_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Val samples  :", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(device: torch.device):
    """Build the DenseNet-121 model with Dropout head, loss, optimizer, and scheduler."""
    num_classes = 2

    # DenseNet-121 with ImageNet weights, full fine-tuning
    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    return model, criterion, optimizer, scheduler


def train_model(
    model: nn.Module,
    criterion,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    models_dir: Path = MODELS_DIR,
):
    """Training loop with early stopping and best-model saving."""
    best_val_acc = 0.0
    best_model_path = models_dir / "cbis_cnn_densenet121_best.pth"

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

            if batch_idx % 20 == 0 or batch_idx == num_train_batches:
                current_loss = running_loss / total_train
                current_acc = running_corrects / total_train
                print(
                    f"  [Train] Batch {batch_idx}/{num_train_batches} - "
                    f"Avg Loss: {current_loss:.4f}  Avg Acc: {current_acc:.4f}",
                    end="\r",
                )

        print()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

                if batch_idx % 10 == 0 or batch_idx == num_val_batches:
                    current_val_loss = val_running_loss / total_val
                    current_val_acc = val_running_corrects / total_val
                    print(
                        f"  [Val]   Batch {batch_idx}/{num_val_batches} - "
                        f"Avg Loss: {current_val_loss:.4f}  Avg Acc: {current_val_acc:.4f}",
                        end="\r",
                    )

        print()

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = val_running_corrects / total_val

        print(
            f"Train Loss: {epoch_train_loss:.4f}  Train Acc: {epoch_train_acc:.4f}"
        )
        print(f"Val   Loss: {epoch_val_loss:.4f}  Val   Acc: {epoch_val_acc:.4f}")

        # LR scheduler step on validation accuracy
        scheduler.step(epoch_val_acc)

        # Save best model + early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"\n✓ New best model saved to: {best_model_path} "
                f"(val_acc={best_val_acc:.4f})"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in val acc for {patience_counter} epoch(s).")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_model_path, best_val_acc


def evaluate_best_model(
    model_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    class_names,
):
    """Quick evaluation of the saved best DenseNet-121 model on the validation set."""
    num_classes = len(class_names)

    model = models.densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    # We saved only state_dict, so weights_only=True is safe
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
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

    print("\n=== Validation Performance (Best Model) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def main():
    device = get_device()
    print(f"TRAIN_DIR: {TRAIN_DIR}")
    print(f"VAL_DIR  : {VAL_DIR}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE
    )
    class_names = train_dataset.classes

    model, criterion, optimizer, scheduler = build_model(device)
    best_model_path, best_val_acc = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
    )

    print("\nReloading best model and evaluating on validation set...")
    evaluate_best_model(best_model_path, val_loader, device, class_names)


if __name__ == "__main__":
    main()


