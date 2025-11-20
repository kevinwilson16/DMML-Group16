import os
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


DATA_DIR = r"C:\Users\gaura\Desktop\CBIS-DDSM_processed\CBIS-DDSM_processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")

CBIS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = CBIS_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

IMG_SIZE = 128  # we downscale for MLP to keep feature size reasonable
INPUT_DIM = 1 * IMG_SIZE * IMG_SIZE  # grayscale


def get_device() -> torch.device:
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
    """Dataloaders for MLP: grayscale + resize to IMG_SIZE, no heavy augmentation."""
    common_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # [1, H, W] in [0,1]
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=common_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=common_transforms)

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


class MLPImageClassifier(nn.Module):
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


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "cbis_mlp_best.pth"
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 30)

        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
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

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # Val
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = val_running_corrects / total_val

        print(f"Train Loss: {epoch_train_loss:.4f}  Train Acc: {epoch_train_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f}  Val   Acc: {epoch_val_acc:.4f}")

        # Early stopping + save best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"\n✓ New best MLP model saved to: {best_model_path} "
                f"(val_acc={best_val_acc:.4f})"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in val acc for {patience_counter} epoch(s).")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nBest validation accuracy (MLP): {best_val_acc:.4f}")
    return best_model_path, best_val_acc


def evaluate_mlp(
    model_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    class_names,
):
    model = MLPImageClassifier(input_dim=INPUT_DIM, num_classes=len(class_names))
    model = model.to(device)

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

    print("\n=== Validation Performance (MLP Best Model) ===")
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

    model = MLPImageClassifier(input_dim=INPUT_DIM, num_classes=len(class_names))
    model = model.to(device)

    best_model_path, best_val_acc = train_mlp(
        model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS
    )

    print("\nReloading best MLP model and evaluating on validation set...")
    evaluate_mlp(best_model_path, val_loader, device, class_names)


if __name__ == "__main__":
    main()


