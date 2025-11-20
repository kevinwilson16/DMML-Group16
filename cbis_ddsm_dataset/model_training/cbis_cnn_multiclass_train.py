
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras.applications import InceptionResNetV2, ResNet50, VGG16, EfficientNetB0
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===========================
# Configuration
# ===========================

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESSED_DIR = SCRIPT_DIR / "preprocessed_data"
MODELS_DIR = SCRIPT_DIR / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 5

# Model selection: 'inceptionresnetv2', 'resnet50', 'vgg16', 'efficientnetb0', 'custom'
MODEL_TYPE = 'inceptionresnetv2'


# ===========================
# Data Loading
# ===========================

def load_preprocessed_data(preprocessed_dir):
    """
    Load preprocessed data from numpy files.
    
    Args:
        preprocessed_dir: Directory containing preprocessed numpy files
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    print("Loading preprocessed data...")
    print(f"  Directory: {preprocessed_dir}")
    
    # Load data
    X_train = np.load(preprocessed_dir / 'X_train.npy')
    X_test = np.load(preprocessed_dir / 'X_test.npy')
    y_train = np.load(preprocessed_dir / 'y_train.npy')
    y_test = np.load(preprocessed_dir / 'y_test.npy')
    
    # Load metadata
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nData loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape:  {y_test.shape}")
    print(f"  Number of classes: {metadata['num_classes']}")
    
    return X_train, X_test, y_train, y_test, metadata


# ===========================
# Data Augmentation
# ===========================

def create_data_augmentation():
    """
    Create data augmentation generator for training.
    
    Returns:
        ImageDataGenerator with augmentation settings
    """
    train_datagen = ImageDataGenerator(
        rotation_range=40,           # Randomly rotate images by 40 degrees
        width_shift_range=0.2,       # Randomly shift images horizontally by 20%
        height_shift_range=0.2,      # Randomly shift images vertically by 20%
        shear_range=0.2,             # Shear transformation
        zoom_range=0.2,              # Randomly zoom images by 20%
        horizontal_flip=True,        # Randomly flip images horizontally
        fill_mode='nearest',         # Fill strategy for new pixels
        channel_shift_range=20       # Random channel shifts
    )
    
    return train_datagen


# ===========================
# Model Building
# ===========================

def build_model(num_classes, image_size=IMAGE_SIZE, model_type=MODEL_TYPE):
    """
    Build CNN model with transfer learning.
    
    Args:
        num_classes: Number of output classes
        image_size: Input image size
        model_type: Type of base model to use
    
    Returns:
        Compiled Keras model
    """
    print(f"\nBuilding model: {model_type}")
    print(f"  Input shape: ({image_size}, {image_size}, 3)")
    print(f"  Output classes: {num_classes}")
    
    input_shape = (image_size, image_size, 3)
    
    # Select base model
    if model_type == 'inceptionresnetv2':
        base_model = InceptionResNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Freeze all layers except last 5
        for layer in base_model.layers[:-5]:
            layer.trainable = False
    
    elif model_type == 'resnet50':
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Freeze all layers except last 10
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    
    elif model_type == 'vgg16':
        base_model = VGG16(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Freeze all layers except last 4
        for layer in base_model.layers[:-4]:
            layer.trainable = False
    
    elif model_type == 'efficientnetb0':
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Freeze all layers except last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    elif model_type == 'custom':
        # Build custom CNN from scratch
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        print(f"\nModel summary:")
        model.summary()
        
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build final model with transfer learning
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    print(f"\nModel summary:")
    model.summary()
    
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    print(f"\nCompiling model...")
    print(f"  Optimizer: Nadam (lr={learning_rate})")
    print(f"  Loss: categorical_crossentropy")
    print(f"  Metrics: accuracy")
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Nadam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model


# ===========================
# Training
# ===========================

def train_model(model, X_train, y_train, X_test, y_test, 
                batch_size=BATCH_SIZE, epochs=EPOCHS, 
                use_augmentation=True):
    """
    Train the model with callbacks.
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels (one-hot encoded)
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        batch_size: Batch size for training
        epochs: Number of epochs
        use_augmentation: Whether to use data augmentation
    
    Returns:
        Training history
    """
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Augmentation: {use_augmentation}")
    
    # Callbacks
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f'best_model_{MODEL_TYPE}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train with or without augmentation
    if use_augmentation:
        print("\nTraining with data augmentation...")
        train_datagen = create_data_augmentation()
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        
        history = model.fit(
            train_generator,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("\nTraining without data augmentation...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    print("\nTraining complete!")
    
    return history


# ===========================
# Evaluation
# ===========================

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
    
    Returns:
        Evaluation metrics
    """
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    return test_loss, test_accuracy, y_pred_classes, cm


def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    print(f"\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Val Loss', marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training history saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    print(f"\nPlotting confusion matrix...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved to: {save_path}")
    plt.close()


def save_model_architecture(model, save_path):
    """
    Save model architecture diagram.
    
    Args:
        model: Keras model
        save_path: Path to save the diagram
    """
    try:
        print(f"\nSaving model architecture...")
        plot_model(model, to_file=save_path, show_shapes=True, 
                  show_layer_names=True, dpi=150)
        print(f"  Model architecture saved to: {save_path}")
    except Exception as e:
        print(f"  Could not save model architecture: {e}")


# ===========================
# Main Execution
# ===========================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("CBIS-DDSM CNN Multi-Class Classification Training")
    print("="*60)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, metadata = load_preprocessed_data(PREPROCESSED_DIR)
    
    num_classes = metadata['num_classes']
    class_names = ['BENIGN', 'MALIGNANT', 'NORMAL'][:num_classes]
    
    print(f"\nClass names: {class_names}")
    
    # Build model
    model = build_model(num_classes=num_classes, model_type=MODEL_TYPE)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    # Save model architecture
    save_model_architecture(
        model, 
        MODELS_DIR / f'model_architecture_{MODEL_TYPE}.png'
    )
    
    # Train model
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        use_augmentation=True
    )
    
    # Plot training history
    plot_training_history(
        history,
        MODELS_DIR / f'training_history_{MODEL_TYPE}.png'
    )
    
    # Evaluate model
    test_loss, test_accuracy, y_pred, cm = evaluate_model(
        model, X_test, y_test, class_names
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        MODELS_DIR / f'confusion_matrix_{MODEL_TYPE}.png'
    )
    
    # Save final model
    final_model_path = MODELS_DIR / f'final_model_{MODEL_TYPE}.h5'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training summary
    summary = {
        'model_type': MODEL_TYPE,
        'num_classes': num_classes,
        'class_names': class_names,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': len(history.history['loss']),
        'learning_rate': LEARNING_RATE,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(MODELS_DIR / f'training_summary_{MODEL_TYPE}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()

