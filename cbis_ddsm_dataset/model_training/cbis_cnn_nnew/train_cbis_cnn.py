"""
CBIS-DDSM Breast Cancer Classification using CNN
Dataset: Curated Breast Imaging Subset of Digital Database for Screening Mammography
Task: Binary classification (Benign vs Malignant)
Model: InceptionResNetV2 with Transfer Learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.utils import to_categorical
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Paths - WSL Linux format
    BASE_PATH = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/archive (4)"
    JPEG_PATH = os.path.join(BASE_PATH, "jpeg")
    CSV_PATH = os.path.join(BASE_PATH, "csv")
    
    # Model parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    
    # Output paths
    OUTPUT_DIR = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/outputs"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

config = Config()

print("=" * 80)
print("CBIS-DDSM Breast Cancer Classification Training")
print("=" * 80)
print(f"Base Path: {config.BASE_PATH}")
print(f"Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Epochs: {config.EPOCHS}")
print("=" * 80)


def load_and_prepare_data():
    """
    Load CSV files and prepare the dataset
    Combines both calcification and mass cases for training
    """
    print("\n[1/7] Loading dataset information...")
    
    # Load training data
    calc_train = pd.read_csv(os.path.join(config.CSV_PATH, "calc_case_description_train_set.csv"))
    mass_train = pd.read_csv(os.path.join(config.CSV_PATH, "mass_case_description_train_set.csv"))
    
    # Load test data
    calc_test = pd.read_csv(os.path.join(config.CSV_PATH, "calc_case_description_test_set.csv"))
    mass_test = pd.read_csv(os.path.join(config.CSV_PATH, "mass_case_description_test_set.csv"))
    
    print(f"Calcification Training Cases: {len(calc_train)}")
    print(f"Mass Training Cases: {len(mass_train)}")
    print(f"Calcification Test Cases: {len(calc_test)}")
    print(f"Mass Test Cases: {len(mass_test)}")
    
    # Combine datasets
    train_df = pd.concat([calc_train, mass_train], ignore_index=True)
    test_df = pd.concat([calc_test, mass_test], ignore_index=True)
    
    print(f"\nTotal Training Samples: {len(train_df)}")
    print(f"Total Test Samples: {len(test_df)}")
    
    # Check pathology distribution
    print("\nTraining Set Pathology Distribution:")
    print(train_df['pathology'].value_counts())
    print("\nTest Set Pathology Distribution:")
    print(test_df['pathology'].value_counts())
    
    return train_df, test_df


def clean_file_path(path):
    """
    Clean the file path from CSV (remove extra quotes and newlines)
    """
    if pd.isna(path):
        return None
    # Remove quotes and newlines
    cleaned = str(path).strip().replace('"', '').replace('\n', '')
    return cleaned


def convert_dicom_to_jpeg_path(dicom_path, base_path):
    """
    Convert DICOM path to JPEG path
    Example: 'Calc-Training_P_00005_RIGHT_CC_1/.../000000.dcm' 
    -> 'CBIS-DDSM/jpeg/[SeriesInstanceUID]/1-xxx.jpg'
    """
    if pd.isna(dicom_path) or dicom_path is None:
        return None
    
    # Extract the directory structure
    parts = dicom_path.split('/')
    if len(parts) >= 2:
        series_uid = parts[-2]  # Get the long UID
        # Find corresponding jpeg file
        jpeg_dir = os.path.join(base_path, series_uid)
        if os.path.exists(jpeg_dir):
            # Get the first jpg file in the directory
            jpg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
            if jpg_files:
                return os.path.join(jpeg_dir, jpg_files[0])
    return None


def prepare_image_paths(df, use_cropped=True):
    """
    Prepare image paths and labels from dataframe
    """
    print("\n[2/7] Preparing image paths...")
    
    image_paths = []
    labels = []
    
    # Use cropped images for better focus on ROI
    path_column = 'cropped image file path' if use_cropped else 'image file path'
    
    for idx, row in df.iterrows():
        # Clean the path
        img_path = clean_file_path(row[path_column])
        
        if img_path is None:
            continue
            
        # Convert DICOM path to JPEG path
        jpeg_path = convert_dicom_to_jpeg_path(img_path, config.JPEG_PATH)
        
        if jpeg_path and os.path.exists(jpeg_path):
            image_paths.append(jpeg_path)
            
            # Convert pathology to binary label
            pathology = row['pathology']
            if pathology == 'MALIGNANT':
                labels.append(1)  # Malignant = 1
            else:  # BENIGN or BENIGN_WITHOUT_CALLBACK
                labels.append(0)  # Benign = 0
    
    print(f"Found {len(image_paths)} valid images")
    print(f"Benign: {labels.count(0)}, Malignant: {labels.count(1)}")
    
    return np.array(image_paths), np.array(labels)


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess a single image
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def create_dataset(image_paths, labels, batch_size=32):
    """
    Create dataset by loading all images into memory
    """
    print(f"\n[3/7] Loading images into memory...")
    
    images = []
    valid_labels = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(image_paths)} images...", end='\r')
        
        img = load_and_preprocess_image(path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        if img is not None:
            images.append(img)
            valid_labels.append(label)
    
    print(f"\nSuccessfully loaded {len(images)} images")
    
    X = np.array(images)
    y = np.array(valid_labels)
    
    # Convert labels to categorical (one-hot encoding)
    y_categorical = to_categorical(y, num_classes=2)
    
    return X, y_categorical, y


def create_data_generators():
    """
    Create data augmentation generators
    """
    print("\n[4/7] Setting up data augmentation...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,          # Reduced from 40
        width_shift_range=0.1,      # Reduced from 0.2
        height_shift_range=0.1,     # Reduced from 0.2
        shear_range=0.1,            # Reduced from 0.2
        zoom_range=0.1,             # Reduced from 0.2
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Validation data - no augmentation
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen


def build_model(input_shape=(224, 224, 3)):
    """
    Build InceptionResNetV2 model with transfer learning
    """
    print("\n[5/7] Building model...")
    
    # Load pre-trained InceptionResNetV2
    base_model = InceptionResNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers, fine-tune last few layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    print(f"Base model: {base_model.name}")
    print(f"Total layers: {len(base_model.layers)}")
    print(f"Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}")
    
    # Build complete model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Nadam(learning_rate=config.LEARNING_RATE),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\nModel compiled successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model


def plot_training_history(history, save_path):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision and Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision')
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def train_model(model, X_train, y_train, X_val, y_val, train_datagen):
    """
    Train the model with callbacks
    """
    print("\n[6/7] Training model...")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, f'best_model_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, f'run_{timestamp}'),
            histogram_freq=1
        )
    ]
    
    # Calculate class weights for imbalanced dataset
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = {}
    for i in range(2):
        class_weights[i] = len(y_train_labels) / (2 * np.sum(y_train_labels == i))
    
    print(f"Class weights: {class_weights}")
    
    # Train with data augmentation
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        steps_per_epoch=len(X_train) // config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history, timestamp


def evaluate_model(model, X_test, y_test, y_test_labels, timestamp):
    """
    Evaluate the model on test set
    """
    print("\n[7/7] Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, 
                                target_names=['Benign', 'Malignant']))
    
    # Save classification report
    report = classification_report(y_test_labels, y_pred, 
                                   target_names=['Benign', 'Malignant'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(config.OUTPUT_DIR, f'classification_report_{timestamp}.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(config.PLOT_DIR, f'confusion_matrix_{timestamp}.png')
    plot_confusion_matrix(y_test_labels, y_pred, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(config.PLOT_DIR, f'roc_curve_{timestamp}.png')
    plot_roc_curve(y_test_labels, y_pred_proba[:, 1], roc_path)
    
    # Calculate final metrics
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")
    print("=" * 80)


def main():
    """
    Main training pipeline
    """
    try:
        # Load data
        train_df, test_df = load_and_prepare_data()
        
        # Prepare image paths and labels
        train_paths, train_labels = prepare_image_paths(train_df, use_cropped=True)
        test_paths, test_labels = prepare_image_paths(test_df, use_cropped=True)
        
        # Load images
        X_train_full, y_train_full_cat, y_train_full = create_dataset(train_paths, train_labels)
        X_test, y_test_cat, y_test = create_dataset(test_paths, test_labels)
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full_cat, 
            test_size=0.15, 
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\nFinal dataset sizes:")
        print(f"Training: {X_train.shape[0]}")
        print(f"Validation: {X_val.shape[0]}")
        print(f"Test: {X_test.shape[0]}")
        
        # Create data generators
        train_datagen, val_datagen = create_data_generators()
        
        # Build model
        model = build_model(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Train model
        history, timestamp = train_model(model, X_train, y_train, X_val, y_val, train_datagen)
        
        # Plot training history
        history_path = os.path.join(config.PLOT_DIR, f'training_history_{timestamp}.png')
        plot_training_history(history, history_path)
        
        # Save final model
        final_model_path = os.path.join(config.MODEL_DIR, f'final_model_{timestamp}.h5')
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Evaluate model
        evaluate_model(model, X_test, y_test_cat, y_test, timestamp)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check GPU availability
    print("\nChecking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU found. Training will run on CPU.")
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Start training
    main()

