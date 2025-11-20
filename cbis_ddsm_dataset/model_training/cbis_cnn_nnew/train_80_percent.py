"""
Enhanced Training Script - Target: 80%+ Accuracy
Fixes overfitting and improves generalization
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# Enhanced Configuration
BASE_PATH = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/archive (4)"
OUTPUT_DIR = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/outputs_80percent"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 128  # Sweet spot for speed vs accuracy
BATCH_SIZE = 32
EPOCHS = 20  # More epochs with better regularization
LEARNING_RATE = 0.0003  # Lower LR for stability

print("="*80)
print("ENHANCED CBIS-DDSM TRAINING - TARGET: 80%+ ACCURACY")
print("="*80)
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} devices")
print(f"Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Epochs: {EPOCHS}")
print("="*80)

def load_data():
    """Load CSV files"""
    print("\n[1/6] Loading CSV files...")
    
    csv_path = os.path.join(BASE_PATH, "csv")
    
    calc_train = pd.read_csv(os.path.join(csv_path, "calc_case_description_train_set.csv"))
    mass_train = pd.read_csv(os.path.join(csv_path, "mass_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(csv_path, "calc_case_description_test_set.csv"))
    mass_test = pd.read_csv(os.path.join(csv_path, "mass_case_description_test_set.csv"))
    
    train_df = pd.concat([calc_train, mass_train], ignore_index=True)
    test_df = pd.concat([calc_test, mass_test], ignore_index=True)
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    
    return train_df, test_df

def get_jpeg_path(dicom_path):
    """Convert DICOM to JPEG path"""
    if pd.isna(dicom_path):
        return None
    
    path_str = str(dicom_path).strip().replace('"', '').replace('\n', '')
    parts = path_str.split('/')
    
    if len(parts) >= 2:
        series_uid = parts[-2]
        jpeg_path = os.path.join(BASE_PATH, "jpeg", series_uid)
        
        if os.path.exists(jpeg_path):
            files = [f for f in os.listdir(jpeg_path) if f.endswith('.jpg')]
            if files:
                return os.path.join(jpeg_path, files[0])
    return None

def load_images_improved(df, max_samples=2500):
    """Load images with better quality control"""
    print("\n[2/6] Loading images...")
    
    # Sample more data for better learning
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    images = []
    labels = []
    failed = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} images ({len(images)} loaded)...", end='\r')
        
        img_path = get_jpeg_path(row['cropped image file path'])
        
        if img_path and os.path.exists(img_path):
            try:
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    img = img.astype('float32') / 255.0
                    
                    # Validate
                    if not (np.isnan(img).any() or np.isinf(img).any()):
                        images.append(img)
                        labels.append(1 if row['pathology'] == 'MALIGNANT' else 0)
                    else:
                        failed += 1
                else:
                    failed += 1
            except:
                failed += 1
        else:
            failed += 1
    
    print(f"\n✓ Loaded {len(images)} images (failed: {failed})")
    
    labels_array = np.array(labels)
    print(f"  Benign: {np.sum(labels_array == 0)}, Malignant: {np.sum(labels_array == 1)}")
    
    return np.array(images), labels_array

def build_better_model():
    """Improved model with better regularization"""
    print("\n[3/6] Building enhanced model...")
    
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Unfreeze last 30 layers for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"✓ Model created with {trainable_params:,} trainable parameters")
    
    return model

def create_augmented_generators():
    """Create data augmentation generators"""
    print("\n[4/6] Setting up data augmentation...")
    
    # Aggressive but realistic augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

def train_improved(model, X_train, y_train, X_val, y_val, train_datagen):
    """Train with augmentation and better callbacks"""
    print("\n[5/6] Training with data augmentation...")
    
    # Calculate class weights
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    total = len(y_train)
    
    weight_0 = total / (2.0 * neg_count)
    weight_1 = total / (2.0 * pos_count)
    class_weight = {0: weight_0, 1: weight_1}
    
    print(f"Class weights: {{0: {weight_0:.2f}, 1: {weight_1:.2f}}}")
    
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=7,  # More patience
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model_80.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train with augmentation
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_final(model, X_test, y_test):
    """Final evaluation"""
    print("\n[6/6] Final Evaluation...")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_auc, test_precision, test_recall = test_results
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")
    print("="*80)
    
    # Calculate specifics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\nAdditional Metrics:")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    
    return test_acc, test_auc

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 0].axhline(y=0.80, color='r', linestyle='--', label='Target 80%')
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Val', linewidth=2)
    axes[1, 0].axhline(y=0.85, color='r', linestyle='--', label='Target 85%')
    axes[1, 0].set_title('AUC', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300)
    print(f"\n✓ Training curves saved to {OUTPUT_DIR}/training_curves.png")
    plt.close()

def main():
    """Enhanced training pipeline"""
    try:
        device = tf.config.list_physical_devices('GPU')
        print(f"\nGPU Devices: {len(device)}")
        
        # Load data
        train_df, test_df = load_data()
        
        # Load MORE images for better learning
        X_train_full, y_train_full = load_images_improved(train_df, max_samples=2500)
        X_test, y_test = load_images_improved(test_df, max_samples=500)
        
        # Split with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.15,
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build improved model
        model = build_better_model()
        
        # Create augmentation
        train_datagen, val_datagen = create_augmented_generators()
        
        # Train
        history = train_improved(model, X_train, y_train, X_val, y_val, train_datagen)
        
        # Plot
        plot_training_history(history)
        
        # Evaluate
        test_acc, test_auc = evaluate_final(model, X_test, y_test)
        
        # Save
        model.save(os.path.join(OUTPUT_DIR, 'final_model_80.keras'))
        
        print("\n" + "="*80)
        if test_acc >= 0.80:
            print("🎉🎉🎉 SUCCESS! 80%+ ACCURACY ACHIEVED! 🎉🎉🎉")
        elif test_acc >= 0.78:
            print("✓ Very close! Almost at 80%")
        elif test_acc >= 0.75:
            print("✓ Good progress, over 75%")
        else:
            print("⚠ Below target, but improved")
        print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
        print(f"Final Test AUC: {test_auc:.4f}")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

