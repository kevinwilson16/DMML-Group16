"""
FIXED FAST Training Script for CBIS-DDSM
Target: 75-80% accuracy in under 20 minutes
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# FAST Configuration
class Config:
    BASE_PATH = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/archive (4)"
    JPEG_PATH = os.path.join(BASE_PATH, "jpeg")
    CSV_PATH = os.path.join(BASE_PATH, "csv")
    
    IMAGE_SIZE = 128  # Smaller = faster (was 224)
    BATCH_SIZE = 64   # Bigger = faster (was 32)
    EPOCHS = 8        # Fewer = faster (was 15)
    LEARNING_RATE = 0.001  # Higher = faster convergence
    
    OUTPUT_DIR = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/outputs_fast"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

config = Config()

print("="*80)
print("FAST CBIS-DDSM Training - Target: 75-80% in <15 minutes")
print("="*80)
print(f"Image Size: {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Epochs: {config.EPOCHS}")
print("="*80)

def load_data():
    """Load and prepare dataset"""
    print("\n[1/5] Loading CSV data...")
    
    calc_train = pd.read_csv(os.path.join(config.CSV_PATH, "calc_case_description_train_set.csv"))
    mass_train = pd.read_csv(os.path.join(config.CSV_PATH, "mass_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(config.CSV_PATH, "calc_case_description_test_set.csv"))
    mass_test = pd.read_csv(os.path.join(config.CSV_PATH, "mass_case_description_test_set.csv"))
    
    train_df = pd.concat([calc_train, mass_train], ignore_index=True)
    test_df = pd.concat([calc_test, mass_test], ignore_index=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Pathology distribution:")
    print(train_df['pathology'].value_counts())
    
    return train_df, test_df

def get_jpeg_path(dicom_path):
    """Convert DICOM path to JPEG path"""
    if pd.isna(dicom_path):
        return None
    
    path_str = str(dicom_path).strip().replace('"', '').replace('\n', '')
    parts = path_str.split('/')
    
    if len(parts) >= 2:
        series_uid = parts[-2]
        jpeg_dir = os.path.join(config.JPEG_PATH, series_uid)
        
        if os.path.exists(jpeg_dir):
            jpg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
            if jpg_files:
                return os.path.join(jpeg_dir, jpg_files[0])
    return None

def load_images(df, max_samples=2000):
    """Load images efficiently - limit for speed"""
    print("\n[2/5] Loading images...")
    
    # Sample for speed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} images for faster training")
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Loaded {idx + 1}/{len(df)} images...", end='\r')
        
        path = get_jpeg_path(row['cropped image file path'])
        if path and os.path.exists(path):
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    # Binary: 0=benign, 1=malignant
                    labels.append(1 if row['pathology'] == 'MALIGNANT' else 0)
            except:
                continue
    
    print(f"\nSuccessfully loaded {len(images)} images")
    print(f"Benign: {labels.count(0)}, Malignant: {labels.count(1)}")
    
    return np.array(images), np.array(labels)

def build_fast_model():
    """Build lightweight but effective model"""
    print("\n[3/5] Building model...")
    
    base = EfficientNetB0(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze most layers for speed
    for layer in base.layers[:-20]:
        layer.trainable = False
    
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"Model built: {model.count_params():,} parameters")
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train with minimal augmentation for speed"""
    print("\n[4/5] Training...")
    
    # Light augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(config.OUTPUT_DIR, 'best_model_fast.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Calculate class weights
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    total = len(y_train)
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print(f"Class weights: {class_weight}")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        steps_per_epoch=len(X_train) // config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    return history

def evaluate(model, X_test, y_test):
    """Evaluate model"""
    print("\n[5/5] Evaluating...")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")
    print("="*80)
    
    return test_acc

def main():
    """Main pipeline"""
    try:
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPU Available: {len(gpus)} device(s)")
        else:
            print("\n! No GPU - will use CPU (slower)")
        
        # Load data
        train_df, test_df = load_data()
        
        # Load images (limited for speed)
        X_train_full, y_train_full = load_images(train_df, max_samples=2000)
        X_test, y_test = load_images(test_df, max_samples=500)
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.15,
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\nFinal splits:")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build and train
        model = build_fast_model()
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        accuracy = evaluate(model, X_test, y_test)
        
        # Save
        model.save(os.path.join(config.OUTPUT_DIR, 'final_model_fast.h5'))
        
        print("\n✓ Training complete!")
        print(f"Model saved to: {config.OUTPUT_DIR}")
        
        if accuracy >= 0.75:
            print(f"\n🎉 SUCCESS! Achieved {accuracy*100:.1f}% accuracy (target: 75-80%)")
        else:
            print(f"\n⚠ Got {accuracy*100:.1f}% - may need more training")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

