

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===========================
# Configuration
# ===========================

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = SCRIPT_DIR / "archive (4)"
CSV_DIR = ARCHIVE_DIR / "csv"
JPEG_DIR = ARCHIVE_DIR / "jpeg"

# Image processing settings
TARGET_SIZE = (224, 224)  # (height, width)
IMAGE_CHANNELS = 3

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "preprocessed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class mapping
CLASS_MAPPER = {
    'MALIGNANT': 1,
    'MALIGNANT_WITHOUT_CALLBACK': 1,
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 0,
    'NORMAL': 2  # For future use if normal cases are added
}

NUM_CLASSES = 3  # BENIGN, MALIGNANT, NORMAL


# ===========================
# Image Processing Functions
# ===========================

def find_jpeg_image(image_dcm_path, jpeg_dir):
    """
    Find the corresponding JPEG image for a DICOM path from CSV.
    
    Args:
        image_dcm_path: Path from CSV (e.g., 'Mass-Training_P_00001_LEFT_CC/...')
        jpeg_dir: Root directory containing JPEG images
    
    Returns:
        Path to JPEG image or None if not found
    """
    # Extract the DICOM UID directories from the path
    parts = image_dcm_path.split('/')
    if len(parts) >= 2:
        # Try all UID parts (typically there are 2-3 UIDs in the path)
        # Start from the second part onwards (skip case folder name)
        for i in range(1, len(parts)):
            uid_folder = parts[i]
            
            # Skip if it looks like a filename
            if uid_folder.endswith('.dcm'):
                continue
            
            # Check if this UID folder exists in jpeg directory
            jpeg_folder = jpeg_dir / uid_folder
            if jpeg_folder.exists() and jpeg_folder.is_dir():
                # Get the first JPEG file in this folder
                jpeg_files = list(jpeg_folder.glob('*.jpg'))
                if jpeg_files:
                    return jpeg_files[0]
    
    return None


def image_processor(image_path, target_size):
    """
    Preprocess images for CNN model.
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (height, width)
    
    Returns:
        Normalized numpy array of shape (height, width, 3)
    """
    try:
        # Read image
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        image_array = image / 255.0
        
        return image_array
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# ===========================
# Data Loading Functions
# ===========================

def load_mass_data(csv_dir, jpeg_dir):
    """
    Load and combine mass train and test datasets.
    
    Args:
        csv_dir: Directory containing CSV files
        jpeg_dir: Directory containing JPEG images
    
    Returns:
        Combined pandas DataFrame with image paths
    """
    print("Loading CSV files...")
    
    # Load train and test CSV files
    mass_train_path = csv_dir / "mass_case_description_train_set.csv"
    mass_test_path = csv_dir / "mass_case_description_test_set.csv"
    
    if not mass_train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {mass_train_path}")
    if not mass_test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {mass_test_path}")
    
    mass_train = pd.read_csv(mass_train_path)
    mass_test = pd.read_csv(mass_test_path)
    
    print(f"  Train samples: {len(mass_train)}")
    print(f"  Test samples: {len(mass_test)}")
    
    # Combine datasets
    full_mass = pd.concat([mass_train, mass_test], axis=0, ignore_index=True)
    print(f"  Total samples: {len(full_mass)}")
    
    # Find corresponding JPEG images
    print("\nFinding JPEG images...")
    jpeg_paths = []
    for idx, row in tqdm(full_mass.iterrows(), total=len(full_mass)):
        dcm_path = row['image file path']
        jpeg_path = find_jpeg_image(dcm_path, jpeg_dir)
        jpeg_paths.append(jpeg_path)
    
    full_mass['jpeg_path'] = jpeg_paths
    
    # Remove rows where JPEG was not found
    initial_count = len(full_mass)
    full_mass = full_mass[full_mass['jpeg_path'].notna()].reset_index(drop=True)
    final_count = len(full_mass)
    
    if initial_count != final_count:
        print(f"  Warning: {initial_count - final_count} images not found. Using {final_count} samples.")
    else:
        print(f"  All images found!")
    
    # Check if we have any samples
    if final_count == 0:
        raise ValueError(
            f"\nERROR: No JPEG images found!\n"
            f"Please check that:\n"
            f"  1. JPEG images exist in: {jpeg_dir}\n"
            f"  2. CSV paths match JPEG folder structure\n"
            f"  3. Image files have .jpg extension"
        )
    
    return full_mass


def add_normal_samples(full_mass, num_normal_samples=0):
    """
    Placeholder function to add normal (healthy) breast images.
    
    Note: The CBIS-DDSM dataset contains only abnormal cases (MALIGNANT/BENIGN).
    If you have normal breast images from another source, add them here.
    
    Args:
        full_mass: DataFrame with existing data
        num_normal_samples: Number of normal samples to add
    
    Returns:
        DataFrame with normal samples added
    """
    if num_normal_samples > 0:
        print(f"\nNote: Adding {num_normal_samples} normal samples...")
        print("  This is a placeholder. Please add your normal breast images.")
        
        # Example structure for adding normal samples:
        # normal_data = {
        #     'pathology': ['NORMAL'] * num_normal_samples,
        #     'jpeg_path': [path1, path2, ...],  # Paths to normal images
        #     'patient_id': ['N_00001', 'N_00002', ...],
        #     ... other required columns
        # }
        # normal_df = pd.DataFrame(normal_data)
        # full_mass = pd.concat([full_mass, normal_df], axis=0, ignore_index=True)
    
    return full_mass


# ===========================
# Preprocessing Pipeline
# ===========================

def preprocess_dataset(csv_dir, jpeg_dir, target_size, output_dir):
    """
    Main preprocessing pipeline.
    
    Args:
        csv_dir: Directory containing CSV files
        jpeg_dir: Directory containing JPEG images
        target_size: Target image size (height, width)
        output_dir: Directory to save preprocessed data
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("="*60)
    print("CBIS-DDSM Preprocessing Pipeline")
    print("="*60)
    
    # Load data
    full_mass = load_mass_data(csv_dir, jpeg_dir)
    
    # Add normal samples if available
    full_mass = add_normal_samples(full_mass, num_normal_samples=0)
    
    # Display class distribution
    print("\nClass distribution:")
    print(full_mass['pathology'].value_counts())
    
    # Process images
    print(f"\nProcessing images to size {target_size}...")
    processed_images = []
    valid_indices = []
    
    for idx, row in tqdm(full_mass.iterrows(), total=len(full_mass)):
        img_path = row['jpeg_path']
        processed_img = image_processor(img_path, target_size)
        
        if processed_img is not None:
            processed_images.append(processed_img)
            valid_indices.append(idx)
    
    print(f"  Successfully processed {len(processed_images)} images")
    
    # Filter dataframe to only valid images
    full_mass = full_mass.iloc[valid_indices].reset_index(drop=True)
    
    # Convert to numpy array
    X_resized = np.array(processed_images)
    print(f"\nImage array shape: {X_resized.shape}")
    
    # Map labels
    print("\nMapping labels...")
    full_mass['labels'] = full_mass['pathology'].map(CLASS_MAPPER)
    
    # Check for unmapped labels
    unmapped = full_mass[full_mass['labels'].isna()]
    if len(unmapped) > 0:
        print(f"  Warning: {len(unmapped)} samples have unmapped labels:")
        print(unmapped['pathology'].value_counts())
        # Remove unmapped samples
        valid_mask = full_mass['labels'].notna()
        full_mass = full_mass[valid_mask].reset_index(drop=True)
        X_resized = X_resized[valid_mask]
    
    # Get labels
    y_labels = full_mass['labels'].values.astype(int)
    
    print("\nLabel distribution:")
    unique, counts = np.unique(y_labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = [k for k, v in CLASS_MAPPER.items() if v == label][0]
        print(f"  Class {label} ({label_name}): {count} samples")
    
    # Check number of classes
    num_classes_found = len(np.unique(y_labels))
    print(f"\nNumber of classes found: {num_classes_found}")
    
    if num_classes_found < NUM_CLASSES:
        print(f"  Warning: Expected {NUM_CLASSES} classes, but found {num_classes_found}")
        print(f"  Note: CBIS-DDSM does not contain NORMAL cases. Only BENIGN and MALIGNANT.")
        print(f"  Adjusting NUM_CLASSES to {num_classes_found}")
        actual_num_classes = num_classes_found
    else:
        actual_num_classes = NUM_CLASSES
    
    # Split data
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resized, 
        y_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=y_labels  # Maintain class distribution
    )
    
    # Convert labels to categorical (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_classes=actual_num_classes)
    y_test_cat = to_categorical(y_test, num_classes=actual_num_classes)
    
    print(f"\nFinal data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train_cat.shape}")
    print(f"  y_test:  {y_test_cat.shape}")
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_dir}...")
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train_cat)
    np.save(output_dir / 'y_test.npy', y_test_cat)
    np.save(output_dir / 'y_train_labels.npy', y_train)  # Save original labels too
    np.save(output_dir / 'y_test_labels.npy', y_test)
    
    # Save metadata
    metadata = {
        'num_classes': actual_num_classes,
        'target_size': target_size,
        'class_mapper': CLASS_MAPPER,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'classes_found': unique.tolist()
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  Done!")
    print(f"\nPreprocessed data saved to: {output_dir}")
    
    # Plot sample images
    plot_sample_images(X_train, y_train, actual_num_classes, output_dir)
    
    return X_train, X_test, y_train_cat, y_test_cat, actual_num_classes


def plot_sample_images(X_train, y_train, num_classes, output_dir, samples_per_class=3):
    """
    Plot sample images from each class for verification.
    
    Args:
        X_train: Training images
        y_train: Training labels (not one-hot)
        num_classes: Number of classes
        output_dir: Directory to save plot
        samples_per_class: Number of samples to show per class
    """
    print("\nPlotting sample images...")
    
    class_names = ['BENIGN', 'MALIGNANT', 'NORMAL'][:num_classes]
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 4*num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx in range(num_classes):
        # Find samples of this class
        class_samples = np.where(y_train == class_idx)[0]
        
        if len(class_samples) == 0:
            continue
        
        # Select random samples
        selected_samples = np.random.choice(
            class_samples, 
            min(samples_per_class, len(class_samples)), 
            replace=False
        )
        
        for i, sample_idx in enumerate(selected_samples):
            if i >= samples_per_class:
                break
                
            ax = axes[class_idx, i] if num_classes > 1 else axes[i]
            ax.imshow(X_train[sample_idx])
            ax.set_title(f'{class_names[class_idx]}')
            ax.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'sample_images.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Sample images saved to: {plot_path}")
    plt.close()


# ===========================
# Main Execution
# ===========================

def main():
    """Main execution function."""
    print("\nStarting preprocessing...")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"JPEG Directory: {JPEG_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Check if directories exist
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")
    if not JPEG_DIR.exists():
        raise FileNotFoundError(f"JPEG directory not found: {JPEG_DIR}")
    
    # Run preprocessing
    X_train, X_test, y_train, y_test, num_classes = preprocess_dataset(
        csv_dir=CSV_DIR,
        jpeg_dir=JPEG_DIR,
        target_size=TARGET_SIZE,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nTo use this data in your CNN training:")
    print(f"  X_train = np.load('{OUTPUT_DIR}/X_train.npy')")
    print(f"  X_test = np.load('{OUTPUT_DIR}/X_test.npy')")
    print(f"  y_train = np.load('{OUTPUT_DIR}/y_train.npy')")
    print(f"  y_test = np.load('{OUTPUT_DIR}/y_test.npy')")
    print(f"\nNumber of classes: {num_classes}")


if __name__ == "__main__":
    main()

