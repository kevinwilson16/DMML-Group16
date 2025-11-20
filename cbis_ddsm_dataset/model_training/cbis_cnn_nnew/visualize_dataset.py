"""
Visualize sample images from CBIS-DDSM dataset
Shows a grid of sample images with their labels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from collections import Counter

# Configuration - WSL Linux format
BASE_PATH = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/archive (4)"
JPEG_PATH = os.path.join(BASE_PATH, "jpeg")
CSV_PATH = os.path.join(BASE_PATH, "csv")


def clean_file_path(path):
    """Clean the file path from CSV"""
    if pd.isna(path):
        return None
    return str(path).strip().replace('"', '').replace('\n', '')


def convert_dicom_to_jpeg_path(dicom_path, base_path):
    """Convert DICOM path to JPEG path"""
    if pd.isna(dicom_path) or dicom_path is None:
        return None
    
    parts = dicom_path.split('/')
    if len(parts) >= 2:
        series_uid = parts[-2]
        jpeg_dir = os.path.join(base_path, series_uid)
        if os.path.exists(jpeg_dir):
            jpg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
            if jpg_files:
                return os.path.join(jpeg_dir, jpg_files[0])
    return None


def load_dataset_info():
    """Load dataset information"""
    print("Loading dataset information...")
    
    # Load all CSV files
    calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
    mass_train = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_test_set.csv"))
    mass_test = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_test_set.csv"))
    
    # Combine
    train_df = pd.concat([calc_train, mass_train], ignore_index=True)
    test_df = pd.concat([calc_test, mass_test], ignore_index=True)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Total cases: {len(all_df)}")
    print(f"Training cases: {len(train_df)}")
    print(f"Test cases: {len(test_df)}")
    
    return all_df, train_df, test_df


def prepare_samples(df, num_samples=20, use_cropped=True):
    """Prepare sample images with labels"""
    print(f"\nPreparing {num_samples} sample images...")
    
    samples = []
    path_column = 'cropped image file path' if use_cropped else 'image file path'
    
    # Shuffle and get samples
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    attempts = 0
    max_attempts = len(df_shuffled)
    
    for idx, row in df_shuffled.iterrows():
        if len(samples) >= num_samples:
            break
        
        attempts += 1
        if attempts > max_attempts:
            break
        
        # Get image path
        img_path = clean_file_path(row[path_column])
        if img_path is None:
            continue
        
        # Convert to JPEG path
        jpeg_path = convert_dicom_to_jpeg_path(img_path, JPEG_PATH)
        
        if jpeg_path and os.path.exists(jpeg_path):
            # Load image
            img = cv2.imread(jpeg_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get metadata
                pathology = row['pathology']
                patient_id = row['patient_id']
                
                # Get abnormality type
                if 'abnormality type' in row:
                    abn_type = row['abnormality type']
                else:
                    abn_type = 'Unknown'
                
                # Simplify pathology labels
                if pathology == 'BENIGN_WITHOUT_CALLBACK':
                    pathology_display = 'NORMAL'
                elif pathology == 'BENIGN':
                    pathology_display = 'BENIGN'
                else:
                    pathology_display = 'MALIGNANT'
                
                samples.append({
                    'image': img,
                    'pathology': pathology,
                    'pathology_display': pathology_display,
                    'patient_id': patient_id,
                    'abnormality_type': abn_type,
                    'path': jpeg_path
                })
    
    print(f"Successfully loaded {len(samples)} images")
    return samples


def visualize_samples(samples, save_path=None):
    """Visualize sample images in a grid"""
    print("\nCreating visualization...")
    
    num_samples = len(samples)
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle('.', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Set background color
    fig.patch.set_facecolor('white')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Display image with better contrast
        ax.imshow(sample['image'], cmap='gray', aspect='auto')
        
        # Add colored border based on pathology
        pathology_display = sample['pathology_display']
        
        if pathology_display == 'MALIGNANT':
            color = '#DC143C'  # Crimson red
            bgcolor = '#FFE4E1'  # Misty rose
        elif pathology_display == 'BENIGN':
            color = '#FFB347'  # Pastel orange
            bgcolor = '#FFF8DC'  # Cornsilk
        else:  # NORMAL
            color = '#32CD32'  # Lime green
            bgcolor = '#F0FFF0'  # Honeydew
        
        # Create title with better formatting
        title = f"ID: {sample['patient_id']}\n"
        title += f"{sample['abnormality_type'].title()}\n"
        title += f"◉ {pathology_display}"
        
        # Style the subplot
        ax.set_title(title, fontsize=11, color=color, fontweight='bold', 
                    pad=10, bbox=dict(boxstyle='round,pad=0.5', 
                                     facecolor=bgcolor, edgecolor=color, linewidth=2))
        ax.axis('off')
        
        # Add border around image
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def plot_dataset_statistics(all_df, train_df, test_df, save_path=None):
    """Plot dataset statistics"""
    print("\nCreating statistics plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CBIS-DDSM Dataset Statistics', fontsize=18, fontweight='bold')
    fig.patch.set_facecolor('white')
    
    # 1. Pathology distribution with simplified labels
    ax = axes[0, 0]
    # Map to simplified labels
    pathology_map = {
        'MALIGNANT': 'Malignant',
        'BENIGN': 'Benign',
        'BENIGN_WITHOUT_CALLBACK': 'Normal'
    }
    pathology_simplified = all_df['pathology'].map(pathology_map)
    pathology_counts = pathology_simplified.value_counts()
    
    # Define colors
    color_map = {
        'Malignant': '#DC143C',
        'Benign': '#FFB347',
        'Normal': '#32CD32'
    }
    colors = [color_map.get(p, 'gray') for p in pathology_counts.index]
    
    bars = ax.bar(pathology_counts.index, pathology_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_title('Pathology Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Train vs Test split
    ax = axes[0, 1]
    split_data = pd.DataFrame({
        'Split': ['Training', 'Test'],
        'Count': [len(train_df), len(test_df)]
    })
    bars = ax.bar(split_data['Split'], split_data['Count'], 
                   color=['#4169E1', '#FF8C00'], edgecolor='black', 
                   linewidth=1.5, alpha=0.8)
    ax.set_title('Train/Test Split', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Abnormality type distribution
    ax = axes[0, 2]
    abn_counts = all_df['abnormality type'].value_counts()
    bars = ax.bar(abn_counts.index, abn_counts.values, 
                   color=['#9370DB', '#BA55D3'], edgecolor='black', 
                   linewidth=1.5, alpha=0.8)
    ax.set_title('Abnormality Type', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Breast side distribution
    ax = axes[1, 0]
    side_counts = all_df['left or right breast'].value_counts()
    colors_pie = ['#87CEEB', '#F08080']
    wedges, texts, autotexts = ax.pie(side_counts.values, labels=side_counts.index, 
                                        autopct='%1.1f%%', colors=colors_pie,
                                        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax.set_title('Left vs Right Breast', fontsize=14, fontweight='bold', pad=10)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # 5. Image view distribution
    ax = axes[1, 1]
    view_counts = all_df['image view'].value_counts()
    bars = ax.bar(view_counts.index, view_counts.values, 
                   color=['#20B2AA', '#48D1CC'], edgecolor='black', 
                   linewidth=1.5, alpha=0.8)
    ax.set_title('Mammogram Views', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('View Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Breast density distribution
    ax = axes[1, 2]
    if 'breast density' in all_df.columns:
        density_counts = all_df['breast density'].value_counts().sort_index()
        bars = ax.bar(density_counts.index, density_counts.values, 
                      color=['#BDB76B', '#DAA520', '#CD853F', '#8B4513'][:len(density_counts)],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_title('Breast Density', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Density Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    elif 'breast_density' in all_df.columns:
        density_counts = all_df['breast_density'].value_counts().sort_index()
        bars = ax.bar(density_counts.index, density_counts.values, 
                      color=['#BDB76B', '#DAA520', '#CD853F', '#8B4513'][:len(density_counts)],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_title('Breast Density', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Density Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Breast Density\nData Not Available', 
                ha='center', va='center', fontsize=13, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Breast Density', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


def print_dataset_summary(all_df, train_df, test_df):
    """Print detailed dataset summary"""
    print("\n" + "=" * 80)
    print("CBIS-DDSM DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Cases: {len(all_df)}")
    print(f"  - Training: {len(train_df)} ({len(train_df)/len(all_df)*100:.1f}%)")
    print(f"  - Test: {len(test_df)} ({len(test_df)/len(all_df)*100:.1f}%)")
    
    print("\nPathology Distribution (Simplified):")
    pathology_map = {
        'MALIGNANT': 'Malignant',
        'BENIGN': 'Benign',
        'BENIGN_WITHOUT_CALLBACK': 'Normal'
    }
    pathology_simplified = all_df['pathology'].map(pathology_map)
    for pathology, count in pathology_simplified.value_counts().items():
        print(f"  - {pathology}: {count} ({count/len(all_df)*100:.1f}%)")
    
    print("\nAbnormality Types:")
    for abn_type, count in all_df['abnormality type'].value_counts().items():
        print(f"  - {abn_type}: {count} ({count/len(all_df)*100:.1f}%)")
    
    print("\nBreast Side:")
    for side, count in all_df['left or right breast'].value_counts().items():
        print(f"  - {side}: {count} ({count/len(all_df)*100:.1f}%)")
    
    print("\nImage Views:")
    for view, count in all_df['image view'].value_counts().items():
        print(f"  - {view}: {count} ({count/len(all_df)*100:.1f}%)")
    
    print("\nUnique Patients: {}".format(all_df['patient_id'].nunique()))
    
    print("=" * 80)


def main():
    """Main visualization function"""
    print("=" * 80)
    print("CBIS-DDSM Dataset Visualization")
    print("=" * 80)
    
    # Create output directory
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    all_df, train_df, test_df = load_dataset_info()
    
    # Print summary
    print_dataset_summary(all_df, train_df, test_df)
    
    # Prepare samples
    print("\n" + "-" * 80)
    print("Preparing image samples...")
    print("-" * 80)
    samples = prepare_samples(all_df, num_samples=20, use_cropped=True)
    
    if samples:
        # Visualize samples
        sample_path = os.path.join(output_dir, "dataset_samples.png")
        visualize_samples(samples, save_path=sample_path)
    else:
        print("⚠ Warning: Could not load any sample images")
    
    # Plot statistics
    print("\n" + "-" * 80)
    print("Creating statistics plots...")
    print("-" * 80)
    stats_path = os.path.join(output_dir, "dataset_statistics.png")
    plot_dataset_statistics(all_df, train_df, test_df, save_path=stats_path)
    
    print("\n" + "=" * 80)
    print("✓ Visualization Complete!")
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

