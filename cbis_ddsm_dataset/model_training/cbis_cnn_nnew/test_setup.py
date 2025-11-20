"""
Quick test script to verify dataset setup before training
"""

import os
import pandas as pd
import cv2
import numpy as np

# Configuration - WSL Linux format
BASE_PATH = "/mnt/c/Users/gaura/Desktop/github/DMML/cbis_ddsm_dataset/model_training/cbis_cnn_nnew/archive (4)"
JPEG_PATH = os.path.join(BASE_PATH, "jpeg")
CSV_PATH = os.path.join(BASE_PATH, "csv")

def test_directory_structure():
    """Test if all required directories exist"""
    print("\n" + "=" * 80)
    print("TEST 1: Directory Structure")
    print("=" * 80)
    
    checks = {
        "Base directory": os.path.exists(BASE_PATH),
        "JPEG directory": os.path.exists(JPEG_PATH),
        "CSV directory": os.path.exists(CSV_PATH),
    }
    
    for check, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {check}")
    
    return all(checks.values())


def test_csv_files():
    """Test if all required CSV files exist and are readable"""
    print("\n" + "=" * 80)
    print("TEST 2: CSV Files")
    print("=" * 80)
    
    csv_files = [
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "meta.csv",
        "dicom_info.csv"
    ]
    
    all_exist = True
    
    for csv_file in csv_files:
        path = os.path.join(CSV_PATH, csv_file)
        exists = os.path.exists(path)
        
        if exists:
            try:
                df = pd.read_csv(path)
                print(f"[PASS]: {csv_file} ({len(df)} rows)")
            except Exception as e:
                print(f"[FAIL]: {csv_file} - Error reading: {str(e)}")
                all_exist = False
        else:
            print(f"[FAIL]: {csv_file} - File not found")
            all_exist = False
    
    return all_exist


def test_jpeg_images():
    """Test if JPEG images are accessible"""
    print("\n" + "=" * 80)
    print("TEST 3: JPEG Images")
    print("=" * 80)
    
    # Count JPEG files
    jpeg_count = 0
    for root, dirs, files in os.walk(JPEG_PATH):
        jpeg_count += sum(1 for f in files if f.endswith('.jpg'))
    
    print(f"Total JPEG files found: {jpeg_count}")
    
    if jpeg_count > 0:
        print("[PASS]: JPEG images found")
        
        # Try to load a sample image
        for root, dirs, files in os.walk(JPEG_PATH):
            jpg_files = [f for f in files if f.endswith('.jpg')]
            if jpg_files:
                sample_path = os.path.join(root, jpg_files[0])
                try:
                    img = cv2.imread(sample_path)
                    if img is not None:
                        print(f"[PASS]: Successfully loaded sample image")
                        print(f"        Path: {sample_path}")
                        print(f"        Shape: {img.shape}")
                        return True
                    else:
                        print(f"[FAIL]: Could not read sample image: {sample_path}")
                        return False
                except Exception as e:
                    print(f"[FAIL]: Error loading sample image: {str(e)}")
                    return False
    else:
        print("[FAIL]: No JPEG images found")
        return False
    
    return False


def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "=" * 80)
    print("TEST 4: Data Loading")
    print("=" * 80)
    
    try:
        # Load CSV files
        calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
        mass_train = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_train_set.csv"))
        
        print(f"[PASS] Loaded calcification training data: {len(calc_train)} cases")
        print(f"[PASS] Loaded mass training data: {len(mass_train)} cases")
        
        # Check for required columns
        required_columns = ['patient_id', 'pathology', 'cropped image file path']
        
        for col in required_columns:
            if col in calc_train.columns:
                print(f"[PASS] Column '{col}' found in calc_train")
            else:
                print(f"[FAIL] Column '{col}' missing in calc_train")
                return False
        
        # Check pathology distribution
        print("\nPathology distribution in training data:")
        combined = pd.concat([calc_train, mass_train])
        print(combined['pathology'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"[FAIL]: Error loading data: {str(e)}")
        return False


def test_image_path_mapping():
    """Test if we can map DICOM paths to JPEG paths"""
    print("\n" + "=" * 80)
    print("TEST 5: Image Path Mapping")
    print("=" * 80)
    
    try:
        # Load a small sample
        calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
        
        # Get first cropped image path
        sample_path = calc_train['cropped image file path'].iloc[0]
        print(f"Sample DICOM path: {sample_path}")
        
        # Clean the path
        cleaned = str(sample_path).strip().replace('"', '').replace('\n', '')
        print(f"Cleaned path: {cleaned}")
        
        # Extract SeriesInstanceUID
        parts = cleaned.split('/')
        if len(parts) >= 2:
            series_uid = parts[-2]
            print(f"Series UID: {series_uid}")
            
            # Check if corresponding JPEG directory exists
            jpeg_dir = os.path.join(JPEG_PATH, series_uid)
            
            if os.path.exists(jpeg_dir):
                jpg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
                if jpg_files:
                    jpeg_path = os.path.join(jpeg_dir, jpg_files[0])
                    print(f"[PASS]: Found corresponding JPEG: {jpeg_path}")
                    
                    # Try to load it
                    img = cv2.imread(jpeg_path)
                    if img is not None:
                        print(f"[PASS]: Successfully loaded image with shape: {img.shape}")
                        return True
                    else:
                        print(f"[FAIL]: Could not read JPEG image")
                        return False
                else:
                    print(f"[FAIL]: No JPEG files in directory: {jpeg_dir}")
                    return False
            else:
                print(f"[FAIL]: JPEG directory not found: {jpeg_dir}")
                return False
        else:
            print(f"[FAIL]: Could not parse DICOM path")
            return False
            
    except Exception as e:
        print(f"[FAIL]: Error in path mapping: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tensorflow():
    """Test TensorFlow/Keras installation"""
    print("\n" + "=" * 80)
    print("TEST 6: TensorFlow/Keras")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print(f"[PASS] TensorFlow version: {tf.__version__}")
        print(f"[PASS] Keras version: {keras.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[PASS] GPU(s) available: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("[WARNING] No GPU found. Training will be slow on CPU.")
        
        return True
        
    except Exception as e:
        print(f"[FAIL]: Error with TensorFlow: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CBIS-DDSM DATASET SETUP VERIFICATION")
    print("=" * 80)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("CSV Files", test_csv_files),
        ("JPEG Images", test_jpeg_images),
        ("Data Loading", test_data_loading),
        ("Image Path Mapping", test_image_path_mapping),
        ("TensorFlow/Keras", test_tensorflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL]: {test_name} - Unexpected error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! You're ready to start training.")
        print("\nRun: python train_cbis_cnn.py")
    else:
        print("\n[ERROR] Some tests failed. Please fix the issues before training.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

