"""
SVM Model Testing Script
Author: Interactive Breast Cancer Diagnosis Testing
Description: Test the trained SVM model on specific rows from the test dataset
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model_and_data():
    """Load the trained model, scaler, and test data"""
    try:
        # Load model and scaler
        model = joblib.load('models/svm_breast_cancer_model.pkl')
        scaler = joblib.load('data/processed/scaler.pkl')
        
        # Load test data
        X_test = pd.read_csv('data/processed/X_test_scaled.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print("✓ Model and data loaded successfully!")
        return model, scaler, X_test, y_test, feature_names
    
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None, None, None

def display_overall_stats(model, X_test, y_test):
    """Display overall model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE ON TEST SET")
    print("="*70)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Benign:  {cm[0][0]}")
    print(f"  False Malignant: {cm[0][1]}")
    print(f"  False Benign: {cm[1][0]}")
    print(f"  True Malignant: {cm[1][1]}")
    print("="*70 + "\n")

def test_specific_row(model, X_test, y_test, feature_names, row_index):
    """Test the model on a specific row"""
    
    if row_index < 0 or row_index >= len(X_test):
        print(f"❌ Invalid row index! Please choose between 0 and {len(X_test)-1}")
        return
    
    # Get the sample
    sample = X_test.iloc[row_index].values.reshape(1, -1)
    actual = y_test[row_index]
    
    # Make prediction
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]
    
    # Format results
    actual_label = "Malignant (Cancer)" if actual == 1 else "Benign (No Cancer)"
    predicted_label = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"
    confidence = probability[prediction] * 100
    
    # Determine if prediction is correct
    is_correct = actual == prediction
    result_symbol = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    result_color = "🟢" if is_correct else "🔴"
    
    # Display results
    print("\n" + "="*70)
    print(f"TEST RESULT FOR ROW #{row_index}")
    print("="*70)
    print(f"\n{result_color} PREDICTION STATUS: {result_symbol}")
    print(f"\n  Actual Diagnosis:    {actual_label}")
    print(f"  Predicted Diagnosis: {predicted_label}")
    print(f"  Confidence:          {confidence:.2f}%")
    print(f"\n  Probability Breakdown:")
    print(f"    Benign:    {probability[0]*100:>6.2f}%")
    print(f"    Malignant: {probability[1]*100:>6.2f}%")
    
    # Show top 5 most important features for this sample
    print(f"\n  Top 5 Feature Values for this sample:")
    sample_features = X_test.iloc[row_index]
    top_features = sample_features.abs().nlargest(5)
    for i, (feat_name, value) in enumerate(top_features.items(), 1):
        print(f"    {i}. {feat_name}: {value:.4f}")
    
    print("="*70 + "\n")

def test_multiple_rows(model, X_test, y_test, feature_names, row_indices):
    """Test multiple rows at once"""
    
    print("\n" + "="*70)
    print(f"BATCH TEST RESULTS FOR {len(row_indices)} SAMPLES")
    print("="*70)
    print(f"\n{'Row':<6} {'Actual':<20} {'Predicted':<20} {'Confidence':<12} {'Status':<10}")
    print("-"*70)
    
    correct_count = 0
    for idx in row_indices:
        if idx < 0 or idx >= len(X_test):
            print(f"{idx:<6} Invalid row index!")
            continue
        
        sample = X_test.iloc[idx].values.reshape(1, -1)
        actual = y_test[idx]
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]
        
        actual_label = "Malignant" if actual == 1 else "Benign"
        predicted_label = "Malignant" if prediction == 1 else "Benign"
        confidence = probability[prediction] * 100
        is_correct = actual == prediction
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            correct_count += 1
        
        print(f"{idx:<6} {actual_label:<20} {predicted_label:<20} {confidence:>6.2f}%      {status}")
    
    batch_accuracy = (correct_count / len(row_indices)) * 100
    print("-"*70)
    print(f"\nBatch Accuracy: {correct_count}/{len(row_indices)} ({batch_accuracy:.2f}%)")
    print("="*70 + "\n")

def test_custom_input(model, scaler, feature_names):
    """Test with manually input feature values"""
    
    print("\n" + "="*70)
    print("CUSTOM INPUT - MANUAL FEATURE ENTRY")
    print("="*70)
    print("\nYou will be asked to enter all 30 feature values.")
    print("Tip: You can also enter values separated by commas all at once.\n")
    
    # Option 1: Enter all at once
    print("Choose input method:")
    print("  1. Enter all 30 values at once (comma-separated)")
    print("  2. Enter each value one by one")
    
    method = input("\nEnter method (1 or 2): ").strip()
    
    features = []
    
    if method == '1':
        print("\nEnter all 30 feature values separated by commas:")
        print("Example: 17.99, 10.38, 122.8, 1001, 0.1184, ...")
        try:
            values_input = input("\nValues: ")
            features = [float(x.strip()) for x in values_input.split(',')]
            
            if len(features) != 30:
                print(f"❌ Error: Expected 30 values, got {len(features)}")
                return
        except ValueError:
            print("❌ Error: Please enter valid numbers!")
            return
    
    elif method == '2':
        print("\n📝 Enter values for each feature:")
        print("-"*70)
        for i, feat_name in enumerate(feature_names, 1):
            while True:
                try:
                    value = float(input(f"{i}. {feat_name}: "))
                    features.append(value)
                    break
                except ValueError:
                    print("   ❌ Please enter a valid number!")
    else:
        print("❌ Invalid method choice!")
        return
    
    # Convert to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features
    try:
        features_scaled = scaler.transform(features_array)
    except Exception as e:
        print(f"❌ Error scaling features: {e}")
        return
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Display results
    predicted_label = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"
    confidence = probability[prediction] * 100
    
    print("\n" + "="*70)
    print("PREDICTION RESULT FOR CUSTOM INPUT")
    print("="*70)
    print(f"\n🔍 Predicted Diagnosis: {predicted_label}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\n  Probability Breakdown:")
    print(f"    Benign:    {probability[0]*100:>6.2f}%")
    print(f"    Malignant: {probability[1]*100:>6.2f}%")
    
    # Show feature values entered
    print(f"\n📋 Feature Values You Entered:")
    print("-"*70)
    for feat_name, value in zip(feature_names, features):
        print(f"  {feat_name:<35} {value:>15.6f}")
    
    print("="*70 + "\n")

def interactive_mode(model, scaler, X_test, y_test, feature_names):
    """Interactive mode for testing specific rows"""
    
    while True:
        print("\n" + "="*70)
        print("INTERACTIVE SVM MODEL TESTING")
        print("="*70)
        print("\nOptions:")
        print("  1. Test a specific row")
        print("  2. Test multiple rows (comma-separated)")
        print("  3. Test random rows")
        print("  4. Input custom values for detection")
        print("  5. Show overall statistics")
        print("  6. Exit")
        print("\n" + "-"*70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            try:
                row_index = int(input(f"\nEnter row index (0-{len(X_test)-1}): "))
                test_specific_row(model, X_test, y_test, feature_names, row_index)
            except ValueError:
                print("❌ Please enter a valid number!")
        
        elif choice == '2':
            try:
                rows_input = input(f"\nEnter row indices separated by commas (e.g., 0,5,10): ")
                row_indices = [int(x.strip()) for x in rows_input.split(',')]
                test_multiple_rows(model, X_test, y_test, feature_names, row_indices)
            except ValueError:
                print("❌ Please enter valid numbers separated by commas!")
        
        elif choice == '3':
            try:
                n_samples = int(input("\nHow many random rows to test? "))
                random_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
                test_multiple_rows(model, X_test, y_test, feature_names, random_indices)
            except ValueError:
                print("❌ Please enter a valid number!")
        
        elif choice == '4':
            test_custom_input(model, scaler, feature_names)
        
        elif choice == '5':
            display_overall_stats(model, X_test, y_test)
        
        elif choice == '6':
            print("\n👋 Thank you for using SVM Model Testing!")
            print("="*70 + "\n")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-6.")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("UCI BREAST CANCER - SVM MODEL TESTING")
    print("="*70)
    print("\nLoading model and data...")
    
    # Load everything
    model, scaler, X_test, y_test, feature_names = load_model_and_data()
    
    if model is None:
        print("\n❌ Failed to load model or data. Please ensure:")
        print("   1. You've trained the model (run uci_svm_training.ipynb)")
        print("   2. The models/ and data/processed/ directories exist")
        return
    
    print(f"\nTest dataset contains {len(X_test)} samples")
    print(f"Features: {len(feature_names)} features")
    
    # Start interactive mode
    interactive_mode(model, scaler, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()

