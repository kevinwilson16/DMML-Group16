
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix

# Available models configuration
AVAILABLE_MODELS = {
    'svm': {
        'name': 'Support Vector Machine (SVM)',
        'model_path': 'models/svm_breast_cancer_model.pkl',
        'metadata_path': 'models/svm_model_metadata.pkl'
    },
    'dt': {
        'name': 'Decision Tree',
        'model_path': 'models/decision_tree_breast_cancer_model.pkl',
        'metadata_path': 'models/decision_tree_model_metadata.pkl'
    },
    'rf': {
        'name': 'Random Forest',
        'model_path': 'models/random_forest_model.pkl',
        'metadata_path': 'models/random_forest_metadata.pkl'
    },
    'nb': {
        'name': 'Naive Bayes',
        'model_path': 'models/naive_bayes_breast_cancer_model.pkl',
        'metadata_path': 'models/naive_bayes_metadata.pkl'
    },
    'lr': {
        'name': 'Logistic Regression',
        'model_path': 'models/logistic_regression_breast_cancer_model.pkl',
        'metadata_path': 'models/logistic_regression_metadata.pkl'
    },
    'km': {
        'name': 'K-Means Clustering',
        'model_path': 'models/kmeans_breast_cancer_model.pkl',
        'metadata_path': None  # K-Means is unsupervised; we typically don't have standard metadata
    }
}

def display_available_models():
    """Display available models for selection"""
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    for key, info in AVAILABLE_MODELS.items():
        model_exists = os.path.exists(info['model_path'])
        status = "✓ Available" if model_exists else "✗ Not Found"
        print(f"  [{key.upper()}] {info['name']:<35} {status}")
    print("="*70 + "\n")

def select_model():
    """Allow user to select which model to use"""
    display_available_models()
    
    valid_keys = "/".join(AVAILABLE_MODELS.keys())
    while True:
        choice = input(f"Select model ({valid_keys}) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice in AVAILABLE_MODELS:
            model_path = AVAILABLE_MODELS[choice]['model_path']
            if os.path.exists(model_path):
                return choice
            else:
                print(f"❌ Model file not found: {model_path}")
                print(f"   Please train the {AVAILABLE_MODELS[choice]['name']} model first.\n")
        else:
            print(f"❌ Invalid choice! Please enter one of: {valid_keys}.\n")

def load_model_and_data(model_type):
    """Load the selected model, scaler, and test data"""
    try:
        model_info = AVAILABLE_MODELS[model_type]
        
        # Load model
        model = joblib.load(model_info['model_path'])

        # Load metadata (optional; some models may not have metadata saved)
        metadata = {}
        metadata_path = model_info.get('metadata_path')
        if metadata_path:
            try:
                metadata = joblib.load(metadata_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not load metadata file ({metadata_path}): {e}")
                print("   Proceeding without training-time metrics.\n")

        scaler = joblib.load('data/processed/scaler.pkl')
        
        # Load test data
        X_test = pd.read_csv('data/processed/X_test_scaled.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

        # For K-Means (unsupervised), compute a simple cluster→label mapping
        # so that we can interpret clusters as benign/malignant.
        if model_type == 'km':
            try:
                cluster_ids = model.predict(X_test)
                cluster_to_label = {}
                for c in range(model.n_clusters):
                    mask = (cluster_ids == c)
                    if mask.sum() == 0:
                        # Default to benign if cluster empty in test set
                        cluster_to_label[c] = 0
                    else:
                        # Majority label in this cluster (0=benign, 1=malignant)
                        majority = int(round(y_test[mask].mean()))
                        cluster_to_label[c] = majority

                # Attach mapping to the model instance so other helpers can use it
                setattr(model, 'cluster_to_label', cluster_to_label)

                print("\nK-Means cluster → label mapping (based on test set):")
                for c, lab in cluster_to_label.items():
                    print(f"  Cluster {c} → Label {lab} (0=Benign, 1=Malignant)")
                print()
            except Exception as e:
                print(f"⚠️ Warning: Failed to compute cluster-to-label mapping for K-Means: {e}")
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✓ {model_info['name']} model and data loaded successfully!")
        
        # Display model metadata
        print("\n" + "="*70)
        print("MODEL INFORMATION")
        print("="*70)
        print(f"  Model Type: {metadata.get('model_type', model_info['name'])}")
        
        if model_type == 'svm':
            print(f"  Kernel: {metadata.get('kernel', 'N/A')}")
            print(f"  C Parameter: {metadata.get('C', 'N/A')}")
            print(f"  Gamma: {metadata.get('gamma', 'N/A')}")
        elif model_type == 'dt':
            print(f"  Criterion: {metadata.get('criterion', 'N/A')}")
            print(f"  Max Depth: {metadata.get('max_depth', 'N/A')}")
            print(f"  Tree Depth: {metadata.get('tree_depth', 'N/A')}")
            print(f"  Number of Leaves: {metadata.get('n_leaves', 'N/A')}")
        elif model_type == 'rf':
            print(f"  n_estimators: {metadata.get('n_estimators', 'N/A')}")
            print(f"  Max Depth: {metadata.get('max_depth', 'N/A')}")
            print(f"  Criterion: {metadata.get('criterion', 'N/A')}")
        elif model_type == 'nb':
            print(f"  Model Variant: {metadata.get('variant', 'Naive Bayes')}")
        elif model_type == 'lr':
            print(f"  Solver: {metadata.get('solver', 'N/A')}")
            print(f"  Class Weight: {metadata.get('class_weight', 'N/A')}")
            print(f"  C Parameter: {metadata.get('C', 'N/A')}")
        
        if model_type != 'km':
            print(f"\n  Performance Metrics (from training):")
            if metadata:
                print(f"    Accuracy:  {metadata.get('accuracy', 0)*100:.2f}%")
                print(f"    Precision: {metadata.get('precision', 0)*100:.2f}%")
                print(f"    Recall:    {metadata.get('recall', 0)*100:.2f}%")
                print(f"    F1-Score:  {metadata.get('f1_score', 0)*100:.2f}%")
                print(f"    ROC-AUC:   {metadata.get('roc_auc', 0)*100:.2f}%")
            else:
                print("    (Not available - metadata file not found.)")
        else:
            print("\n  Performance Metrics (from training):")
            print("    (Not available for unsupervised K-Means. We will compute stats on the test set.)")
        print("="*70)
        
        return model, scaler, X_test, y_test, feature_names, model_info['name']
    
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return None, None, None, None, None, None

def display_overall_stats(model, X_test, y_test, model_name):
    """Display overall model performance"""
    # Special handling for K-Means (unsupervised)
    if hasattr(model, 'cluster_to_label'):
        clusters = model.predict(X_test)
        mapping = getattr(model, 'cluster_to_label', {})
        y_pred = np.vectorize(mapping.get)(clusters)
    else:
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*70)
    print(f"OVERALL {model_name.upper()} PERFORMANCE ON TEST SET")
    print("="*70)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Benign:     {cm[0][0]}")
    print(f"  False Malignant: {cm[0][1]}")
    print(f"  False Benign:    {cm[1][0]}")
    print(f"  True Malignant:  {cm[1][1]}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print("="*70 + "\n")

def test_specific_row(model, X_test, y_test, feature_names, row_index, model_name):
    """Test the model on a specific row"""
    
    if row_index < 0 or row_index >= len(X_test):
        print(f"❌ Invalid row index! Please choose between 0 and {len(X_test)-1}")
        return
    
    # Map Python index to Excel row number (account for header row)
    # If you exported X_test to Excel, row 2 corresponds to index 0, row 3 to index 1, etc.
    excel_row = row_index + 2

    # Get the sample
    sample = X_test.iloc[row_index].values.reshape(1, -1)
    actual = y_test[row_index]
    
    # Make prediction
    if hasattr(model, 'cluster_to_label'):
        # K-Means path: map clusters to labels, and use a simple 0/1 one-hot as "probability"
        cluster = model.predict(sample)[0]
        mapping = getattr(model, 'cluster_to_label', {})
        prediction = mapping.get(cluster, 0)
        probability = np.zeros(2)
        probability[prediction] = 1.0
    else:
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
    print(f"TEST RESULT FOR ROW #{row_index} (Excel row {excel_row}) - {model_name}")
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

def test_multiple_rows(model, X_test, y_test, feature_names, row_indices, model_name):
    """Test multiple rows at once"""
    
    print("\n" + "="*70)
    print(f"BATCH TEST RESULTS FOR {len(row_indices)} SAMPLES - {model_name}")
    print("="*70)
    print("\nNote: Excel row number = index + 2 (because row 1 is the header).")
    print(f"\n{'Idx':<6} {'ExcelRow':<10} {'Actual':<20} {'Predicted':<20} {'Confidence':<12} {'Status':<10}")
    print("-"*70)
    
    correct_count = 0
    for idx in row_indices:
        if idx < 0 or idx >= len(X_test):
            print(f"{idx:<6} Invalid row index!")
            continue
        
        excel_row = idx + 2  # Excel row mapping
        sample = X_test.iloc[idx].values.reshape(1, -1)
        actual = y_test[idx]

        if hasattr(model, 'cluster_to_label'):
            cluster = model.predict(sample)[0]
            mapping = getattr(model, 'cluster_to_label', {})
            prediction = mapping.get(cluster, 0)
            probability = np.zeros(2)
            probability[prediction] = 1.0
        else:
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]
        
        actual_label = "Malignant" if actual == 1 else "Benign"
        predicted_label = "Malignant" if prediction == 1 else "Benign"
        confidence = probability[prediction] * 100
        is_correct = actual == prediction
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            correct_count += 1
        
        print(f"{idx:<6} {excel_row:<10} {actual_label:<20} {predicted_label:<20} {confidence:>6.2f}%      {status}")
    
    batch_accuracy = (correct_count / len(row_indices)) * 100
    print("-"*70)
    print(f"\nBatch Accuracy: {correct_count}/{len(row_indices)} ({batch_accuracy:.2f}%)")
    print("="*70 + "\n")

def test_custom_input(model, scaler, feature_names, model_name):
    """Test with manually input feature values"""
    
    print("\n" + "="*70)
    print("CUSTOM INPUT - MANUAL FEATURE ENTRY")
    print("="*70)
    print("\nYou will be asked to enter all 30 feature values.")
    print("Tip: You can also enter values separated by commas all at once.\n")
    
    # Check if there's an example file
    example_file = 'example_custom_values.txt'
    if os.path.exists(example_file):
        print(f"📄 Found example file: {example_file}")
        use_file = input("Load values from this file? (y/n): ").strip().lower()
        if use_file == 'y':
            try:
                with open(example_file, 'r') as f:
                    content = f.read().strip()
                    features = [float(x.strip()) for x in content.replace('\n', ',').split(',') if x.strip()]
                    if len(features) == 30:
                        print(f"✓ Loaded {len(features)} feature values from file")
                    else:
                        print(f"❌ File contains {len(features)} values, expected 30")
                        return
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return
        else:
            features = get_manual_input(feature_names)
    else:
        features = get_manual_input(feature_names)
    
    if features is None:
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
    if hasattr(model, 'cluster_to_label'):
        cluster = model.predict(features_scaled)[0]
        mapping = getattr(model, 'cluster_to_label', {})
        prediction = mapping.get(cluster, 0)
        probability = np.zeros(2)
        probability[prediction] = 1.0
    else:
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
    
    # Display results
    predicted_label = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"
    confidence = probability[prediction] * 100
    
    print("\n" + "="*70)
    print(f"PREDICTION RESULT FOR CUSTOM INPUT - {model_name}")
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

def get_manual_input(feature_names):
    """Get manual input from user"""
    print("\nChoose input method:")
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
                return None
        except ValueError:
            print("❌ Error: Please enter valid numbers!")
            return None
    
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
        return None
    
    return features

def compare_models_on_row(models_data, row_index):
    """Compare predictions from multiple models on the same row"""
    
    print("\n" + "="*70)
    print(f"MODEL COMPARISON FOR ROW #{row_index}")
    print("="*70)
    
    results = []
    actual = None
    
    for model_type, (model, X_test, y_test, model_name) in models_data.items():
        if row_index >= len(X_test):
            continue
            
        sample = X_test.iloc[row_index].values.reshape(1, -1)
        actual = y_test[row_index]

        if hasattr(model, 'cluster_to_label'):
            cluster = model.predict(sample)[0]
            mapping = getattr(model, 'cluster_to_label', {})
            prediction = mapping.get(cluster, 0)
            probability = np.zeros(2)
            probability[prediction] = 1.0
        else:
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]
        
        results.append({
            'model': model_name,
            'prediction': prediction,
            'confidence': probability[prediction] * 100,
            'prob_benign': probability[0] * 100,
            'prob_malignant': probability[1] * 100
        })
    
    if actual is not None:
        actual_label = "Malignant (Cancer)" if actual == 1 else "Benign (No Cancer)"
        print(f"\nActual Diagnosis: {actual_label}\n")
    
    print(f"{'Model':<30} {'Prediction':<20} {'Confidence':<12} {'Benign %':<12} {'Malignant %'}")
    print("-"*70)
    
    for r in results:
        pred_label = "Malignant" if r['prediction'] == 1 else "Benign"
        correct = "✓" if r['prediction'] == actual else "✗"
        print(f"{r['model']:<30} {pred_label:<20} {r['confidence']:>6.2f}% {correct:>3}  "
              f"{r['prob_benign']:>6.2f}%     {r['prob_malignant']:>6.2f}%")
    
    print("="*70 + "\n")

def interactive_mode(model, scaler, X_test, y_test, feature_names, model_name, model_type):
    """Interactive mode for testing"""
    
    while True:
        print("\n" + "="*70)
        print(f"INTERACTIVE MODEL TESTING - {model_name}")
        print("="*70)
        print("\nOptions:")
        print("  1. Test a specific row (by index; Excel row = index + 2)")
        print("  2. Test multiple rows (comma-separated indices)")
        print("  3. Test random rows")
        print("  4. Input custom values for detection")
        print("  5. Test whole test dataset (overall accuracy)")
        print("  6. Switch model")
        print("  7. Exit")
        print("\n" + "-"*70)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            try:
                row_index = int(input(f"\nEnter row index (0-{len(X_test)-1}): "))
                test_specific_row(model, X_test, y_test, feature_names, row_index, model_name)
            except ValueError:
                print("❌ Please enter a valid number!")
        
        elif choice == '2':
            try:
                rows_input = input(f"\nEnter row indices separated by commas (e.g., 0,5,10): ")
                row_indices = [int(x.strip()) for x in rows_input.split(',')]
                test_multiple_rows(model, X_test, y_test, feature_names, row_indices, model_name)
            except ValueError:
                print("❌ Please enter valid numbers separated by commas!")
        
        elif choice == '3':
            try:
                n_samples = int(input("\nHow many random rows to test? "))
                random_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
                test_multiple_rows(model, X_test, y_test, feature_names, random_indices, model_name)
            except ValueError:
                print("❌ Please enter a valid number!")
        
        elif choice == '4':
            test_custom_input(model, scaler, feature_names, model_name)
        
        elif choice == '5':
            display_overall_stats(model, X_test, y_test, model_name)
        
        elif choice == '6':
            print("\n🔄 Switching model...")
            return 'switch'
        
        elif choice == '7':
            print("\n👋 Thank you for using Model Testing!")
            print("="*70 + "\n")
            return 'exit'
        
        else:
            print("❌ Invalid choice! Please enter 1-7.")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("UCI BREAST CANCER - UNIFIED MODEL TESTING")
    print("="*70)
    print("\nThis tool allows you to test both SVM and Decision Tree models")
    print("on the UCI Breast Cancer dataset.\n")
    
    while True:
        # Select model
        model_type = select_model()
        
        if model_type is None:
            print("\n👋 Exiting...")
            break
        
        print(f"\nLoading {AVAILABLE_MODELS[model_type]['name']} model...")
        
        # Load model and data
        model, scaler, X_test, y_test, feature_names, model_name = load_model_and_data(model_type)
        
        if model is None:
            print("\n❌ Failed to load model or data. Please ensure:")
            print("   1. You've trained the model")
            print("   2. The models/ and data/processed/ directories exist")
            continue
        
        print(f"\nTest dataset contains {len(X_test)} samples")
        print(f"Features: {len(feature_names)} features")
        
        # Start interactive mode
        result = interactive_mode(model, scaler, X_test, y_test, feature_names, model_name, model_type)
        
        if result == 'exit':
            break

if __name__ == "__main__":
    main()

