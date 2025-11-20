"""
CBIS-DDSM Breast Cancer Prediction Script
Use trained model to make predictions on new mammogram images
"""

import os
import numpy as np
import cv2
import argparse
from tensorflow import keras
import tensorflow as tf

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise Exception(f"Error processing image {img_path}: {str(e)}")


def predict_single_image(model, img_path, threshold=0.5):
    """
    Predict on a single image
    
    Args:
        model: Trained Keras model
        img_path: Path to image file
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Prediction results
    """
    # Load and preprocess image
    img = load_and_preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img, verbose=0)
    
    # Extract probabilities
    benign_prob = prediction[0][0]
    malignant_prob = prediction[0][1]
    
    # Determine class
    if malignant_prob > threshold:
        predicted_class = "MALIGNANT"
        confidence = malignant_prob
    else:
        predicted_class = "BENIGN"
        confidence = benign_prob
    
    results = {
        'image_path': img_path,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'benign_probability': float(benign_prob),
        'malignant_probability': float(malignant_prob)
    }
    
    return results


def predict_batch(model, image_paths, threshold=0.5):
    """
    Predict on multiple images
    
    Args:
        model: Trained Keras model
        image_paths: List of image file paths
        threshold: Classification threshold (default 0.5)
    
    Returns:
        list: List of prediction results for each image
    """
    results = []
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, img_path in enumerate(image_paths):
        try:
            result = predict_single_image(model, img_path, threshold)
            results.append(result)
            
            print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}: "
                  f"{result['predicted_class']} ({result['confidence']:.4f})")
        except Exception as e:
            print(f"[{i+1}/{len(image_paths)}] Error processing {img_path}: {str(e)}")
            results.append({
                'image_path': img_path,
                'error': str(e)
            })
    
    return results


def print_results(results):
    """
    Print prediction results in a formatted way
    """
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Image: {os.path.basename(result['image_path'])}")
        
        if 'error' in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Predicted Class: {result['predicted_class']}")
            print(f"    Confidence: {result['confidence']:.4f}")
            print(f"    Benign Probability: {result['benign_probability']:.4f}")
            print(f"    Malignant Probability: {result['malignant_probability']:.4f}")
    
    print("\n" + "=" * 80)
    
    # Summary statistics
    successful = [r for r in results if 'error' not in r]
    if successful:
        benign_count = sum(1 for r in successful if r['predicted_class'] == 'BENIGN')
        malignant_count = sum(1 for r in successful if r['predicted_class'] == 'MALIGNANT')
        
        print("\nSUMMARY:")
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {len(successful)}")
        print(f"Predicted as BENIGN: {benign_count}")
        print(f"Predicted as MALIGNANT: {malignant_count}")
        
        avg_confidence = np.mean([r['confidence'] for r in successful])
        print(f"Average confidence: {avg_confidence:.4f}")
    
    print("=" * 80)


def main():
    """
    Main prediction function
    """
    parser = argparse.ArgumentParser(description='CBIS-DDSM Breast Cancer Prediction')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5 file)')
    parser.add_argument('--image', type=str, 
                        help='Path to single image file')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--output', type=str,
                        help='Path to save results as CSV file')
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    if args.image and args.image_dir:
        parser.error("Specify either --image or --image_dir, not both")
    
    # Check GPU availability
    print("\nChecking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
    else:
        print("No GPU found. Inference will run on CPU.")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    try:
        model = keras.models.load_model(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Collect image paths
    image_paths = []
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        image_paths = [args.image]
    
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm')
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        if not image_paths:
            print(f"No image files found in: {args.image_dir}")
            return
    
    # Make predictions
    results = predict_batch(model, image_paths, args.threshold)
    
    # Print results
    print_results(results)
    
    # Save results to CSV if requested
    if args.output:
        import pandas as pd
        
        # Filter out errors
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            df = pd.DataFrame(successful_results)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        else:
            print("\nNo successful predictions to save.")


if __name__ == "__main__":
    main()

