# Getting Started - Step by Step Guide

## 🎯 Overview

This guide will walk you through the complete process of training a breast cancer classification model using the CBIS-DDSM dataset, from installation to making predictions.

**Estimated Time**: 10 minutes setup + 2-4 hours training (with GPU)

---

## ✅ Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] pip (Python package manager) installed
- [ ] At least 20 GB free disk space
- [ ] 16 GB RAM (32 GB recommended)
- [ ] NVIDIA GPU with CUDA support (optional but recommended)
- [ ] The CBIS-DDSM dataset downloaded and extracted in `archive (4)/` directory

---

## 📋 Step-by-Step Instructions

### Step 1: Install Dependencies (5 minutes)

Open your terminal/command prompt and navigate to the project directory:

```bash
cd c:\Users\gaura\Desktop\github\DMML\cbis_ddsm_dataset\model_training\cbis_cnn_nnew
```

Install required packages:

```bash
pip install -r requirements.txt
```

**Expected Output**:
```
Collecting tensorflow==2.15.0
Collecting keras==2.15.0
...
Successfully installed tensorflow-2.15.0 keras-2.15.0 ...
```

**Troubleshooting**:
- If you get permission errors, try: `pip install --user -r requirements.txt`
- If you're behind a proxy, use: `pip install -r requirements.txt --proxy your_proxy`

---

### Step 2: Verify Your Setup (2 minutes)

Run the setup verification script:

```bash
python test_setup.py
```

**Expected Output**:
```
================================================================================
CBIS-DDSM DATASET SETUP VERIFICATION
================================================================================

TEST 1: Directory Structure
================================================================================
✓ PASS: Base directory
✓ PASS: JPEG directory
✓ PASS: CSV directory

TEST 2: CSV Files
================================================================================
✓ PASS: calc_case_description_train_set.csv (3093 rows)
✓ PASS: calc_case_description_test_set.csv (763 rows)
...

TOTAL: 6/6 tests passed

✓ All tests passed! You're ready to start training.
```

**If Tests Fail**:
- Check that `archive (4)/` directory exists and contains `csv/` and `jpeg/` folders
- Verify CSV files are not corrupted
- Ensure JPEG images are present in the `jpeg/` directory

---

### Step 3: Explore the Dataset (Optional, 3 minutes)

Visualize sample images and statistics:

```bash
python visualize_dataset.py
```

This will:
1. Display dataset statistics in the terminal
2. Create a grid of 20 sample mammogram images
3. Generate statistical plots
4. Save all visualizations to `visualizations/` directory

**Output Files**:
- `visualizations/dataset_samples.png` - Sample images with labels
- `visualizations/dataset_statistics.png` - Statistical plots

**What You'll See**:
- Pathology distribution (Benign vs Malignant)
- Abnormality types (Calcification vs Mass)
- Train/Test split
- Image view distribution (CC, MLO)
- Sample mammogram images with labels

---

### Step 4: Train the Model (2-20 hours)

Now you're ready to train! Run the training script:

```bash
python train_cbis_cnn.py
```

**Training Progress**:

You'll see output like this:
```
================================================================================
CBIS-DDSM Breast Cancer Classification Training
================================================================================

[1/7] Loading dataset information...
Calcification Training Cases: 3093
Mass Training Cases: 1318
...

[2/7] Preparing image paths...
Found 4411 valid images
Benign: 2416, Malignant: 1995

[3/7] Loading images into memory...
Loaded 4411 images

[4/7] Setting up data augmentation...

[5/7] Building model...
Base model: inception_resnet_v2
Total layers: 572
Trainable layers: 10

[6/7] Training model...
Epoch 1/50
157/157 [==============================] - 120s - loss: 0.5234 - accuracy: 0.7456
Epoch 2/50
157/157 [==============================] - 95s - loss: 0.4123 - accuracy: 0.8123
...
```

**Training Time Estimates**:
- **With GPU (RTX 3080/4090)**: 2-4 hours
- **With GPU (GTX 1080)**: 4-6 hours
- **With CPU**: 15-20 hours

**What's Happening**:
1. Loading dataset metadata from CSV files
2. Converting DICOM paths to JPEG paths
3. Loading all images into memory
4. Setting up data augmentation
5. Building InceptionResNetV2 model
6. Training with callbacks (EarlyStopping, ModelCheckpoint)
7. Evaluating on test set
8. Generating plots and reports

**Monitoring Training**:

Option 1 - Watch terminal output:
- Watch accuracy and loss decrease each epoch
- Training will stop early if no improvement for 10 epochs

Option 2 - Use TensorBoard:
```bash
# Open a new terminal
tensorboard --logdir=outputs/logs
# Open browser: http://localhost:6006
```

**Don't Interrupt Training**:
- Let it run until completion or early stopping
- Models are saved automatically at checkpoints
- If interrupted, you'll lose current training progress

---

### Step 5: Review Results (5 minutes)

After training completes, check the outputs:

```bash
outputs/
├── models/
│   ├── best_model_20241120_143052.h5      ← Best model (use this!)
│   └── final_model_20241120_143052.h5     ← Final model
├── plots/
│   ├── training_history_20241120_143052.png   ← Training curves
│   ├── confusion_matrix_20241120_143052.png   ← Confusion matrix
│   └── roc_curve_20241120_143052.png          ← ROC curve
└── classification_report_20241120_143052.csv  ← Metrics
```

**Review Training History**:
Open `outputs/plots/training_history_[timestamp].png`:
- Check if accuracy is increasing
- Check if loss is decreasing
- Verify no overfitting (train/val gap)
- Look at AUC, precision, recall trends

**Review Confusion Matrix**:
Open `outputs/plots/confusion_matrix_[timestamp].png`:
- See true positives and true negatives (diagonal)
- See false positives and false negatives (off-diagonal)
- Evaluate model's classification performance

**Review ROC Curve**:
Open `outputs/plots/roc_curve_[timestamp].png`:
- AUC score should be > 0.80
- Higher AUC = better model performance

**Check Classification Report**:
Open `outputs/classification_report_[timestamp].csv`:
```
                precision  recall  f1-score  support
Benign            0.84      0.87     0.85      600
Malignant         0.81      0.77     0.79      450
```

---

### Step 6: Make Predictions (1 minute)

Use your trained model to classify new mammograms:

#### Predict on Single Image

```bash
python predict.py --model outputs/models/best_model_20241120_143052.h5 --image path/to/mammogram.jpg
```

**Output**:
```
================================================================================
PREDICTION RESULTS
================================================================================

[1] Image: mammogram.jpg
    Predicted Class: MALIGNANT
    Confidence: 0.8732
    Benign Probability: 0.1268
    Malignant Probability: 0.8732

================================================================================
```

#### Predict on Multiple Images

```bash
python predict.py --model outputs/models/best_model_20241120_143052.h5 --image_dir path/to/images/
```

**Output**:
```
Processing 10 images...
[1/10] image1.jpg: BENIGN (0.9234)
[2/10] image2.jpg: MALIGNANT (0.8567)
[3/10] image3.jpg: BENIGN (0.7891)
...

SUMMARY:
Total images processed: 10
Predicted as BENIGN: 6
Predicted as MALIGNANT: 4
Average confidence: 0.8523
```

#### Save Results to CSV

```bash
python predict.py --model outputs/models/best_model_20241120_143052.h5 --image_dir path/to/images/ --output results.csv
```

This creates a CSV file with:
- Image path
- Predicted class
- Confidence score
- Individual probabilities

---

## 🎓 What You've Learned

After completing this guide, you've:

✅ Installed all dependencies  
✅ Verified your dataset setup  
✅ Explored the CBIS-DDSM dataset  
✅ Trained a deep learning model for breast cancer classification  
✅ Evaluated model performance with multiple metrics  
✅ Made predictions on new mammogram images  

---

## 🚀 Next Steps

### Beginner Level
1. **Experiment with thresholds**: Try different classification thresholds (0.3, 0.5, 0.7)
2. **Test more images**: Use the prediction script on various test images
3. **Understand the metrics**: Learn about precision, recall, F1-score, AUC

### Intermediate Level
1. **Tune hyperparameters**: 
   - Try different learning rates (0.001, 0.0001, 0.00001)
   - Adjust batch sizes (16, 32, 64)
   - Modify data augmentation parameters

2. **Try different models**:
   - ResNet50
   - VGG16
   - EfficientNetB0
   - DenseNet121

3. **Implement cross-validation**: Split data into multiple folds

### Advanced Level
1. **Ensemble methods**: Combine predictions from multiple models
2. **Explainability**: Implement GradCAM to visualize what the model is looking at
3. **Multi-task learning**: Predict both pathology and abnormality type
4. **Deploy the model**: Create a web application or REST API
5. **Publish results**: Write a paper comparing your approach with others

---

## 🐛 Common Issues & Quick Fixes

### Issue 1: "No module named 'tensorflow'"
**Fix**: 
```bash
pip install tensorflow==2.15.0
```

### Issue 2: Out of Memory Error
**Fix**: Edit `train_cbis_cnn.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
IMAGE_SIZE = 128  # Reduce from 224
```

### Issue 3: Training is Very Slow
**Check GPU**:
```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU detected:
- Install CUDA Toolkit
- Install cuDNN
- Reinstall tensorflow-gpu

### Issue 4: Test Setup Fails
**Fix**: Check paths in `test_setup.py`:
```python
BASE_PATH = r"c:\Users\gaura\Desktop\github\DMML\cbis_ddsm_dataset\model_training\cbis_cnn_nnew\archive (4)"
```
Ensure this matches your actual dataset location.

### Issue 5: Images Not Found
**Check directory structure**:
```
archive (4)/
├── csv/
│   └── [CSV files]
└── jpeg/
    └── [SeriesInstanceUID folders]/
        └── [JPEG files]
```

---

## 📊 Expected Performance Benchmarks

After training, you should see results similar to:

| Metric | Target Range | Your Result |
|--------|-------------|-------------|
| Training Accuracy | 80-90% | ___% |
| Validation Accuracy | 75-85% | ___% |
| Test Accuracy | 75-85% | ___% |
| AUC-ROC | 0.80-0.90 | ___ |
| Precision (Malignant) | 70-80% | ___% |
| Recall (Malignant) | 75-85% | ___% |

If your results are significantly different:
- **Much lower**: Check data preprocessing, increase training time
- **Much higher**: Possible overfitting, check train/val gap
- **Unstable**: Reduce learning rate, adjust augmentation

---

## 💡 Pro Tips

1. **Save your work**: Keep all output files organized by timestamp
2. **Document changes**: Note any modifications you make to hyperparameters
3. **Compare runs**: Train multiple times with different settings
4. **Use version control**: Git is your friend for code changes
5. **Monitor resources**: Use `nvidia-smi` to monitor GPU usage
6. **Start small**: Test with a small subset first, then scale up
7. **Read the papers**: Understand the InceptionResNetV2 architecture
8. **Validate carefully**: Make sure test set is truly held out

---

## 📚 Recommended Reading

### For Understanding the Dataset
- Original Paper: "A curated mammography data set..." (Lee et al., 2017)
- CBIS-DDSM Documentation on TCIA website

### For Understanding the Model
- InceptionResNetV2 Paper: "Inception-v4, Inception-ResNet..." (Szegedy et al., 2016)
- Transfer Learning Guide: TensorFlow documentation

### For Improving Performance
- Data Augmentation in Medical Imaging
- Handling Imbalanced Datasets in Healthcare
- Explainable AI in Medical Image Analysis

---

## 🎉 Congratulations!

You've successfully completed the CBIS-DDSM CNN training pipeline! 

You now have:
- A trained breast cancer classification model
- Comprehensive evaluation metrics
- Ability to make predictions on new images
- Foundation for further research and development

**Keep learning, keep improving, and good luck with your research!** 🚀

---

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review `QUICKSTART.md` for quick reference
- Read `PROJECT_SUMMARY.md` for project overview
- Run `test_setup.py` to diagnose issues

---

*Last Updated: November 2024*  
*Guide Version: 1.0*

