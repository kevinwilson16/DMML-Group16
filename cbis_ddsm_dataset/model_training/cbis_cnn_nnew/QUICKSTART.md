# Quick Start Guide - CBIS-DDSM CNN Training

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python test_setup.py
```

This will check if your dataset is properly configured. All tests should pass before proceeding.

### Step 3: Start Training
```bash
python train_cbis_cnn.py
```

The training will:
- Load 5,000+ mammogram images
- Train for up to 50 epochs (with early stopping)
- Save the best model automatically
- Generate evaluation plots and reports

Expected training time:
- **With GPU**: 2-4 hours
- **With CPU**: 15-20 hours

### Step 4: Make Predictions
After training completes, use your model:

```bash
# Single image prediction
python predict.py --model outputs/models/best_model_TIMESTAMP.h5 --image path/to/mammogram.jpg

# Batch prediction
python predict.py --model outputs/models/best_model_TIMESTAMP.h5 --image_dir path/to/images/
```

---

## 📊 What to Expect

### Dataset Info
- **Training samples**: ~5,000 images
- **Test samples**: ~1,200 images
- **Classes**: Benign (0) and Malignant (1)
- **Image size**: Resized to 224×224 pixels

### Model Architecture
```
Input (224×224×3)
    ↓
InceptionResNetV2 (pretrained)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Output(2) - Softmax
```

### Expected Performance
Based on the CBIS-DDSM dataset characteristics:
- **Accuracy**: 75-85%
- **AUC**: 0.80-0.90
- **Precision/Recall**: Varies by class due to imbalance

---

## 📁 Output Files

After training, you'll find:

```
outputs/
├── models/
│   ├── best_model_20241120_143052.h5     # Best model during training
│   └── final_model_20241120_143052.h5    # Final model after all epochs
├── plots/
│   ├── training_history_20241120_143052.png
│   ├── confusion_matrix_20241120_143052.png
│   └── roc_curve_20241120_143052.png
├── logs/
│   └── run_20241120_143052/              # TensorBoard logs
└── classification_report_20241120_143052.csv
```

---

## 🎯 Key Features

### 1. **Automatic Data Handling**
- Combines calcification and mass cases
- Handles DICOM → JPEG path conversion
- Preprocesses images automatically
- Uses cropped ROI images for better focus

### 2. **Data Augmentation**
- Rotation: ±20°
- Shifts: ±10%
- Zoom: ±10%
- Horizontal flip
- Brightness variation

### 3. **Class Imbalance Handling**
- Automatic class weight calculation
- Balanced training despite imbalanced data

### 4. **Smart Training**
- **Early Stopping**: Stops if no improvement
- **Model Checkpoint**: Saves best model only
- **Learning Rate Reduction**: Adapts LR automatically
- **TensorBoard**: Live training visualization

### 5. **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve with AUC
- Classification Report (saved as CSV)

---

## 🔧 Customization

### Adjust Training Parameters
Edit `train_cbis_cnn.py` → `Config` class:

```python
class Config:
    IMAGE_SIZE = 224        # Change image size
    BATCH_SIZE = 32         # Adjust batch size
    EPOCHS = 50             # Maximum epochs
    LEARNING_RATE = 0.0001  # Initial learning rate
```

### Use Full Mammograms Instead of Cropped
In `train_cbis_cnn.py`, line ~150:
```python
train_paths, train_labels = prepare_image_paths(train_df, use_cropped=False)
```

### Change Model Architecture
Replace `InceptionResNetV2` with other models:
- `ResNet50`
- `VGG16`
- `EfficientNetB0`
- `DenseNet121`

---

## 🐛 Common Issues & Solutions

### Issue: Out of Memory Error
**Solution 1**: Reduce batch size
```python
BATCH_SIZE = 16  # or even 8
```

**Solution 2**: Reduce image size
```python
IMAGE_SIZE = 128  # instead of 224
```

### Issue: Training Too Slow
**Solution**: Check GPU usage
```bash
# Should show GPU(s) at startup
python train_cbis_cnn.py
```

If no GPU detected:
- Install CUDA toolkit
- Install cuDNN
- Reinstall tensorflow-gpu: `pip install tensorflow-gpu`

### Issue: Test Setup Fails
**Solution**: Check paths in `test_setup.py`
```python
BASE_PATH = r"c:\Users\gaura\Desktop\github\DMML\cbis_ddsm_dataset\model_training\cbis_cnn_nnew\archive (4)"
```

Make sure this matches your actual dataset location.

### Issue: Images Not Found
**Solution**: The script expects:
```
archive (4)/
├── csv/
│   └── [CSV files]
└── jpeg/
    └── [SeriesInstanceUID folders]/
        └── [JPEG files]
```

---

## 📈 Monitor Training

### Option 1: Terminal Output
Watch real-time progress in the terminal:
```
Epoch 1/50
157/157 [==============================] - 120s - loss: 0.5234 - accuracy: 0.7456
```

### Option 2: TensorBoard
Open TensorBoard to visualize training:
```bash
tensorboard --logdir=outputs/logs
```
Then open: http://localhost:6006

---

## 🎓 Next Steps After Training

1. **Analyze Results**
   - Check confusion matrix
   - Review misclassified cases
   - Examine ROC curve

2. **Improve Model**
   - Increase epochs
   - Adjust data augmentation
   - Try different architectures
   - Fine-tune more layers

3. **Deploy Model**
   - Use `predict.py` for inference
   - Integrate into web application
   - Create REST API endpoint

4. **Research & Compare**
   - Compare with published results
   - Try ensemble methods
   - Experiment with attention mechanisms

---

## 📚 Additional Resources

### Dataset Paper
Lee, R. S., et al. (2017). "A curated mammography data set for use in computer-aided detection and diagnosis research." Scientific Data, 4, 170177.

### Useful Links
- [TCIA - Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
- [Original CBIS-DDSM Paper](https://www.nature.com/articles/sdata2017177)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)

---

## ✅ Checklist Before Training

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted to correct location
- [ ] `test_setup.py` passes all checks
- [ ] Sufficient disk space (~20 GB free)
- [ ] GPU drivers installed (optional but recommended)

---

## 🆘 Need Help?

If you encounter issues:
1. Run `test_setup.py` to diagnose problems
2. Check error messages carefully
3. Verify file paths and dataset structure
4. Ensure all dependencies are installed
5. Check GPU compatibility for TensorFlow

---

**Happy Training! 🎉**

