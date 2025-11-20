# CBIS-DDSM Breast Cancer Classification using CNN

## Dataset Overview

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** is an updated and standardized version of the Digital Database for Screening Mammography (DDSM).

### Dataset Statistics
- **Number of Studies**: 6,775
- **Number of Participants**: 1,566
- **Number of Images**: 10,239
- **Modality**: Mammography (MG)
- **Image Format**: JPEG
- **Dataset Size**: ~6 GB

### Dataset Structure
The dataset contains:
- **Calcification cases**: Abnormalities related to calcium deposits
- **Mass cases**: Abnormalities related to masses/tumors
- Each case includes:
  - Full mammogram images
  - Cropped images (Region of Interest)
  - ROI mask images

### Pathology Labels
- **MALIGNANT**: Cancer confirmed by pathology
- **BENIGN**: Non-cancerous abnormality
- **BENIGN_WITHOUT_CALLBACK**: Benign case without need for follow-up

## Model Architecture

### Base Model: InceptionResNetV2
- Pre-trained on ImageNet
- Transfer learning approach
- Last 10 layers fine-tuned for our specific task

### Custom Layers
```
InceptionResNetV2 (base)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) + L2 regularization + Dropout(0.5)
    ↓
Dense(128, relu) + L2 regularization + Dropout(0.3)
    ↓
Dense(2, softmax) - Binary Classification
```

### Training Configuration
- **Image Size**: 224x224x3
- **Batch Size**: 32
- **Optimizer**: Nadam
- **Learning Rate**: 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall

### Data Augmentation
- Rotation: ±20°
- Width/Height Shift: ±10%
- Shear: ±10%
- Zoom: ±10%
- Horizontal Flip: Yes
- Brightness Variation: 80%-120%

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Directory Structure
Ensure your dataset is organized as follows:
```
cbis_cnn_nnew/
├── archive (4)/
│   ├── csv/
│   │   ├── calc_case_description_train_set.csv
│   │   ├── calc_case_description_test_set.csv
│   │   ├── mass_case_description_train_set.csv
│   │   ├── mass_case_description_test_set.csv
│   │   ├── meta.csv
│   │   └── dicom_info.csv
│   └── jpeg/
│       └── [image files organized by SeriesInstanceUID]
├── train_cbis_cnn.py
├── requirements.txt
└── README.md
```

### Before Training: Explore the Dataset
It's recommended to run these scripts first:

#### 1. Verify Setup
```bash
python test_setup.py
```

This will check:
- Directory structure
- CSV file integrity
- JPEG image accessibility
- Data loading functionality
- Image path mapping
- TensorFlow/Keras installation

#### 2. Visualize Dataset (Optional)
```bash
python visualize_dataset.py
```

This will generate:
- Sample image grid showing 20 random cases
- Statistical plots (pathology distribution, train/test split, etc.)
- Dataset summary statistics
- All visualizations saved to `visualizations/` directory

### Training
Run the training script:
```bash
python train_cbis_cnn.py
```

### Training Features
1. **Automatic Data Loading**: Loads both calcification and mass cases
2. **Image Preprocessing**: Resizes, normalizes, and converts DICOM paths to JPEG
3. **Data Augmentation**: Real-time augmentation during training
4. **Class Imbalance Handling**: Uses class weights
5. **Callbacks**:
   - ModelCheckpoint: Saves best model
   - EarlyStopping: Prevents overfitting
   - ReduceLROnPlateau: Adaptive learning rate
   - TensorBoard: Training visualization

### Output Directory Structure
After training, outputs will be saved in:
```
outputs/
├── models/
│   ├── best_model_[timestamp].h5
│   └── final_model_[timestamp].h5
├── plots/
│   ├── training_history_[timestamp].png
│   ├── confusion_matrix_[timestamp].png
│   └── roc_curve_[timestamp].png
├── logs/
│   └── run_[timestamp]/
│       └── [TensorBoard logs]
└── classification_report_[timestamp].csv
```

## Model Evaluation

The script automatically evaluates the model on the test set and provides:
- **Classification Report**: Precision, Recall, F1-score for each class
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve**: With AUC score
- **Training History**: Accuracy, Loss, AUC, Precision, Recall over epochs

## Making Predictions with Trained Model

### Single Image Prediction
```bash
python predict.py --model outputs/models/best_model_[timestamp].h5 --image path/to/image.jpg
```

### Batch Prediction (Directory)
```bash
python predict.py --model outputs/models/best_model_[timestamp].h5 --image_dir path/to/images/
```

### Save Predictions to CSV
```bash
python predict.py --model outputs/models/best_model_[timestamp].h5 --image_dir path/to/images/ --output results.csv
```

### Custom Threshold
```bash
python predict.py --model outputs/models/best_model_[timestamp].h5 --image path/to/image.jpg --threshold 0.7
```

The prediction script will output:
- Predicted class (BENIGN or MALIGNANT)
- Confidence score
- Individual probabilities for each class
- Summary statistics for batch predictions

## Configuration

You can modify training parameters in the `Config` class:

```python
class Config:
    IMAGE_SIZE = 224      # Input image size
    BATCH_SIZE = 32       # Batch size for training
    EPOCHS = 50           # Maximum number of epochs
    LEARNING_RATE = 0.0001  # Initial learning rate
```

## Hardware Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 16 GB
- Storage: 10 GB free space

### Recommended
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 32 GB
- Storage: 20 GB free space (for outputs and logs)

## Training Time Estimates

- **With GPU (RTX 3080/4090)**: ~2-4 hours for 50 epochs
- **With CPU**: ~15-20 hours for 50 epochs

## Tips for Better Performance

1. **GPU Acceleration**: Use a CUDA-compatible GPU
2. **Batch Size**: Increase if you have more GPU memory
3. **Data Augmentation**: Adjust parameters if overfitting occurs
4. **Fine-tuning**: Experiment with unfreezing more/fewer layers
5. **Learning Rate**: Use ReduceLROnPlateau for adaptive learning
6. **Class Weights**: Already implemented for handling class imbalance

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in Config
- Reduce `IMAGE_SIZE` (e.g., from 224 to 128)

### Slow Training
- Enable GPU acceleration
- Reduce number of training samples for quick testing
- Use mixed precision training (add to script if needed)

### Poor Performance
- Increase number of epochs
- Adjust data augmentation parameters
- Unfreeze more layers for fine-tuning
- Try different optimizers (Adam, SGD with momentum)

## Citation

If you use this dataset, please cite:

```
Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017).
A curated mammography data set for use in computer-aided detection and diagnosis research.
Scientific Data, 4, 170177.
https://www.nature.com/articles/sdata2017177
```

## License

The CBIS-DDSM dataset is available through The Cancer Imaging Archive (TCIA).
Please refer to TCIA's data usage policies.

## Contact

For scientific inquiries about the dataset:
- Dr. Daniel Rubin
- Department of Biomedical Data Science, Radiology, and Medicine
- Stanford University School of Medicine
- Email: dlrubin@stanford.edu

## Acknowledgments

- The Cancer Imaging Archive (TCIA)
- University of South Florida Digital Mammography Lab
- Stanford University

