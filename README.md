# DMML — Breast Cancer detection using machine learning

Course: DMML

## Project Title

Breast Cancer detection using machine learning

## Description

Our project focuses on building a machine learning system that uses both numerical data and medical images to detect breast cancer more accurately. We use the UCI Breast Cancer Wisconsin Diagnostic dataset, which contains 30 measurable features of breast tumors, to train classical machine learning models that distinguish between benign and malignant cases. Alongside this, the CBIS-DDSM mammogram dataset provides real medical images that allow us to train deep learning models using CNNs to detect the presence of tumors. By combining insights from both datasets, our project creates a more complete and reliable approach to early breast cancer detection.

## Research Question

This project examines the effectiveness of different data modalities - tabular data (clinical features from the Breast Cancer Wisconsin Diagnostic dataset), image data (mammogram images from the CBIS-DDSM dataset), and their integration - in improving breast cancer detection accuracy. By comparing models trained on each modality individually and in combination, the study aims to evaluate whether multimodal learning enhances diagnostic performance and to identify the most significant clinical attributes and image patterns contributing to early detection.

## Datasets

- UCI Breast Cancer Wisconsin (Diagnostic) dataset — tabular data with 30 numeric features per sample (used for classical ML models)
- CBIS-DDSM mammogram dataset — medical imaging dataset (used to train CNN-based deep learning models)

## Members

|   # | Name   |
| --: | ------ |
|   1 | Aadi   |
|   2 | Andre  |
|   3 | Gaurav |
|   4 | Kevin  |
|   5 | Rhea   |

## Data Preprocessing

We have created a comprehensive Jupyter notebook (`uci_breast_cancer_preprocessing.ipynb`) that performs the following preprocessing steps on the UCI Breast Cancer dataset:

1. **Data Loading**: Loaded 569 samples with 30 features from `uci_breast_cancer_dataset/wdbc.data`
2. **Exploratory Data Analysis**: Statistical analysis and visualization of features
3. **Data Cleaning**: Checked for missing values, duplicates, and inconsistencies (none found)
4. **Label Encoding**: Encoded diagnosis labels (M=Malignant→1, B=Benign→0)
5. **Feature Standardization**: Standardized all features using StandardScaler (mean=0, std=1)
6. **Train-Test Split**: Split data into 80% training (455 samples) and 20% testing (114 samples)

### Preprocessed Data Files

All preprocessed data is saved under `data/processed/`:
- `X_train_scaled.csv` — Standardized training features (455 × 30)
- `y_train.csv` — Training labels (455 × 1)
- `X_test_scaled.csv` — Standardized testing features (114 × 30)
- `y_test.csv` — Testing labels (114 × 1)
- `scaler.pkl` — Saved StandardScaler for future use
- `feature_names.txt` — List of all 30 feature names



to run -
1) pip install -r requirements.txt
2) jupyter notebook