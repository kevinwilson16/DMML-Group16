UCI Breast Cancer Dataset (Tabular)

- Clean the dataset and check for any missing or inconsistent values.
- Encode target labels (“M” = Malignant, “B” = Benign) into numeric form.
- Standardize all features using StandardScaler to normalize the data.
- Split the dataset into training and testing sets for model evaluation.

CBIS-DDSM Dataset (Images)

- Organize images into folders based on their class (benign, malignant, normal).
- Resize and normalize all images to 224×224 pixels for CNN compatibility.
- Apply data augmentation to balance the dataset and reduce overfitting.