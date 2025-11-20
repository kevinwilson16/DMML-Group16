import os
import random
import pandas as pd
import numpy as np
import pydicom
from PIL import Image, ImageEnhance

ROOT_DIR = r"D:\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"
OUT_DIR = r"D:\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM_processed"

os.makedirs(OUT_DIR, exist_ok=True)

# Load CSV metadata
calc_train = pd.read_csv(os.path.join(ROOT_DIR, "calc_case_description_train_set.csv"))
calc_test  = pd.read_csv(os.path.join(ROOT_DIR, "calc_case_description_test_set.csv"))
mass_train = pd.read_csv(os.path.join(ROOT_DIR, "mass_case_description_train_set.csv"))
mass_test  = pd.read_csv(os.path.join(ROOT_DIR, "mass_case_description_test_set.csv"))

df = pd.concat([calc_train, calc_test, mass_train, mass_test], ignore_index=True)

df = df[["pathology", "image file path"]]
df.columns = ["pathology", "rel_path"]

df["case_folder"] = df["rel_path"].apply(lambda x: str(x).split("/")[0])
df["pathology"] = df["pathology"].str.lower().str.strip()

LABEL_MAP = {
    "malignant": "malignant",
    "benign": "benign",
    "benign_without_callback": "benign"
}

df["class"] = df["pathology"].map(LABEL_MAP)
df = df.dropna(subset=["class"])

# ----- Train/Test Split (80/20) on CASES -----

cases = df[["case_folder", "class"]].drop_duplicates()
case_list = cases["case_folder"].tolist()
random.shuffle(case_list)

split_idx = int(0.8 * len(case_list))
train_cases = set(case_list[:split_idx])
test_cases = set(case_list[split_idx:])

cases["split"] = cases["case_folder"].apply(
    lambda c: "train" if c in train_cases else "test"
)

# Save which cases are in test (tracking)
test_cases_df = cases[cases["split"] == "test"]
test_cases_df.to_csv(os.path.join(OUT_DIR, "test_cases.csv"), index=False)

# Paths for train/test subfolders
TRAIN_DIR = os.path.join(OUT_DIR, "train")
TEST_DIR = os.path.join(OUT_DIR, "test")

for base in [TRAIN_DIR, TEST_DIR]:
    for c in ["benign", "malignant"]:
        os.makedirs(os.path.join(base, c), exist_ok=True)


# ----- Image Utils -----

def load_dicom(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    m = arr.max()
    if m > 0:
        arr /= m
    return (arr * 255).astype(np.uint8)

def resize_img(arr):
    return Image.fromarray(arr).resize((224, 224))

def augment(img):
    out = []
    if random.random() > 0.5:
        out.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    out.append(img.rotate(random.randint(-10, 10)))
    enh = ImageEnhance.Brightness(img)
    out.append(enh.enhance(random.uniform(0.8, 1.2)))
    return out

AUG_PER_IMAGE = {
    "malignant": 3,
    "benign": 1
}

processed = 0
missing = 0

# ----- Main Processing Loop -----

for _, row in df.iterrows():
    cls = row["class"]
    folder = row["case_folder"]

    split = "train" if folder in train_cases else "test"
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR

    full_case_path = os.path.join(ROOT_DIR, folder)

    if not os.path.exists(full_case_path):
        missing += 1
        continue

    dicoms = []
    for root, _, files in os.walk(full_case_path):
        for f in files:
            if f.lower().endswith(".dcm") and "full mammogram" in root.lower():
                dicoms.append(os.path.join(root, f))

    if not dicoms:
        missing += 1
        continue

    for dcm_path in dicoms:
        arr = load_dicom(dcm_path)
        img = resize_img(arr)

        case = folder.replace("/", "_").replace("\\", "_")
        file = os.path.basename(dcm_path).replace(".dcm", "")
        base_name = f"{case}_{file}.png"

        out_path = os.path.join(base_dir, cls, base_name)
        img.save(out_path)

        # Augment ONLY training images
        if split == "train":
            for i in range(AUG_PER_IMAGE[cls]):
                for j, a in enumerate(augment(img)):
                    ap = os.path.join(
                        base_dir,
                        cls,
                        f"{case}_{file}_aug{i}_{j}.png"
                    )
                    a.save(ap)

        processed += 1

print("Processed:", processed)
print("Missing:", missing)
print("Train cases:", len(train_cases))
print("Test cases:", len(test_cases))
