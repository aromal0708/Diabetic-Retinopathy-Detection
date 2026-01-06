import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).parent.parent.parent

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "iDRID" / "B.%20Disease%20Grading" / "B. Disease Grading" / "1. Original Images" / "a. Training Set"
LABEL_CSV = PROJECT_ROOT / "data" / "iDRID" / "B.%20Disease%20Grading" / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed_idrid"
IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42
# ==========================================


def crop_fundus(img):
    """Crop circular fundus region and remove black borders"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]


def apply_clahe(img):
    """Mild CLAHE on L-channel"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = crop_fundus(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


def main():
    os.makedirs(OUT_DIR / "images" / "train", exist_ok=True)
    os.makedirs(OUT_DIR / "images" / "val", exist_ok=True)

    # Verify paths exist
    assert LABEL_CSV.exists(), f"CSV not found: {LABEL_CSV}"
    assert RAW_IMAGE_DIR.exists(), f"Image folder not found: {RAW_IMAGE_DIR}"

    df = pd.read_csv(LABEL_CSV)

    # Normalize filename format
    df["Image name"] = df["Image name"].apply(
        lambda x: f"IDRiD_{int(x.split('_')[-1]):03d}.jpg"
    )

    images = []
    labels = []

    print("🔍 Preprocessing images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["Image name"]
        label = int(row["Retinopathy grade"])

        img_path = RAW_IMAGE_DIR / img_name
        if not img_path.exists():
            continue

        processed = preprocess_image(img_path)
        if processed is None:
            continue

        images.append((img_name, processed))
        labels.append(label)

    labels = np.array(labels)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_rows, val_rows = [], []

    print("💾 Saving processed images...")
    for idx in train_idx:
        name, img = images[idx]
        cv2.imwrite(str(OUT_DIR / "images" / "train" / name), img)
        train_rows.append([name, labels[idx]])

    for idx in val_idx:
        name, img = images[idx]
        cv2.imwrite(str(OUT_DIR / "images" / "val" / name), img)
        val_rows.append([name, labels[idx]])

    pd.DataFrame(train_rows, columns=["image", "label"]).to_csv(
        OUT_DIR / "train_labels.csv", index=False
    )
    pd.DataFrame(val_rows, columns=["image", "label"]).to_csv(
        OUT_DIR / "val_labels.csv", index=False
    )

    print("\n✅ IDRiD preprocessing COMPLETE")
    print(f"Train samples: {len(train_rows)}")
    print(f"Val samples  : {len(val_rows)}")


if __name__ == "__main__":
    main()
