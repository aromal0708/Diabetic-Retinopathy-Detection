# ============================================================
# Process Messidor-2 Dataset (CSV → train/val + labels)
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

RAW_ROOT = PROJECT_ROOT / "data" / "messidor-2"
CSV_PATH = RAW_ROOT / "messidor_data.csv"
IMG_ROOT = RAW_ROOT / "messidor-2" / "messidor-2" / "preprocess"

OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "messidor"
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "val"

# ---------------- CHECKS ----------------
assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
assert IMG_ROOT.exists(), f"Image folder not found: {IMG_ROOT}"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

print("✔ Messidor paths verified")

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} records")
print("CSV Columns:", df.columns.tolist())

# Expected columns (Messidor standard)
# id_code / diagnosis
img_col = "id_code"
label_col = "diagnosis"

assert img_col in df.columns
assert label_col in df.columns

# ---------------- VERIFY IMAGES ----------------
valid_rows = []

for _, row in df.iterrows():
    img_name = row[img_col]

    # Messidor images already include extension
    img_path = IMG_ROOT / img_name

    if img_path.exists():
        valid_rows.append(row)

df = pd.DataFrame(valid_rows)
print(f"Valid images found: {len(df)}")

# ---------------- STRATIFIED SPLIT ----------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[label_col],
    random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ---------------- COPY FILES ----------------
def copy_split(split_df, split_dir):
    copied = 0
    for _, row in split_df.iterrows():
        src = IMG_ROOT / row[img_col]
        dst = split_dir / row[img_col]
        shutil.copy(src, dst)
        copied += 1
    return copied

print("Copying training images...")
copy_split(train_df, TRAIN_DIR)

print("Copying validation images...")
copy_split(val_df, VAL_DIR)

# ---------------- SAVE CSVs ----------------
train_labels = train_df[[img_col, label_col]].rename(
    columns={img_col: "image", label_col: "label"}
)
val_labels = val_df[[img_col, label_col]].rename(
    columns={img_col: "image", label_col: "label"}
)

train_labels.to_csv(OUT_ROOT / "train_labels.csv", index=False)
val_labels.to_csv(OUT_ROOT / "val_labels.csv", index=False)

print("✔ train_labels.csv & val_labels.csv created")

print("\n✅ Messidor processing COMPLETE")
