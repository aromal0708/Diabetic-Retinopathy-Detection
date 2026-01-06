# ============================================================
# DDR PROCESSING — CORRECT PATHS, LOW LOAD
# ============================================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# -------- CORRECT PATHS --------
ROOT = Path("data/DDR")
CSV = ROOT / "DR_grading.csv"                    # ✅ CORRECT
IMG_DIR = ROOT / "DR_grading" / "DR_grading"    # ✅ CORRECT
OUT = Path("data/processed/ddr")

print(f"Looking for CSV at: {CSV}")
print(f"Looking for images at: {IMG_DIR}")

assert CSV.exists(), f"CSV file not found: {CSV}"
assert IMG_DIR.exists(), f"Image directory not found: {IMG_DIR}"

# -------- OUTPUT STRUCTURE --------
for split in ["train", "val"]:
    for c in range(5):
        (OUT / split / str(c)).mkdir(parents=True, exist_ok=True)

# -------- LOAD CSV (ONLY 2 COLUMNS) --------
df = pd.read_csv(CSV)
df = df.iloc[:, :2]
df.columns = ["image", "label"]

print(f"Total CSV rows: {len(df)}")

# -------- FILTER VALID FILES --------
df["path"] = df["image"].apply(lambda x: IMG_DIR / str(x))
df = df[df["path"].apply(lambda p: p.exists())]

print(f"Valid images found: {len(df)}")

# -------- STRATIFIED SPLIT --------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# -------- COPY FILES --------
def copy_split(rows, split):
    count = 0
    for _, r in rows.iterrows():
        dst = OUT / split / str(int(r["label"])) / r["path"].name
        shutil.copyfile(r["path"], dst)
        count += 1
    print(f"{split}: copied {count} images")

copy_split(train_df, "train")
copy_split(val_df, "val")

# -------- SAVE CSVs --------
train_df[["image", "label"]].to_csv(OUT / "train_labels.csv", index=False)
val_df[["image", "label"]].to_csv(OUT / "val_labels.csv", index=False)

print("✅ DDR processed successfully (correct paths)")
