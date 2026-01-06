# ============================================================
# FLATTEN DDR TRAIN & VAL FOLDERS
# ============================================================

from pathlib import Path
import shutil

ROOT = Path("data/processed/ddr")

for split in ["train", "val"]:
    split_dir = ROOT / split
    print(f"\nProcessing {split}...")

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for img in class_dir.iterdir():
            dst = split_dir / img.name
            if not dst.exists():
                shutil.move(str(img), str(dst))

        class_dir.rmdir()

    print(f"✔ {split} flattened")

print("\n✅ DDR flattening complete")

