import torch
import pandas as pd
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

# ================= PATH SETUP =================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONFIG =================
MODEL_PATH = PROJECT_ROOT / "models" / "archive" / "clanet_stage2.pth"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "IDRiD": {
        "img_dir": PROJECT_ROOT / "data" / "processed_idrid" / "images" / "val",
        "csv": PROJECT_ROOT / "data" / "processed_idrid" / "val_labels.csv"
    },
    "Messidor": {
        "img_dir": PROJECT_ROOT / "data" / "processed" / "messidor" / "val",
        "csv": PROJECT_ROOT / "data" / "processed" / "messidor" / "val_labels.csv"
    },
    "DDR": {
        "img_dir": PROJECT_ROOT / "data" / "processed" / "ddr" / "val",
        "csv": PROJECT_ROOT / "data" / "processed" / "ddr" / "val_labels.csv"
    }
}
# =========================================


class FundusDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image"]
        label = int(self.df.iloc[idx]["label"])

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Missing image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        return img, label


def load_model():
    from src.model.clanet import CLANet_DenseNet  

    model = CLANet_DenseNet(num_classes=NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader):
    y_true, y_pred = [], []

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred) * 100
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return acc, qwk


def main():
    print("\n========== STAGE-2 DATASET-WISE (VAL) EVALUATION ==========\n")

    model = load_model()

    for name, cfg in DATASETS.items():
        dataset = FundusDataset(cfg["csv"], cfg["img_dir"])
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        acc, qwk = evaluate(model, loader)

        print(f"{name}")
        print(f"Accuracy          : {acc:.2f}%")
        print(f"Quadratic Kappa   : {qwk:.4f}\n")


if __name__ == "__main__":
    main()
