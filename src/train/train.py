import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score

# -------------------------------------------------
# Path setup
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import FundusDataset, get_transforms
from src.modelzoo import build_classifier, save_checkpoint
from src.losses.focal_loss import FocalLoss

# -------------------------------------------------
# Utility: compute class weights
# -------------------------------------------------
def compute_class_weights(labels, num_classes=5):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

# -------------------------------------------------
# Training epoch
# -------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_preds, all_labels = [], []
    losses = []

    for imgs, labels, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

# -------------------------------------------------
# Validation epoch
# -------------------------------------------------
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    losses = []

    for imgs, labels, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        losses.append(loss.item())
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic",
        labels=[0, 1, 2, 3, 4]
    )

    return np.mean(losses), acc, qwk

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. GPU not found.")

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    train_ds = FundusDataset(
        args.train_dir,
        labels_csv=args.train_csv,
        transform=get_transforms(train=True)
    )
    val_ds = FundusDataset(
        args.val_dir,
        labels_csv=args.val_csv,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = build_classifier(
        backbone=args.backbone,
        num_classes=5,
        pretrained=True
    ).to(device)

    # -------------------------------------------------
    # Loss (FOCAL LOSS WITH CLASS WEIGHTS)
    # -------------------------------------------------
    train_df = pd.read_csv(args.train_csv)
    class_weights = compute_class_weights(
        train_df["label"].values
    ).to(device)

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    best_kappa = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_kappa = val_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"QWK: {val_kappa:.4f}"
        )

        # Save best model by QWK (paper-faithful)
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(args.out_dir, "best_model.pth")
            )
            print("✅ Saved new best model")

    print("\nTraining complete.")
    print("Best Quadratic Kappa:", best_kappa)

# -------------------------------------------------
if __name__ == "__main__":
    main()
