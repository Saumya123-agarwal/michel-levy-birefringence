"""
Step 4: Training pipeline with Leave-One-Out / K-Fold Cross-Validation.

Two-phase training strategy:
  Phase A: Freeze backbone, train only fusion head (10 epochs)
  Phase B: Unfreeze last backbone blocks, fine-tune everything (50+ epochs)

Usage:
  python src/train.py --csv data/dataset_unified.csv --epochs 60 --backbone efficientnet_b0
  python src/train.py --csv data/dataset_unified.csv --cv-mode loocv
  python src/train.py --csv data/dataset_unified.csv --cv-mode kfold --n-folds 5
"""

import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import POMPatchDataset, POMWholeImageDataset
from model import HybridCNN, PhysicsInformedLoss, build_model, count_parameters


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slower)")
    return device


def train_one_epoch(model, loader, optimizer, criterion, device, temperatures=None):
    model.train()
    total_loss = 0
    n_batches = 0

    for images, scalars, targets in loader:
        images = images.to(device)
        scalars = scalars.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images, scalars)

        # Extract temperature from scalars for physics loss
        temps = scalars[:, 0] * 60.0  # denormalize

        if isinstance(criterion, PhysicsInformedLoss):
            loss = criterion(preds, targets, temps)
        else:
            loss = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, scalars, targets in loader:
        images = images.to(device)
        scalars = scalars.to(device)

        preds = model(images, scalars)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds) if len(all_targets) > 1 else 0.0

    return mae, rmse, r2, all_preds, all_targets


def aggregate_patch_predictions(preds, dataset, method="mean"):
    """
    Aggregate patch-level predictions to image-level.
    Multiple patches from the same image should be averaged.
    """
    img_preds = {}
    for flat_idx, pred in enumerate(preds):
        img_idx = dataset.get_image_index(flat_idx)
        if img_idx not in img_preds:
            img_preds[img_idx] = []
        img_preds[img_idx].append(pred)

    aggregated = {}
    for img_idx, vals in img_preds.items():
        if method == "mean":
            aggregated[img_idx] = np.mean(vals)
        elif method == "median":
            aggregated[img_idx] = np.median(vals)

    return aggregated


def run_kfold_cv(args, device):
    """Run K-Fold or LOOCV cross-validation."""

    df = pd.read_csv(args.csv)
    n_samples = len(df)

    if args.cv_mode == "loocv":
        n_folds = n_samples
        splitter = LeaveOneOut()
        print(f"Running Leave-One-Out CV ({n_samples} folds)")
    else:
        n_folds = args.n_folds
        # Group by material to ensure material-balanced folds
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        print(f"Running {n_folds}-Fold CV")

    all_fold_preds = []
    all_fold_targets = []
    all_fold_materials = []
    all_fold_temps = []
    fold_metrics = []

    indices = np.arange(n_samples)

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_i + 1}/{n_folds}")
        print(f"  Train: {len(train_idx)} images, Val: {len(val_idx)} images")

        # Create train/val CSVs for this fold
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_csv = f"/tmp/fold_{fold_i}_train.csv"
        val_csv = f"/tmp/fold_{fold_i}_val.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        # Build datasets
        train_ds = POMPatchDataset(
            train_csv, mode="train",
            patch_size=224, n_patches=args.n_patches,
        )
        val_ds = POMPatchDataset(
            val_csv, mode="val",
            patch_size=224, n_patches=5,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        # Build model (fresh for each fold)
        model = build_model(
            backbone=args.backbone,
            pretrained=True,
            freeze_backbone=True,
        ).to(device)

        # Phase A: Train fusion head only
        optimizer_a = Adam([
            {"params": model.scalar_branch.parameters(), "lr": args.lr},
            {"params": model.fusion_head.parameters(), "lr": args.lr},
        ])
        criterion = PhysicsInformedLoss(alpha_nonneg=1.0, alpha_mono=0.05)
        scheduler_a = CosineAnnealingLR(optimizer_a, T_max=args.warmup_epochs)

        print(f"  Phase A: Training fusion head ({args.warmup_epochs} epochs)...")
        for epoch in range(args.warmup_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer_a, criterion, device)
            scheduler_a.step()
            if (epoch + 1) % 5 == 0:
                mae, rmse, r2, _, _ = evaluate(model, val_loader, device)
                print(f"    Epoch {epoch+1}: loss={train_loss:.6f}, "
                      f"val_MAE={mae:.6f}, val_RMSE={rmse:.6f}")

        # Phase B: Fine-tune backbone
        model.unfreeze_backbone(n_layers_from_end=2)
        optimizer_b = Adam([
            {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
            {"params": model.scalar_branch.parameters(), "lr": args.lr * 0.5},
            {"params": model.fusion_head.parameters(), "lr": args.lr * 0.5},
        ])
        scheduler_b = CosineAnnealingLR(optimizer_b, T_max=args.epochs - args.warmup_epochs)

        best_val_mae = float("inf")
        patience_counter = 0

        print(f"  Phase B: Fine-tuning ({args.epochs - args.warmup_epochs} epochs)...")
        for epoch in range(args.warmup_epochs, args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer_b, criterion, device)
            scheduler_b.step()

            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                mae, rmse, r2, val_preds, val_targets = evaluate(model, val_loader, device)
                print(f"    Epoch {epoch+1}: loss={train_loss:.6f}, "
                      f"val_MAE={mae:.6f}, val_RMSE={rmse:.6f}, R2={r2:.4f}")

                if mae < best_val_mae:
                    best_val_mae = mae
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
        mae, rmse, r2, val_preds, val_targets = evaluate(model, val_loader, device)

        # Aggregate patch predictions to image level
        img_level_preds = aggregate_patch_predictions(val_preds, val_ds)
        img_level_targets = {
            img_idx: val_ds._targets_cache[img_idx]
            for img_idx in img_level_preds
        }

        for img_idx in img_level_preds:
            pred = img_level_preds[img_idx]
            true = float(img_level_targets[img_idx])
            row = val_df.iloc[img_idx]

            all_fold_preds.append(pred)
            all_fold_targets.append(true)
            all_fold_materials.append(row["material"])
            all_fold_temps.append(row["temperature_C"])

        fold_metrics.append({"fold": fold_i, "mae": mae, "rmse": rmse, "r2": r2})
        print(f"  Fold {fold_i+1} result: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")

        # Cleanup
        del model, optimizer_a, optimizer_b, train_ds, val_ds
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        os.remove(train_csv)
        os.remove(val_csv)

    # Overall results
    all_fold_preds = np.array(all_fold_preds)
    all_fold_targets = np.array(all_fold_targets)

    overall_mae = mean_absolute_error(all_fold_targets, all_fold_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_fold_targets, all_fold_preds))
    overall_r2 = r2_score(all_fold_targets, all_fold_preds)

    print(f"\n{'='*60}")
    print(f"OVERALL CV RESULTS ({args.cv_mode})")
    print(f"  MAE  = {overall_mae:.6f}")
    print(f"  RMSE = {overall_rmse:.6f}")
    print(f"  R²   = {overall_r2:.4f}")
    print(f"  Mean fold MAE: {np.mean([m['mae'] for m in fold_metrics]):.6f}")

    # Save results
    results = {
        "cv_mode": args.cv_mode,
        "backbone": args.backbone,
        "n_patches": args.n_patches,
        "epochs": args.epochs,
        "overall_mae": float(overall_mae),
        "overall_rmse": float(overall_rmse),
        "overall_r2": float(overall_r2),
        "fold_metrics": fold_metrics,
        "predictions": [
            {
                "pred": float(p), "true": float(t),
                "material": m, "temperature": float(tmp)
            }
            for p, t, m, tmp in zip(
                all_fold_preds, all_fold_targets,
                all_fold_materials, all_fold_temps
            )
        ],
    }

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/cv_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    return results


def train_final_model(args, device):
    """Train final model on ALL data and save weights."""
    print("\n" + "="*60)
    print("Training FINAL model on all data...")

    ds = POMPatchDataset(
        args.csv, mode="train",
        patch_size=224, n_patches=args.n_patches,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    model = build_model(
        backbone=args.backbone, pretrained=True, freeze_backbone=True,
    ).to(device)

    criterion = PhysicsInformedLoss(alpha_nonneg=1.0, alpha_mono=0.05)

    # Phase A
    optimizer = Adam([
        {"params": model.scalar_branch.parameters(), "lr": args.lr},
        {"params": model.fusion_head.parameters(), "lr": args.lr},
    ])
    for epoch in range(args.warmup_epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Phase A epoch {epoch+1}: loss={loss:.6f}")

    # Phase B
    model.unfreeze_backbone(n_layers_from_end=2)
    optimizer = Adam([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.scalar_branch.parameters(), "lr": args.lr * 0.5},
        {"params": model.fusion_head.parameters(), "lr": args.lr * 0.5},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    for epoch in range(args.warmup_epochs, args.epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Phase B epoch {epoch+1}: loss={loss:.6f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    save_path = f"models/birefringence_model_{args.backbone}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone": args.backbone,
        "args": vars(args),
    }, save_path)
    print(f"Final model saved: {save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train birefringence prediction model")
    parser.add_argument("--csv", type=str, default="data/dataset_unified.csv")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet18", "mobilenetv3_small_100"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Epochs with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-patches", type=int, default=25,
                        help="Patches per image for training")
    parser.add_argument("--cv-mode", type=str, default="kfold",
                        choices=["kfold", "loocv", "none"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save-final", action="store_true",
                        help="Also train final model on all data")
    args = parser.parse_args()

    device = get_device()

    # Run cross-validation
    if args.cv_mode != "none":
        results = run_kfold_cv(args, device)

    # Train final model
    if args.save_final or args.cv_mode == "none":
        train_final_model(args, device)


if __name__ == "__main__":
    main()
