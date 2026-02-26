"""
Step 2B: Baseline model using color statistics + XGBoost/RandomForest.

Extracts handcrafted features from the circular ROI of each POM image:
  - Mean RGB within ROI
  - Color histogram (16 bins per channel)
  - HSV statistics (mean hue, saturation, value)
  - Texture: mean intensity, std intensity
  - Scalar inputs: temperature, thickness, order

Then trains XGBoost/RandomForest for regression.

This is a STRONG BASELINE that often works well with small datasets.

Usage:
  python src/baseline_xgboost.py --csv data/dataset_unified.csv
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] xgboost not installed; using GradientBoostingRegressor instead")


def detect_circular_roi_mask(img_np, threshold=30):
    """Create binary mask of the circular FOV."""
    gray = np.mean(img_np, axis=2)
    mask = gray > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.ones(img_np.shape[:2], dtype=bool)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cy, cx = (rmin + rmax) // 2, (cmin + cmax) // 2
    radius = min(rmax - rmin, cmax - cmin) // 2 * 0.85

    h, w = img_np.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
    return circle_mask


def extract_color_features(img_path: str) -> np.ndarray:
    """Extract handcrafted color/texture features from a POM image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32)

    # Get circular ROI mask
    mask = detect_circular_roi_mask(img_np)
    mask_3d = np.stack([mask]*3, axis=2)

    # Masked pixels only
    roi_pixels = img_np[mask]  # (N, 3)

    if len(roi_pixels) == 0:
        return np.zeros(80)

    features = []

    # --- RGB statistics ---
    r, g, b = roi_pixels[:, 0], roi_pixels[:, 1], roi_pixels[:, 2]
    features.extend([
        np.mean(r), np.std(r), np.median(r),
        np.mean(g), np.std(g), np.median(g),
        np.mean(b), np.std(b), np.median(b),
    ])

    # Normalized RGB
    total = r + g + b + 1e-6
    features.extend([
        np.mean(r / total), np.mean(g / total), np.mean(b / total),
    ])

    # --- RGB Histograms (16 bins per channel) ---
    for ch in range(3):
        hist, _ = np.histogram(roi_pixels[:, ch], bins=16, range=(0, 255))
        hist = hist / (hist.sum() + 1e-6)  # normalize
        features.extend(hist.tolist())

    # --- HSV statistics ---
    img_hsv = Image.open(img_path).convert("HSV")
    hsv_np = np.array(img_hsv, dtype=np.float32)
    roi_hsv = hsv_np[mask]

    h_ch, s_ch, v_ch = roi_hsv[:, 0], roi_hsv[:, 1], roi_hsv[:, 2]
    features.extend([
        np.mean(h_ch), np.std(h_ch),  # circular mean would be better but this works
        np.mean(s_ch), np.std(s_ch),
        np.mean(v_ch), np.std(v_ch),
    ])

    # --- Intensity statistics ---
    gray = np.mean(roi_pixels, axis=1)
    features.extend([
        np.mean(gray), np.std(gray),
        np.percentile(gray, 10), np.percentile(gray, 90),
        np.percentile(gray, 90) - np.percentile(gray, 10),  # dynamic range
    ])

    # --- Color ratios (informative for Michel-Levy) ---
    mean_rgb = np.array([np.mean(r), np.mean(g), np.mean(b)])
    max_ch = np.max(mean_rgb) + 1e-6
    features.extend([
        mean_rgb[0] / max_ch,  # R dominance
        mean_rgb[1] / max_ch,  # G dominance
        mean_rgb[2] / max_ch,  # B dominance
        mean_rgb[0] / (mean_rgb[1] + 1e-6),  # R/G ratio
        mean_rgb[1] / (mean_rgb[2] + 1e-6),  # G/B ratio
        mean_rgb[0] / (mean_rgb[2] + 1e-6),  # R/B ratio
    ])

    return np.array(features, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Build feature matrix from dataframe."""
    print("Extracting features from images...")
    img_features = []
    for i, row in df.iterrows():
        feat = extract_color_features(row["image_path"])
        img_features.append(feat)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(df)} images")

    img_features = np.array(img_features)
    print(f"  Image feature dim: {img_features.shape[1]}")

    # Scalar features
    scalar_features = df[["temperature_C", "thickness_um", "order"]].values.astype(np.float32)

    # Add derived features
    derived = np.column_stack([
        df["temperature_C"].values / df["Tc_approx"].values,  # T/Tc
        (1 - df["temperature_C"].values / df["Tc_approx"].values).clip(0),  # 1-T/Tc
    ])

    X = np.hstack([img_features, scalar_features, derived])
    y = df["delta_n"].values.astype(np.float32)

    print(f"  Total feature dim: {X.shape[1]}")
    return X, y


def run_baseline(args):
    df = pd.read_csv(args.csv)
    X, y = build_feature_matrix(df)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose model
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
        model_name = "XGBoost"
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model_name = "GradientBoosting"

    # Leave-One-Out CV (best for tiny dataset)
    print(f"\nRunning LOOCV with {model_name}...")
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(model, X_scaled, y, cv=loo)

    mae = mean_absolute_error(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    r2 = r2_score(y, y_pred_loo)

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS ({model_name} + Color Features, LOOCV)")
    print(f"  MAE  = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  R²   = {r2:.4f}")

    # Per-material results
    for mat in df["material"].unique():
        mask = df["material"] == mat
        mat_mae = mean_absolute_error(y[mask], y_pred_loo[mask])
        mat_r2 = r2_score(y[mask], y_pred_loo[mask]) if mask.sum() > 1 else 0
        print(f"  {mat}: MAE={mat_mae:.6f}, R²={mat_r2:.4f}")

    # Also try Random Forest
    print(f"\nRunning LOOCV with RandomForest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=loo)

    rf_mae = mean_absolute_error(y, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y, y_pred_rf))
    rf_r2 = r2_score(y, y_pred_rf)

    print(f"  RandomForest: MAE={rf_mae:.6f}, RMSE={rf_rmse:.6f}, R²={rf_r2:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "model": model_name,
        "cv": "LOOCV",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "rf_mae": float(rf_mae),
        "rf_rmse": float(rf_rmse),
        "rf_r2": float(rf_r2),
        "predictions": [
            {
                "image": df.iloc[i]["image_path"],
                "temp": float(df.iloc[i]["temperature_C"]),
                "material": df.iloc[i]["material"],
                "true_dn": float(y[i]),
                "pred_dn_xgb": float(y_pred_loo[i]),
                "pred_dn_rf": float(y_pred_rf[i]),
            }
            for i in range(len(y))
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/baseline_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")

    # Train final model on all data for feature importance
    model.fit(X_scaled, y)

    if HAS_XGBOOST:
        imp = model.feature_importances_
        n_img_feat = X.shape[1] - 5  # 5 scalar+derived features
        img_imp = imp[:n_img_feat].sum()
        scalar_imp = imp[n_img_feat:].sum()
        print(f"\nFeature importance: image={img_imp:.3f}, scalar={scalar_imp:.3f}")
        top_idx = np.argsort(imp)[-10:][::-1]
        print("Top 10 features:", top_idx.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/dataset_unified.csv")
    args = parser.parse_args()
    run_baseline(args)
