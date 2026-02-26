"""
Step 6: Visualization — plot results, temperature curves, Grad-CAM.

Usage:
  python src/visualize.py --results results/cv_results_*.json
  python src/visualize.py --results results/baseline_results_*.json --type baseline
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_pred_vs_true(predictions, title, save_path):
    """Scatter plot: predicted vs true delta_n."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    materials = list(set(p["material"] for p in predictions))
    colors = {"P100-7CB+": "#2196F3", "5PCH": "#FF5722"}
    markers = {"P100-7CB+": "o", "5PCH": "s"}

    for mat in materials:
        mat_preds = [p for p in predictions if p["material"] == mat]
        true_vals = [p["true"] if "true" in p else p["true_dn"] for p in mat_preds]
        pred_key = "pred" if "pred" in mat_preds[0] else "pred_dn_xgb"
        pred_vals = [p[pred_key] for p in mat_preds]

        ax.scatter(true_vals, pred_vals, c=colors.get(mat, "#666"),
                   marker=markers.get(mat, "o"), s=80, alpha=0.8,
                   label=mat, edgecolors="white", linewidth=0.5)

    # Perfect prediction line
    all_vals = [p["true"] if "true" in p else p["true_dn"] for p in predictions]
    lims = [min(all_vals) - 0.01, max(all_vals) + 0.01]
    ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect prediction")

    ax.set_xlabel("True Δn", fontsize=14)
    ax.set_ylabel("Predicted Δn", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_temperature_curves(predictions, title, save_path):
    """Plot delta_n vs temperature curves (true vs predicted)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    materials = list(set(p["material"] for p in predictions))

    for i, mat in enumerate(materials):
        ax = axes[i] if len(materials) > 1 else axes[0]
        mat_preds = [p for p in predictions if p["material"] == mat]

        temps = [p["temperature"] if "temperature" in p else p["temp"] for p in mat_preds]
        true_vals = [p["true"] if "true" in p else p["true_dn"] for p in mat_preds]
        pred_key = "pred" if "pred" in mat_preds[0] else "pred_dn_xgb"
        pred_vals = [p[pred_key] for p in mat_preds]

        # Sort by temperature
        sort_idx = np.argsort(temps)
        temps = np.array(temps)[sort_idx]
        true_vals = np.array(true_vals)[sort_idx]
        pred_vals = np.array(pred_vals)[sort_idx]

        ax.plot(temps, true_vals, "o-", color="#2196F3", markersize=6,
                label="True Δn", linewidth=2)
        ax.plot(temps, pred_vals, "s--", color="#FF5722", markersize=6,
                label="Predicted Δn", linewidth=2, alpha=0.8)

        # Error bars
        errors = np.abs(true_vals - pred_vals)
        ax.fill_between(temps, pred_vals - errors, pred_vals + errors,
                         alpha=0.15, color="#FF5722")

        ax.set_xlabel("Temperature (°C)", fontsize=13)
        ax.set_ylabel("Δn (birefringence)", fontsize=13)
        ax.set_title(f"{mat}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_distribution(predictions, title, save_path):
    """Histogram of prediction errors."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    true_vals = [p["true"] if "true" in p else p["true_dn"] for p in predictions]
    pred_key = "pred" if "pred" in predictions[0] else "pred_dn_xgb"
    pred_vals = [p[pred_key] for p in predictions]

    errors = np.array(pred_vals) - np.array(true_vals)

    ax.hist(errors, bins=15, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(errors), color="blue", linestyle="-.", linewidth=1.5,
               label=f"Mean error: {np.mean(errors):.5f}")

    ax.set_xlabel("Prediction Error (Δn_pred - Δn_true)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--type", type=str, default="dl",
                        choices=["dl", "baseline"])
    parser.add_argument("--outdir", type=str, default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    predictions = data["predictions"]
    model_name = data.get("backbone", data.get("model", "model"))

    print(f"Loaded {len(predictions)} predictions")
    print(f"Overall: MAE={data.get('overall_mae', data.get('mae', '?')):.6f}, "
          f"R²={data.get('overall_r2', data.get('r2', '?')):.4f}")

    # Plot 1: Predicted vs True
    plot_pred_vs_true(
        predictions,
        f"Predicted vs True Δn ({model_name})",
        os.path.join(args.outdir, "pred_vs_true.png"),
    )

    # Plot 2: Temperature curves
    plot_temperature_curves(
        predictions,
        f"Δn vs Temperature ({model_name})",
        os.path.join(args.outdir, "temperature_curves.png"),
    )

    # Plot 3: Error distribution
    plot_error_distribution(
        predictions,
        f"Error Distribution ({model_name})",
        os.path.join(args.outdir, "error_distribution.png"),
    )


if __name__ == "__main__":
    main()
