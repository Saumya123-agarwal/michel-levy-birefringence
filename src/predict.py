"""
Step 5: Inference — predict birefringence from a new POM image.

Usage:
  python src/predict.py \
    --image path/to/pom_image.jpg \
    --temperature 35.0 \
    --thickness 10.0 \
    --order 1 \
    --model models/birefringence_model_efficientnet_b0.pth

Outputs: Predicted delta_n with confidence estimate.
"""

import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from dataset import detect_circular_roi, sample_patches_from_circle
from model import HybridCNN


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    backbone = checkpoint.get("backbone", "efficientnet_b0")
    model = HybridCNN(backbone_name=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model: {backbone} from {model_path}")
    return model


@torch.no_grad()
def predict_birefringence(
    model,
    image_path: str,
    temperature_C: float,
    thickness_um: float,
    order: int,
    device: torch.device,
    n_patches: int = 20,
    patch_size: int = 224,
):
    """
    Predict birefringence from a POM image.

    Returns:
        mean_dn: Mean predicted delta_n across patches
        std_dn: Std of predictions (uncertainty estimate)
        all_preds: All patch-level predictions
    """
    # Load and process image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    cx, cy, r = detect_circular_roi(img_np)

    print(f"Image: {img.size}, ROI center=({cx},{cy}), radius={r}")

    # Extract patches
    patches = sample_patches_from_circle(
        img, cx, cy, r, patch_size=patch_size, n_patches=n_patches,
    )

    # Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build batch
    Tc_approx = 53.7 if temperature_C > 50 else 40.0  # rough heuristic
    scalars = torch.tensor([
        temperature_C / 60.0,
        thickness_um / 15.0,
        order / 6.0,
        temperature_C / Tc_approx,
    ], dtype=torch.float32).unsqueeze(0).repeat(len(patches), 1).to(device)

    images = torch.stack([transform(p) for p in patches]).to(device)

    # Predict
    preds = model(images, scalars).cpu().numpy()

    # Clip negative predictions (physically impossible)
    preds = np.clip(preds, 0, None)

    mean_dn = float(np.mean(preds))
    std_dn = float(np.std(preds))
    median_dn = float(np.median(preds))

    return mean_dn, std_dn, median_dn, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in C")
    parser.add_argument("--thickness", type=float, required=True, help="Thickness in um")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--model", type=str, default="models/birefringence_model_efficientnet_b0.pth")
    parser.add_argument("--n-patches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    mean_dn, std_dn, median_dn, all_preds = predict_birefringence(
        model, args.image, args.temperature, args.thickness, args.order,
        device, n_patches=args.n_patches,
    )

    # Confidence based on prediction spread
    if std_dn < 0.005:
        confidence = "HIGH"
    elif std_dn < 0.015:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    print(f"\n{'='*50}")
    print(f"PREDICTION RESULT")
    print(f"  Input:")
    print(f"    Image:       {args.image}")
    print(f"    Temperature: {args.temperature} C")
    print(f"    Thickness:   {args.thickness} um")
    print(f"    Order:       {args.order}")
    print(f"  Output:")
    print(f"    delta_n (mean):   {mean_dn:.6f}")
    print(f"    delta_n (median): {median_dn:.6f}")
    print(f"    Uncertainty:      +/- {std_dn:.6f}")
    print(f"    Confidence:       {confidence}")
    print(f"    Retardation:      {mean_dn * args.thickness * 1000:.1f} nm")
    print(f"  Patch predictions: min={all_preds.min():.6f}, max={all_preds.max():.6f}")


if __name__ == "__main__":
    main()
