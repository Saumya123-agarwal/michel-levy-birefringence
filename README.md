# Phase 2: Deep Learning Birefringence Prediction from POM Images

Predict birefringence (Δn) of liquid crystals from **polarized optical microscopy (POM) images** using a hybrid CNN + scalar fusion model.

## Overview

| Input | Output |
|---|---|
| POM image (crossed polars) | Birefringence Δn |
| Temperature (°C) | Confidence score |
| Thickness (μm) | Retardation Γ (nm) |
| Order (integer) | |

### Architecture

```
POM Image (224×224) ──► EfficientNet-B0 (pretrained) ──► 1280-d features ──┐
                                                                            ├──► Fusion MLP ──► Δn
Temperature, Thickness, Order ──► Scalar MLP ──► 32-d features ────────────┘
```

**Key techniques:**
- **Transfer learning**: EfficientNet-B0 pretrained on ImageNet (only 35 POM images available)
- **Patch extraction**: Each 1920×1080 POM image → 25 random 224×224 patches from circular FOV
- **Physics-informed loss**: Non-negativity constraint + monotonicity with temperature
- **Two-phase training**: (A) freeze backbone, train head → (B) fine-tune last backbone blocks

### Two Models

| Model | Description | When to use |
|---|---|---|
| **XGBoost Baseline** | Color statistics + gradient boosting | Quick baseline, interpretable |
| **Hybrid CNN** | EfficientNet-B0 + scalar fusion | Best accuracy, learns textures |

---

## Quick Start

### Option A: Google Colab (Recommended — Free GPU)

1. Upload this project to Google Drive
2. Open `notebooks/train_colab.ipynb` in Colab
3. Set GPU runtime: Runtime → Change runtime type → T4 GPU
4. Run all cells

### Option B: Local Machine

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/phase2_birefringence.git
cd phase2_birefringence
pip install -r requirements.txt

# 2. Place dataset
# Extract dataset_physics_SOP_liquid_crystal.zip into dataset/ folder

# 3. Prepare data
python src/prepare_dataset.py \
    --data-root dataset/dataset_physics_SOP_liquid_crystal \
    --p100-thickness 9.415 --p100-order 1 \
    --pch-thickness 10.0 --pch-order 1

# 4. Run baseline
python src/baseline_xgboost.py --csv data/dataset_unified.csv

# 5. Train DL model (5-fold CV)
python src/train.py \
    --csv data/dataset_unified.csv \
    --backbone efficientnet_b0 \
    --epochs 60 --warmup-epochs 10 \
    --batch-size 16 --n-patches 25 \
    --cv-mode kfold --n-folds 5 \
    --save-final

# 6. Predict on new image
python src/predict.py \
    --image path/to/new_pom_image.jpg \
    --temperature 35.0 \
    --thickness 10.0 \
    --order 1 \
    --model models/birefringence_model_efficientnet_b0.pth

# 7. Visualize results
python src/visualize.py --results results/cv_results_*.json
```

---

## Project Structure

```
phase2_birefringence/
├── src/
│   ├── prepare_dataset.py   # Step 1: Parse Excel + images → unified CSV
│   ├── dataset.py           # Step 2: PyTorch Dataset with patch extraction
│   ├── model.py             # Step 3: HybridCNN architecture + physics loss
│   ├── train.py             # Step 4: Training pipeline with CV
│   ├── baseline_xgboost.py  # Step 2B: XGBoost baseline (comparison)
│   ├── predict.py           # Step 5: Inference on new images
│   └── visualize.py         # Step 6: Plot results
├── notebooks/
│   └── train_colab.ipynb    # One-click Colab notebook
├── data/                    # Generated CSV files
├── dataset/                 # Raw POM images + Excel (not in git)
├── models/                  # Saved model weights
├── results/                 # Evaluation metrics + plots
├── configs/                 # (optional) experiment configs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## GPU Requirements

| Setup | Training Time (5-fold CV) | Notes |
|---|---|---|
| **Google Colab T4** | ~15-25 min | **Recommended** (free) |
| **Local GPU (RTX 3060+)** | ~10-20 min | Best experience |
| **Apple M1/M2 (MPS)** | ~20-30 min | Works well |
| **CPU only** | ~2-4 hours | Feasible but slow |

The model is lightweight (EfficientNet-B0 = 5.3M params). Training is fast because:
- Only 35 source images (875 patches at 25 patches/image)
- Backbone is frozen in Phase A (only training ~100K params)
- Small batch size (16)

**CPU training is fully feasible** — just slower. For a quick test, reduce `--n-patches 10` and `--epochs 30`.

---

## Dataset Details

### P100-7CB+ (folder: `P100-7CB+ Str +t3318`)
- **Material**: 7CB liquid crystal mixture
- **Thickness**: 9.415 μm
- **Order**: 1
- **Images**: 23 (1920×1080, JPG)
- **Temperature**: 30.0°C – 45.0°C
- **Δn range**: 0.0 (isotropic, >40°C) → 0.211 (25°C)

### Pure 5PCH (folder: `M7 - pure 5PCH`)
- **Material**: 5PCH
- **Thickness**: 10.0 μm
- **Order**: 1
- **Images**: 12 (1920×1200, JPG)
- **Temperature**: 35.5°C – 55.5°C
- **Δn range**: 0.0 (isotropic, >53.7°C) → 0.101 (32°C)

---

## How It Works

### Phase 1 (Previous work)
Single pixel color → Michel-Lévy LUT → retardation → Δn

### Phase 2 (This project)
Full POM image + physical parameters → CNN → Δn

The CNN learns to extract relevant features from the **entire texture pattern** — not just a single pixel color. This captures:
- Spatial color distribution across the field of view
- Texture patterns (schlieren, domains, defects)
- Intensity variations that correlate with birefringence
- Temperature-dependent optical changes

### Why Patch Extraction?

With only 35 images, direct CNN training would massively overfit. By extracting 25 random patches per image:
- **35 images → 875 training patches** (25× data multiplication)
- Each patch captures local texture from different parts of the FOV
- Random augmentation (rotation, flip) further diversifies training data
- At inference, predictions from multiple patches are averaged for robustness

---

## GitHub Setup

```bash
# Initialize repository
cd phase2_birefringence
git init
git add .
git commit -m "Initial commit: Phase 2 birefringence prediction"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/phase2_birefringence.git
git branch -M main
git push -u origin main
```

**Note**: The `.gitignore` excludes large files (dataset, model weights). For the dataset:
- Either add a download link in the README
- Or use [Git LFS](https://git-lfs.github.com/) for large files
- Or keep the dataset separate and reference it

For model weights after training:
- Use GitHub Releases to attach `.pth` files
- Or use Git LFS

---

## Extending to New Materials

To add a new liquid crystal material:

1. **Collect POM images** at different temperatures (10+ images recommended)
2. **Measure birefringence** at each temperature (Berek compensator or known method)
3. **Create Excel file** with columns: Temperature, Δn
4. **Update `prepare_dataset.py`** to parse your new material
5. **Retrain** the model

The model generalizes across materials because it learns the physics-to-color mapping through the pretrained backbone.

---

## References

1. Sigaki et al. (2020). "Learning physical properties of liquid crystals with deep CNNs." *Scientific Reports*. [DOI](https://doi.org/10.1038/s41598-020-63662-9)
2. Tasinkevych et al. (2025). "CNN analysis of optical texture patterns in LC skyrmions." *Scientific Reports*. [DOI](https://doi.org/10.1038/s41598-025-89699-2)
3. Chattha et al. (2025). "The use of AI in liquid crystal applications." *Can. J. Chem. Eng.* [DOI](https://doi.org/10.1002/cjce.25452)
