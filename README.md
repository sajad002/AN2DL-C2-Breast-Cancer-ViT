# AN2DL 2025–2026 — Image Classification 2.0 (Kaggle)

Course project for **Artificial Neural Networks and Deep Learning (AN2DL)** at Politecnico di Milano (A.Y. 2025–2026).

**Goal:** slide-level classification of breast cancer **molecular subtypes** from low-magnification histopathology images using a **patch-based, mask-aware** pipeline.

## Team — *How I Met Your Model*

- Sajjad Shaffaf
- Farnoosh AfshinRad
- Dorsa Motiallah
- Morteza Kazemi



## Task

Each whole-slide image (WSI) is classified into one of four classes:

- **Luminal A**
- **Luminal B**
- **HER2-positive**
- **Triple Negative**

**Metric:** slide-level **Macro F1-score**.

**Data (as described in our report):** 691 training images and 477 test images, each paired with a binary tumor mask.

> Note: The Kaggle competition is private (course competition). Dataset files are not included in this repository.

## Method (high level)

Our approach is built around the idea that **what** the tissue looks like (RGB appearance) and **where** the lesion is (mask geometry) are both important.

- **Data cleaning**
  - Removed artifact-heavy “Shrek” slides.
  - Automated **green-ink removal** via HSV detection + morphological dilation + inpainting.
  - Zeroed corresponding pixels in masks to avoid spurious supervision.

- **Patch extraction (“bag of patches”)**
  - Sliding window of **224×224** with stride **95** (~50% overlap).
  - Kept patches with **> 5%** mask coverage.
  - Filtered overly dark/bright/low-texture patches.
  - Removed near-duplicate patches using cosine similarity (**> 0.98**).

- **Dual-path model (RGB + mask fusion)**
  - **RGB branch:** ViT-Small backbone pretrained with **DINO on histopathology** (Lunit DINO).
  - **Mask branch:** lightweight CNN encoder for lesion geometry.
  - **Fusion:** concatenation + MLP head.

- **Slide-level inference**
  - Average softmax probabilities across patches for the same slide.
  - Test-time augmentation (flips/rotations) for robustness.

## Results (from our report)

Macro F1 (%) — **Local / Kaggle**:

- **Best single model:** `DP + Lunit DINO + mask gen.` → **41.98 / 41.18**
- **Final ensemble:** `Ensemble (Best 4 + Diverse 4)` → **43.71** on Kaggle (**18th** on the leaderboard)

## Repository contents

- `patch_extraction_preprocessing_with_mask.ipynb` — data cleaning + patch extraction (image/mask pairs)
- `[DualPath_Lunit_dino]_with_masks_gen_Histo_final.ipynb` — dual-path training/inference (mask-aware ViT + mask encoder)
- `ensembling_submissions.ipynb` — analyze multiple Kaggle submissions and create ensembles
- `Grumpy_Doctogres_EDA.ipynb` — EDA (class imbalance, artifacts, leakage/duplicates checks)
- `AN2DL_Homeworks_Report_Second_Challenge.pdf` — final project report

Optional reading:

- `How I Met Your Model_ Navigating Cancer Subtypes, Green Ink, and… Shrek_ 🧬 _ by Farnoosh AfshinRad _ Medium.pdf`

## How to run

### Option A — Google Colab (recommended)

These notebooks were developed in Colab and may mount Google Drive.

1. Open a notebook in Colab.
2. Mount Drive (already included in some notebooks).
3. Place the competition dataset in your Drive and update the path variables in the first cells.
   - Example paths used in the notebooks: `/content/drive/MyDrive/an2dl-2` or `/content/drive/MyDrive/an2dl-c2`.
4. Run top-to-bottom.

### Option B — Local

A GPU-enabled environment is strongly recommended.

- Install common dependencies (approximate; see notebook imports for exact versions):

```bash
pip install -U numpy pandas matplotlib seaborn plotly opencv-python pillow scipy scikit-learn tqdm \
  torch torchvision timm torch-optimizer torchsummary
```

Then open the notebooks locally and update the dataset paths.



## Links

- Kaggle competition (original): https://www.kaggle.com/competitions/an2dl2526c2
- Kaggle competition (updated / v2): https://www.kaggle.com/competitions/an2dl2526c2v2
- Medium post: https://medium.com/@radfarnush/how-i-met-your-model-navigating-cancer-subtypes-green-ink-and-shrek-12473ad290ba

## Disclaimer

This repository is for educational/research purposes as part of the AN2DL course challenge. It is **not** a medical device and should not be used for clinical decision-making.
