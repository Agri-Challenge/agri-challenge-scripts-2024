# AgrI Challenge -- Scripts and Baselines (2024)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official code repository for the paper:

> **AgrI Challenge: Cross-Team Insights from a Data-Centric AI Competition in Agricultural Vision**  
> Mohammed Brahimi, Karim Laabassi, Mohamed Seghir Hadj Ameur, Aicha Boutorh, Badia Siab-Farsi,  
> Amin Khouani, Omar Farouk Zouak, Seif Eddine Bouziane, Kheira Lakhdari, Abdelkader Nabil Benghanem  
> *arXiv preprint, 2026*  
> Paper: https://arxiv.org/abs/2603.07356
> Website: https://www.ensia.edu.dz/agri-challenge.html

---

## Overview

This repository contains the data processing pipelines and the Cross-Team Validation (CTV) training framework used in the AgrI Challenge. The AgrI Challenge is a data-centric competition in which twelve interdisciplinary teams independently collected field images of six tree species across a two-day campaign. The resulting dataset (50,673 images) is used to study domain generalization in agricultural vision.

The code is organized into three data preparation notebooks and one CTV training script, corresponding directly to the experimental pipeline described in the paper.

---

## Repository Structure

```
agri-challenge-scripts-2024/
|-- 1.dataset_stats.ipynb           # Scan raw images, extract metadata, detect duplicates
|-- 2.clean_duplicates.ipynb        # Remove duplicate images using perceptual hash rules
|-- 3.build_resized_dataset.ipynb   # Resize and normalize the dataset to a uniform resolution
|-- 4.agri_challenge_ctv_cli_v3.py  # CTV training script (TOTO and LOTO protocols)
|-- README.md
```

---

## Dataset

The AgrI Challenge dataset contains **50,673 field images** across **6 tree species** collected by **12 independent teams** at the National Higher School of Agronomy (ENSA), El Harrach, Algeria.

**Tree species:**
| Common Name | Scientific Name | French Name |
|---|---|---|
| Carob tree | *Ceratonia siliqua* | Caroubier |
| Oak | *Quercus* spp. | Chene |
| Peruvian pepper tree | *Schinus molle* | Faux poivrier |
| Ash | *Fraxinus* spp. | Frene |
| Pistachio tree | *Pistacia vera* | Pistachier |
| Tipu tree | *Tipuana tipu* | Tipu |

The dataset is available upon request. Please submit an access request at the project website: https://www.ensia.edu.dz/agri-challenge.html

---

## Requirements

```bash
pip install torch torchvision timm pandas Pillow pillow-heif imagehash \
            scikit-learn matplotlib seaborn tqdm wandb
```

- Python 3.9 or later
- PyTorch 2.0 or later (recommended for `torch.compile` support)
- A CUDA-capable GPU is strongly recommended for `4.agri_challenge_ctv_cli_v3.py`

---

## Data Preparation Pipeline

The three notebooks must be run in order on the raw dataset.

### Step 1: Dataset Statistics (`1.dataset_stats.ipynb`)

Recursively scans the raw dataset directory, extracts per-image metadata (file size, resolution, EXIF device information, perceptual hash), and writes a summary CSV.

**Configuration (Cell 2):**

| Parameter | Description |
|---|---|
| `DATASET_ROOT` | Path to the root folder containing team subdirectories |
| `TEAMS_TO_SCAN` | List of team folder names to scan; leave empty for all teams |
| `CALCULATE_HASH` | Set `True` to compute perceptual hashes (required for duplicate detection) |

**Outputs:**
- `image_dataset_summary_with_hash.csv` (or `..._no_hash.csv`)
- `duplicates.csv` -- images sharing the same perceptual hash

---

### Step 2: Duplicate Removal (`2.clean_duplicates.ipynb`)

Reads `duplicates.csv` and retains one representative image per duplicate group according to a deterministic priority: largest file size > highest resolution > known capture device. All other duplicates are moved (not deleted) to a configurable destination folder.

**Configuration (Cell 4):**

| Parameter | Description |
|---|---|
| `destination_folder` | Folder to which duplicate images are moved |
| `debug_mode` | Set `True` to simulate moves without modifying any files |

---

### Step 3: Build Resized Dataset (`3.build_resized_dataset.ipynb`)

Reads the metadata CSV, resizes every image to a uniform square resolution (default: 336x336 px), normalizes class folder names via a configurable mapping, and saves the processed dataset under a new directory tree organised as `<output_root>/<team>/<class>/`. Output filenames are the perceptual hash of the source image.

**Configuration (Cell 2):**

| Parameter | Description |
|---|---|
| `METADATA_CSV_PATH` | Path to the summary CSV from Step 1 |
| `TARGET_SIZE` | Output resolution in pixels (square) |
| `OUTPUT_FORMAT` | `"JPEG"` or `"PNG"` |
| `JPEG_QUALITY` | JPEG quality (1-95); only used when `OUTPUT_FORMAT="JPEG"` |
| `CLASS_MAPPING` | Dictionary mapping raw folder names to normalized class names |

**Outputs:**
- Resized image tree at `OUTPUT_ROOT_DIR/`
- `resizing_report.csv`
- `dataset_summary_clean.csv`
- `dataset_statistics.csv`

---

## CTV Training Framework (`4.agri_challenge_ctv_cli_v3.py`)

Implements the two Cross-Team Validation protocols introduced in the paper.

**TOTO (Train-on-One-Team-Only):** For each team T, a model is trained exclusively on T's data and evaluated on the combined data of all other teams. This measures single-source generalization.

**LOTO (Leave-One-Team-Out):** For each team T, a model is trained on the combined data of all other teams and evaluated on T's data. This measures collaborative multi-source generalization.

Both protocols are implemented for two backbone architectures: DenseNet121 and Swin Transformer (via `timm`).

### Usage

```bash
# TOTO with Swin Transformer, 20 epochs
python 4.agri_challenge_ctv_cli_v3.py --scenario TOTO --architecture swin --epochs 20

# LOTO with DenseNet121, 30 epochs
python 4.agri_challenge_ctv_cli_v3.py --scenario LOTO --architecture densenet --epochs 30

# Quick test run
python 4.agri_challenge_ctv_cli_v3.py --scenario TOTO --architecture swin --epochs 5 --batch-size 16

# Disable W&B tracking
python 4.agri_challenge_ctv_cli_v3.py --scenario LOTO --architecture densenet --epochs 20 --no-wandb
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--scenario` | required | `TOTO` or `LOTO` |
| `--architecture` | required | `densenet` or `swin` |
| `--data-dir` | `./data/...` | Path to the resized dataset directory |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--val-split` | `0.3` | Fraction of training data reserved for validation |
| `--dropout` | `0.3` | Dropout rate in the classification head |
| `--num-workers` | `4` | Number of DataLoader workers |
| `--seed` | `42` | Random seed for reproducibility |
| `--output-dir` | auto | Output directory (timestamped by default) |
| `--no-wandb` | -- | Disable Weights and Biases logging |
| `--no-augmentation` | -- | Disable training data augmentation |
| `--no-compile` | -- | Disable `torch.compile` (required for PyTorch < 2.0) |
| `--no-mixed-precision` | -- | Disable automatic mixed precision (AMP) |

**Weights and Biases:** If W&B logging is enabled (default when `wandb` is installed), authentication is handled via the `WANDB_API_KEY` environment variable or an interactive `wandb login` prompt. No API keys are stored in the code.

### Outputs

Each run creates a timestamped experiment directory under `./experiments_ctv/` containing:

```
CTV_<SCENARIO>_<ARCH>_epochs_<N>_<timestamp>/
|-- models/          # Best model checkpoints (.pth)
|-- plots/           # Training curves and confusion matrices
|-- results/         # Per-iteration result JSON files and summary
|-- matrices/        # Cross-team accuracy matrix (CSV + heatmap PNG)
|-- predictions/     # Per-image prediction CSVs (train / val / test splits)
|-- logs/
```

---

## Training Details

The following hyperparameters were used to produce the results reported in the paper.

| Hyperparameter | Value |
|---|---|
| Input resolution | 224 x 224 px |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Learning rate schedule | Cosine annealing (eta_min = 1e-6) |
| Early stopping patience | 30 epochs |
| Validation split | 30% of training data |
| Data augmentation | Random resized crop, horizontal/vertical flip, rotation, color jitter, random erasing |
| Random seed | 42 |

---

## Experimental Results

Results from the paper (top-1 accuracy, averaged across teams):

**TOTO -- Single-source generalization**

| Architecture | Val Accuracy | Test Accuracy | Val-Test Gap |
|---|---|---|---|
| DenseNet121 | ~98% | ~82% | up to 16.20% |
| Swin Transformer | ~97% | ~86% | up to 11.37% |

**LOTO -- Multi-source collaborative training**

| Architecture | Val Accuracy | Test Accuracy | Val-Test Gap |
|---|---|---|---|
| DenseNet121 | ~97% | ~94% | 2.82% |
| Swin Transformer | ~98% | ~96% | 1.78% |

Collaborative multi-source training (LOTO) substantially reduces the generalization gap compared to single-source training (TOTO), demonstrating the value of diverse data collection across teams.

---

## Citation

If you use this code or the AgrI Challenge dataset in your research, please cite:

```bibtex
@article{brahimi2026agrichallenge,
  title   = {AgrI Challenge: Cross-Team Insights from a Data-Centric
             AI Competition in Agricultural Vision},
  author  = {Brahimi, Mohammed and Laabassi, Karim and
             {Hadj Ameur}, Mohamed Seghir and Boutorh, Aicha and
             Siab-Farsi, Badia and Khouani, Amin and
             Zouak, Omar Farouk Zouak and Bouziane, Seif Eddine and
             Lakhdari, Kheira and Benghanem, Abdelkader Nabil},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.07356}
}
```

---

## License

The code in this repository is released under the [MIT License](LICENSE).

The AgrI Challenge dataset is made available for **non-commercial research use only** under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Dataset access is managed via the project website. Redistribution of the dataset without the authors' written permission is not permitted.

---

## Contact

Corresponding author: Mohammed Brahimi  
Email: mohamed.brahimi@ensia.edu.dz  
Institution: National School of Artificial Intelligence (ENSIA), Algiers, Algeria

Project website: https://www.ensia.edu.dz/agri-challenge.html  
GitHub: https://github.com/Agri-Challenge/agri-challenge-scripts-2024
