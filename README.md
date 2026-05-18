# Smart Manufacturing — Milling Chatter Detection & Control Current Prediction

> **Course project** applying Neural Networks and Random Forest to synthetic milling-process data for chatter detection and electromagnetic actuator control-current prediction, grounded in peer-reviewed research on active chatter suppression systems.

---

## Table of Contents

- [Overview](#overview)
- [Background & Motivation](#background--motivation)
- [Dataset](#dataset)
- [Data Engineering — Derived Features](#data-engineering--derived-features)
- [Models](#models)
  - [1. Chatter Classification NN](#1-chatter-classification-nn)
  - [2. Counter-Force Regression NN](#2-counter-force-regression-nn)
  - [3. Control Current Prediction — Random Forest](#3-control-current-prediction--random-forest)
- [Project Structure](#project-structure)
- [Requirements & Installation](#requirements--installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Overview

This notebook implements three machine-learning models targeting two core problems in **smart CNC milling**:

| Task | Model | Target |
|---|---|---|
| Chatter Detection | Binary Classification NN | `label_chatter` (0 / 1) |
| Counter-Force Estimation | Regression NN | `counter_force` (N) |
| Control Current Prediction | Random Forest Regressor | `current` (A) |

All models are trained on **synthetic data** that was first generated to match the statistical envelope of real milling experiments, then extended with physically meaningful derived features using equations from the reference papers listed below.

---

## Background & Motivation

Milling chatter is a self-excited vibration that arises from the regenerative effect between the tool and workpiece. Left uncontrolled, it causes poor surface finishes, accelerated tool wear, and loud noise. Modern active suppression systems — such as those employing **electromagnetic actuators (AMB/EMA)** integrated into the spindle — counteract chatter by injecting a real-time control force into the rotating shaft.

This project draws on three research papers to ground the synthetic data and derived features in real physics:

- **Wan et al. (2019)** — PD-controlled electromagnetic actuator integrated into a motorized spindle; linearised force model `F = ki·Ic − ks·s`.
- **Li et al. (2021)** — LMI-based discrete robust controller with the same EMA; system modal parameters (stiffness `k = 2.07 × 10⁶ N/m`, damping `c = 2400 N·s/m`) are adopted directly in the feature engineering step of this notebook.
- **Wan et al. (2022)** — Projection-based robust adaptive controller with Active Magnetic Bearing for chatter mitigation under model-parameter uncertainty.

The derived features (displacement, velocity, counter-force, control current) therefore mirror the physical quantities that a real AMB/EMA controller would compute.

---

## Dataset

### Primary file: `synthetic_data.csv`

500 rows, originally 6 columns:

| Column | Unit | Description |
|---|---|---|
| `spindle_speed_rpm` | RPM | Spindle rotational speed |
| `depth_of_cut_mm` | mm | Axial depth of cut |
| `feed_rate_mm_per_tooth` | mm/tooth | Feed per tooth |
| `vibration_rms` | m/s² (RMS) | RMS vibration amplitude measured at spindle |
| `cutting_force_N` | N | Resultant cutting force |
| `label_chatter` | 0 / 1 | Ground-truth chatter label (1 = chatter detected) |

The dataset was synthetically generated to reflect cutting conditions used in the reference experiments (AL6061 workpiece, 3-flute carbide end mill, Ø 7 mm, 35 mm overhang).

---

## Data Engineering — Derived Features

The regression and Random Forest tasks operate on an **extended dataset** (`df_reg` / `df_rf`) with features derived from the primary data using the mechanical model of the electromagnetic actuator.

### Derived columns

| Column | Formula | Physical meaning |
|---|---|---|
| `displacement` | `vibration_rms × dt` (`dt = 1×10⁻⁴ s`) | Peak spindle shaft displacement |
| `frequency` | `spindle_speed_rpm / 60` | Rotation frequency (Hz) |
| `velocity` | `2π × frequency × displacement` | Shaft surface velocity (m/s) |
| `counter_force` | `k × displacement + c × velocity` | Required active damping force (N); `k = 2.07×10⁶`, `c = 2400` (from Li et al., 2021) |
| `force` *(RF only)* | same as counter_force with `k=1000`, `c=10` | Simplified actuator force for current mapping |
| `current` *(RF target)* | `(force − Kq × displacement) / Ki` | Control current command to EMA; `Ki = 10`, `Kq = 5` (linearised actuator model, Wan et al., 2019) |

---

## Models

### 1. Chatter Classification NN

**Task:** Binary classification — predict whether milling conditions produce chatter.

**Architecture:**

```
Input (5 features)
  → Linear(5 → 32) → ReLU
  → Linear(32 → 16) → ReLU
  → Linear(16 → 1)
  → BCEWithLogitsLoss (sigmoid threshold 0.5)
```

**Training config:**

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Epochs | 150 |
| Train / Test split | 80 / 20 |
| Feature scaling | `StandardScaler` |
| Loss function | `BCEWithLogitsLoss` |
| Random seed | 42 |

**Input features:** `spindle_speed_rpm`, `depth_of_cut_mm`, `feed_rate_mm_per_tooth`, `vibration_rms`, `cutting_force_N`

**Evaluation:** Loss curve, accuracy curve, confusion matrix (via `torchmetrics` + `mlxtend`)

---

### 2. Counter-Force Regression NN

**Task:** Predict the required active counter-force (N) given spindle state measurements.

**Architecture:**

```
Input (6 features)
  → Linear(6 → 64) → ReLU
  → Linear(64 → 32) → ReLU
  → Linear(32 → 1)
  → MSELoss
```

**Training config:**

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 1×10⁻⁴ |
| Epochs | 1000 |
| Train / Test split | 80 / 20 |
| Feature scaling | `StandardScaler` on both X and y |
| Loss function | `MSELoss` |
| Random seed | 42 |

**Input features:** `vibration_rms`, `spindle_speed_rpm`, `depth_of_cut_mm`, `feed_rate_mm_per_tooth`, `displacement`, `velocity`

**Target:** `counter_force`

---

### 3. Control Current Prediction — Random Forest

**Task:** Predict the control current command (A) that an electromagnetic actuator must deliver to cancel chatter forces.

**Model:** `RandomForestRegressor`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `random_state` | 42 |
| Train / Test split | 80 / 20 |

**Input features:** `vibration_rms`, `displacement`, `velocity`, `force`, `spindle_speed_rpm`, `depth_of_cut_mm`, `feed_rate_mm_per_tooth`

**Target:** `current`

**Evaluation:** MSE, MAE, scatter plot (actual vs. predicted), feature importance bar chart.

---

## Project Structure

```
.
├── models.ipynb          # Main notebook (all models)
├── synthetic_data.csv    # Primary dataset (500 rows × 6 columns)
└── README.md             # This file
```

> **Note:** `synthetic_data.csv` must sit in the **same directory** as `models.ipynb` for the `pd.read_csv("synthetic_data.csv")` call to work.

---

## Requirements & Installation

Python **3.10+** is recommended (the notebook was developed on Python 3.13.5).

### Install all dependencies

```bash
pip install torch torchvision torchaudio          # PyTorch (CPU)
# OR for GPU (CUDA 12.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install pandas numpy matplotlib scikit-learn
pip install tqdm torchinfo torchmetrics mlxtend
```

### Full `requirements.txt` (pin versions used in development)

```text
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
torchinfo>=1.8.0
torchmetrics>=1.0.0
mlxtend>=0.22.0
```

Install from this file:

```bash
pip install -r requirements.txt
```

### Hardware note

The notebook auto-selects the best available backend:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()   # Apple Silicon
    else "cpu"
)
```

All models are small enough to train comfortably on CPU in under a few minutes.

---

## How to Run

### 1. Clone / download the repository

```bash
git clone https://github.com/keerthi806/SM_project.git
cd SM_project
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Ensure `synthetic_data.csv` is in the **same folder** as `models.ipynb`.

### 5. Launch Jupyter and run the notebook

```bash
jupyter notebook models.ipynb
# or
jupyter lab models.ipynb
```

Then run all cells in order: **Kernel → Restart & Run All**.

### Cell execution order

| Cell range | Description |
|---|---|
| Cell 0 | Imports & device detection |
| Cells 1 – 4 | Load data, derive features, build tensor utility |
| Cells 6 – 14 | Chatter Classification NN — define, train, evaluate, confusion matrix |
| Cells 15 – 19 | Counter-Force Regression NN — define, train, evaluate |
| Cells 20 – 21 | Random Forest — build features, train, evaluate, feature importance |

---

## Results

### Chatter Classification NN
- Trained for 150 epochs; convergence visible in train/test loss & accuracy curves.
- Confusion matrix plotted for No-Chatter vs. Chatter classes.
- Model summary: **737 total parameters**.

### Counter-Force Regression NN
- Trained for 1000 epochs with scaled targets.
- Train and test MSE loss curves plotted.
- Model summary: **4,353 total parameters**.

### Random Forest — Control Current
- Evaluated with **MSE** and **MAE** on the held-out 20% test set.
- Scatter plot of actual vs. predicted current.
- Feature importance chart identifying `vibration_rms`, `force`, and `displacement` as the dominant predictors.

---

## References

1. **Wan, S., Li, X., Su, W., Yuan, J., Hong, J., & Jin, X.** (2019). Active damping of milling chatter vibration via a novel spindle system with an integrated electromagnetic actuator. *Precision Engineering*, 57, 203–210. https://doi.org/10.1016/j.precisioneng.2019.04.007

2. **Li, X., Wan, S., Yuan, J., Yin, Y., & Hong, J.** (2021). Active suppression of milling chatter with LMI-based robust controller and electromagnetic actuator. *Journal of Materials Processing Technology*, 297, 117238. https://doi.org/10.1016/j.jmatprotec.2021.117238

3. **Wan, S., Li, X., Su, W., & Hong, J.** (2022). Milling chatter mitigation with projection-based robust adaptive controller and active magnetic bearing. *International Journal of Precision Engineering and Manufacturing*, 23, 1453–1463. https://doi.org/10.1007/s12541-022-00710-6

---

> **Course:** Smart Manufacturing  
> **Kernel used in development:** `ml_env` (Python 3.13.5)