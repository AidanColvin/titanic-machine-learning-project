# Kaggle — Titanic: Machine Learning from Disaster

[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/competitions/titanic)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Predict passenger survival on the Titanic using an ensemble of classical and gradient-boosted machine learning models. Best single model: **XGBoost at 84.18% accuracy**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Models & Results](#models--results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
- [Citation](#citation)

---

## Overview

This repository contains a complete machine learning pipeline for the **Kaggle Titanic competition** — one of the most well-known introductory competitions in data science. The goal is binary classification: predict whether a passenger survived the sinking of the Titanic.

Six models are trained, cross-validated, and benchmarked. A voting ensemble combines the top performers to reduce variance and improve generalization.

---

## Dataset

The dataset contains passenger information including age, sex, ticket class, fare, cabin, and embarkation port. The target variable is `Survived` (1 = survived, 0 = did not survive).

| File | Description |
|---|---|
| `data/train.csv` | Labeled training data |
| `data/test.csv` | Unlabeled test data for submission |
| `data/train_cleaned.parquet` | Cleaned training set |
| `data/train_processed.parquet` | Feature-engineered training set |
| `data/test_cleaned.parquet` | Cleaned test set |
| `data/test_processed.parquet` | Feature-engineered test set |

---

## Repository Structure

```
.
├── data/
│   ├── train.csv / test.csv                  # Raw competition data
│   ├── train.parquet / test.parquet           # Parquet versions of raw data
│   ├── train_cleaned.parquet                  # After cleaning
│   ├── train_processed.parquet                # After feature engineering
│   ├── test_cleaned.parquet
│   ├── test_processed.parquet
│   ├── submission_logreg.csv
│   ├── submission_svm.csv
│   ├── submission_rf.csv
│   ├── submission_gb.csv
│   ├── submission_ensemble.csv
│   └── submission_ultimate.csv
├── scripts/
│   ├── clean_data.py                          # Data cleaning
│   ├── convert_data.py                        # CSV → Parquet conversion
│   ├── process_data.py                        # Feature engineering
│   ├── train_logreg.py                        # Logistic Regression
│   ├── train_svm.py                           # Support Vector Machine
│   ├── train_rf.py                            # Random Forest
│   ├── train_gb.py                            # Gradient Boosting
│   ├── run_xgboost.py                         # XGBoost
│   ├── train_model.py                         # Generic model trainer
│   ├── train_ensemble.py                      # Voting ensemble
│   ├── train_final_ensemble.py                # Final ensemble for submission
│   ├── train_final_robust.py                  # Robust final model
│   ├── setup_ensemble.py                      # Ensemble configuration
│   ├── setup_repo.py                          # Repo initialization
│   ├── benchmark_models.py                    # Cross-validated benchmarking
│   ├── benchmark_advanced.py                  # Advanced benchmarking
│   └── compare_models.py                      # Model comparison & leaderboard
└── docs/
    ├── Kaggle_README.md                       # Competition reference
    ├── Rules.md                               # Competition rules
    ├── data_files.md                          # Data file descriptions
    ├── data_pipeline.md                       # Pipeline documentation
    ├── model_experiments.md                   # Experiment log
    └── pipeline_data.md                       # Pipeline data reference
```

---

## Models & Results

All models were evaluated using **5-fold stratified cross-validation**. Accuracy and standard deviation are reported across folds.

| Model | CV Accuracy | Std Dev |
|---|:---:|:---:|
| Logistic Regression | 0.8137 | ±0.0280 |
| Support Vector Machine | 0.8316 | ±0.0192 |
| LightGBM | 0.8339 | ±0.0274 |
| Random Forest | 0.8384 | ±0.0326 |
| CatBoost | 0.8417 | ±0.0281 |
| **XGBoost** | **0.8418** | **±0.0332** ✅ |

> **Best model:** XGBoost at **84.18% accuracy**.
>
> **Voting Ensemble** combines the mathematical strengths of all models to reduce variance and generalization error across unseen data.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AidanColvin/titanic.git
   cd titanic
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

Scripts must be run **in order**. Data must be cleaned and processed before training.

```bash
# 1. Convert and clean raw data
python3 scripts/convert_data.py && \
python3 scripts/clean_data.py && \
python3 scripts/process_data.py

# 2. Train all models
python3 scripts/train_logreg.py && \
python3 scripts/train_svm.py && \
python3 scripts/train_rf.py && \
python3 scripts/train_gb.py && \
python3 scripts/run_xgboost.py

# 3. Train ensemble and compare
python3 scripts/train_ensemble.py && \
python3 scripts/compare_models.py
```

Submission CSVs will be saved to the `data/` directory.

---

## Citation

```bibtex
@misc{titanic,
    author       = {Aidan Colvin},
    title        = {Titanic - Machine Learning from Disaster},
    year         = {2026},
    howpublished = {\url{https://kaggle.com/competitions/titanic}},
    note         = {Kaggle}
}
```
