<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=32&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&width=800&height=70&lines=Customer+Churn+Prediction;Modular+ML+%7C+SMOTE+%7C+Recall-Optimized)](https://git.io/typing-svg)

<img src="https://capsule-render.vercel.app/api?type=rect&height=3&color=gradient&customColorList=12" width="100%"/>

![Python](https://img.shields.io/badge/Python-3.10%2B-A78BFA?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-818CF8?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-7C3AED?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-C084FC?style=for-the-badge&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-7.0%2B-22C55E?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-818CF8?style=for-the-badge)

> **Predicting telecom churners before they leave — powered by modular ML architecture, SMOTE resampling, and threshold optimization.**

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

---

## `◈` Key Highlights & Benchmarks

| Metric | Baseline (Threshold = 0.50) | Tuned Model (Threshold = 0.26) | Gain |
|:---|:---:|:---:|:---:|
| **Churn Recall** | 62.8% | **84.0%** | **+21.2%** |
| **ROC-AUC Score** | 0.8418 | **0.8418** | High Discrimination |
| **F1-Score (Churn)** | 0.6088 | **0.6312** | Optimized |
| **Caught Churners** | 235 / 374 | **314 / 374** | +79 Churners Saved |

> **Business Impact**: Lowering the decision threshold from `0.50` to `0.26` boosts customer churn recall to **84.0%**, identifying **314 out of 374 at-risk customers** for retention outreach.

---

## `◈` Architectural Design & Principles

```
raw dataset
    │
    ▼
[DataLoader] ───────────► Clean NaNs, coerce types, stratified train/test split
    │
    ▼
[FeatureEngineer] ──────► Generate domain features (tenure_years, service_count, bundles)
    │
    ▼
[ColumnTransformer] ────► Scale numeric features, one-hot encode categoricals (fit on X_train ONLY)
    │
    ▼
[ModelTrainer + SMOTE] ─► Over-sample X_train ONLY, compare RandomForest vs GradientBoosting vs XGBoost
    │
    ▼
[ThresholdOptimizer] ──► Sweep probability thresholds [0.10, 0.90] for max recall/F1
    │
    ▼
[Model Persistence] ───► Save preprocessor.joblib, best_model.joblib, threshold.joblib
```

1. **Zero Data Leakage**: Scalers, imputers, categorical encoders, and SMOTE resampling are fit **strictly on training splits (`X_train`)**, ensuring clean evaluation on unseen test data.
2. **Modular Architecture**: Decoupled modules for configuration (`src/config.py`), data loading (`src/data.py`), feature engineering (`src/features.py`), modeling (`src/models.py`), evaluation (`src/evaluate.py`), and CLI inference (`predict.py`).
3. **Automated Testing**: 100% passing `pytest` test suite verifying data cleaning, feature outputs, pipeline transformations, model training, and threshold optimization.

---

## `◈` Tech Stack

| Component | Library / Framework |
|:---|:---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost, imbalanced-learn (SMOTE) |
| **Serialization** | Joblib |
| **Testing** | pytest |

---

## `◈` Quick Start & Usage

### 1. Installation
```bash
git clone https://github.com/swikarb69/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

### 2. Run the Full ML Training Pipeline
```bash
python main.py
```

### 3. Run CLI Inference
Score a sample customer record:
```bash
python predict.py --sample
```
Or batch score a CSV file:
```bash
python predict.py --csv data/Telco-Customer-Churn.csv
```

### 4. Run Automated Unit Tests
```bash
pytest tests/ -v
```

---

## `◈` Repository Structure

```
Customer-Churn-Prediction/
├── data/
│   └── Telco-Customer-Churn.csv      # IBM Telco Churn Dataset
├── src/
│   ├── __init__.py
│   ├── config.py                     # Centralized paths, columns & hyperparameters
│   ├── utils.py                      # Logging and joblib serialization helpers
│   ├── data.py                       # DataLoader & cleaning pipeline
│   ├── features.py                   # Domain feature engineering & ColumnTransformer
│   ├── models.py                     # ModelTrainer with SMOTE & ThresholdOptimizer
│   └── evaluate.py                   # Evaluation metrics & feature importances
├── tests/
│   ├── __init__.py
│   ├── test_data.py                  # Unit tests for data pipeline
│   ├── test_features.py              # Unit tests for feature pipeline
│   └── test_models.py                # Unit tests for ML engine & threshold optimizer
├── models/                           # Saved trained artifacts (.joblib)
├── outputs/                          # Batch inference exports
├── main.py                           # Full pipeline orchestrator
├── predict.py                        # CLI inference interface
├── requirements.txt                  # Environment dependencies
├── .gitignore
└── README.md                         # Documentation
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

**Built by [Swikar Bhattarai](https://github.com/swikarb69) · Nepal 🇳🇵**

[![GitHub](https://img.shields.io/badge/GitHub-swikarb69-A78BFA?style=for-the-badge&logo=github&logoColor=white)](https://github.com/swikarb69)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Swikar_Bhattarai-818CF8?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swikar-bhattarai-11178b240)

</div>