<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=32&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&width=800&height=70&lines=Customer+Churn+Prediction;Telecom+%7C+ML+%7C+Recall-Optimized)](https://git.io/typing-svg)

<img src="https://capsule-render.vercel.app/api?type=rect&height=3&color=gradient&customColorList=12" width="100%"/>

![Python](https://img.shields.io/badge/Python-A78BFA?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-818CF8?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-C084FC?style=for-the-badge&logo=pandas&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-7C3AED?style=for-the-badge&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-818CF8?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Shipped-A78BFA?style=for-the-badge)

> **Predicting which telecom customers will leave — before they do.**
> Built around maximizing recall on the churn class, because a missed churner costs far more than a false alarm.

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

---

## `◈` The Problem

Telecom churn datasets are naturally imbalanced — typically **~27% churn rate**. A naive model that always predicts "stays" hits 73% accuracy without learning anything useful. It looks good on paper while failing the retention team entirely.

```
Standard accuracy trap:
  Predict "No Churn" for everyone → 73% accuracy ✓ (but useless)

What actually matters:
  Did we catch the customers who WILL leave? → Recall on churn class
```

A missed churner = lost lifetime revenue. An unnecessary retention offer = small coupon cost.
The asymmetry is obvious. This project is built around that asymmetry.

---

## `◈` Approach

```python
strategy = {
    "problem"    : "Binary classification — Churn vs No Churn",
    "imbalance"  : "SMOTE applied to training set only (no data leakage)",
    "model"      : "GradientBoostingClassifier",
    "objective"  : "Maximize recall on the churn class",
    "technique"  : "Threshold tuning via predict_proba",
}
```

**Why threshold tuning?**
The default `predict_proba` threshold of `0.5` is arbitrary. By lowering it to `~0.35`, we flag more customers as at-risk — trading some precision for a significant recall gain. This is a deliberate business decision, not a mistake.

---

## `◈` Results

<div align="center">

| Metric | Baseline (threshold = 0.50) | Tuned (threshold ≈ 0.35) |
|:---|:---:|:---:|
| **Churn Recall** | 0.45 | **0.62** |
| **Decision Threshold** | 0.50 | ~0.35 |
| **What it means** | Miss 55% of churners | Catch 62% of churners |

</div>

> Lowering the threshold from `0.50` → `0.35` improved churn recall by **+17 percentage points** — meaning the model now correctly flags 38% more at-risk customers for the retention team to act on.

---

## `◈` Pipeline

```
raw data
   │
   ▼
① Load & Clean
   ├── TotalCharges: coerce to numeric, fill missing with median
   └── Drop customerID (no signal)
   │
   ▼
② Feature Engineering
   ├── One-hot encode all categorical columns (pd.get_dummies)
   └── Stratified train/test split (80/20)
   │
   ▼
③ Handle Class Imbalance
   └── SMOTE on training set only → balanced classes, no leakage
   │
   ▼
④ Model Training
   └── GradientBoostingClassifier
   │
   ▼
⑤ Evaluation
   ├── Confusion matrix
   ├── Classification report (precision / recall / F1)
   └── ROC-AUC score
   │
   ▼
⑥ Threshold Tuning
   └── predict_proba → sweep thresholds → pick ~0.35 for recall gain
```

---

## `◈` Tech Stack

<div align="center">

| Tool | Purpose |
|:---|:---|
| `Python` | Core language |
| `Pandas` | Data loading, cleaning, feature engineering |
| `Scikit-Learn` | Model training, evaluation, metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `NumPy` | Numerical operations |

</div>

---

## `◈` Dataset

**[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** — IBM sample dataset

- **7,043** customer records
- **21** features: demographics, account info, services subscribed
- **Target:** `Churn` (Yes/No) — ~27% positive class

Place the CSV at `data/Telco-Customer-Churn.csv` before running.

---

## `◈` Run It

```bash
# Clone the repo
git clone https://github.com/swikarb69/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

**Expected output:**
```
Confusion Matrix
Accuracy score
Classification Report (precision / recall / F1 per class)
Top 10 Feature Importances
ROC-AUC Score
Model trained!
```

---

## `◈` Key Design Decisions

**1. SMOTE on training set only**
Applying SMOTE before the train/test split would leak synthetic samples into evaluation — inflating metrics. SMOTE is fit and applied only on `X_train`, never touching `X_test`.

**2. Recall over accuracy**
In a subscription business, the cost matrix is asymmetric:
- False Negative (miss a churner) → lose the customer entirely
- False Positive (flag a non-churner) → send an unnecessary retention offer

Optimizing recall directly reflects this business reality.

**3. Threshold tuning over resampling alone**
SMOTE helps during training. Threshold tuning at inference time gives a second lever to push recall further without retraining — useful for production scenarios where the model is frozen.

---

## `◈` Next Steps

- [ ] **XGBoost / LightGBM** — compare against GradientBoosting baseline
- [ ] **SHAP values** — explain individual predictions for the retention team
- [ ] **Cost-based threshold optimization** — assign real $ values to FP/FN and find the profit-maximizing threshold
- [ ] **Feature selection** — remove low-importance features to reduce noise
- [ ] **Streamlit dashboard** — let a non-technical user input customer data and get churn probability + explanation

---

## `◈` Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv     # Dataset (download from Kaggle)
│
├── notebooks/                        # Exploratory analysis (optional)
│
├── outputs/                          # Saved plots / results
│
├── main.py                           # Full ML pipeline
├── requirements.txt                  # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&height=2&color=gradient&customColorList=12" width="100%"/>

**Built by [Swikar Bhattarai](https://github.com/swikarb69) · Nepal 🇳🇵**

[![GitHub](https://img.shields.io/badge/GitHub-swikarb69-A78BFA?style=for-the-badge&logo=github&logoColor=white)](https://github.com/swikarb69)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Swikar_Bhattarai-818CF8?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swikar-bhattarai-11178b240)

*"The universe speaks in patterns. I speak Python."*

</div>