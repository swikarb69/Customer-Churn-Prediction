import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# File Paths for Saved Artifacts
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.joblib")

# Pipeline Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.35

# Column Definitions
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ENGINEERED_NUMERICAL_COLS = [
    "tenure_years",
    "avg_monthly_charges",
    "total_services_subscribed",
]

ENGINEERED_CATEGORICAL_COLS = [
    "has_security_bundle",
    "has_streaming_bundle",
    "is_automatic_payment",
]
