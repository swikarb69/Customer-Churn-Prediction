"""
Customer Churn Prediction — Inference CLI.
"""

import argparse
import pandas as pd
import numpy as np

from src.config import PREPROCESSOR_PATH, MODEL_PATH, THRESHOLD_PATH
from src.utils import load_artifact, setup_logger

logger = setup_logger("inference")

def load_pipeline_artifacts():
    """Loads saved preprocessor, best model, and tuned threshold."""
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    model = load_artifact(MODEL_PATH)
    try:
        threshold = load_artifact(THRESHOLD_PATH)
    except Exception:
        threshold = 0.35
    return preprocessor, model, threshold

def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Makes churn predictions on a raw dataframe."""
    preprocessor, model, threshold = load_pipeline_artifacts()

    # Preprocess features
    X_proc = preprocessor.transform(df)

    # Predict probabilities
    probs = model.predict_proba(X_proc)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = df.copy()
    results["Churn_Probability"] = probs.round(4)
    results["Predicted_Churn"] = preds
    results["Risk_Level"] = np.where(preds == 1, "HIGH RISK", "LOW RISK")

    return results

def get_sample_customer() -> pd.DataFrame:
    """Returns a sample customer dataframe for demonstration."""
    return pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0
    }])

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Inference CLI")
    parser.add_argument("--csv", type=str, help="Path to input CSV file for batch prediction")
    parser.add_argument("--sample", action="store_true", help="Run inference on a sample customer payload")

    args = parser.parse_args()

    if args.sample or not args.csv:
        logger.info("Running inference on sample customer record...")
        sample_df = get_sample_customer()
        res = predict_dataframe(sample_df)
        print("\n" + "="*50)
        print("Customer Churn Inference Result:")
        print("="*50)
        for col in ["tenure", "Contract", "MonthlyCharges", "Churn_Probability", "Predicted_Churn", "Risk_Level"]:
            print(f"  {col}: {res[col].iloc[0]}")
        print("="*50 + "\n")
    elif args.csv:
        logger.info(f"Running batch inference on CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        res = predict_dataframe(df)
        out_file = "outputs/predictions_output.csv"
        res.to_csv(out_file, index=False)
        logger.info(f"Batch predictions saved to '{out_file}'.")

if __name__ == "__main__":
    main()
