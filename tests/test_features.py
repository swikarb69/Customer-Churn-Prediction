import pytest
import pandas as pd
import numpy as np
from src.features import engineer_features, FeaturePreprocessor

def sample_df():
    return pd.DataFrame({
        "customerID": ["001", "002"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 100.0],
        "TotalCharges": [600.0, 2400.0],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["Yes", "Yes"],
        "StreamingMovies": ["Yes", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"]
    })

def test_engineer_features():
    df = sample_df()
    eng = engineer_features(df)

    assert "customerID" not in eng.columns
    assert "tenure_years" in eng.columns
    assert eng["tenure_years"].iloc[0] == 1.0
    assert "avg_monthly_charges" in eng.columns
    assert "total_services_subscribed" in eng.columns
    assert eng["has_security_bundle"].iloc[0] == 1
    assert eng["has_security_bundle"].iloc[1] == 0
    assert eng["has_streaming_bundle"].iloc[0] == 1
    assert eng["is_automatic_payment"].iloc[1] == 1

def test_feature_preprocessor_fit_transform():
    df = sample_df()
    prep = FeaturePreprocessor()
    arr = prep.fit_transform(df)

    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 2
    assert not np.isnan(arr).any()

    names = prep.get_feature_names_out()
    assert len(names) == arr.shape[1]
