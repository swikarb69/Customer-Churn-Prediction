import pytest
import pandas as pd
import numpy as np
from src.data import DataLoader
from src.config import DATA_PATH, TARGET_COL

def test_data_loader_init():
    loader = DataLoader(DATA_PATH)
    assert loader.data_path == DATA_PATH

def test_clean_data():
    raw_df = pd.DataFrame({
        "customerID": ["1001-TEST", "1002-TEST", "1003-TEST"],
        "gender": ["Male", "Female", "Male"],
        "TotalCharges": ["100.5", " ", "300.0"],
        "Churn": ["No", "Yes", "No"]
    })
    loader = DataLoader()
    cleaned = loader.clean_data(raw_df)

    assert cleaned["TotalCharges"].dtype in [np.float64, float]
    assert cleaned["TotalCharges"].isnull().sum() == 0
    assert cleaned["TotalCharges"].iloc[1] == 200.25 # median of [100.5, 300.0]
    assert cleaned[TARGET_COL].tolist() == [0, 1, 0]

def test_load_and_split():
    loader = DataLoader()
    X_tr, X_te, y_tr, y_te = loader.load_and_split(test_size=0.2, random_state=42)
    assert len(X_tr) > 0
    assert len(X_te) > 0
    assert len(X_tr) + len(X_te) == 7043
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
