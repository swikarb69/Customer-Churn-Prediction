import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE
from src.utils import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Handles dataset loading, cleaning, and train/test splitting."""

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw CSV data from path."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Raw data loaded with shape: {df.shape}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans dataset: handles type conversions and target encoding."""
        df = df.copy()

        # Type conversion for TotalCharges (coercing spaces/invalid strings to NaN)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            missing_count = df["TotalCharges"].isnull().sum()
            if missing_count > 0:
                logger.info(f"Found {missing_count} missing values in TotalCharges. Imputing with median.")
                median_val = df["TotalCharges"].median()
                df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

        # Encode target column if present
        if TARGET_COL in df.columns:
            if df[TARGET_COL].dtype == object or isinstance(df[TARGET_COL].iloc[0], str):
                df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

        return df

    def load_and_split(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads raw data, cleans it, and returns stratified train and test splits."""
        df = self.load_raw_data()
        df = self.clean_data(df)

        if TARGET_COL not in df.columns:
            raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(
            f"Data split into train shape: {X_train.shape}, test shape: {X_test.shape}. "
            f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}"
        )

        return X_train, X_test, y_train, y_test
