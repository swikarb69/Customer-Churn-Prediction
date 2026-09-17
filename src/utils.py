import os
import logging
import joblib

def setup_logger(name: str = "customer_churn") -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def ensure_directories(*dirs: str) -> None:
    """Creates directories if they do not exist."""
    for directory in dirs:
        if directory:
            os.makedirs(directory, exist_ok=True)

def save_artifact(obj: object, filepath: str) -> None:
    """Saves a python object using joblib."""
    ensure_directories(os.path.dirname(filepath))
    joblib.dump(obj, filepath)

def load_artifact(filepath: str) -> object:
    """Loads a saved python object using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)
