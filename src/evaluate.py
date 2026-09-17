import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.utils import setup_logger

logger = setup_logger(__name__)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Calculates evaluation metrics for classification predictions."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_true, y_pred, zero_division=0)
    }
    return metrics


def get_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extracts feature importances or coefficients from a fitted model."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def log_metrics_summary(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    """Prints a formatted summary of evaluation metrics."""
    logger.info(f"========== {model_name} Evaluation Summary ==========")
    logger.info(f"Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall   : {metrics['recall']:.4f}")
    logger.info(f"F1 Score : {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        logger.info(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
