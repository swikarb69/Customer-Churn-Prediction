import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, DEFAULT_THRESHOLD
from src.evaluate import evaluate_predictions, log_metrics_summary
from src.utils import setup_logger

logger = setup_logger(__name__)

class ThresholdOptimizer:
    """Optimizes probability decision thresholds for binary classification."""

    @staticmethod
    def optimize_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_recall: float = 0.62
    ) -> Tuple[float, Dict[str, Any]]:
        """Finds decision threshold maximizing F1 score while meeting target recall."""
        best_threshold = 0.5
        best_metrics = None
        best_f1 = -1.0

        thresholds = np.arange(0.10, 0.90, 0.01)

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            metrics = evaluate_predictions(y_true, y_pred, y_prob)

            # Prioritize meeting or exceeding target recall, then max F1
            if metrics["recall"] >= target_recall:
                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_threshold = th
                    best_metrics = metrics

        # Fallback to threshold with highest F1 score if target recall not reached
        if best_metrics is None:
            for th in thresholds:
                y_pred = (y_prob >= th).astype(int)
                metrics = evaluate_predictions(y_true, y_pred, y_prob)
                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_threshold = th
                    best_metrics = metrics

        logger.info(f"Optimal threshold found: {best_threshold:.2f} (F1: {best_f1:.4f}, Recall: {best_metrics['recall']:.4f})")
        return best_threshold, best_metrics


class ModelTrainer:
    """Trains candidate models with SMOTE resampling and selects best model."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def get_candidate_models(self) -> Dict[str, Any]:
        """Returns dictionary of candidate model instances."""
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state,
                class_weight="balanced"
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            )
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        return models

    def train_with_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: Any
    ) -> Any:
        """Applies SMOTE to training data only and fits the given model."""
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied: expanded training set from {X_train.shape[0]} to {X_resampled.shape[0]} samples.")
        model.fit(X_resampled, y_resampled)
        return model

    def compare_and_select_best(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """Evaluates all candidate models using SMOTE on X_train and selects the best by ROC-AUC."""
        candidates = self.get_candidate_models()
        best_name = None
        best_model = None
        best_metrics = None
        best_score = -1.0

        for name, model in candidates.items():
            logger.info(f"Training candidate model: {name}")
            trained_model = self.train_with_smote(X_train, y_train, model)

            y_prob = trained_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1_score"]
            logger.info(f"Candidate {name} -> ROC-AUC: {score:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

            if score > best_score:
                best_score = score
                best_name = name
                best_model = trained_model
                best_metrics = metrics

        logger.info(f"Selected Best Model: {best_name} with score {best_score:.4f}")
        return best_name, best_model, best_metrics
