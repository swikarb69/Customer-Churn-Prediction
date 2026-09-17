import pytest
import numpy as np
from src.evaluate import evaluate_predictions, get_feature_importances
from src.models import ModelTrainer, ThresholdOptimizer
from sklearn.ensemble import RandomForestClassifier

def test_evaluate_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.85])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.5

def test_get_feature_importances():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    rf.fit(X, y)

    df_imp = get_feature_importances(rf, ["f1", "f2", "f3"])
    assert len(df_imp) == 3
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns

def test_threshold_optimizer():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.65, 0.3, 0.75, 0.15, 0.4])

    best_th, best_metrics = ThresholdOptimizer.optimize_threshold(y_true, y_prob, target_recall=0.60)

    assert 0.10 <= best_th <= 0.90
    assert "recall" in best_metrics
    assert best_metrics["recall"] >= 0.60
