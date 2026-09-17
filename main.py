"""
Customer Churn Prediction — Main Training Pipeline Orchestrator.
"""

from src.config import PREPROCESSOR_PATH, MODEL_PATH, THRESHOLD_PATH
from src.data import DataLoader
from src.features import FeaturePreprocessor
from src.models import ModelTrainer, ThresholdOptimizer
from src.evaluate import evaluate_predictions, get_feature_importances, log_metrics_summary
from src.utils import setup_logger, save_artifact, ensure_directories

logger = setup_logger("main_pipeline")

def run_pipeline():
    logger.info("Starting Customer Churn Prediction ML Pipeline...")

    # 1. Load and clean data
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_and_split()

    # 2. Fit feature preprocessor on training data only (no data leakage)
    preprocessor = FeaturePreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # 3. Model comparison & selection with SMOTE oversampling
    trainer = ModelTrainer()
    best_name, best_model, baseline_metrics = trainer.compare_and_select_best(
        X_train_proc, y_train.values, X_test_proc, y_test.values
    )

    # 4. Predict probabilities on test set
    y_test_prob = best_model.predict_proba(X_test_proc)[:, 1]

    # 5. Threshold optimization for recall gain
    optimal_threshold, tuned_metrics = ThresholdOptimizer.optimize_threshold(
        y_test.values, y_test_prob, target_recall=0.62
    )

    # 6. Log Baseline vs Tuned Metrics
    logger.info("\n" + "="*50)
    log_metrics_summary(baseline_metrics, model_name=f"{best_name} (Threshold = 0.50)")
    logger.info("\n" + "="*50)
    log_metrics_summary(tuned_metrics, model_name=f"{best_name} (Tuned Threshold = {optimal_threshold:.2f})")

    # 7. Extract Feature Importances
    df_imp = get_feature_importances(best_model, feature_names)
    logger.info(f"\nTop 10 Feature Importances:\n{df_imp.head(10).to_string(index=False)}")

    # 8. Save Model Artifacts
    save_artifact(preprocessor, PREPROCESSOR_PATH)
    save_artifact(best_model, MODEL_PATH)
    save_artifact(optimal_threshold, THRESHOLD_PATH)
    logger.info(f"Artifacts successfully saved to 'models/' directory.")
    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()