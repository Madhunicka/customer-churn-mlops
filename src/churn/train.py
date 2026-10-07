"""
Model Training and MLflow Experiment Tracking Script.
"""
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.churn.config import DEFAULT_CONFIG, METRICS_DIR, ModelConfig
from src.churn.dataset import get_train_test_splits
from src.churn.evaluate import evaluate_pipeline, plot_confusion_matrix, save_metrics_to_json
from src.churn.pipeline import build_model_pipeline, save_pipeline


def train_and_track(
    model_type: str = "logistic",
    register_model: bool = False,
    config: ModelConfig = DEFAULT_CONFIG,
):
    """
    Executes training workflow:
    1. Loads train/test data
    2. Builds preprocessing and estimator pipeline
    3. Fits pipeline
    4. Evaluates performance metrics
    5. Logs experiment parameters, metrics, plots, and artifacts to MLflow
    6. Serializes model for production serving
    """
    print(f"\n[INFO] Starting training pipeline for model: {model_type}")

    # Set up MLflow
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    # 1. Load data
    print("[INFO] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test = get_train_test_splits(config=config)
    print(f"       Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Build Pipeline
    pipeline = build_model_pipeline(model_type=model_type, config=config)

    # 3. Train & Log with MLflow
    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        run_id = run.info.run_id
        print(f"       MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("numerical_features_count", len(config.numerical_features))
        mlflow.log_param("categorical_features_count", len(config.categorical_features))

        if model_type == "logistic":
            mlflow.log_param("C", config.lr_c)
            mlflow.log_param("max_iter", config.lr_max_iter)
        elif model_type == "random_forest":
            mlflow.log_param("n_estimators", config.rf_n_estimators)
            mlflow.log_param("max_depth", config.rf_max_depth)

        # Train
        print("[INFO] Fitting Scikit-Learn Pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        print("[INFO] Evaluating model performance...")
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)
            print(f"       {metric_name.upper():<10}: {metric_val:.4f}")

        # Confusion Matrix artifact
        cm_path = METRICS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(pipeline, X_test, y_test, output_path=cm_path)
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

        # Save metrics to local JSON
        metrics_json_path = METRICS_DIR / "latest_metrics.json"
        save_metrics_to_json(metrics, metrics_json_path)
        mlflow.log_artifact(str(metrics_json_path), artifact_path="metrics")

        # Infer model signature for production validation
        signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))

        # Log model artifact to MLflow
        model_name = config.registered_model_name if register_model else None
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name=model_name,
            serialization_format="cloudpickle",
        )

        # 4. Save serving model artifact locally
        model_output_path = config.model_artifact_path
        save_pipeline(pipeline, model_output_path)
        print(f"[INFO] Production model artifact saved to: {model_output_path}")

    print("[SUCCESS] Training and MLflow tracking complete!\n")
    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer churn model with MLflow tracking")
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Model algorithm to train",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLflow model registry",
    )
    args = parser.parse_args()

    train_and_track(model_type=args.model_type, register_model=args.register)
