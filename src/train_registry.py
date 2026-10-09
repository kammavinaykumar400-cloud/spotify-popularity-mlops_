"""Lab 6 - Train a model and register it in the MLflow Model Registry.

Run it more than once with different settings to create several model versions:
    python src/train_registry.py --run-name RandomForest_Registry_V1
    python src/train_registry.py --run-name RandomForest_Registry_V2 --n-estimators 200 --max-depth 5
"""
import argparse
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

EXPERIMENT_NAME = "Spotify_Popularity_Prediction"
REGISTERED_MODEL_NAME = "Spotify_Popularity_Production_Model"


def train_and_register_model(run_name, n_estimators, max_depth):
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        return False

    params = {"n_estimators": n_estimators, "max_depth": max_depth,
              "random_state": 42, "class_weight": "balanced", "n_jobs": -1}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],  # we trained this model ourselves
        )

        print(f"[SUCCESS] Metrics logged: recall = {metrics['recall']:.4f} | F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: '{REGISTERED_MODEL_NAME}'")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="RandomForest_Registry_V1")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=10)
    a = ap.parse_args()
    sys.exit(0 if train_and_register_model(a.run_name, a.n_estimators, a.max_depth) else 1)
