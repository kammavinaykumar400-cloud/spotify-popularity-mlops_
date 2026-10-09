"""Lab 4 - Experiment tracking with MLflow."""
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)

EXPERIMENT_NAME = "Spotify_Popularity_Prediction"

DEFAULT_RF_PARAMS = {"n_estimators": 100, "max_depth": 10, "random_state": 42,
                     "class_weight": "balanced", "n_jobs": -1}


def load_data():
    return (np.load('data/processed/X_train_final.npy'), np.load('data/processed/X_test_final.npy'),
            np.load('data/processed/y_train.npy'), np.load('data/processed/y_test.npy'))


def train_and_track(run_name="RandomForest_Baseline", params=None, model_family="RandomForest"):
    """Train one model, log everything to MLflow, return the metrics dict."""
    if params is None:
        params = dict(DEFAULT_RF_PARAMS)
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_family)

        if model_family == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_family == "LogisticRegression":
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

        os.makedirs("artifacts", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {run_name}")
        fig.savefig("artifacts/confusion_matrix.png", bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact("artifacts/confusion_matrix.png", artifact_path="plots")

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f"ROC Curve - {run_name}")
        fig.savefig("artifacts/roc_curve.png", bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact("artifacts/roc_curve.png", artifact_path="plots")

        if os.path.exists('data/processed/dataset_metadata.json'):
            mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")

        # Newer MLflow saves sklearn models with skops and requires trusted types to be
        # declared. We trained this model ourselves, so tree nodes are safe to trust.
        mlflow.sklearn.log_model(sk_model=model, name="model",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, 'models/random_forest_model.pkl' if model_family == "RandomForest"
                    else 'models/logistic_regression_model.pkl')
        print(f"Run '{run_name}' successfully tracked!")
    return metrics


if __name__ == "__main__":
    # Run 1: baseline
    train_and_track(run_name="RandomForest_Baseline")

    # Run 2: shallower, more trees (change tree depth / count, seed fixed)
    train_and_track(run_name="RandomForest_Shallow",
                    params={**DEFAULT_RF_PARAMS, "n_estimators": 200, "max_depth": 5})

    # Run 3: deeper trees
    train_and_track(run_name="RandomForest_Deep",
                    params={**DEFAULT_RF_PARAMS, "max_depth": 20})

    # Run 4: different model family
    train_and_track(run_name="LogisticRegression_Baseline",
                    params={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                    model_family="LogisticRegression")
