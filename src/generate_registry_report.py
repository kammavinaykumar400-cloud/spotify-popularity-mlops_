"""Lab 6 - Write a deployment-ready JSON report for the Production model."""
import os
import sys
import json
import warnings
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

MODEL_NAME = "Spotify_Popularity_Production_Model"


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    champion = next((mv for mv in versions if mv.current_stage == "Production"), None)
    if not champion:
        print("[ERROR] No model found in Production stage!")
        return False

    run = client.get_run(champion.run_id)
    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "model_lineage": {
            "registered_name": MODEL_NAME,
            "version": int(champion.version),
            "current_stage": champion.current_stage,
            "run_id": champion.run_id,
            "artifact_uri": champion.source,
        },
        "performance_metrics": {k: run.data.metrics.get(k)
                                for k in ["recall", "f1_score", "roc_auc", "accuracy", "precision"]},
        "hyperparameters": run.data.params,
        "preprocessing_dependency": "models/preprocessor.pkl",
    }
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/production_model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
    print(f"[INFO] Saved to: {report_path}")
    return True


if __name__ == "__main__":
    try:
        ok = generate_registry_report()
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        ok = False
    sys.exit(0 if ok else 1)
