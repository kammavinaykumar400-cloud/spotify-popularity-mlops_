"""Lab 6 - Champion / Challenger lifecycle: compare registered versions and promote the best."""
import sys
import warnings
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")   # MLflow stages API prints deprecation warnings

MODEL_NAME = "Spotify_Popularity_Production_Model"
METRIC_TO_OPTIMIZE = "recall"   # decision metric (same as the churn lab); change to "f1_score" if you prefer


def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    client = MlflowClient()

    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    # 1. New models (stage None) -> Staging
    new_models = [mv for mv in all_versions if mv.current_stage == "None"]
    if new_models:
        print(f"\n[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")
        for mv in new_models:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Staging", archive_existing_versions=False)
            print(f"  -> Version {mv.version} is now in Staging.")

    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
    if not staging_models:
        print("\n[INFO] No models in Staging to evaluate. Exiting.")
        return

    # 2. Best challenger
    print("\n[INFO] Evaluating Staging candidates...")
    best_challenger, best_score = None, -1.0
    for mv in staging_models:
        score = client.get_run(mv.run_id).data.metrics.get(METRIC_TO_OPTIMIZE, 0.0)
        print(f"  -> Staging Candidate: Version {mv.version} | {METRIC_TO_OPTIMIZE}: {score:.4f}")
        if score > best_score:
            best_score, best_challenger = score, mv
    print(f"\n[INFO] Best Challenger is Version {best_challenger.version} ({METRIC_TO_OPTIMIZE}: {best_score:.4f})")

    # 3. Current champion
    production = [mv for mv in all_versions if mv.current_stage == "Production"]
    champion = production[0] if production else None
    promote = False
    if not champion:
        print("[INFO] No model currently in Production. Challenger wins by default!")
        promote = True
    else:
        champ_score = client.get_run(champion.run_id).data.metrics.get(METRIC_TO_OPTIMIZE, 0.0)
        print(f"[INFO] Current Production Champion: Version {champion.version} | {METRIC_TO_OPTIMIZE}: {champ_score:.4f}")
        if best_score > champ_score:
            print("[SUCCESS] Challenger beat the Champion!")
            promote = True
        else:
            print("[INFO] Challenger failed to beat the Champion. Champion retains its title.")

    # 4. Promote / archive
    if promote:
        print(f"\n[INFO] Promoting Version {best_challenger.version} to Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=best_challenger.version, stage="Production", archive_existing_versions=False)
        if champion:
            print(f"[INFO] Archiving defeated Champion (Version {champion.version})...")
            client.transition_model_version_stage(
                name=MODEL_NAME, version=champion.version, stage="Archived", archive_existing_versions=False)

    # 5. Clean up Staging
    leftovers = [mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'") if mv.current_stage == "Staging"]
    if leftovers:
        print("\n[INFO] Cleaning up Staging area...")
        for mv in leftovers:
            print(f"  -> Archiving Version {mv.version} (Lost the competition)")
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Archived", archive_existing_versions=False)

    print("\n[SUCCESS] Automated Model Lifecycle execution complete!")


if __name__ == "__main__":
    try:
        automate_champion_challenger()
    except Exception as e:
        print(f"[ERROR] Failed to automate lifecycle: {e}")
        sys.exit(1)
