# Spotify Track Popularity - MLOps Project (Labs 1-6)

Same structure as the Telco Churn project, rebuilt for `dataset.csv` (Spotify tracks).

**Task:** binary classification - is a track popular? `is_popular = popularity >= 50` (~24% positive).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Lab 1 - Data preparation
`python src/preprocess.py`
- drops rows with missing names, removes duplicate `track_id` rows (same song listed under many genres -> would leak between train/test)
- creates target, stratified 80/20 split (`random_state=42`)
- scales 10 numeric audio features, one-hot encodes `explicit, key, mode, time_signature, track_genre`
- writes `data/processed/*.npy`, `dataset_metadata.json`, `models/scaler.pkl`, `models/ohe.pkl`
- EDA: `notebooks/project_implementation.ipynb`

## Lab 2 - Baseline model
`python src/train.py` then `python src/evaluate.py`
(or `python pipelines/run_lab3_baseline.py` for preprocess + train + evaluate)
- Random Forest (100 trees, depth 10, balanced class weights)
- error analysis -> `outputs/false_negatives.csv`, `outputs/false_positives.csv`

## Lab 3 - Git workflow
```bash
git init && git add . && git commit -m "Initial project structure and Lab 1 setup"
```
then practise branches, merge, conflict resolution, `git revert`, and tags as in the guide.

## Lab 4 - MLflow tracking
`python src/train_mlflow.py` -> 4 runs in experiment `Spotify_Popularity_Prediction`
(RandomForest_Baseline, RandomForest_Shallow, RandomForest_Deep, LogisticRegression_Baseline)

`python src/validate_reproducibility.py` -> trains twice with the same seed and checks metrics are identical.

Full lab 4 pipeline: `python pipelines/run_lab4_tracking.py`

View runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Lab 5 - Production data pipeline
`python pipelines/run_lab5_pipeline.py`
1. `src/validate_data.py` - Pandera schema check on the raw CSV (types, 0-1 ranges, key 0-11, mode 0/1 ...). Stops the pipeline if the data is bad.
2. `src/preprocess_pipeline.py` - one scikit-learn `ColumnTransformer` (impute + scale + one-hot), saved as `models/preprocessor.pkl`.
3. `src/validate_outputs.py` - checks no NaNs, matching columns/rows, 0/1 target; writes `artifacts/preprocessing_summary_report.json`.

## Lab 6 - Model registry and lifecycle
`python pipelines/run_lab6_registry.py`
1. `src/train_registry.py` - trains a Random Forest and registers it as `Spotify_Popularity_Production_Model`.
2. `src/automate_lifecycle.py` - new version -> Staging -> compared with the Production champion on **recall** -> promoted or archived.
3. `src/generate_registry_report.py` - writes `artifacts/production_model_report.json`.

Create a second version to see champion vs challenger:
`python pipelines/run_lab6_registry.py --run-name RandomForest_Registry_V2 --n-estimators 200 --max-depth 5`

## Results (test set, 17,948 tracks)
| Run | F1 | ROC-AUC |
|---|---|---|
| RandomForest_Baseline (depth 10) | 0.5225 | 0.7927 |
| RandomForest_Shallow (depth 5, 200 trees) | 0.5038 | 0.7686 |
| RandomForest_Deep (depth 20) | 0.5594 | 0.8209 |
| LogisticRegression_Baseline | 0.5790 | 0.8305 |
