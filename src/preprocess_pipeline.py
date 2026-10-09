"""Lab 5 - Production preprocessing with a single scikit-learn Pipeline/ColumnTransformer."""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

RAW_PATH = 'data/raw/spotify_tracks.csv'
POPULARITY_THRESHOLD = 50
RANDOM_STATE = 42
DROP_COLS = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'popularity']
NUM_COLS = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
CAT_COLS = ['explicit', 'key', 'mode', 'time_signature', 'track_genre']


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load + basic cleaning
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    # same track appears under several genres -> keep one row to avoid train/test leakage
    df = df.drop_duplicates(subset='track_id', keep='first').reset_index(drop=True)

    # 2. Target
    df['is_popular'] = (df['popularity'] >= POPULARITY_THRESHOLD).astype(int)

    X = df.drop(columns=DROP_COLS + ['is_popular'])
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    y = df['is_popular']

    # 3. Split BEFORE transformations (prevents leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 4. Modular pipelines
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS),
    ])

    # 5. Fit on train, transform both
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 6. Save artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    df.loc[X_test.index, ['track_id', 'artists', 'track_name', 'track_genre', 'popularity']] \
        .to_csv('data/processed/test_raw.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    metadata = {
        "dataset_name": "Spotify Tracks - Popularity",
        "target": f"is_popular (popularity >= {POPULARITY_THRESHOLD})",
        "n_unique_tracks": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUM_COLS,
        "categorical_features": CAT_COLS,
        "random_state": RANDOM_STATE,
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
