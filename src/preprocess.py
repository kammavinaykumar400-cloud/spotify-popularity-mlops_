"""Lab 1 - Dataset preparation for Spotify Track Popularity.

Target: is_popular = 1 if popularity >= 50 else 0  (~24% positive, similar
imbalance to the Telco churn project).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RAW_PATH = 'data/raw/spotify_tracks.csv'
POPULARITY_THRESHOLD = 50
RANDOM_STATE = 42

# Columns that identify a track / are free text -> not usable as model features
DROP_COLS = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'popularity']
NUM_COLS = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
CAT_COLS = ['explicit', 'key', 'mode', 'time_signature', 'track_genre']


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Load raw data
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")

    # 2. Data cleaning
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    # The same track_id appears under several genres. Keep one row per track so
    # the same song cannot land in both train and test (data leakage).
    before = len(df)
    df = df.drop_duplicates(subset='track_id', keep='first').reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate track_id rows -> {len(df)} unique tracks")

    # 3. Build target
    df['is_popular'] = (df['popularity'] >= POPULARITY_THRESHOLD).astype(int)

    # 4. Features / target
    X = df.drop(columns=DROP_COLS + ['is_popular'])
    X['explicit'] = X['explicit'].astype(str)
    for c in ['key', 'mode', 'time_signature']:
        X[c] = X[c].astype(str)
    y = df['is_popular']

    # 5. Reproducible stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Scale numeric + one-hot encode categorical (fit on train only)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[NUM_COLS])
    x_test_scaled = scaler.transform(X_test[NUM_COLS])

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[CAT_COLS])
    x_test_encoded = ohe.transform(X_test[CAT_COLS])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 7. Save artifacts
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    # Keep the original test rows (with names) for error analysis
    test_raw = df.loc[X_test.index, ['track_id', 'artists', 'track_name', 'track_genre', 'popularity']]
    test_raw.to_csv('data/processed/test_raw.csv', index=False)

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')

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

    print(f"Train: {X_train_final.shape}  Test: {X_test_final.shape}  Positive rate: {y.mean():.3f}")
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
