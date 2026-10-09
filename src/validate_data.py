"""Lab 5 - Raw data validation with Pandera (schema for the Spotify dataset)."""
import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_spotify_schema():
    """Strict schema specification for the Spotify tracks dataset."""
    unit = Check.in_range(0.0, 1.0)
    return pa.DataFrameSchema({
        "Unnamed: 0": Column(pa.Int),
        "track_id": Column(pa.String),
        "artists": Column(pa.String, nullable=True),
        "album_name": Column(pa.String, nullable=True),
        "track_name": Column(pa.String, nullable=True),
        "popularity": Column(pa.Int, Check.in_range(0, 100)),
        "duration_ms": Column(pa.Int, Check.ge(0)),
        "explicit": Column(pa.Bool),
        "danceability": Column(pa.Float, unit),
        "energy": Column(pa.Float, unit),
        "key": Column(pa.Int, Check.in_range(0, 11)),
        "loudness": Column(pa.Float, Check.in_range(-60.0, 5.0)),
        "mode": Column(pa.Int, Check.isin([0, 1])),
        "speechiness": Column(pa.Float, unit),
        "acousticness": Column(pa.Float, unit),
        "instrumentalness": Column(pa.Float, unit),
        "liveness": Column(pa.Float, unit),
        "valence": Column(pa.Float, unit),
        "tempo": Column(pa.Float, Check.ge(0.0)),
        "time_signature": Column(pa.Int, Check.in_range(0, 5)),
        "track_genre": Column(pa.String),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validate a DataFrame against the schema. Returns True/False."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_spotify_schema()
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.head(30).to_string())
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report ({len(failures)} issues) saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    import sys
    data_path = 'data/raw/spotify_tracks.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)
    ok = validate_schema(pd.read_csv(data_path), output_report_name="baseline_validation.csv")
    sys.exit(0 if ok else 1)   # non-zero exit stops the pipeline when data is bad
