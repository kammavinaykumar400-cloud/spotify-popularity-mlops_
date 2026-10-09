"""Lab 2 - Evaluation + error analysis."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_evaluation():
    print("Starting Model Evaluation...")
    X_test = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = joblib.load('models/random_forest_baseline.pkl')

    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("-------------------------------\n")

    # Error analysis: original test rows saved by preprocess.py
    errors_df = pd.read_csv('data/processed/test_raw.csv')
    errors_df['Actual_Popular'] = y_test
    errors_df['Predicted_Popular'] = y_pred

    os.makedirs('outputs', exist_ok=True)
    errors_df[(errors_df.Actual_Popular == 1) & (errors_df.Predicted_Popular == 0)] \
        .to_csv('outputs/false_negatives.csv', index=False)
    errors_df[(errors_df.Actual_Popular == 0) & (errors_df.Predicted_Popular == 1)] \
        .to_csv('outputs/false_positives.csv', index=False)

    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
