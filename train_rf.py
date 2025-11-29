# train_rf.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dataset_path = DATA_DIR / "flight_phase_dataset.csv"
    df = pd.read_csv(dataset_path)

    # features basate SOLO su misure (non true)
    feature_cols = [
        "alt_agl_meas",
        "vz_meas",
        "az_meas",
        "acc_norm_meas",
    ]
    X = df[feature_cols].values
    y = df["phase"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    print("\nValidation metrics:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    model_path = MODELS_DIR / "flight_phase_rf.pkl"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    print(f"\nSaved model to {model_path}")

if __name__ == "__main__":
    main()
