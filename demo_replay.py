# demo_replay.py
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")

def main():
    model_bundle = joblib.load(MODELS_DIR / "flight_phase_rf.pkl")
    clf = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    df = pd.read_csv(DATA_DIR / "flight_phase_dataset.csv")

    # prendo un solo flight_id per demo
    flight_id = 0
    df_flight = df[df["flight_id"] == flight_id].copy()
    df_flight.sort_values("time", inplace=True)

    print(f"Demo su flight_id={flight_id}, samples={len(df_flight)}")

    for _, row in df_flight.iterrows():
        x = row[feature_cols].values.reshape(1, -1)
        pred_phase = int(clf.predict(x)[0])

        t = row["time"]
        alt = row["alt_agl_meas"]
        vz = row["vz_meas"]

        # mappa fase -> nome
        phase_names = {
            0: "GROUND / PRE-LAUNCH",
            1: "BOOST",
            2: "COAST ASCENT",
            3: "DESCENT",
        }

        print(
            f"t={t:6.2f}s | alt={alt:7.1f} m | vz={vz:7.1f} m/s | phase={phase_names[pred_phase]}"
        )

        # se vuoi simulare realtime, decommenta:
        # time.sleep(0.05)

if __name__ == "__main__":
    main()
