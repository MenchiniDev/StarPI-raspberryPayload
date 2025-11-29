# generate_dataset.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from rocket_model import simulate_flight

np.random.seed(42)

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- definizione fasi ----
# 0 = pre-launch/ground
# 1 = boost
# 2 = coast (salita fino ad apogeo)
# 3 = descent

### DATASET SINTETICO DA SIMULAZIONI ROCKETPY ###
### DA RIADDESTRARE CON NOSTRI DATI REALI ###

def label_phases(time, liftoff_time, burn_out_time, apogee_time):
    """
    Ritorna array di int (phase) per ogni istante temporale.
    """
    phases = np.zeros_like(time, dtype=int)
    boost_end = liftoff_time + burn_out_time

    for i, t in enumerate(time):
        if t < liftoff_time:
            phases[i] = 0
        elif t < boost_end:
            phases[i] = 1
        elif t < apogee_time:
            phases[i] = 2
        else:
            phases[i] = 3
    return phases

def add_sensor_noise(alt_agl_true, vz_true, dt):
    """
    Simula rumore barometrico e IMU verticale.
    """
    alt_true = np.array(alt_agl_true)
    vz_true = np.array(vz_true)

    # barometro: noise gauss + drift
    sigma_baro = 1.5  # m
    baro_noise = np.random.normal(0.0, sigma_baro, size=alt_true.shape)

    drift_rate = 0.02  # m/s drift "lento"
    drift = np.cumsum(np.random.normal(0.0, drift_rate * dt, size=alt_true.shape))

    alt_meas = alt_true + baro_noise + drift

    # IMU: accel noise -> velocità rumorosa
    sigma_accel = 0.03 * 9.81  # 0.03 g
    accel_noise = np.random.normal(0.0, sigma_accel, size=vz_true.shape)
    # integra il rumore accel per ottenere err sulla velocità
    vz_noise = np.cumsum(accel_noise * dt)
    vz_meas = vz_true + vz_noise

    # accel_z "misurata" (derivata vz_true + rumore)
    az_true = np.gradient(vz_true, dt)
    az_meas = az_true + np.random.normal(0.0, 0.5 * 9.81, size=az_true.shape)  # rumore IMU

    return alt_meas, vz_meas, az_meas

def generate_flights_dataset(
    n_flights=100,
    sample_dt=0.05,  # 20 Hz
    out_csv=DATA_DIR / "flight_phase_dataset.csv",
):
    """
    Genera un dataset concatenando n_flights simulati.
    Ogni riga = un istante di un volo con features + label fase.
    """
    all_rows = []

    for flight_id in range(n_flights):
        print(f"Simulating flight {flight_id}/{n_flights-1}...")

        # piccoli random sulle condizioni
        mass_scale = np.random.uniform(0.98, 1.02)
        rail_incl = np.random.uniform(80, 88)
        wind_u = np.random.uniform(-5, 5)
        wind_v = np.random.uniform(-5, 5)

        env, rocket, motor, flight = simulate_flight(
            mass_scale=mass_scale,
            rail_inclination_deg=rail_incl,
            wind_u=wind_u,
            wind_v=wind_v,
        )

        time = flight.time  # array np
        dt_array = np.diff(time)
        dt = np.mean(dt_array)

        # prendo solo campioni ogni ~sample_dt (downsampling)
        idx = np.arange(0, len(time), max(1, int(sample_dt / dt)))
        time_ds = time[idx]

        # altitudine e velocità
        z_asl = flight.z(time_ds)           # ASL
        vz = flight.vz(time_ds)
        alt_agl_true = z_asl - env.elevation

        # info fasi
        liftoff_time = getattr(flight, "liftoff_time", time_ds[0])
        burn_out_time = motor.burn_out_time
        apogee_time = flight.apogee_time

        phases = label_phases(time_ds, liftoff_time, burn_out_time, apogee_time)

        # sensori rumorosi
        alt_meas, vz_meas, az_meas = add_sensor_noise(
            alt_agl_true, vz, dt=sample_dt
        )

        acc_norm = np.abs(az_meas)  # qui solo z, ma puoi espandere a 3D

        for t, phase, h_true, h_m, vz_t, vz_m, az_m, an in zip(
            time_ds, phases, alt_agl_true, alt_meas, vz, vz_meas, az_meas, acc_norm
        ):
            all_rows.append(
                {
                    "flight_id": flight_id,
                    "time": t,
                    "phase": phase,
                    "alt_agl_true": h_true,
                    "alt_agl_meas": h_m,
                    "vz_true": vz_t,
                    "vz_meas": vz_m,
                    "az_meas": az_m,
                    "acc_norm_meas": an,
                }
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved dataset to {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    generate_flights_dataset(
        n_flights=100,
        sample_dt=0.05,
    )
