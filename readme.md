# Flight Phase Classification – Random Forest + RocketPy

Questo progetto implementa una pipeline completa per **riconoscere automaticamente le fasi di volo di un razzo** usando:

- simulazioni fisiche con **RocketPy**
- sensori virtuali (barometro + IMU) con rumore realistico
- un classificatore **Random Forest**
- un’API semplice da portare su **Raspberry Pi** come payload AI.

L’obiettivo finale è usare questo modello come “cervello” a bordo per:
- riconoscere in tempo reale in che fase di volo si trova il razzo,
- loggare meglio i dati,
- supportare logiche avanzate (diagnostica, safety, R&D).

>  Questa repo è pensata come **prototipo offline**.  
> Per usarla su un razzo reale bisogna fare alcune personalizzazioni (vedi sezione “Cosa manca per usarlo sul razzo").

---

## 📁 Struttura del progetto

### `requirements.txt`
Lista delle dipendenze Python:

- `rocketpy` – simulazione dinamica del razzo
- `numpy`, `pandas` – gestione dati
- `scikit-learn` – Random Forest
- `matplotlib` – grafici
- `joblib` – salvataggio del modello

---

### `rocket_model.py`

Contiene il **modello fisico del razzo** usato per generare i dati:

- costruzione dell’**ambiente** (lat/long del sito tipo EuroC, quota, modello atmosferico)
- definizione di un **razzo stile Calisto** (massa, inerzie, pinne, nosecone ecc.)
- definizione del **motore** (Cesaroni M1670 nel codice di esempio)
- funzione `simulate_flight(...)` che:

  - crea ambiente + razzo + motore,
  - imposta inclinazione del rail e vento,
  - lancia una simulazione `Flight` completa (ascesa + discesa),
  - restituisce oggetti `env, rocket, motor, flight`.

> questo file andrà adattato: masse, motore, curve di drag, ecc.

---

### `generate_dataset.py`

Qui si costruisce il **dataset sintetico** con cui allenare il modello.

Cosa fa:

1. Esegue **n_flights** simulazioni con RocketPy (`simulate_flight`), ogni volta con piccole variazioni:
   - massa (mass_scale),
   - inclinazione rail,
   - vento (componenti wind_u, wind_v).

2. Per ogni volo:
   - campiona la traiettoria a **20 Hz** (ogni 0.05 s),
   - calcola:
     - tempo,
     - quota vera AGL,
     - velocità verticale vera (`vz_true`).

3. Etichetta le fasi del volo con `label_phases(...)`:
   - `phase = 0` → ground / pre-launch
   - `phase = 1` → boost (motore acceso)
   - `phase = 2` → coast in salita (fino all’apogeo)
   - `phase = 3` → discesa

4. Simula i **sensori di bordo** con rumore:

   - **Barometro**:
     - quota misurata `alt_agl_meas = alt_true + rumore_gauss + drift_lento`
   - **IMU verticale**:
     - velocità misurata `vz_meas = vz_true + rumore_integrato_da_acc`
     - accelerazione misurata `az_meas = az_true + rumore_IMU`
   - **Norma accelerazione verticale** `acc_norm_meas` (per avere una feature legata ai carichi).

5. Salva tutto in `flight_phase_dataset.csv` con colonne tipo:

   - `flight_id` – indice del volo simulato
   - `time`
   - `phase` – etichetta di classe (0–3)
   - `alt_agl_true`, `alt_agl_meas`
   - `vz_true`, `vz_meas`
   - `az_meas`, `acc_norm_meas`

---

### `train_rf.py`

Allena il **Random Forest** a partire dal dataset.

Passi principali:

1. Carica `data/flight_phase_dataset.csv`.
2. Usa come **feature di input** SOLO dati “misurati” (cioè simulati come sensori di bordo):
   - `alt_agl_meas`
   - `vz_meas`
   - `az_meas`
   - `acc_norm_meas`
3. **classi** contenute in `phase`.
4. Divide in **train/validation**.
5. Allena un `RandomForestClassifier` con:
   - ~300 alberi
   - profondità limitata
   - `class_weight="balanced"` per gestire sbilanciamenti tra fasi.
6. Stampa:
   - `classification_report` → precision, recall, f1 per ogni fase.
   - `confusion_matrix`.
7. Salva il modello e la lista delle feature in:
   - `models/flight_phase_rf.pkl`

Questo file `.pkl` sarà quello che poi caricherai anche sul Raspberry Pi.

---

### `demo_replay.py`

Serve come **demo di inference “in streaming”**:

- Carica il modello salvato (`flight_phase_rf.pkl`).
- Carica il dataset e seleziona un singolo `flight_id`.
- Scorre riga per riga e, ad ogni step:
  - prende le feature (`alt_agl_meas`, `vz_meas`, `az_meas`, `acc_norm_meas`),
  - chiama `clf.predict(x)` per ottenere la fase,
  - stampa una riga tipo:

    ```text
    t=  12.35s | alt= 1234.5 m | vz=   45.3 m/s | phase=COAST ASCENT
    ```

Nel mondo reale, il loop su `df_flight.iterrows()` sarà sostituito da:

- lettura in tempo reale dei sensori sul Raspberry,
- pre-processing,
- chiamata al modello Random Forest,
- logging + eventuale logica decisionale.

## Cosa manca per usarlo con razzo (nome da definire)

Al momento il modello è basato su:

- un razzo tipo Calisto (massa, forma, drag, altre caratteristiche),
- un motore M1670,
- un modello di rumore sensori generico.

Per renderlo **utilizzabile e credibile** sul vostro razzo EuroC, bisognerà:

### 1. Adattare il modello RocketPy

Nel file `rocket_model.py`:

- sostituire massa, distribuzione di massa e momenti d’inerzia con quelli del CAD;
- sostituire la geometria (lunghezze, diametri, pinne);
- usare le **curve di drag** (senza airbrake) ottenute da:
  - CFD sul modello 3D, oppure
  - software tipo OpenRocket/rasAero/altro, oppure
  - galleria del vento (se bona);
- usare il **motore reale**:
  - file `.eng` corretto,
  - spinta e durata verificate.

Questo rende il dataset sintetico **coerente con il razzo**.

---

### 2. Modellare il rumore dei sensori REALI

Sul banco, con il Raspberry Pi e la sensoristica reale (ad es.):

- IMU (ICM42688 / BMI088 / MPU6050 / BNO085 / ecc.)
- Barometro (BMP390 / MS5611 / DPS310 / ecc.)

dovrete:

1. Registrare qualche minuto di dati **a razzo fermo**.
2. Analizzare:
   - deviazione standard del rumore,
   - drift,
   - eventuali picchi o pattern strani.

Poi, rimpiazzare in `generate_dataset.py` i valori di:

- `sigma_baro`
- `sigma_accel`
- drift rate, ecc.

così che i sensori simulati assomiglino il più possibile a quelli reali.

---

### 3. Generare un dataset nuovo e ri-addestrare

Dopo aver:

- adattato il modello del razzo,
- aggiornato i modelli di rumore sensori,

farai:

```bash
python generate_dataset.py
python train_rf.py
