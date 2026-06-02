"""Real-time local weather-station API integration.

Polls a local weather station endpoint every POLL_INTERVAL_SEC seconds,
feeds the live reading into the trained ML model, and returns a real-time
attenuation + optimized-link prediction. Falls back to a simulated reading
if no station is reachable (useful for offline thesis demos).
"""
import time
import requests
import numpy as np
import config
from src.models.traditional import itur_total
from src.simulation.link_simulation import link_snr, spectral_efficiency, ber_qpsk

def fetch_reading():
    """GET the latest reading from the local station. Returns a dict."""
    try:
        r = requests.get(config.WEATHER_API_URL, timeout=3,
                         headers={"Authorization": config.WEATHER_API_KEY})
        r.raise_for_status()
        return r.json()
    except Exception:
        # Offline fallback: synthesise a plausible live reading
        rng = np.random.default_rng()
        return {"temperature": float(rng.uniform(20, 40)),
                "humidity": float(rng.uniform(20, 80)),
                "wind_speed": float(rng.uniform(0, 20)),
                "precipitation": float(rng.uniform(0, 50)),
                "condition": rng.choice(["Sunny", "Cloudy", "Rainy"]),
                "_simulated": True}

def predict_realtime(model, scaler, reading: dict):
    """Predict attenuation + optimized link KPIs from one live reading."""
    cond = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}.get(reading.get("condition", "Sunny"), 0)
    x = np.array([[reading["temperature"], reading["humidity"], reading["wind_speed"],
                   reading["precipitation"], cond]])
    att_ml = float(model.predict(scaler.transform(x))[0])
    att_itu = float(itur_total(reading["precipitation"], reading["temperature"],
                               reading["humidity"], config.CARRIER_GHZ, config.LINK_DISTANCE_KM))
    snr = float(link_snr(att_ml, beamforming=True))
    return {"attenuation_ml_db": round(att_ml, 4),
            "attenuation_itu_db": round(att_itu, 4),
            "snr_db": round(snr, 2),
            "spectral_eff_bps_hz": round(float(spectral_efficiency(snr)), 3),
            "ber": float(ber_qpsk(snr)),
            "reading": reading}

def run_loop(model, scaler, n_iter=None):
    """Continuously poll the station and print real-time predictions."""
    i = 0
    while n_iter is None or i < n_iter:
        out = predict_realtime(model, scaler, fetch_reading())
        print(f"[{time.strftime('%H:%M:%S')}] ATT(ML)={out['attenuation_ml_db']} dB  "
              f"SNR={out['snr_db']} dB  SE={out['spectral_eff_bps_hz']} bps/Hz")
        i += 1
        if n_iter is None or i < n_iter:
            time.sleep(config.POLL_INTERVAL_SEC)
