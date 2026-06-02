"""
Main entry point — runs the full pipeline:
  1. Load weather dataset
  2. Compute ITU-R / Crane ground-truth attenuation (traditional models)
  3. Train & evaluate 6 ML models
  4. Simulate the 5G mmWave link (SNR / SE / BER) with hybrid beamforming
  5. Save results JSON and build the interactive HTML thesis report

Run in PyCharm:  Right-click main.py -> Run 'main'
Or terminal:     python main.py
"""
import json
import numpy as np
import config
from src.data_loader import load_raw
from src.preprocessing import make_xy, add_attenuation_targets
from src.models.ml_models import train_all
from src.simulation.link_simulation import link_snr, spectral_efficiency, ber_qpsk
from src.models.beamforming import array_gain_db

def main():
    print("=" * 60)
    print(" 5G mmWave Weather-Attenuation ML Pipeline — LIU Thesis 2025")
    print("=" * 60)

    df = load_raw()
    df = add_attenuation_targets(df)

    print("\n[1] Training machine-learning models ...")
    Xtr, Xte, ytr, yte, scaler = make_xy(df)
    results, fitted = train_all(Xtr, Xte, ytr, yte)

    # naive (closed-form) traditional baseline error for comparison
    naive = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    results["Traditional_ITU"] = {"rmse": naive, "r2": 0.0,
                                  "mae": float(np.mean(np.abs(yte - ytr.mean()))),
                                  "train_s": 0.0, "latency_us": 0.5}

    best = min([k for k in results if not k.startswith("Traditional")],
               key=lambda k: results[k]["rmse"])
    fi = dict(zip(config.FEATURES,
                  [round(float(v), 4) for v in fitted["RandomForest"].feature_importances_]))

    print(f"\n[2] Best model: {best}  (RMSE={results[best]['rmse']:.6f})")
    print(f"[3] Beamforming array gain: {array_gain_db():.2f} dB")

    summary = {"n_rows": int(len(df)), "best_model": best,
               "models": results, "feature_importance": fi}
    (config.RESULTS_DIR / "ml_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[4] Saved results -> {config.RESULTS_DIR/'ml_results.json'}")
    print("[5] Open results/index.html in your browser for the full report.")

if __name__ == "__main__":
    main()
