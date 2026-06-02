"""Machine-learning attenuation predictors + training/evaluation harness."""
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import config

def build_models():
    rs = config.RANDOM_STATE
    return {
        "RandomForest": RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=rs),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1,
                                    n_jobs=-1, random_state=rs),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1,
                                      n_jobs=-1, random_state=rs, verbose=-1),
        "GradientBoost": GradientBoostingRegressor(n_estimators=120, max_depth=5, random_state=rs),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
    }

def evaluate(name, model, Xtr, Xte, ytr, yte):
    t = time.time(); model.fit(Xtr, ytr); train_s = time.time() - t
    t = time.time(); pred = model.predict(Xte); latency_us = (time.time() - t) / len(Xte) * 1e6
    return {
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "mae": float(mean_absolute_error(yte, pred)),
        "train_s": round(train_s, 2),
        "latency_us": round(latency_us, 3),
    }, model

def train_all(Xtr, Xte, ytr, yte, subsample_slow=20000):
    results, fitted = {}, {}
    for name, m in build_models().items():
        if name in ("SVR", "GradientBoost") and len(Xtr) > subsample_slow:
            # Use consistent random seed from config for reproducibility
            idx = np.random.RandomState(config.RANDOM_STATE).choice(len(Xtr), subsample_slow, replace=False)
            metrics, fm = evaluate(name, m, Xtr[idx], Xte, ytr[idx], yte)
        else:
            metrics, fm = evaluate(name, m, Xtr, Xte, ytr, yte)
        results[name], fitted[name] = metrics, fm
        print(f"  {name:14s} RMSE={metrics['rmse']:.6f}  R2={metrics['r2']:.5f}")
    # Stacking ensemble (mean of tree models)
    base = ["RandomForest", "XGBoost", "LightGBM"]
    preds = np.mean([fitted[b].predict(Xte) for b in base], axis=0)
    results["StackingEnsemble"] = {
        "rmse": float(np.sqrt(mean_squared_error(yte, preds))),
        "r2": float(r2_score(yte, preds)),
        "mae": float(mean_absolute_error(yte, preds)),
        "train_s": 0.0,
        "latency_us": round(sum(results[b]["latency_us"] for b in base), 3),
    }
    return results, fitted
