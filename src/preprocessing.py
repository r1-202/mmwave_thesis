"""Preprocessing & feature engineering for attenuation prediction.

Note on class imbalance: The CoMMon dataset has imbalanced weather conditions:
  - Sunny: ~74% (381,010 samples)
  - Cloudy: ~19% (96,213 samples)  
  - Rainy: ~7% (35,721 samples)

With temporal split, stratification is not applicable (would break time ordering).
This imbalance should be documented in the paper methodology section.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config
from src.models.traditional import itur_total, crane_total

COND_MAP = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}

def add_attenuation_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical ground-truth attenuation from weather using ITU-R/Crane."""
    R, T, RH = df["Precipitation"].values, df["Temperature"].values, df["Humidity"].values
    d = config.LINK_DISTANCE_KM
    if "ITU_38" not in df:
        df["ITU_38"] = itur_total(R, T, RH, 38, d)
        df["ITU_60"] = itur_total(R, T, RH, 60, d)
        df["Crane_38"] = crane_total(R, T, RH, 38, d)
    df["WC"] = df["Weather Condition"].map(COND_MAP)
    return df

def make_xy(df: pd.DataFrame, temporal_split: bool = True, verbose: bool = True):
    """Prepare features and target for ML training.
    
    Args:
        df: DataFrame with weather and attenuation data
        temporal_split: If True, use chronological split (recommended for time-series).
                       If False, use random split (not recommended, causes data leakage).
        verbose: If True, print class distribution info
    
    Returns:
        Xtr, Xte, ytr, yte, scaler
    """
    # Prefer a REAL measured attenuation column (e.g. from the CoMMon dataset);
    # otherwise fall back to the physics-computed ITU-R target.
    if "Attenuation" in df.columns:
        target = "Attenuation"
        df = df.copy()
    else:
        df = add_attenuation_targets(df)
        target = config.TARGET
    df["WC"] = df["Weather Condition"].map(COND_MAP)
    feats = [f for f in config.FEATURES if f in df.columns]
    sub = df[feats + [target, "Weather Condition"] if "Weather Condition" in df.columns else feats + [target]].dropna()
    
    # Report class distribution
    if verbose and "Weather Condition" in df.columns:
        cond_counts = df["Weather Condition"].value_counts()
        total = cond_counts.sum()
        print("Weather condition distribution:")
        for cond, count in cond_counts.items():
            print(f"  {cond}: {count:,} ({100*count/total:.1f}%)")
    
    X = sub[feats].values
    y = sub[target].values
    
    if temporal_split:
        # Chronological split: train on earlier data, test on later data
        # This prevents temporal data leakage in time-series data
        split_idx = int(len(X) * (1 - config.TEST_SIZE))
        Xtr, Xte = X[:split_idx], X[split_idx:]
        ytr, yte = y[:split_idx], y[split_idx:]
        if verbose:
            print(f"Temporal split: train={len(Xtr):,} (first {100*(1-config.TEST_SIZE):.0f}%), "
                  f"test={len(Xte):,} (last {100*config.TEST_SIZE:.0f}%)")
    else:
        # Random split (NOT recommended for time-series - causes data leakage)
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        if verbose:
            print(f"Random split (WARNING: may cause data leakage): "
                  f"train={len(Xtr):,}, test={len(Xte):,}")
    
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte, scaler
