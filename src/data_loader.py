"""Data loading. Reads weather columns from the raw dataset, skipping the
heavy I/Q-data column for speed. Falls back to the slim bundled dataset."""
import pandas as pd
from pathlib import Path
import config

WEATHER_COLS = ["Timestamp", "Frequency", "Signal Strength", "Bandwidth",
                "Temperature", "Humidity", "Wind Speed", "Precipitation",
                "Weather Condition", "Latitude", "Longitude", "Altitude(m)"]

def load_raw(path: str | Path = None) -> pd.DataFrame:
    """Load the dataset. If the full 505 MB CSV is present we read only the
    weather columns (usecols) to avoid loading the giant I/Q column."""
    path = Path(path) if path else config.DATASET
    try:
        keep = set(WEATHER_COLS) | {"Attenuation", "A_total", "Pressure", "Frequency_GHz", "Link_km"}
        df = pd.read_csv(path, usecols=lambda c: c in keep or c.startswith("ITU") or c.startswith("Crane"))
    except ValueError:
        df = pd.read_csv(path)
    print(f"[data_loader] Loaded {len(df):,} rows from {path.name}")
    return df

if __name__ == "__main__":
    print(load_raw().head())
