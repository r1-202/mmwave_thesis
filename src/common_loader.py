"""
CoMMon dataset adapter — converts the raw CoMMon field-campaign .dat files into
the project's `weather_attenuation_dataset.csv` with a REAL measured attenuation target.

Dataset: Spackova et al. (2021), "One year of attenuation data from a commercial
dual-polarized duplex microwave link...", Zenodo 10.5281/zenodo.4923125 (CC-BY-4.0).
38-GHz CML, 1.85 km, Dubendorf CH, 4-second resolution.

USAGE
-----
1. Download Common_dataset.zip from https://doi.org/10.5281/zenodo.4923125
2. Unzip it into  data/common_raw/   so you have:
       data/common_raw/CML/CML_YYYYMMDD.dat
       data/common_raw/Airport_weather_stations/Site_11_YYYY-MM-DD.dat
3. Run:  python -m src.common_loader
   -> writes data/weather_attenuation_dataset.csv  (real attenuation + weather)

The script:
  * reads every CML file, computes total path attenuation A = Tx - Rx (dB),
  * estimates the dry-weather baseline and the rain-induced attenuation,
  * reads the airport weather station (Temperature, Humidity, Wind, Rain, Pressure),
  * aligns both to a common 1-minute grid and merges on timestamp.
"""
import glob, os
import numpy as np
import pandas as pd

RAW = "data/common_raw"
CML_DIR = os.path.join(RAW, "CML")
WX_DIR = os.path.join(RAW, "Airport_weather_stations")
RESAMPLE = "1min"          # common time grid; use "4s" to keep ~7.9M rows
WX_SITE = "Site_11"        # airport weather station to use

CML_COLS = ["Time", "Tx_WD_H", "Rx_WD_H", "Tx_WD_V", "Rx_WD_V",
            "Tx_DW_H", "Rx_DW_H", "Tx_DW_V", "Rx_DW_V"]

def load_cml() -> pd.DataFrame:
    """Read all CML_*.dat files, compute attenuation = Tx - Rx (W->D, H polarization)."""
    files = sorted(glob.glob(os.path.join(CML_DIR, "CML_*.dat")))
    if not files:
        raise FileNotFoundError(f"No CML files in {CML_DIR} — download & unzip the dataset first.")
    import re
    tok = re.compile(r'"([^"]*)"')
    frames = []
    for f in files:
        rows = []
        with open(f) as fh:
            next(fh, None)  # skip header
            for line in fh:
                vals = tok.findall(line)
                if len(vals) >= 3:
                    rows.append(vals[:9])
        d = pd.DataFrame(rows, columns=CML_COLS[:len(rows[0])] if rows else CML_COLS)
        frames.append(d)
    cml = pd.concat(frames, ignore_index=True)
    cml["Time"] = pd.to_datetime(cml["Time"], errors="coerce")
    cml = cml.dropna(subset=["Time"]).set_index("Time").sort_index()
    for c in CML_COLS[1:]:
        cml[c] = pd.to_numeric(cml[c], errors="coerce")
    # Total path attenuation (dB): transmitted minus received power
    cml["A_total"] = cml["Tx_WD_H"] - cml["Rx_WD_H"]
    cml = cml.resample(RESAMPLE)["A_total"].mean().to_frame()
    # Dry-weather baseline = rolling 24h minimum (standard CML baseline separation)
    base = cml["A_total"].rolling("24h", min_periods=10).quantile(0.05)
    cml["Attenuation"] = (cml["A_total"] - base).clip(lower=0)   # rain-induced attenuation (dB)
    return cml

def load_weather() -> pd.DataFrame:
    """Read airport weather station: temp, humidity, rain intensity, wind, pressure."""
    files = sorted(glob.glob(os.path.join(WX_DIR, f"{WX_SITE}_*.dat")))
    if not files:
        raise FileNotFoundError(f"No weather files in {WX_DIR}")
    cols = ["Time", "Temperature", "DewPoint", "Humidity", "Pressure",
            "Precipitation", "Wind Speed", "WindDir"]
    frames = [pd.read_csv(f, header=0, names=cols, quotechar='"', na_values=["NA", ""]) for f in files]
    wx = pd.concat(frames, ignore_index=True)
    wx["Time"] = pd.to_datetime(wx["Time"], errors="coerce")
    wx = wx.dropna(subset=["Time"]).set_index("Time").sort_index()
    for c in cols[1:]:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    return wx[["Temperature", "Humidity", "Wind Speed", "Precipitation", "Pressure"]].resample(RESAMPLE).mean()

def build():
    print("[common_loader] reading CML attenuation ...")
    cml = load_cml()
    print(f"  CML rows: {len(cml):,}")
    print("[common_loader] reading weather station ...")
    wx = load_weather()
    print(f"  weather rows: {len(wx):,}")
    df = cml.join(wx, how="inner").dropna(subset=["Attenuation", "Precipitation", "Temperature"])
    df["Weather Condition"] = np.where(df["Precipitation"] > 0.1, "Rainy",
                              np.where(df["Humidity"] > 80, "Cloudy", "Sunny"))
    df = df.reset_index().rename(columns={"Time": "Timestamp"})
    out = "data/weather_attenuation_dataset.csv"
    df[["Timestamp", "Temperature", "Humidity", "Wind Speed", "Precipitation",
        "Pressure", "Weather Condition", "Attenuation"]].to_csv(out, index=False)
    print(f"[common_loader] wrote {out}  ({len(df):,} rows, REAL 38 GHz attenuation)")
    print(df[["Temperature","Humidity","Wind Speed","Precipitation","Attenuation"]].describe().round(2))

if __name__ == "__main__":
    build()
