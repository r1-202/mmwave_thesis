# Using a Real Measured-Attenuation Dataset (recommended for the thesis)

The bundled dataset derives attenuation from rain rate via the ITU-R formula, so ML
reaches R² ≈ 1.0 (it re-learns a clean equation). To get **realistic, defensible
results**, switch to a dataset with *physically measured* attenuation.

## ⭐ Recommended: the CoMMon dataset (38 GHz, real mmWave)

**Spackova et al. (2021)** — One year of attenuation data from a 38-GHz dual-polarized
commercial microwave link, Dübendorf, Switzerland, 4-second resolution.
DOI: **10.5281/zenodo.4923125** · License: **CC-BY-4.0** (free with citation)
Download: https://doi.org/10.5281/zenodo.4923125

Why it fits this thesis:
- **Real measured attenuation** — the link logs Tx and Rx power, so attenuation = Tx − Rx.
- **38 GHz = 5G mmWave (FR2)**, close to the 28 GHz target band.
- **~7.9 M rows** at 4 s resolution (≈ 525,600 rows when resampled to 1-minute).
- Airport weather stations provide **Temperature, Humidity, Wind Speed, Rain intensity,
  Pressure** — the exact features used in this project.

### Steps
1. Download `Common_dataset.zip` from the link above.
2. Unzip into `data/common_raw/` so you have:
   ```
   data/common_raw/CML/CML_YYYYMMDD.dat
   data/common_raw/Airport_weather_stations/Site_11_YYYY-MM-DD.dat
   ```
3. Build the project dataset (computes real attenuation, merges weather):
   ```bash
   python -m src.common_loader
   ```
   This overwrites `data/weather_attenuation_dataset.csv` with a real-attenuation target.
4. Run the pipeline as usual:
   ```bash
   python main.py
   ```
   The pipeline auto-detects the real `Attenuation` column and trains against it,
   so you will now get a realistic R² (typically ~0.7–0.9) and a meaningful
   ML-vs-ITU-R comparison.

Tip: in `src/common_loader.py` set `RESAMPLE = "4s"` to keep the full ~7.9 M rows,
or `"1min"` (default) for ~525k rows.

## Alternative: E-band CML dataset (71–86 GHz)
**Fencl et al. (2020)**, DOI **10.5281/zenodo.4090953**, CC-BY-4.0, 85 MB.
Real total losses + rainfall, air temperature, relative humidity at 1-min/5-min
resolution. Simpler and smaller; higher frequency, no wind-speed column.

## How to cite (CoMMon)
> Špačková, A., Bareš, V., Fencl, M., Schleiss, M., Jaffrain, J., Berne, A.,
> Rieckermann, J. (2021). One year of attenuation data from a commercial
> dual-polarized duplex microwave link… Zenodo. https://doi.org/10.5281/zenodo.4923125
