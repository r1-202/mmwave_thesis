"""Central configuration for the 5G mmWave attenuation thesis project."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATASET = DATA_DIR / "weather_attenuation_dataset.csv"

# ---- 5G mmWave link parameters ----
CARRIER_GHZ = 38            # 5G NR FR2 mmWave (CoMMon link freq)
SECONDARY_GHZ = 60          # mmWave/WiGig band
LINK_DISTANCE_KM = 1.85      # CoMMon link actual distance (km)
TX_POWER_DBM = 30.0
TX_GAIN_DBI = 24.0
RX_GAIN_DBI = 24.0
BANDWIDTH_HZ = 400e6        # 400 MHz NR channel
NOISE_FIGURE_DB = 7.0

# ---- Hybrid beamforming ----
BF_ROWS = 8                 # 8x8 planar array
BF_COLS = 8

# ---- ML ----
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ["Temperature", "Humidity", "Wind Speed", "Precipitation", "WC"]
TARGET = "Attenuation"   # real measured rain-induced attenuation (dB)

# ---- Real-time weather station API ----
WEATHER_API_URL = "http://localhost:8000/weather"   # local station endpoint
WEATHER_API_KEY = ""                                  # set if required
POLL_INTERVAL_SEC = 120                               # update every 2 minutes
