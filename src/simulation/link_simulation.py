"""5G mmWave link-budget simulation: SNR, spectral efficiency, BER."""
import numpy as np
from scipy.special import erfc
import config
from src.models.beamforming import array_gain_db

C = 3e8

def fspl_db(f_hz, d_m):
    return 20*np.log10(d_m) + 20*np.log10(f_hz) + 20*np.log10(4*np.pi/C)

def noise_floor_dbm(bw_hz=None, nf_db=None):
    bw = bw_hz or config.BANDWIDTH_HZ; nf = nf_db or config.NOISE_FIGURE_DB
    return 10*np.log10(1.38e-23*290*bw*1e3) + nf

def link_snr(attenuation_db, f_ghz=None, d_km=None, beamforming=False):
    f = (f_ghz or config.CARRIER_GHZ)*1e9; d = (d_km or config.LINK_DISTANCE_KM)*1000
    snr = (config.TX_POWER_DBM + config.TX_GAIN_DBI + config.RX_GAIN_DBI
           - fspl_db(f, d) - attenuation_db - noise_floor_dbm())
    if beamforming:
        snr = snr + array_gain_db()
    return snr

def spectral_efficiency(snr_db, cap=12.0):
    return np.clip(np.log2(1 + 10**(snr_db/10)), 0, cap)

def ber_qpsk(snr_db):
    return 0.5*erfc(np.sqrt(np.clip(10**(snr_db/10), 0, 50)))
