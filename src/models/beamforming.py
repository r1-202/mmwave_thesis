"""Hybrid (analog+digital) beamforming optimization for the mmWave link."""
import numpy as np
import config

def array_gain_db(rows=None, cols=None):
    """Ideal planar-array beamforming gain = 10log10(N_elements)."""
    rows = rows or config.BF_ROWS; cols = cols or config.BF_COLS
    return 10 * np.log10(rows * cols)

def steering_vector(n, theta):
    k = np.arange(n)
    return np.exp(1j * np.pi * k * np.sin(theta))

def hybrid_beamforming_gain(theta_deg=0.0, rows=None, cols=None):
    """Realised gain of an analog steering beam toward theta (degrees)."""
    rows = rows or config.BF_ROWS; cols = cols or config.BF_COLS
    th = np.deg2rad(theta_deg)
    a = np.kron(steering_vector(rows, th), steering_vector(cols, th))
    w = a / np.linalg.norm(a)
    g = np.abs(np.vdot(w, a)) ** 2
    return float(10 * np.log10(g))

def optimize_link(snr_no_bf_db):
    """Add beamforming gain to a raw-link SNR array."""
    return snr_no_bf_db + array_gain_db()
