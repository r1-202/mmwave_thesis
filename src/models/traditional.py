"""Physics-based mmWave attenuation: ITU-R P.838-3 (rain) + P.676 (gases) + Crane (1980)."""
import numpy as np

# ITU-R P.838-3 rain coefficients (horizontal polarization)
# Source: ITU-R P.838-3 Table 1 (k) and Table 2 (alpha)
ITU_K = {28: 0.2051, 38: 0.3282, 60: 0.8606}
ITU_A = {28: 0.9679, 38: 0.9272, 60: 0.7656}

def itur_specific_rain(R, f):
    """Specific rain attenuation gamma_R = k*R^alpha (dB/km), ITU-R P.838-3."""
    return ITU_K[f] * np.power(np.maximum(R, 0.0), ITU_A[f])

def gaseous_attenuation(T, RH, f):
    """Simplified ITU-R P.676 gaseous specific attenuation (dB/km).
    Oxygen peaks ~60 GHz; water-vapour scales with absolute humidity."""
    # Absolute humidity (g/m^3) via Magnus formula
    es = 6.112 * np.exp(17.67 * T / (T + 243.5))      # sat vapour pressure hPa
    e = es * (RH / 100.0)
    rho = 216.7 * e / (T + 273.15)                     # water vapour density g/m^3
    # Oxygen (dry) term - resonance near 60 GHz
    gamma_o = 0.0 if f == 28 else 15.0 * np.exp(-((f - 60) ** 2) / 50.0)
    if f == 28:
        gamma_o = 0.10 + 0.0  # ~0.1 dB/km dry air at 28 GHz
    # Water vapour term (grows with frequency & rho)
    gamma_w = (0.0021 * f) * rho / 7.5
    return gamma_o + gamma_w

def crane_path_factor(R, d):
    """Crane (1980) effective path-length reduction factor."""
    # Crane two-component empirical reduction
    delta = 3.8 - 0.6 * np.log(np.maximum(R, 1.0))
    r = 1.0 / (1.0 + d / np.maximum(delta, 0.5))
    return np.clip(r, 0.3, 1.0)

def itur_total(R, T, RH, f, d_km):
    """ITU-R total path attenuation (dB) = rain*pathfactor + gaseous."""
    gamma_r = itur_specific_rain(R, f)
    r = 1.0 / (1.0 + d_km / 35.0 * np.power(np.maximum(R,0.0), 0.1))  # P.530 path reduction
    A_rain = gamma_r * d_km * np.clip(r, 0.4, 1.0)
    A_gas = gaseous_attenuation(T, RH, f) * d_km
    return A_rain + A_gas

def crane_total(R, T, RH, f, d_km):
    gamma_r = itur_specific_rain(R, f)
    A_rain = gamma_r * d_km * crane_path_factor(R, d_km)
    A_gas = gaseous_attenuation(T, RH, f) * d_km
    return A_rain + A_gas
