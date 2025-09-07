from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

def rayleigh_chan(alpha: float, sigma: float, N: int) -> np.ndarray:
    """
    Generates a complex Rayleigh fading channel.
    """
    h = np.zeros(N, dtype=complex)
    h[0] = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
    for i in range(1, N):
        noise = (np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma)) / np.sqrt(2)
        h[i] = alpha * h[i-1] + noise
    return h

def add_wgn(h: np.ndarray, C: np.ndarray, SNR: float, N: int) -> Tuple[np.ndarray, float]:
    """
    Adds white Gaussian noise to the signal.
    """
    y = h * C
    noise_power = np.mean(np.abs(y)**2) / (10**(SNR/10))
    noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N))
    y_noisy = y + noise
    return y_noisy, noise_power

def create_data(alpha: float, sigma: float, SNR: float, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Creates the dataset for the Rayleigh fading channel.
    """
    C = np.random.choice([-1, 1], N)
    h = rayleigh_chan(alpha, sigma, N)
    y, noise_power = add_wgn(h, C, SNR, N)
    return h, y, C, noise_power

def kalman_filter_complex(y: np.ndarray, C: np.ndarray, alpha: float, sigma: float, noise_power: float, N: int) -> Dict[str, np.ndarray]:
    """
    Performs Kalman filtering and prediction for complex-valued signals.
    """
    h_pred = np.zeros(N, dtype=complex)
    P_pred = np.zeros(N)
    h_filt = np.zeros(N, dtype=complex)
    P_filt = np.zeros(N)

    h_filt[0] = 0
    P_filt[0] = 1
    
    for k in range(1, N):
        # Prediction
        h_pred[k] = alpha * h_filt[k-1]
        P_pred[k] = alpha**2 * P_filt[k-1] + sigma**2
        
        # Filtering
        K = P_pred[k] * C[k].conj() / (C[k] * P_pred[k] * C[k].conj() + noise_power)
        h_filt[k] = h_pred[k] + K * (y[k] - C[k] * h_pred[k])
        P_filt[k] = (1 - K * C[k]) * P_pred[k]
        
    return {'h_pred': h_pred, 'P_pred': P_pred, 'h_filt': h_filt, 'P_filt': P_filt}

def rts_smoother_complex(h_filt: np.ndarray, P_filt: np.ndarray, h_pred: np.ndarray, P_pred: np.ndarray, alpha: float, N: int) -> Dict[str, np.ndarray]:
    """
    Performs RTS smoothing for complex-valued signals.
    """
    h_smooth = np.zeros(N, dtype=complex)
    P_smooth = np.zeros(N)
    
    h_smooth[-1] = h_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for k in range(N-2, -1, -1):
        J = alpha * P_filt[k] / P_pred[k+1]
        h_smooth[k] = h_filt[k] + J * (h_smooth[k+1] - h_pred[k+1])
        P_smooth[k] = P_filt[k] + J**2 * (P_smooth[k+1] - P_pred[k+1])
        
    return {'h_smooth': h_smooth, 'P_smooth': P_smooth}

def run_monte_carlo_simulation(n_simulations: int, alpha: float, sigma: float, SNR: float, N: int) -> Tuple[list, list, list]:
    """
    Runs a Monte Carlo simulation for the Kalman filter.
    """
    mse_preds = []
    mse_filts = []
    mse_smooths = []

    for _ in range(n_simulations):
        h, y, C, noise_power = create_data(alpha, sigma, SNR, N)
        
        kf_results = kalman_filter_complex(y, C, alpha, sigma, noise_power, N)
        h_pred = kf_results['h_pred']
        h_filt = kf_results['h_filt']
        P_filt = kf_results['P_filt']
        P_pred = kf_results['P_pred']
        
        # Prediction only (as a baseline)
        h_pred_only = np.zeros(N, dtype=complex)
        for k in range(1, N):
            h_pred_only[k] = alpha * h_pred_only[k-1]

        smooth_results = rts_smoother_complex(h_filt, P_filt, h_pred, P_pred, alpha, N)
        h_smooth = smooth_results['h_smooth']
        
        mse_preds.append(np.mean(np.abs(h - h_pred_only)**2))
        mse_filts.append(np.mean(np.abs(h - h_filt)**2))
        mse_smooths.append(np.mean(np.abs(h - h_smooth)**2))
        
    return mse_preds, mse_filts, mse_smooths
