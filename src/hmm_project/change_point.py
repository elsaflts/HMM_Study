from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


# ---------------------------
# 0) Helpers
# ---------------------------

def AR_from_C(Ck: int, params: Dict[str, float]) -> Tuple[float, float]:
    """Return (A, R_std) for a given change indicator Ck.

    - If Ck == 0: A=1 (carry state), R_std=q0_std (tiny drift)
    - If Ck == 1: A=0 (reset),      R_std=R_std (fresh draw)
    """
    if Ck == 0:
        return 1.0, float(params["q0_std"])  # tiny drift when no change
    else:
        return 0.0, float(params["R_std"])   # fresh draw on change


# ---------------------------
# 1) Simulation
# ---------------------------

def simulate_change_point(
    n: int, params: Dict[str, float], seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (C[0..n-1], W[0..n], Y[0..n-1]) for the change-point model.

    Model:
      C_k ~ Bernoulli(p)
      W_{k+1} = A(C_k) W_k + R(C_k) U_k
      Y_k     = W_k + S V_k
    """
    rng = np.random.default_rng(seed)
    p = float(params["p"])
    w0_mean, w0_var = float(params["w0_mean"]), float(params["w0_var"])
    S_std = float(params["S_std"])

    C = rng.binomial(1, p, size=n)
    W = np.empty(n + 1)
    W[0] = rng.normal(w0_mean, np.sqrt(w0_var))
    for k in range(n):
        A_k, R_k = AR_from_C(int(C[k]), params)  # transition k -> k+1
        W[k + 1] = A_k * W[k] + R_k * rng.standard_normal()
    Y = W[:-1] + S_std * rng.standard_normal(size=n)
    return C, W, Y


# ----------------------------------------
# 2) Kalman filter (known C)
# ----------------------------------------

def kf_known_C(
    Y: np.ndarray,
    C: np.ndarray,
    params: Dict[str, float],
    w0_mean: Optional[float] = None,
    w0_var: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Forward Kalman filter with time-varying (A, R) picked by C.

    Returns dict with sequences of predictions, filters, innovations, and (A,R).
    """
    n = len(Y)
    B = 1.0
    S2 = float(params["S_std"]) ** 2
    eps = 1e-12

    m_prev = float(params["w0_mean"]) if w0_mean is None else float(w0_mean)
    P_prev = float(params["w0_var"]) if w0_var is None else float(w0_var)

    W_pred = np.empty(n)  # m_{k|k-1}
    P_pred = np.empty(n)  # P_{k|k-1}
    W_filt = np.empty(n)  # m_{k|k}
    P_filt = np.empty(n)  # P_{k|k}
    innov = np.empty(n)   # eps_k
    innov_v = np.empty(n) # Gamma_k
    K_gain = np.empty(n)  # K_k

    # store actual transition A_k, R_k for k -> k+1 (length n)
    A_seq = np.empty(n)
    R_seq = np.empty(n)
    for k in range(n):
        A_seq[k], R_seq[k] = AR_from_C(int(C[k]), params)

    for k in range(n):
        if k == 0:
            m_pred = m_prev
            P_pred_k = P_prev
        else:
            A_prev, R_prev = A_seq[k - 1], R_seq[k - 1]
            m_pred = A_prev * m_prev
            P_pred_k = A_prev * P_prev * A_prev + R_prev ** 2

        eps_k = Y[k] - B * m_pred
        Gamma_k = B * P_pred_k * B + S2
        K_k = (P_pred_k * B) / (Gamma_k + eps)

        m_filt = m_pred + K_k * eps_k
        P_filt_k = P_pred_k - K_k * B * P_pred_k

        W_pred[k], P_pred[k] = m_pred, P_pred_k
        W_filt[k], P_filt[k] = m_filt, P_filt_k
        innov[k], innov_v[k] = eps_k, Gamma_k
        K_gain[k] = K_k

        m_prev, P_prev = m_filt, P_filt_k

    return dict(
        W_pred=W_pred,
        P_pred=P_pred,
        W_filt=W_filt,
        P_filt=P_filt,
        innov=innov,
        innov_var=innov_v,
        K=K_gain,
        A_seq=A_seq,
        R_seq=R_seq,
    )


# ----------------------------------------
# 3) RTS fixed-interval smoother
# ----------------------------------------

def rts_smoother(
    W_pred: np.ndarray,
    W_filt: np.ndarray,
    P_pred: np.ndarray,
    P_filt: np.ndarray,
    A_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rauch–Tung–Striebel smoother (scalar, time-varying A)."""
    n = len(W_filt)
    W_smooth = np.empty(n)
    P_smooth = np.empty(n)
    eps = 1e-12

    W_smooth[-1] = W_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for k in range(n - 2, -1, -1):
        A_k = A_seq[k]  # transition k -> k+1
        J_k = (P_filt[k] * A_k) / (P_pred[k + 1] + eps)

        W_smooth[k] = W_filt[k] + J_k * (W_smooth[k + 1] - W_pred[k + 1])
        P_smooth[k] = P_filt[k] + J_k * (P_smooth[k + 1] - P_pred[k + 1]) * J_k
    return W_smooth, P_smooth


# ------------------------------------------------
# 4) FFBS (simulation smoother) and W | C, Y draw
# ------------------------------------------------

def ffbs_sample(
    W_pred: np.ndarray,
    W_filt: np.ndarray,
    P_pred: np.ndarray,
    P_filt: np.ndarray,
    A_seq: np.ndarray,
    rng=None,
) -> np.ndarray:
    """Forward-Filter Backward-Sample for a single trajectory {W_k}."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(W_filt)
    W_s = np.empty(n)
    eps = 1e-12

    W_s[-1] = rng.normal(W_filt[-1], np.sqrt(max(P_filt[-1], 0.0)))
    for k in range(n - 2, -1, -1):
        A_k = A_seq[k]
        J_k = (P_filt[k] * A_k) / (P_pred[k + 1] + eps)
        mean_cond = W_filt[k] + J_k * (W_s[k + 1] - W_pred[k + 1])
        cov_cond = P_filt[k] - J_k * P_pred[k + 1] * J_k
        cov_cond = max(cov_cond, 0.0)
        W_s[k] = rng.normal(mean_cond, np.sqrt(cov_cond))
    return W_s


def sample_W_given_C(
    Y: np.ndarray, C: np.ndarray, params: Dict[str, float], rng=None
) -> np.ndarray:
    """One Gibbs block for W | C, Y: draw W_0..W_{n-1} via FFBS, then W_n."""
    rng = np.random.default_rng() if rng is None else rng
    out = kf_known_C(Y, C, params)
    W_pred, P_pred = out["W_pred"], out["P_pred"]
    W_filt, P_filt = out["W_filt"], out["P_filt"]
    A_seq, R_seq = out["A_seq"], out["R_seq"]

    W_draw = ffbs_sample(W_pred, W_filt, P_pred, P_filt, A_seq, rng=rng)

    # Append W_n from transition (n-1 -> n)
    n = len(Y)
    A_last, R_last = A_seq[-1], R_seq[-1]
    Wn = rng.normal(A_last * W_draw[-1], R_last)
    return np.concatenate([W_draw, [Wn]])


# ------------------------------------------------------------
# 5) Gibbs: update C | W  (site-wise Bernoulli with log-scores)
# ------------------------------------------------------------

def _log_norm_pdf(x: float, mean: float, var: float, eps: float = 1e-12) -> float:
    var = max(var, eps)
    return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)


def site_logpost_Ck(
    k: int, W: np.ndarray, params: Dict[str, float], p_change: float
) -> Tuple[float, float]:
    """Log posterior scores for C_k in {0,1}.

    prior: Bernoulli(p_change)
    transition:
      if C_k=0: W_{k+1} ~ N(W_k, q0_std^2)
      if C_k=1: W_{k+1} ~ N(0,    R_std^2)
    """
    w_k, w_k1 = W[k], W[k + 1]
    ll0 = _log_norm_pdf(w_k1, mean=w_k, var=float(params["q0_std"]) ** 2)
    ll1 = _log_norm_pdf(w_k1, mean=0.0, var=float(params["R_std"]) ** 2)
    lp0 = np.log(1.0 - p_change + 1e-15) + ll0
    lp1 = np.log(p_change + 1e-15) + ll1
    return lp0, lp1


def gibbs_change_points(
    Y: np.ndarray,
    params: Dict[str, float],
    n_iter: int = 2000,
    burn_in: int = 1000,
    thin: int = 1,
    init_C: Optional[np.ndarray] = None,
    rng=None,
) -> Dict[str, np.ndarray]:
    """Gibbs for (C, W) | Y.

    Iterates between:
      1) W | C, Y  via KF + FFBS
      2) C | W, Y  via local Bernoulli updates (transition density)
    Returns posterior mean of C and MAP labels.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(Y)
    p = float(params["p"])

    C = rng.binomial(1, p, size=n) if init_C is None else init_C.astype(int).copy()

    keep = []
    count = np.zeros(n, dtype=float)

    for it in range(n_iter):
        W = sample_W_given_C(Y, C, params, rng=rng)
        for k in range(n):
            lp0, lp1 = site_logpost_Ck(k, W, params, p)
            m = max(lp0, lp1)
            p1 = np.exp(lp1 - m) / (np.exp(lp0 - m) + np.exp(lp1 - m))
            C[k] = rng.binomial(1, p1)

        if it >= burn_in and ((it - burn_in) % thin == 0):
            count += C
            keep.append(C.copy())

    draws = max(len(keep), 1)
    post_prob = count / draws
    C_map = (post_prob >= 0.5).astype(int)
    return dict(post_prob=post_prob, C_map=C_map, samples=np.array(keep))


# ------------------------------------------------------------
# 6) Online change-point detection (particle filter)
# ------------------------------------------------------------

def _systematic_resample(weights: np.ndarray, rng=None) -> np.ndarray:
    """Systematic resampling. Returns ancestor indices."""
    rng = np.random.default_rng() if rng is None else rng
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    idx = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def pf_detect(
    Y: np.ndarray,
    params: Dict[str, float],
    N: int = 500,
    lag: int = 0,
    ess_threshold: float = 0.5,
    rng=None,
) -> Dict[str, np.ndarray]:
    """Bootstrap particle filter for online change-point probability.

    Reports P(C_{k-lag}=1 | Y_{0:k}) using a ring-buffer of the most recent C's.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(Y)
    p = float(params["p"])
    S2 = float(params["S_std"]) ** 2
    eps = 1e-15

    # Initialize W_0 ~ prior; no C sampled yet
    Wp = rng.normal(float(params["w0_mean"]), np.sqrt(float(params["w0_var"])), size=N)
    w = np.ones(N) / N
    # ring buffer of size lag+1; fill from right; leftmost is time k-lag
    recent_C = -np.ones((N, lag + 1), dtype=int)

    prob_change_at_lag = np.full(n, np.nan)

    for k in range(n):
        # 1) Weight with current observation Y_k
        ll = -0.5 * (np.log(2 * np.pi * S2) + (Y[k] - Wp) ** 2 / S2)
        ll -= ll.max()  # stabilize
        w *= np.exp(ll)
        s = w.sum()
        w = (w / s) if s > eps else (np.ones_like(w) / len(w))

        # 2) Report probability at lag if available
        if k >= lag and lag >= 0 and recent_C[:, 0].min() >= 0:
            prob_change_at_lag[k] = float((w * (recent_C[:, 0] == 1)).sum())

        # 3) Resample if ESS low
        ess = 1.0 / (w @ w)
        if ess < ess_threshold * len(w):
            idx = _systematic_resample(w, rng)
            Wp = Wp[idx]
            recent_C = recent_C[idx]
            w[:] = 1.0 / len(w)

        # 4) Sample C_k and propagate to W_{k+1}
        Ck = rng.binomial(1, p, size=len(w))

        # shift buffer left, append new C_k to rightmost column
        if lag >= 0:
            if lag > 0:
                recent_C[:, :-1] = recent_C[:, 1:]
            recent_C[:, -1] = Ck

        A_k = np.where(Ck == 0, 1.0, 0.0)
        R_k = np.where(Ck == 0, float(params["q0_std"]), float(params["R_std"]))
        Wp = A_k * Wp + R_k * rng.standard_normal(size=len(w))

    return dict(prob_change_at_lag=prob_change_at_lag)

