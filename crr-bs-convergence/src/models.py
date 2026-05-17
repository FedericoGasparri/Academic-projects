"""
models.py
---------
Cox-Ross-Rubinstein (CRR) binomial option pricing model and
Black-Scholes (BS) closed-form formula.

Both European and American options are supported.
Parametrisations: CRR (Cox-Ross-Rubinstein) and JR (Jarrow-Rudd).
Discounting: simple rate (1+r)^dt  or  continuously compounded e^(r*dt).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations and config
# ---------------------------------------------------------------------------

class OptionType(str, Enum):
    CALL = "call"
    PUT  = "put"

class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN  = "american"

class Parametrisation(str, Enum):
    CRR = "crr"   # Cox-Ross-Rubinstein
    JR  = "jr"    # Jarrow-Rudd

class DiscountingMode(str, Enum):
    SIMPLE      = "simple"       # (1+r)^dt
    CONTINUOUS  = "continuous"   # e^(r*dt)


@dataclass(frozen=True)
class OptionParams:
    """Container for all option parameters."""
    S0:     float           # initial stock price
    K:      float           # strike price
    r:      float           # annualised risk-free rate (decimal)
    sigma:  float           # annualised volatility (decimal)
    T:      float           # time to maturity (years)
    option_type:     OptionType      = OptionType.PUT
    exercise_style:  ExerciseStyle   = ExerciseStyle.EUROPEAN
    param:           Parametrisation = Parametrisation.CRR
    discounting:     DiscountingMode = DiscountingMode.CONTINUOUS


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def black_scholes(params: OptionParams) -> float:
    """
    Closed-form Black-Scholes price for a European option.

    Uses continuous compounding and the standard BS formula:
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 − σ√T
        Call = S·N(d1) − K·e^{-rT}·N(d2)
        Put  = K·e^{-rT}·N(−d2) − S·N(−d1)
    """
    S, K, r, sigma, T = params.S0, params.K, params.r, params.sigma, params.T

    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if params.option_type == OptionType.CALL:
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# CRR helper: u, d factors
# ---------------------------------------------------------------------------

def _up_down_factors(params: OptionParams, dt: float) -> tuple[float, float]:
    """Return (u, d) factors for a single time step dt."""
    sigma, r = params.sigma, params.r

    if params.param == Parametrisation.CRR:
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))        # d = 1/u exactly
    else:
        # Jarrow-Rudd: drift-corrected so that E[S_{t+dt}] = S_t · e^{r·dt}
        drift = (r - 0.5 * sigma ** 2) * dt
        u = np.exp(drift + sigma * np.sqrt(dt))
        d = np.exp(drift - sigma * np.sqrt(dt))

    return u, d


# ---------------------------------------------------------------------------
# CRR binomial pricer
# ---------------------------------------------------------------------------

def crr_price(params: OptionParams, N: int) -> float:
    """
    Price a European or American option with the CRR binomial model.

    Parameters
    ----------
    params : OptionParams
    N      : int — number of time steps (more steps → closer to BS)

    Returns
    -------
    float — option price at t=0
    """
    S, K, r, sigma, T = params.S0, params.K, params.r, params.sigma, params.T
    dt = T / N
    u, d = _up_down_factors(params, dt)

    # Discount factor and risk-neutral probability
    if params.discounting == DiscountingMode.CONTINUOUS:
        disc = np.exp(-r * dt)
        pu   = (np.exp(r * dt) - d) / (u - d)
    else:
        disc = 1.0 / (1.0 + r) ** dt
        pu   = ((1.0 + r) ** dt - d) / (u - d)

    pd = 1.0 - pu

    # --- Terminal payoffs (vectorised) ---
    j     = np.arange(N + 1)              # number of up-moves: 0 … N
    S_T   = S * (u ** (N - j)) * (d ** j)

    if params.option_type == OptionType.CALL:
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # --- Backward induction ---
    if params.exercise_style == ExerciseStyle.EUROPEAN:
        for _ in range(N):
            V = disc * (pu * V[:-1] + pd * V[1:])
    else:
        # American: compare holding vs immediate exercise at each node
        for step in range(N - 1, -1, -1):
            j_inner = np.arange(step + 1)
            S_node  = S * (u ** (step - j_inner)) * (d ** j_inner)
            hold    = disc * (pu * V[:-1] + pd * V[1:])
            if params.option_type == OptionType.CALL:
                intrinsic = np.maximum(S_node - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_node, 0.0)
            V = np.maximum(hold, intrinsic)

    return float(V[0])


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def convergence_series(
    params: OptionParams,
    n_steps: list[int] | None = None,
) -> dict[str, list]:
    """
    Compute CRR prices (both CRR and JR parametrisations) and BS price
    for a range of N values, returning a dict suitable for plotting.

    Parameters
    ----------
    params  : OptionParams  (param field is ignored; both are computed)
    n_steps : list of N values to evaluate (default: 1..200 log-spaced)

    Returns
    -------
    dict with keys:
        'N'       — list of N values
        'crr'     — CRR prices
        'jr'      — JR prices
        'bs'      — Black-Scholes price (constant)
        'err_crr' — |crr − bs|
        'err_jr'  — |jr  − bs|
    """
    if n_steps is None:
        n_steps = sorted(set(
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30,
             40, 50, 75, 100, 150, 200]
        ))

    bs = black_scholes(params)

    params_crr = OptionParams(**{**params.__dict__, "param": Parametrisation.CRR,
                                  "discounting": DiscountingMode.CONTINUOUS})
    params_jr  = OptionParams(**{**params.__dict__, "param": Parametrisation.JR,
                                  "discounting": DiscountingMode.CONTINUOUS})

    crr_prices, jr_prices = [], []
    for N in n_steps:
        crr_prices.append(crr_price(params_crr, N))
        jr_prices.append(crr_price(params_jr,  N))

    return {
        "N":       n_steps,
        "crr":     crr_prices,
        "jr":      jr_prices,
        "bs":      [bs] * len(n_steps),
        "err_crr": [abs(p - bs) for p in crr_prices],
        "err_jr":  [abs(p - bs) for p in jr_prices],
    }


# ---------------------------------------------------------------------------
# Stock price tree (for visualisation)
# ---------------------------------------------------------------------------

def stock_tree(params: OptionParams, N: int) -> np.ndarray:
    """
    Return the full binomial stock-price tree as a 2-D array.
    tree[t, j] = S0 * u^(t-j) * d^j  for j = 0 … t
    (upper-triangular; lower entries are NaN)
    """
    dt = params.T / N
    u, d = _up_down_factors(params, dt)
    tree = np.full((N + 1, N + 1), np.nan)
    for t in range(N + 1):
        for j in range(t + 1):
            tree[t, j] = params.S0 * (u ** (t - j)) * (d ** j)
    return tree
