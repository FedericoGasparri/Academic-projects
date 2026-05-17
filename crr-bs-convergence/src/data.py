"""
data.py
-------
Fetch real market data via yfinance for use in the CRR / BS analysis.

Provides:
    - Historical stock prices and implied volatility estimation
    - Options chain with market prices for comparison
    - Convenience function to build OptionParams from live data
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    yf = None

from .models import OptionParams, OptionType, ExerciseStyle, Parametrisation, DiscountingMode


# Internal helpers

def _require_yfinance():
    if not _YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required for market data: pip install yfinance")


def _to_series(col) -> pd.Series:
    """
    Collapse a DataFrame column that yfinance >= 0.2.38 may return as a
    single-column DataFrame (MultiIndex) instead of a plain Series.
    """
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    if hasattr(col, "squeeze"):
        squeezed = col.squeeze()
        if isinstance(squeezed, pd.Series):
            return squeezed
    return pd.Series(col)


def _scalar(val) -> float:
    """Convert a scalar-like (possibly 0-d array or single-element Series) to float."""
    if isinstance(val, pd.Series):
        return float(val.iloc[-1])
    return float(val)

# Historical volatility

def historical_volatility(ticker: str, window_days: int = 252) -> float:
    _require_yfinance()
    end   = date.today()
    start = end - timedelta(days=window_days)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = yf.download(ticker, start=str(start), end=str(end),
                           progress=False, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    close   = _to_series(hist["Close"])
    log_ret = np.log(close / close.shift(1)).dropna()
    sigma   = float(np.asarray(log_ret).std() * np.sqrt(252))
    return sigma

# Risk-free rate proxy

def risk_free_rate(region: str = "us") -> float:
    if region == "us":
        _require_yfinance()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tbill = yf.Ticker("^IRX")
                hist  = tbill.history(period="5d")
            if not hist.empty:
                raw = _scalar(_to_series(hist["Close"]).iloc[-1])
                return raw / 100.0
        except Exception:
            pass
        return 0.053

    elif region == "eu":
        return 0.040

    raise ValueError(f"Unknown region '{region}'. Use 'us' or 'eu'.")

# Options chain

def options_chain(ticker: str, expiry: str | None = None) -> pd.DataFrame:
    _require_yfinance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk = yf.Ticker(ticker)

    expiries = tk.options
    if not expiries:
        raise ValueError(f"No options data available for '{ticker}'.")

    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        raise ValueError(f"Expiry '{expiry}' not found. Available: {expiries}")

    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    puts  = chain.puts.copy()
    calls["optionType"] = "call"
    puts["optionType"]  = "put"

    df = pd.concat([calls, puts], ignore_index=True)

    try:
        spot = float(tk.fast_info["lastPrice"])
    except Exception:
        spot = _scalar(_to_series(tk.history(period="1d")["Close"]))

    df["expiry"]          = expiry
    df["underlyingPrice"] = spot

    cols = ["strike", "lastPrice", "bid", "ask",
            "impliedVolatility", "inTheMoney", "optionType",
            "expiry", "underlyingPrice"]
    return df[[c for c in cols if c in df.columns]]

# Build OptionParams from live market data

def option_params_from_market(
    ticker:         str,
    strike:         float | None = None,
    expiry:         str   | None = None,
    option_type:    str = "put",
    exercise_style: str = "european",
    hist_vol_days:  int = 252,
    r_region:       str = "eu",
    param:          str = "crr",
) -> tuple[OptionParams, dict]:
    _require_yfinance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="5d")

    if hist.empty:
        raise ValueError(f"No price history for '{ticker}'. Check the ticker symbol.")

    close = _to_series(hist["Close"])
    S0    = float(close.iloc[-1])
    r     = risk_free_rate(r_region)
    sigma = historical_volatility(ticker, hist_vol_days)

    expiries = tk.options
    if expiry is None and expiries:
        expiry = expiries[0]

    if expiry:
        today  = date.today()
        exp_dt = date.fromisoformat(expiry)
        T      = max((exp_dt - today).days / 365.0, 1e-6)
    else:
        T = 0.25

    K = strike if strike is not None else round(S0)

    params = OptionParams(
        S0    = S0,
        K     = K,
        r     = r,
        sigma = sigma,
        T     = T,
        option_type    = OptionType(option_type),
        exercise_style = ExerciseStyle(exercise_style),
        param          = Parametrisation(param),
        discounting    = DiscountingMode.CONTINUOUS,
    )

    meta = {
        "ticker":  ticker,
        "spot":    S0,
        "strike":  K,
        "expiry":  expiry,
        "T_years": T,
        "r":       r,
        "sigma":   sigma,
    }
    return params, meta

# Summary helper

def print_market_summary(meta: dict) -> None:
    print("\n" + "=" * 50)
    print(f"  Market data  —  {meta['ticker']}")
    print("=" * 50)
    print(f"  Spot price   S₀ = {meta['spot']:.4f}")
    print(f"  Strike price  K = {meta['strike']:.4f}")
    print(f"  Expiry           = {meta['expiry']}")
    print(f"  Time to mat.  T = {meta['T_years']:.4f} yr")
    print(f"  Risk-free     r = {meta['r']*100:.2f}%")
    print(f"  Hist. vol.    σ = {meta['sigma']*100:.2f}%")
    print("=" * 50 + "\n")