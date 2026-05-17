# CRR → Black-Scholes Convergence

**Binomial option pricing and its continuous-time limit**
Sapienza University of Rome — Quantitative Financial Modelling

---

## Overview

This repository implements the **Cox-Ross-Rubinstein (CRR)** binomial option pricing model and demonstrates its convergence to the **Black-Scholes (BS)** closed-form formula as the number of time steps N → ∞.

Key features:
- European and American options (call and put)
- Two parametrisations: **CRR** and **Jarrow-Rudd (JR)**
- Both simple and continuous compounding
- Live market data via **yfinance**
- Interactive **Plotly** convergence dashboard

---

## Project structure

```
crr_bs_convergence/
├── main.py                  # CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py            # CRR + Black-Scholes (pure, vectorised)
│   ├── data.py              # yfinance integration
│   └── visualization.py     # Plotly figures + HTML dashboard
├── notebooks/
│   └── analysis.ipynb       # End-to-end Jupyter notebook
├── tests/
    └── test_models.py       # pytest unit tests

```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/crr-bs-convergence.git
cd crr-bs-convergence
pip install -r requirements.txt

# 2. Run with paper parameters (S0=27, K=25, σ=20%, r=1.5%, T=0.41yr)
python main.py

# 3. Run with live market data
python main.py --ticker AAPL --option-type put

# 4. Open the interactive dashboard
open crr_bs_dashboard.html    # macOS
start crr_bs_dashboard.html   # Windows

# 5. Run tests
pytest tests/
```

---

## Usage (Python API)

```python
from src import (
    OptionParams, OptionType, ExerciseStyle,
    black_scholes, crr_price, convergence_plot,
)

params = OptionParams(
    S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.EUROPEAN,
)

print(black_scholes(params))          # 5.5735
print(crr_price(params, N=100))       # ≈ 5.57xx (converging)

fig = convergence_plot(params)
fig.show()                            # interactive Plotly figure
```

### Live data

```python
from src import option_params_from_market, print_market_summary

params, meta = option_params_from_market(ticker="AAPL", option_type="put")
print_market_summary(meta)

from src import black_scholes, convergence_plot
print(black_scholes(params))
convergence_plot(params).show()
```

---

## Theoretical background

### CRR model

At each time step Δt = T/N the stock price moves:

```
S_{t+Δt} = S_t × u   (with risk-neutral probability p)
S_{t+Δt} = S_t × d   (with probability 1 − p)
```

**CRR parametrisation:**
```
u = exp(σ√Δt),   d = 1/u
```

**Jarrow-Rudd parametrisation:**
```
u = exp((r − σ²/2)Δt + σ√Δt)
d = exp((r − σ²/2)Δt − σ√Δt)
```

The risk-neutral probability ensures no-arbitrage:
```
p = (e^{rΔt} − d) / (u − d)
```

### Convergence theorem (Cutland & Roux)

For any bounded continuous payoff function D and the JR parametrisation:

```
lim_{N→∞} D₀^(N) = e^{-rT} · E^Q[D(S_T)]
```

where the right-hand side is the Black-Scholes price. The key ingredient is the weak convergence of the normalised random walk W_n^(N) to a Brownian motion W_T under the risk-neutral measure Q.

---

## Results (paper parameters)

| N   | CRR price | JR price  | \|err CRR\| | \|err JR\|  |
|-----|-----------|-----------|-------------|-------------|
| 5   | 0.398xx   | 0.401xx   | ~0.020      | ~0.017      |
| 10  | 0.406xx   | 0.408xx   | ~0.012      | ~0.010      |
| 50  | 0.415xx   | 0.416xx   | ~0.003      | ~0.002      |
| 100 | 0.417xx   | 0.418xx   | ~0.001      | ~0.001      |
| BS  | 0.4186xx  | —         | 0.000000    | 0.000000    |

---

## Authors

- Simone Cuonzo
- Federico Gasparri
- Xia Tian
- Loris Diotallevi
- Sahar Shirazi
- Danilo Capaldi

Sapienza University of Rome — FINASS curriculum

---

## License

MIT — free for academic and educational use.
Data fetched via `yfinance` is subject to Yahoo Finance's terms of use.
