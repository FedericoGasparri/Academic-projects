"""
tests/test_models.py
--------------------
Unit tests for the CRR and Black-Scholes implementations.
Run with:  pytest tests/
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    OptionParams, OptionType, ExerciseStyle, Parametrisation, DiscountingMode,
    black_scholes, crr_price, convergence_series, stock_tree,
)

# Fixtures

@pytest.fixture
def paper_params():
    """Lecture parameters."""
    return OptionParams(
        S0=27.0, K=25.0, r=0.015, sigma=0.20, T=0.410959,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        param=Parametrisation.CRR,
        discounting=DiscountingMode.CONTINUOUS,
    )

@pytest.fixture
def call_params():
    return OptionParams(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        param=Parametrisation.CRR,
        discounting=DiscountingMode.CONTINUOUS,
    )

# Black-Scholes

class TestBlackScholes:

    def test_atm_call_approx(self, call_params):
        """ATM call price should be approximately 10.45 (textbook value)."""
        price = black_scholes(call_params)
        assert abs(price - 10.4506) < 0.001, f"Got {price:.4f}"

    def test_put_call_parity(self, call_params):
        """C - P = S*e^{0} - K*e^{-rT}  (put-call parity)."""
        call = black_scholes(call_params)
        put_p = OptionParams(**{**call_params.__dict__, "option_type": OptionType.PUT})
        put  = black_scholes(put_p)
        lhs  = call - put
        rhs  = call_params.S0 - call_params.K * np.exp(-call_params.r * call_params.T)
        assert abs(lhs - rhs) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*e^{-rT}."""
        p = OptionParams(S0=200, K=50, r=0.05, sigma=0.20, T=1.0,
                         option_type=OptionType.CALL,
                         exercise_style=ExerciseStyle.EUROPEAN,
                         param=Parametrisation.CRR,
                         discounting=DiscountingMode.CONTINUOUS)
        price    = black_scholes(p)
        intrinsic = p.S0 - p.K * np.exp(-p.r * p.T)
        assert abs(price - intrinsic) < 0.05

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be close to 0."""
        p = OptionParams(S0=50, K=200, r=0.05, sigma=0.20, T=1.0,
                         option_type=OptionType.CALL,
                         exercise_style=ExerciseStyle.EUROPEAN,
                         param=Parametrisation.CRR,
                         discounting=DiscountingMode.CONTINUOUS)
        assert black_scholes(p) < 0.01

    def test_invalid_T(self, call_params):
        p = OptionParams(**{**call_params.__dict__, "T": 0.0})
        with pytest.raises(ValueError):
            black_scholes(p)

# CRR model

class TestCRRPrice:

    def test_convergence_to_bs(self, call_params):
        """CRR price should converge to BS price as N grows."""
        bs    = black_scholes(call_params)
        err50 = abs(crr_price(call_params, 50)  - bs)
        err200= abs(crr_price(call_params, 200) - bs)
        assert err50  < 0.05
        assert err200 < 0.01
        assert err200 < err50  # monotonically improving (on average)

    def test_crr_paper_put(self, paper_params):
        """CRR N=5 put price should match original paper value ≈ 0.40."""
        price = crr_price(paper_params, 5)
        assert abs(price - 0.40) < 0.05, f"Got {price:.5f}"

    def test_american_ge_european(self, paper_params):
        """American put ≥ European put (early exercise premium)."""
        eur = crr_price(paper_params, 50)
        am_p = OptionParams(**{**paper_params.__dict__,
                                "exercise_style": ExerciseStyle.AMERICAN})
        amer = crr_price(am_p, 50)
        assert amer >= eur - 1e-10

    def test_jr_vs_crr_converge(self, call_params):
        """Both JR and CRR parametrisations converge to the same BS limit."""
        bs   = black_scholes(call_params)
        crr  = crr_price(call_params, 200)
        jr_p = OptionParams(**{**call_params.__dict__,
                                "param": Parametrisation.JR})
        jr   = crr_price(jr_p, 200)
        assert abs(crr - bs) < 0.01
        assert abs(jr  - bs) < 0.01

    def test_n1_call(self):
        """N=1 binomial call price (hand-verifiable)."""
        p = OptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
                         option_type=OptionType.CALL,
                         exercise_style=ExerciseStyle.EUROPEAN,
                         param=Parametrisation.CRR,
                         discounting=DiscountingMode.CONTINUOUS)
        price = crr_price(p, 1)
        # Compute by hand: u=e^0.2, d=e^{-0.2}
        u  = np.exp(0.20)
        d  = np.exp(-0.20)
        pu = (np.exp(0.05) - d) / (u - d)
        Vu = max(100 * u - 100, 0)
        Vd = max(100 * d - 100, 0)
        expected = np.exp(-0.05) * (pu * Vu + (1 - pu) * Vd)
        assert abs(price - expected) < 1e-10


# Convergence series

class TestConvergenceSeries:

    def test_output_keys(self, call_params):
        data = convergence_series(call_params, n_steps=[5, 10, 50])
        for key in ("N", "crr", "jr", "bs", "err_crr", "err_jr"):
            assert key in data

    def test_errors_decrease(self, call_params):
        """Mean error at N=100 should be lower than at N=5."""
        data = convergence_series(call_params, n_steps=[5, 50, 100, 200])
        assert data["err_crr"][-1] < data["err_crr"][0]

    def test_bs_constant(self, call_params):
        """BS values should all be identical (it's a constant)."""
        data = convergence_series(call_params, n_steps=[1, 5, 100])
        assert len(set(data["bs"])) == 1

# Stock tree

class TestStockTree:

    def test_shape(self, paper_params):
        N    = 4
        tree = stock_tree(paper_params, N)
        assert tree.shape == (N + 1, N + 1)

    def test_initial_price(self, paper_params):
        tree = stock_tree(paper_params, 3)
        assert abs(tree[0, 0] - paper_params.S0) < 1e-10

    def test_recombining(self, paper_params):
        """up then down == down then up  (recombining tree)."""
        dt  = paper_params.T / 4
        u   = np.exp(paper_params.sigma * np.sqrt(dt))
        d   = 1.0 / u
        assert abs(paper_params.S0 * u * d - paper_params.S0) < 1e-10
