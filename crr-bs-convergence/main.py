"""
main.py
-------
Entry point for the CRR / Black-Scholes convergence project.

Run modes:
    python main.py                   → use paper parameters (S0=27, K=25, ...)
    python main.py --ticker AAPL     → fetch live data from Yahoo Finance
    python main.py --help            → show all options

Outputs:
    - Printed table of prices and errors
    - Interactive HTML dashboard  (crr_bs_dashboard.html)
    - Individual Plotly figures   (.show() if --show flag is set)
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    OptionParams, OptionType, ExerciseStyle, Parametrisation, DiscountingMode,
    black_scholes, crr_price, convergence_series,
    option_params_from_market, print_market_summary,
    convergence_plot, parity_oscillation_plot, sensitivity_plot,
    stock_tree_plot, full_dashboard,
)

# Default parameters (from the original university lecture)
PAPER_PARAMS = OptionParams(
    S0    = 27.0,
    K     = 25.0,
    r     = 0.015,
    sigma = 0.20,
    T     = 0.410959,
    option_type    = OptionType.PUT,
    exercise_style = ExerciseStyle.EUROPEAN,
    param          = Parametrisation.CRR,
    discounting    = DiscountingMode.CONTINUOUS,
)

# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="CRR / Black-Scholes convergence analysis"
    )
    p.add_argument("--ticker",  type=str,   default=None,
                   help="Yahoo Finance ticker (e.g. AAPL, SPY, ENI.MI). "
                        "If omitted, uses paper parameters.")
    p.add_argument("--expiry",  type=str,   default=None,
                   help="Option expiry date YYYY-MM-DD (used with --ticker).")
    p.add_argument("--strike",  type=float, default=None,
                   help="Strike price override.")
    p.add_argument("--option-type", choices=["call", "put"], default="put",
                   help="Option type (default: put).")
    p.add_argument("--N-max",   type=int,   default=200,
                   help="Maximum number of CRR steps to evaluate.")
    p.add_argument("--show",    action="store_true",
                   help="Open figures in browser interactively.")
    p.add_argument("--output",  type=str,   default="crr_bs_dashboard.html",
                   help="Output HTML file for the dashboard.")
    return p.parse_args()

# Print summary table
def print_summary(params: OptionParams, N_values=(5, 10, 25, 50, 100, 200)):
    bs = black_scholes(params)
    print("\n" + "=" * 65)
    print(f"  Black-Scholes price: {bs:.6f}")
    print("=" * 65)
    print(f"  {'N':>5}  {'CRR (CRR param)':>18}  {'CRR (JR param)':>16}  {'|err CRR|':>10}  {'|err JR|':>9}")
    print("  " + "-" * 63)

    from src.models import Parametrisation, DiscountingMode
    for N in N_values:
        p_crr = OptionParams(**{**params.__dict__,
                                 "param": Parametrisation.CRR,
                                 "discounting": DiscountingMode.CONTINUOUS})
        p_jr  = OptionParams(**{**params.__dict__,
                                 "param": Parametrisation.JR,
                                 "discounting": DiscountingMode.CONTINUOUS})
        v_crr = crr_price(p_crr, N)
        v_jr  = crr_price(p_jr,  N)
        print(f"  {N:>5}  {v_crr:>18.6f}  {v_jr:>16.6f}  {abs(v_crr-bs):>10.6f}  {abs(v_jr-bs):>9.6f}")

    print("=" * 65 + "\n")

# Main
def main():
    args = parse_args()

    # --- Resolve parameters ---
    if args.ticker:
        print(f"\nFetching market data for {args.ticker} …")
        params, meta = option_params_from_market(
            ticker      = args.ticker,
            strike      = args.strike,
            expiry      = args.expiry,
            option_type = args.option_type,
        )
        print_market_summary(meta)
    else:
        print("\nUsing paper parameters (S0=27, K=25, r=1.5%, σ=20%, T=0.41yr)")
        params = PAPER_PARAMS

    # --- Numerical summary ---
    n_steps = sorted(set(
        [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, args.N_max]
    ))
    print_summary(params)

    # --- Figures ---
    print("Building figures …")
    fig_conv  = convergence_plot(params, n_steps)
    fig_osc   = parity_oscillation_plot(params, N_max=min(args.N_max, 100))
    fig_sens  = sensitivity_plot(params, vary="sigma")
    fig_tree  = stock_tree_plot(params, N=5)

    if args.show:
        fig_conv.show()
        fig_osc.show()
        fig_sens.show()
        fig_tree.show()

    # --- Dashboard ---
    full_dashboard(params, output_path=args.output)
    print(f"\nDone. Open '{args.output}' in your browser.")


if __name__ == "__main__":
    main()
