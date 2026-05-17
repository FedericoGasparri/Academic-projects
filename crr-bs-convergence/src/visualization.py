"""
visualization.py
----------------
Interactive Plotly figures for the CRR / Black-Scholes convergence analysis.

All functions return a plotly.graph_objects.Figure object that can be:
    - Displayed in Jupyter:  fig.show()
    - Saved as HTML:         fig.write_html("out.html")
    - Saved as PNG/PDF:      fig.write_image("out.png")  (requires kaleido)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import OptionParams, convergence_series, stock_tree, crr_price, black_scholes

# Colour palette (consistent across all figures)

_BLUE   = "#2563EB"   # CRR parametrisation
_GREEN  = "#059669"   # JR parametrisation
_ORANGE = "#EA580C"   # Black-Scholes
_PURPLE = "#7C3AED"   # error CRR
_PINK   = "#DB2777"   # error JR
_GREY   = "#94A3B8"

_LAYOUT = dict(
    font       = dict(family="IBM Plex Mono, monospace", size=13),
    paper_bgcolor = "white",
    plot_bgcolor  = "#F8FAFC",
    margin     = dict(l=60, r=40, t=70, b=60),
    legend     = dict(
        bgcolor     = "rgba(255,255,255,0.85)",
        bordercolor = "#E2E8F0",
        borderwidth = 1,
        font        = dict(size=12),
    ),
)

# 1. Convergence plot  (main figure)

def convergence_plot(
    params:   OptionParams,
    n_steps:  list[int] | None = None,
    title:    str | None       = None,
) -> go.Figure:
    """
    Two-panel figure:
      Top    — CRR (CRR param), CRR (JR param), and BS price vs N
      Bottom — absolute error |CRR − BS| and |JR − BS| vs N  (log scale)

    Parameters
    ----------
    params  : OptionParams
    n_steps : list of N values to evaluate
    title   : optional figure title
    """
    data = convergence_series(params, n_steps)
    N    = data["N"]
    bs   = data["bs"][0]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=("Option price vs number of steps N",
                        "Absolute error  |CRR − BS|  (log scale)"),
    )

    # --- Top panel ---
    fig.add_trace(go.Scatter(
        x=N, y=data["crr"],
        mode="lines+markers",
        name="CRR parametrisation",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=6, symbol="circle"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=N, y=data["jr"],
        mode="lines+markers",
        name="JR parametrisation",
        line=dict(color=_GREEN, width=2),
        marker=dict(size=6, symbol="diamond"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[N[0], N[-1]], y=[bs, bs],
        mode="lines",
        name=f"Black-Scholes  ({bs:.4f})",
        line=dict(color=_ORANGE, width=2, dash="dash"),
    ), row=1, col=1)

    # Shaded band ±0.01 around BS for visual reference
    fig.add_hrect(
        y0=bs - 0.01, y1=bs + 0.01,
        fillcolor=_ORANGE, opacity=0.06,
        line_width=0, row=1, col=1,
    )

    # --- Bottom panel (log scale) ---
    fig.add_trace(go.Scatter(
        x=N, y=data["err_crr"],
        mode="lines+markers",
        name="Error — CRR param.",
        line=dict(color=_PURPLE, width=1.8),
        marker=dict(size=5, symbol="circle"),
        showlegend=True,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=N, y=data["err_jr"],
        mode="lines+markers",
        name="Error — JR param.",
        line=dict(color=_PINK, width=1.8),
        marker=dict(size=5, symbol="diamond"),
        showlegend=True,
    ), row=2, col=1)

    # --- Layout ---
    opt_label = f"{params.option_type.value.capitalize()} " \
                f"({'European' if params.exercise_style.value == 'european' else 'American'})"
    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=title or f"CRR → Black-Scholes convergence  |  {opt_label}  "
                          f"S₀={params.S0}  K={params.K}  "
                          f"σ={params.sigma:.0%}  r={params.r:.1%}  T={params.T:.2f}yr",
            font=dict(size=14),
            x=0.02,
        ),
        height=620,
    )
    fig.update_yaxes(title_text="Option price", row=1, col=1)
    fig.update_yaxes(title_text="|Error|", type="log", row=2, col=1)
    fig.update_xaxes(title_text="N  (number of steps)", row=2, col=1)

    return fig

# 2. Parity oscillation: even vs odd N

def parity_oscillation_plot(params: OptionParams, N_max: int = 80) -> go.Figure:
    """
    Show the even/odd oscillation of CRR prices that disappears as N grows.
    Separate traces for even and odd N values.
    """
    all_N   = list(range(1, N_max + 1))
    from .models import Parametrisation, DiscountingMode
    p_crr = OptionParams(**{**params.__dict__,
                             "param": Parametrisation.CRR,
                             "discounting": DiscountingMode.CONTINUOUS})
    prices = [crr_price(p_crr, n) for n in all_N]
    bs     = black_scholes(params)

    even_N = [n for n in all_N if n % 2 == 0]
    odd_N  = [n for n in all_N if n % 2 != 0]
    even_p = [prices[n - 1] for n in even_N]
    odd_p  = [prices[n - 1] for n in odd_N]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=even_N, y=even_p, mode="lines+markers",
        name="Even N", line=dict(color=_BLUE, width=1.5),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=odd_N, y=odd_p, mode="lines+markers",
        name="Odd N", line=dict(color=_GREEN, width=1.5),
        marker=dict(size=5),
    ))
    fig.add_hline(y=bs, line_dash="dash", line_color=_ORANGE,
                  annotation_text=f"BS = {bs:.4f}",
                  annotation_position="top right")

    fig.update_layout(
        **_LAYOUT,
        title="Even / odd oscillation of CRR prices",
        xaxis_title="N (steps)",
        yaxis_title="Option price",
        height=420,
    )
    return fig

# 3. Sensitivity: BS vs CRR(N=50) across a range of σ or S0

def sensitivity_plot(
    params: OptionParams,
    vary:   str = "sigma",     # "sigma" | "S0" | "K"
    n_points: int = 50,
    N_crr: int = 50,
) -> go.Figure:
    """
    Compare BS and CRR(N=N_crr) prices as one parameter varies.

    vary : which parameter to sweep  ('sigma', 'S0', or 'K')
    """
    ranges = {
        "sigma": np.linspace(0.05, 0.80, n_points),
        "S0":    np.linspace(params.K * 0.5, params.K * 1.5, n_points),
        "K":     np.linspace(params.S0 * 0.5, params.S0 * 1.5, n_points),
    }
    if vary not in ranges:
        raise ValueError(f"vary must be one of {list(ranges.keys())}")

    xs     = ranges[vary]
    bs_v   = []
    crr_v  = []

    for x in xs:
        kw      = params.__dict__.copy()
        kw[vary] = float(x)
        p_bs    = OptionParams(**kw)
        p_crr   = OptionParams(**{**kw, "param": "crr", "discounting": "continuous"})
        bs_v.append(black_scholes(p_bs))
        crr_v.append(crr_price(p_crr, N_crr))

    labels = {"sigma": "σ (volatility)", "S0": "S₀ (spot price)", "K": "K (strike)"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=bs_v, mode="lines",
        name="Black-Scholes", line=dict(color=_ORANGE, width=2.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=crr_v, mode="lines+markers",
        name=f"CRR  (N={N_crr})", line=dict(color=_BLUE, width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=[abs(b - c) for b, c in zip(bs_v, crr_v)],
        mode="lines",
        name="|Error|", line=dict(color=_PURPLE, width=1.5, dash="dot"),
        yaxis="y2",
    ))

    sens_layout = {k: v for k, v in _LAYOUT.items() if k != "legend"}
    fig.update_layout(
        **sens_layout,
        title=f"Sensitivity to {labels[vary]}  |  CRR(N={N_crr}) vs Black-Scholes",
        xaxis_title=labels[vary],
        yaxis_title="Option price",
        yaxis2=dict(
            title="|Error|", overlaying="y", side="right",
            showgrid=False, tickformat=".4f",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#E2E8F0", borderwidth=1, font=dict(size=12)),
        height=450,
    )
    return fig

# 4. Binomial stock tree (for small N)

def stock_tree_plot(params: OptionParams, N: int = 5) -> go.Figure:
    """
    Draw the recombining stock price tree for N ≤ 10.
    Nodes are coloured by moneyness relative to K.
    """
    if N > 10:
        raise ValueError("stock_tree_plot is intended for N ≤ 10 for readability.")

    tree = stock_tree(params, N)
    K    = params.K

    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []

    for t in range(N + 1):
        prices = [tree[t, j] for j in range(t + 1) if not np.isnan(tree[t, j])]
        # centre the prices vertically
        n      = len(prices)
        y_vals = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

        for k, (price, y) in enumerate(zip(prices, y_vals)):
            node_x.append(t)
            node_y.append(y)
            node_text.append(f"<b>{price:.2f}</b>")
            # colour: green = ITM call / OTM put, red = OTM call / ITM put
            if params.option_type.value == "call":
                node_color.append("#16A34A" if price > K else "#DC2626")
            else:
                node_color.append("#16A34A" if price < K else "#DC2626")

            # Edges to children
            if t < N:
                n_next   = t + 2
                y_next   = np.linspace(-(n_next - 1) / 2, (n_next - 1) / 2, n_next)
                for child_y in [y_next[k], y_next[k + 1]]:
                    edge_x += [t, t + 1, None]
                    edge_y += [y, child_y, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=_GREY, width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=22, color=node_color, line=dict(color="white", width=1.5)),
        hoverinfo="text",
        showlegend=False,
    ))

    # Strike price annotation
    fig.add_annotation(
        x=N, y=max(node_y) + 0.7,
        text=f"K = {K}",
        showarrow=False,
        font=dict(color=_ORANGE, size=12),
    )

    fig.update_layout(
        **_LAYOUT,
        title=f"Binomial stock price tree  (N={N})  |  S₀={params.S0}  K={K}",
        xaxis=dict(title="Time step", tickmode="linear", dtick=1, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=420,
    )
    return fig

# 5. All-in-one dashboard (saved as HTML)

def full_dashboard(params: OptionParams, output_path: str = "dashboard.html") -> None:
    """
    Export a self-contained HTML file with all four figures stacked.
    Open in any browser — no server needed.
    """
    figs = [
        convergence_plot(params),
        parity_oscillation_plot(params),
        sensitivity_plot(params, vary="sigma"),
        stock_tree_plot(params, N=5),
    ]

    html_parts = []
    for i, fig in enumerate(figs):
        html_parts.append(fig.to_html(
            full_html=(i == 0),
            include_plotlyjs=(i == 0),
            div_id=f"fig_{i}",
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        if len(figs) == 1:
            f.write(html_parts[0])
        else:
            # Merge: write first fig (has full HTML wrapper), inject rest before </body>
            combined = html_parts[0]
            insert   = "\n".join(html_parts[1:]) + "\n</body>"
            combined = combined.replace("</body>", insert)
            f.write(combined)

    print(f"Dashboard saved → {output_path}")
