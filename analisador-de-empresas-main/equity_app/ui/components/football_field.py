"""
Football-field valuation chart.

For each surviving model (DCF / Comparables / Monte Carlo / RI / DDM)
draws a vertical bar from its low estimate to its high estimate, with a
midpoint marker. Adds three reference lines:
- ``Current price`` (solid white)
- ``52W high / 52W low`` (dotted grey)
- ``Aggregator intrinsic`` (gold horizontal across the chart)

Bars are coloured green when the midpoint is above the current price
(undervalued by that model) and red otherwise.

Ranges per model:
    DCF             midpoint ± 15% (deterministic; no built-in band)
    Comparables     P25 / median / P75 from the model
    Monte Carlo     P5 / median / P95
    Residual Income midpoint ± 15%
    DDM             midpoint ± 15%
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from core.valuation_pipeline import ValuationResults
from ui.theme import (
    SURFACE, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, GAINS, LOSSES,
)


_BAND_FALLBACK_PCT = 0.15      # ±15% when the model has no native range


def _safe(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _model_range(midpoint: Optional[float]) -> Optional[tuple[float, float, float]]:
    m = _safe(midpoint)
    if m is None:
        return None
    return (m * (1 - _BAND_FALLBACK_PCT), m, m * (1 + _BAND_FALLBACK_PCT))


def _gather_ranges(
    results: ValuationResults,
) -> list[tuple[str, float, float, float, str]]:
    """Return list of (label, low, mid, high, model_key) for surviving models."""
    rows: list[tuple[str, float, float, float, str]] = []

    if results.dcf is not None:
        r = _model_range(results.dcf.intrinsic_value_per_share)
        if r is not None:
            rows.append(("DCF", *r, "dcf"))

    cmp_res = results.comparables
    if (cmp_res is not None
            and _safe(cmp_res.implied_per_share_median) is not None):
        lo = _safe(cmp_res.implied_per_share_low)
        hi = _safe(cmp_res.implied_per_share_high)
        mid = _safe(cmp_res.implied_per_share_median)
        if lo is None or hi is None:
            lo, mid, hi = mid * 0.85, mid, mid * 1.15
        rows.append(("Comparables", lo, mid, hi, "comps"))

    if results.monte_carlo is not None:
        mc = results.monte_carlo
        lo = _safe(mc.percentiles.get(5))
        hi = _safe(mc.percentiles.get(95))
        mid = _safe(mc.median)
        if mid is not None and lo is not None and hi is not None:
            rows.append(("Monte Carlo", lo, mid, hi, "monte_carlo"))

    if results.ddm is not None:
        r = _model_range(results.ddm.intrinsic_value_per_share)
        if r is not None:
            rows.append(("DDM", *r, "ddm"))

    if results.residual_income is not None:
        r = _model_range(results.residual_income.intrinsic_value_per_share)
        if r is not None:
            rows.append(("RI", *r, "ri"))

    if results.epv is not None:
        r = _model_range(results.epv.intrinsic_value_per_share)
        if r is not None:
            rows.append(("EPV", *r, "epv"))

    if results.multiples is not None:
        r = _model_range(results.multiples.implied_per_share_median)
        if r is not None:
            rows.append(("Multiples", *r, "multiples"))

    return rows


def _empty(fig: go.Figure, height: int, msg: str) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=0, r=0, t=10, b=0),
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(color=TEXT_MUTED, size=12),
                          x=0.5, y=0.5, xref="paper", yref="paper")],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def build_football_field_figure(
    results: ValuationResults,
    *,
    week52_low: Optional[float] = None,
    week52_high: Optional[float] = None,
    height: int = 360,
) -> go.Figure:
    fig = go.Figure()
    rows = _gather_ranges(results)
    if not rows:
        return _empty(fig, height, "No valuation models produced a range")

    current = _safe(results.current_price)
    agg = results.aggregator
    aggregator = (_safe(agg.intrinsic_per_share)
                  if agg is not None else None)
    agg_p25 = (_safe(agg.range_p25) if agg is not None else None)
    agg_p75 = (_safe(agg.range_p75) if agg is not None else None)
    clipped: set[str] = (set(agg.clipped_models)
                         if agg is not None and agg.clipped_models else set())

    labels = [r[0] for r in rows]
    lows   = [r[1] for r in rows]
    mids   = [r[2] for r in rows]
    highs  = [r[3] for r in rows]
    keys   = [r[4] for r in rows]

    # Bar colours: clipped models render in muted grey (40% opacity) so
    # the eye reads "present but excluded from the aggregator". Live
    # models use the gains/losses colour driven by midpoint-vs-price.
    bar_colors: list[str] = []
    bar_opacities: list[float] = []
    for mid, key in zip(mids, keys):
        if key in clipped:
            bar_colors.append(TEXT_SECONDARY)
            bar_opacities.append(0.30)
        elif current is None:
            bar_colors.append(ACCENT)
            bar_opacities.append(0.55)
        elif mid >= current:
            bar_colors.append(GAINS)
            bar_opacities.append(0.55)
        else:
            bar_colors.append(LOSSES)
            bar_opacities.append(0.55)

    fig.add_trace(go.Bar(
        x=labels,
        y=[h - l for l, h in zip(lows, highs)],
        base=lows,
        marker=dict(
            color=bar_colors,
            opacity=bar_opacities,
            line=dict(color=BORDER, width=1),
        ),
        width=0.55,
        name="Range",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Low  $%{customdata[0]:,.2f}<br>"
            "Mid  $%{customdata[1]:,.2f}<br>"
            "High $%{customdata[2]:,.2f}"
            "%{customdata[3]}"
            "<extra></extra>"
        ),
        customdata=[
            [f"{lo:.2f}", f"{mid:.2f}", f"{hi:.2f}",
             "<br><i>excluded from aggregator (sanity clip)</i>"
             if key in clipped else ""]
            for lo, mid, hi, key in zip(lows, mids, highs, keys)
        ],
    ))

    # Midpoint markers
    fig.add_trace(go.Scatter(
        x=labels, y=mids,
        mode="markers",
        marker=dict(color=TEXT_PRIMARY, size=10, symbol="line-ew",
                    line=dict(color=TEXT_PRIMARY, width=3)),
        name="Midpoint",
        hovertemplate="<b>%{x}</b><br>Mid $%{y:,.2f}<extra></extra>",
        showlegend=False,
    ))

    # Aggregator P25-P75 band — light gold rectangle across the chart
    if agg_p25 is not None and agg_p75 is not None and agg_p25 < agg_p75:
        fig.add_hrect(
            y0=agg_p25, y1=agg_p75,
            fillcolor=ACCENT,
            opacity=0.10,
            line_width=0,
            annotation=dict(
                text=f"Aggregator range  ${agg_p25:,.0f}–${agg_p75:,.0f}",
                font=dict(color=ACCENT, size=10),
                bgcolor=SURFACE,
            ),
            annotation_position="top left",
        )

    # Reference lines — current price + aggregator + 52W band
    if current is not None:
        fig.add_hline(
            y=current,
            line=dict(color=TEXT_PRIMARY, width=2, dash="solid"),
            annotation=dict(text=f"Current  ${current:,.2f}",
                            font=dict(color=TEXT_PRIMARY, size=11),
                            bgcolor=SURFACE),
            annotation_position="top right",
        )
    if aggregator is not None:
        fig.add_hline(
            y=aggregator,
            line=dict(color=ACCENT, width=2, dash="dot"),
            annotation=dict(text=f"Aggregator  ${aggregator:,.2f}",
                            font=dict(color=ACCENT, size=11),
                            bgcolor=SURFACE),
            annotation_position="bottom right",
        )
    if week52_low is not None and week52_low > 0:
        fig.add_hline(
            y=week52_low,
            line=dict(color=TEXT_MUTED, width=1, dash="dot"),
            annotation=dict(text=f"52W low  ${week52_low:,.2f}",
                            font=dict(color=TEXT_MUTED, size=10),
                            bgcolor=SURFACE),
            annotation_position="bottom left",
        )
    if week52_high is not None and week52_high > 0:
        fig.add_hline(
            y=week52_high,
            line=dict(color=TEXT_MUTED, width=1, dash="dot"),
            annotation=dict(text=f"52W high  ${week52_high:,.2f}",
                            font=dict(color=TEXT_MUTED, size=10),
                            bgcolor=SURFACE),
            annotation_position="top left",
        )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT_SECONDARY, family="Inter, sans-serif", size=11),
        showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                        font=dict(color=TEXT_PRIMARY, size=12)),
        xaxis=dict(color=TEXT_MUTED, showgrid=False, zeroline=False,
                   type="category"),
        yaxis=dict(color=TEXT_MUTED, showgrid=True, gridcolor=BORDER,
                   zeroline=False, tickprefix="$", tickformat=",.0f"),
        bargap=0.4,
    )
    return fig
