"""
Damodaran lifecycle stage classifier.

Companies move through stages: idea -> young growth -> high growth ->
mature growth -> mature stable -> decline. Valuation approach and
multiple weighting depends on stage.

Ref: Damodaran, "The Little Book of Valuation", Ch 7 ("Valuing the
Whole World"); also "Investment Valuation", Part V.
"""
from __future__ import annotations
from typing import Literal, Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get, free_cash_flow, cagr


LifecycleStage = Literal[
    "young_growth",
    "high_growth",
    "mature_growth",
    "mature_stable",
    "declining",
    "cyclical",
]


def _safe_mean(s: Optional[pd.Series]) -> float:
    if s is None:
        return float("nan")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if not s.empty else float("nan")


def _safe_std(s: Optional[pd.Series]) -> float:
    if s is None:
        return float("nan")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.std(ddof=0)) if len(s) >= 2 else float("nan")


def classify_lifecycle(
    income: pd.DataFrame,
    cash: pd.DataFrame,
    *,
    ticker: str,
    sector: Optional[str] = None,
) -> dict:
    """Classify a company into a Damodaran-style lifecycle stage.

    Returns a dict with the stage label, the underlying metrics that
    drove the decision, and a human-readable rationale. Never raises:
    when the inputs are too thin to classify, returns
    ``stage='mature_stable'`` with a rationale flagging the gap.
    """
    revenue = _get(income, "revenue")
    net_income = _get(income, "net_income")
    fcf = free_cash_flow(cash)

    rev_cagr = cagr(revenue, periods=4) if revenue is not None else float("nan")
    if not np.isfinite(rev_cagr) and revenue is not None:
        # Fall back to whatever history we have (minimum 2 points).
        rev_cagr = cagr(revenue)

    # FCF margin: per-year FCF/Revenue, then 5y mean.
    fcf_margin_avg = float("nan")
    if revenue is not None and fcf is not None:
        aligned_rev = revenue.where(revenue > 0)
        fcf_margin = (fcf / aligned_rev).tail(5)
        fcf_margin_avg = _safe_mean(fcf_margin)

    # Earnings volatility: std of NI/Revenue over up to 7y of history.
    earn_vol = float("nan")
    if revenue is not None and net_income is not None:
        aligned_rev = revenue.where(revenue > 0)
        ni_margin = (net_income / aligned_rev).tail(7)
        earn_vol = _safe_std(ni_margin)

    # ---- Sector / OEM cyclical override (precede the rule chain) ----
    # Damodaran Ch 22: sigma-based detection misses commodity-driven
    # cyclicality. Energy / Materials have stable NET margins (cost &
    # revenue move together) yet ARE cyclical via underlying commodity
    # prices. Auto OEMs are structurally cyclical via the consumer
    # credit cycle, regardless of recent CAGR.
    sector_str = (sector or "").lower()
    CYCLICAL_SECTOR_KEYWORDS = (
        "energy",            # XOM, CVX, COP, OXY, EOG, MPC, PSX, SLB
        "materials",         # FCX, NUE, NEM, X, DD, LIN, APD
        "basic materials",   # alternative naming convention
    )
    AUTO_OEM_TICKERS = {"F", "GM", "STLA", "TSLA", "RIVN", "LCID"}
    tkr_upper = (ticker or "").upper()

    def _override(stage_str: str, why: str) -> dict:
        return {
            "stage": stage_str,
            "revenue_5y_cagr": float(rev_cagr) if np.isfinite(rev_cagr) else float("nan"),
            "fcf_5y_avg_margin": (float(fcf_margin_avg)
                                  if np.isfinite(fcf_margin_avg) else float("nan")),
            "earnings_volatility": (float(earn_vol)
                                    if np.isfinite(earn_vol) else float("nan")),
            "rationale": why,
        }

    if any(kw in sector_str for kw in CYCLICAL_SECTOR_KEYWORDS):
        return _override("cyclical", f"cyclical - sector '{sector}' is commodity-driven")
    if tkr_upper in AUTO_OEM_TICKERS:
        return _override("cyclical", f"cyclical - auto OEM ({tkr_upper})")

    # ---- Stage classification ----
    # Recent YoY (most recent observation only) — used to confirm
    # whether a negative CAGR is structural decline or a one-off
    # divestiture / restructuring year already in recovery.
    last_yoy = float("nan")
    if revenue is not None and len(revenue.dropna()) >= 2:
        rev_clean = revenue.dropna()
        last_yoy = float(rev_clean.iloc[-1] / rev_clean.iloc[-2]) - 1.0

    stage: LifecycleStage
    if (np.isfinite(earn_vol) and earn_vol > 0.12
            and np.isfinite(rev_cagr) and rev_cagr < 0.25):
        stage = "cyclical"
    elif (np.isfinite(rev_cagr) and rev_cagr < -0.03
          and np.isfinite(last_yoy) and last_yoy < 0.02):
        # Sustained negative CAGR AND most recent year not recovering.
        # If the last YoY shows >=2% growth, we treat it as a one-off
        # decline already reversed and fall through to the mature
        # branches instead of labeling it 'declining' permanently.
        stage = "declining"
    elif (np.isfinite(rev_cagr) and rev_cagr > 0.25
          and np.isfinite(fcf_margin_avg) and fcf_margin_avg < 0.05):
        stage = "young_growth"
    elif np.isfinite(rev_cagr) and rev_cagr > 0.15:
        stage = "high_growth"
    elif np.isfinite(rev_cagr) and 0.07 <= rev_cagr <= 0.15:
        stage = "mature_growth"
    elif np.isfinite(rev_cagr) and rev_cagr < 0.07:
        stage = "mature_stable"
    else:
        # Not enough history to compute CAGR cleanly — default to the
        # least-aggressive stage and let the rationale say why.
        stage = "mature_stable"

    rationale = (
        f"{stage} - revenue CAGR "
        f"{(rev_cagr if np.isfinite(rev_cagr) else float('nan')):.1%}, "
        f"FCF margin "
        f"{(fcf_margin_avg if np.isfinite(fcf_margin_avg) else float('nan')):.1%}, "
        f"earnings sigma "
        f"{(earn_vol if np.isfinite(earn_vol) else float('nan')):.1%}"
    )

    return {
        "stage": stage,
        "revenue_5y_cagr": float(rev_cagr) if np.isfinite(rev_cagr) else float("nan"),
        "fcf_5y_avg_margin": float(fcf_margin_avg) if np.isfinite(fcf_margin_avg) else float("nan"),
        "earnings_volatility": float(earn_vol) if np.isfinite(earn_vol) else float("nan"),
        "rationale": rationale,
    }
