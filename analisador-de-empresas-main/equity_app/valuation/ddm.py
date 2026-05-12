"""
Dividend Discount Models — Gordon (single stage) and 2-stage.

DDM is the right tool when a company's value is mostly returned to
shareholders as cash (mature dividend payers, banks, REITs, utilities).
For non-dividend payers it returns a near-zero value, so we expose an
``is_applicable`` helper that the caller (or the aggregator) can use to
decide whether to include the model in the blended valuation.

Stage 1 (high growth) ⇒ explicit projection of dividends per share.
Stage 2 (terminal)    ⇒ Gordon-growth at the long-run rate.

Inputs: income/balance/cash DataFrames + cost of equity (NOT WACC —
DDM discounts equity cash flows so the equity rate is the right one).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get, cagr
from core.constants import DCF_DEFAULTS
from core.exceptions import InsufficientDataError, ValuationError


# ============================================================
# Result
# ============================================================
@dataclass
class DDMResult:
    intrinsic_value_per_share: float
    pv_explicit: float
    pv_terminal: float
    base_dividend: float
    cost_of_equity: float
    stage1_growth: float
    terminal_growth: float
    stage1_years: int
    payout_ratio: Optional[float] = None
    method: str = "two_stage"           # "gordon" | "two_stage"
    projected_dividends: list[float] = field(default_factory=list)


# ============================================================
# Internals
# ============================================================
def _shares_outstanding(income: pd.DataFrame, balance: pd.DataFrame) -> float:
    sh = _get(income, "weighted_avg_shares")
    if sh is not None and not sh.dropna().empty:
        return float(sh.dropna().iloc[-1])
    sh = _get(balance, "common_shares_outstanding")
    if sh is not None and not sh.dropna().empty:
        return float(sh.dropna().iloc[-1])
    raise InsufficientDataError("Cannot find share count")


def _dividends_per_share(
    cash: pd.DataFrame, income: pd.DataFrame, balance: pd.DataFrame,
    *,
    shares_override: Optional[float] = None,
) -> Optional[pd.Series]:
    """Convert ``dividendsPaid`` (negative cash outflow) into DPS using
    the weighted-average share count.

    Resolution order for shares:
      1. ``weighted_avg_shares`` series from the income statement
         (per-period — most accurate for DPS history).
      2. ``shares_override`` scalar from caller (broadcast across the
         dividend periods). Use this for thin-XBRL filers (V, MA, …)
         whose income/balance don't expose a share series; the
         pipeline has already resolved their count via the 4-tier
         cascade in ``core.valuation_pipeline``.
      3. Local ``_shares_outstanding(income, balance)`` (back-compat).

    Returns None when dividends are absent / zero / un-resolvable.
    """
    div = _get(cash, "dividends_paid")
    if div is None or div.dropna().empty:
        return None
    div = div.abs()                       # cash flow comes in negative
    if (div.dropna() <= 0).all():
        return None

    # Tier 1: per-period series from income statement
    shares = _get(income, "weighted_avg_shares")
    if shares is None or shares.dropna().empty:
        # Tier 2: caller-provided scalar (V, MA, … via pipeline)
        sh: Optional[float] = None
        if shares_override is not None and shares_override > 0:
            sh = float(shares_override)
        else:
            # Tier 3: local resolver (back-compat for legacy callers)
            try:
                sh = _shares_outstanding(income, balance)
            except InsufficientDataError:
                sh = None
        if sh is None or sh <= 0:
            return None
        shares = pd.Series(sh, index=div.index)

    aligned = pd.concat([div, shares], axis=1).dropna()
    if aligned.empty:
        return None
    aligned.columns = ["div", "shares"]
    return aligned["div"] / aligned["shares"]


def _payout_ratio(income: pd.DataFrame, cash: pd.DataFrame) -> Optional[float]:
    """3-year average payout ratio, capped at [0, 1.5]."""
    div = _get(cash, "dividends_paid")
    ni = _get(income, "net_income")
    if div is None or ni is None:
        return None
    df = pd.concat([div.abs(), ni], axis=1).dropna().tail(3)
    if df.empty:
        return None
    df.columns = ["div", "ni"]
    df = df[df["ni"] > 0]
    if df.empty:
        return None
    pr = (df["div"] / df["ni"]).mean()
    return float(np.clip(pr, 0.0, 1.5))


# ============================================================
# Public API
# ============================================================
def is_applicable(
    cash: pd.DataFrame,
    income: pd.DataFrame,
    *,
    shares_outstanding: Optional[float] = None,
    min_payout: float = 0.15,
    min_dps: float = 0.10,
) -> bool:
    """DDM is meaningful when the company pays *meaningful* dividends.

    "Meaningful" uses two floors with OR logic to catch both low-payout
    payers (V, MA, AAPL: payout 14-23%, but $1-3/share absolute) AND
    high-payout-low-yield edge cases. NVDA / GOOGL with token quarterly
    dividends correctly stay below both floors.

    - ``min_payout``: payout-ratio floor (default 15%). Damodaran's
      threshold for "the company prioritizes returning cash via
      dividends over buybacks". Lowered from 20% so AAPL (~14%) and
      MA (~18%) pass; they shouldn't be excluded just because their
      buyback program is larger than their dividend program.
    - ``min_dps``: absolute dollars-per-share-per-year floor. Filters
      out token dividends (NVDA $0.04/share/yr) that pass a high
      payout ratio purely due to low NI in a given year.

    Uses the MOST RECENT year's data (not 3y avg) — averaging mutes
    transitional cases like MU where dividends were re-initiated only
    recently with a tiny payout.
    """
    div = _get(cash, "dividends_paid")
    if div is None:
        return False
    s = div.abs().dropna()
    if (s > 0).sum() < 2:
        return False

    ni = _get(income, "net_income")
    if ni is None:
        return False
    df = pd.concat([div.abs(), ni], axis=1).dropna().tail(1)
    if df.empty:
        return False
    div_last, ni_last = float(df.iloc[0, 0]), float(df.iloc[0, 1])

    payout_ok = ni_last > 0 and (div_last / ni_last) >= min_payout

    # Per-share floor: catches absolute-dividend payers whose payout
    # ratio is depressed (high buyback) but who still distribute real
    # cash per share. Falls back to the in-module share resolver when
    # the caller didn't supply ``shares_outstanding``.
    dps_ok = False
    try:
        sh = float(shares_outstanding) if shares_outstanding else 0.0
        if sh <= 0:
            sh = 0.0
    except (TypeError, ValueError):
        sh = 0.0
    if sh > 0:
        dps_ok = (div_last / sh) >= min_dps

    return payout_ok or dps_ok


def gordon(
    *,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    cost_of_equity: float,
    terminal_growth: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
) -> DDMResult:
    """Single-stage Gordon model: D1 / (re − g).

    ``shares_outstanding`` — see ``two_stage`` docstring; same purpose
    (thin-XBRL filers like V whose share count needs the pipeline's
    4-tier resolver)."""
    g_t = float(terminal_growth if terminal_growth is not None
                else DCF_DEFAULTS["terminal_growth"])
    spread = float(DCF_DEFAULTS["min_wacc_terminal_spread"])
    if cost_of_equity - g_t < spread:
        raise ValuationError(
            f"Cost of equity ({cost_of_equity:.2%}) must exceed terminal growth "
            f"({g_t:.2%}) by at least {spread:.0%}"
        )

    dps = _dividends_per_share(
        cash, income, balance, shares_override=shares_outstanding,
    )
    if dps is None:
        raise InsufficientDataError("Company does not pay dividends")
    base_dps = float(dps.dropna().iloc[-1])
    if base_dps <= 0:
        raise InsufficientDataError("Most recent dividend per share is non-positive")

    next_dps = base_dps * (1.0 + g_t)
    intrinsic = next_dps / (cost_of_equity - g_t)

    return DDMResult(
        intrinsic_value_per_share=float(intrinsic),
        pv_explicit=0.0,
        pv_terminal=float(intrinsic),
        base_dividend=base_dps,
        cost_of_equity=float(cost_of_equity),
        stage1_growth=g_t,
        terminal_growth=g_t,
        stage1_years=0,
        payout_ratio=_payout_ratio(income, cash),
        method="gordon",
        projected_dividends=[float(next_dps)],
    )


def two_stage(
    *,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    cost_of_equity: float,
    stage1_growth: Optional[float] = None,
    stage1_years: int = 5,
    terminal_growth: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
) -> DDMResult:
    """
    2-stage DDM:
        - explicit DPS for ``stage1_years`` at ``stage1_growth``
        - terminal Gordon at ``terminal_growth`` from year stage1_years+1
    Stage-1 growth defaults to historical DPS CAGR, clipped to DCF caps.

    ``shares_outstanding`` is an optional override for thin-XBRL filers
    (V, MA, …) whose income/balance frames don't expose a share series;
    the pipeline resolves it once via its 4-tier cascade and passes it
    here so this module doesn't have to duplicate that logic.
    """
    g_t = float(terminal_growth if terminal_growth is not None
                else DCF_DEFAULTS["terminal_growth"])
    spread = float(DCF_DEFAULTS["min_wacc_terminal_spread"])
    if cost_of_equity - g_t < spread:
        raise ValuationError(
            f"Cost of equity ({cost_of_equity:.2%}) must exceed terminal growth "
            f"({g_t:.2%}) by at least {spread:.0%}"
        )

    dps = _dividends_per_share(
        cash, income, balance, shares_override=shares_outstanding,
    )
    if dps is None:
        raise InsufficientDataError("Company does not pay dividends")
    s = dps.dropna()
    base_dps = float(s.iloc[-1])
    if base_dps <= 0:
        raise InsufficientDataError("Most recent DPS is non-positive")

    if stage1_growth is None:
        cg = cagr(s)
        stage1_growth = cg if np.isfinite(cg) else g_t
    g1 = float(np.clip(
        stage1_growth,
        DCF_DEFAULTS["growth_cap_lower"],
        DCF_DEFAULTS["growth_cap_upper"],
    ))

    projected: list[float] = []
    pv_explicit = 0.0
    d_t = base_dps
    for t in range(1, stage1_years + 1):
        d_t = d_t * (1.0 + g1)
        projected.append(d_t)
        pv_explicit += d_t / (1.0 + cost_of_equity) ** t

    terminal_dps = projected[-1] * (1.0 + g_t)
    terminal_value = terminal_dps / (cost_of_equity - g_t)
    pv_terminal = terminal_value / (1.0 + cost_of_equity) ** stage1_years

    intrinsic = pv_explicit + pv_terminal
    return DDMResult(
        intrinsic_value_per_share=float(intrinsic),
        pv_explicit=float(pv_explicit),
        pv_terminal=float(pv_terminal),
        base_dividend=base_dps,
        cost_of_equity=float(cost_of_equity),
        stage1_growth=g1,
        terminal_growth=g_t,
        stage1_years=int(stage1_years),
        payout_ratio=_payout_ratio(income, cash),
        method="two_stage",
        projected_dividends=[float(x) for x in projected],
    )
