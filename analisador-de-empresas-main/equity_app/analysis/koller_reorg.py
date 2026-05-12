"""
Koller-style reorganization of financial statements.

Separa OPERATING de FINANCING items para computar NOPAT (Net Operating
Profit After Tax) e Invested Capital limpios. Esta es la base de
todas las metricas de value creation (ROIC, value driver formula).

Ref: Koller, Goedhart & Wessels, "Valuation: Measuring and Managing
the Value of Companies", 7th ed, Ch 11 ("Reorganizing the Financial
Statements").
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get


# Internal floor used to clamp RONIC. A return on new invested capital
# below WACC means new investments destroy value; Koller / Damodaran
# both anchor RONIC plausibility against the cost of capital. The
# downstream caller (phase 2+) can pipe a ticker-specific WACC in via
# the `wacc` kwarg; this default is only the fallback floor.
_DEFAULT_WACC_FOR_RONIC = 0.08


@dataclass
class ReorganizedFinancials:
    """NOPAT-, IC- and FCFF-level view of the business, free of financing
    noise. All per-year series share the income statement's fiscal index
    (ascending). Latest-* fields are convenience snapshots."""
    nopat: pd.Series                       # Net Operating Profit After Tax
    invested_capital: pd.Series            # Operating capital (financing-side)
    roic: pd.Series                        # NOPAT / avg(IC)
    ronic: Optional[float]                 # Return on New IC (last 3y window)
    fcff: pd.Series                        # NOPAT - dIC
    reinvestment_rate: pd.Series           # dIC / NOPAT
    effective_tax_rate: pd.Series          # Tax / Pre-tax income (per year)
    avg_tax_rate_3y: float                 # Smoothed 3y mean of effective rate
    latest_year: Optional[int]
    latest_nopat: float
    latest_ic: float
    latest_roic: float


# ============================================================
# Internals
# ============================================================
def _resolve_total_debt(balance: pd.DataFrame) -> Optional[pd.Series]:
    debt = _get(balance, "total_debt")
    if debt is not None:
        return debt
    ltd = _get(balance, "long_term_debt")
    std = _get(balance, "short_term_debt")
    if ltd is None and std is None:
        return None
    if ltd is None:
        return std
    if std is None:
        return ltd
    return ltd.add(std, fill_value=0.0)


def _effective_tax_rate_series(
    income: pd.DataFrame, marginal: float,
) -> pd.Series:
    """Per-year effective tax rate. Falls back to marginal when the ratio
    is NaN or pre-tax income is zero/negative. Always clipped to
    [0.10, 0.40] — outliers (one-time benefits, NOL releases) distort
    NOPAT trends and the Koller / Damodaran convention is to bound them.
    """
    tax = _get(income, "income_tax")
    pretax = _get(income, "pretax_income")
    if tax is None or pretax is None:
        idx = income.index if income is not None else pd.Index([])
        return pd.Series(marginal, index=idx, dtype=float)
    raw = tax.where(pretax > 0) / pretax.where(pretax > 0)
    raw = raw.where(~raw.isna(), marginal)
    return raw.clip(lower=0.10, upper=0.40)


def _latest_index_label(s: pd.Series) -> Optional[int]:
    """Return the latest period label as an int year if possible, else
    the raw label, else None."""
    if s is None or s.dropna().empty:
        return None
    label = s.dropna().index[-1]
    try:
        if hasattr(label, "year"):
            return int(label.year)
        return int(label)
    except (TypeError, ValueError):
        return label


# ============================================================
# Public API
# ============================================================
def reorganize(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    *,
    marginal_tax_rate: float = 0.21,
    wacc: float = _DEFAULT_WACC_FOR_RONIC,
) -> ReorganizedFinancials:
    """Reorganize the three statements into NOPAT / IC / FCFF series.

    Parameters
    ----------
    income, balance, cash : pd.DataFrame
        Period-indexed (ascending) frames in FMP camelCase shape.
    marginal_tax_rate : float
        Fallback tax rate when per-year effective rate is unrecoverable.
    wacc : float
        Used only to floor the RONIC clamp (RONIC < WACC means new
        capital destroys value; the spec clamps to WACC * 0.7 as the
        lower bound).
    """
    ebit = _get(income, "ebit")
    if ebit is None or ebit.dropna().empty:
        raise ValueError(
            "Cannot reorganize statements: operating income / EBIT is "
            "missing from the income statement."
        )

    # ---- Tax rate ----
    eff_tax = _effective_tax_rate_series(income, marginal_tax_rate)
    eff_tax = eff_tax.reindex(ebit.index).fillna(marginal_tax_rate)
    avg_tax_3y = float(eff_tax.tail(3).mean()) if not eff_tax.empty else marginal_tax_rate

    # ---- NOPAT ----
    nopat = ebit * (1.0 - eff_tax)

    # ---- Invested Capital (financing side) ----
    equity = _get(balance, "total_equity")
    debt = _resolve_total_debt(balance)
    cash_eq = _get(balance, "cash_eq")
    if equity is None or debt is None:
        raise ValueError(
            "Cannot reorganize: invested capital requires total equity "
            "and total debt on the balance sheet."
        )
    ic = equity.add(debt, fill_value=0.0)
    if cash_eq is not None:
        ic = ic.sub(cash_eq, fill_value=0.0)

    # Align IC to the NOPAT index so per-year math doesn't drift.
    ic = ic.reindex(ebit.index)

    # ---- ROIC ----
    avg_ic = ic.rolling(window=2, min_periods=1).mean()
    roic = nopat / avg_ic.where(avg_ic != 0)

    # ---- FCFF & Reinvestment ----
    d_ic = ic.diff()
    fcff = nopat - d_ic.fillna(0.0)
    rr_raw = d_ic / nopat.where(nopat != 0)
    reinvestment_rate = rr_raw.clip(lower=-0.5, upper=1.5)

    # ---- RONIC (last 3 years window) ----
    ronic = _compute_ronic(nopat, ic, roic, wacc=wacc)

    # ---- Snapshot ----
    latest_year = _latest_index_label(nopat)
    latest_nopat = float(nopat.dropna().iloc[-1]) if not nopat.dropna().empty else float("nan")
    latest_ic = float(ic.dropna().iloc[-1]) if not ic.dropna().empty else float("nan")
    latest_roic = float(roic.dropna().iloc[-1]) if not roic.dropna().empty else float("nan")

    return ReorganizedFinancials(
        nopat=nopat,
        invested_capital=ic,
        roic=roic,
        ronic=ronic,
        fcff=fcff,
        reinvestment_rate=reinvestment_rate,
        effective_tax_rate=eff_tax,
        avg_tax_rate_3y=avg_tax_3y,
        latest_year=latest_year,
        latest_nopat=latest_nopat,
        latest_ic=latest_ic,
        latest_roic=latest_roic,
    )


def _compute_ronic(
    nopat: pd.Series,
    ic: pd.Series,
    roic: pd.Series,
    *,
    wacc: float,
) -> Optional[float]:
    """Return on new invested capital over the last 3y window:
        RONIC ~= sum(dNOPAT_3y) / sum(dIC_3y)
    Falls back to the 3y average ROIC when net dIC is negative (the
    business disinvested — incremental returns aren't observable, so
    Koller's convention is to assume new capital matches the average).
    Final value is clamped to [WACC*0.7, ROIC*1.0]."""
    d_nopat = nopat.diff().dropna().tail(3)
    d_ic = ic.diff().dropna().tail(3)
    if d_nopat.empty or d_ic.empty:
        return None
    roic_avg = float(roic.dropna().tail(3).mean()) if not roic.dropna().empty else None
    sum_dic = float(d_ic.sum())
    sum_dnopat = float(d_nopat.sum())
    if sum_dic <= 0:
        ronic = roic_avg
    else:
        ronic = sum_dnopat / sum_dic
    if ronic is None or not np.isfinite(ronic):
        return None
    lower = wacc * 0.7
    upper = roic_avg if (roic_avg is not None and np.isfinite(roic_avg)) else ronic
    if upper < lower:
        upper = lower
    return float(np.clip(ronic, lower, upper))
