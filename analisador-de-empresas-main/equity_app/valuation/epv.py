"""
Earnings Power Value — Koller no-growth edition.

Conceptually EPV is Koller's continuing value with ``g=0`` and
``reinvestment=0``: the value of the business "as it stands today",
ignoring growth.

    EV_EPV = NOPAT_normalized / WACC

For consistency with the Damodaran-Koller DCF (Phase 2), the
normalization respects the lifecycle stage:

- ``cyclical``: through-cycle winsorized margin x latest revenue
  (matches DCF's _through_cycle_nopat — Damodaran Ch 22).
- ``declining``: 3y average NOPAT (conservative; assumes no recovery).
- else (mature_*, growth_*): 3y average NOPAT (smooths one-off
  charges without leaning on a single year).

The result is *robust for mature compounders* (consumer staples,
healthcare, retail) where DCF/Monte Carlo can mis-price because growth
assumptions dominate. It is also a *floor* for growth companies — when
the DCF is much higher than EPV, the gap is the value-of-growth, which
is exactly Koller's central diagnostic.

Ref: Greenwald, "Value Investing: From Graham to Buffett and Beyond",
Ch 4-6; Koller, "Valuation", Ch 11 (value-driver decomposition).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get
from analysis.koller_reorg import ReorganizedFinancials
from core.exceptions import InsufficientDataError, ValuationError


@dataclass
class EPVResult:
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    nopat_normalized: float
    wacc: float
    net_debt: float
    # Diagnostic
    current_nopat: float                       # latest single year
    normalization_factor: float                # current / normalized

    # ---- Back-compat fields (the old EPVResult had these) ----
    avg_ebit: float = 0.0
    tax_rate: float = 0.0
    normalization_years: int = 0


def _last_or_zero(series: Optional[pd.Series]) -> float:
    if series is None:
        return 0.0
    s = series.dropna()
    if s.empty:
        return 0.0
    v = float(s.iloc[-1])
    return v if np.isfinite(v) else 0.0


def _resolve_total_debt(balance: pd.DataFrame) -> float:
    debt = _get(balance, "total_debt")
    if debt is not None and not debt.dropna().empty:
        return _last_or_zero(debt)
    ltd = _last_or_zero(_get(balance, "long_term_debt"))
    std = _last_or_zero(_get(balance, "short_term_debt"))
    return ltd + std


def _through_cycle_nopat_epv(
    income: pd.DataFrame, reorg: ReorganizedFinancials,
) -> Optional[float]:
    """Winsorized through-cycle NOPAT. Same algorithm as the DCF's
    helper so both models agree on cyclicals' base."""
    revenue = _get(income, "revenue")
    if revenue is None or revenue.dropna().empty:
        return None
    avg_tax_full = float(reorg.effective_tax_rate.mean())
    if not np.isfinite(avg_tax_full) or avg_tax_full <= 0:
        avg_tax_full = 0.21
    ebit_series = reorg.nopat / (1.0 - avg_tax_full)
    valid_idx = revenue.index.intersection(ebit_series.index)
    if len(valid_idx) < 2:
        return None
    n_years = min(10, len(valid_idx))
    rev_window = revenue.loc[valid_idx].tail(n_years)
    ebit_window = ebit_series.loc[valid_idx].tail(n_years)
    margins = (ebit_window / rev_window.where(rev_window > 0)).dropna()
    if len(margins) >= 5:
        sorted_m = margins.sort_values()
        through_cycle_margin = float(sorted_m.iloc[1:-1].mean())
    elif len(margins) >= 3:
        through_cycle_margin = float(margins.median())
    elif not margins.empty:
        through_cycle_margin = float(margins.iloc[-1])
    else:
        return None
    latest_revenue = float(revenue.dropna().iloc[-1])
    fwd_tax = (reorg.avg_tax_rate_3y
               if np.isfinite(reorg.avg_tax_rate_3y) and reorg.avg_tax_rate_3y > 0
               else avg_tax_full)
    return float(latest_revenue * through_cycle_margin * (1.0 - fwd_tax))


# ============================================================
# Public API
# ============================================================
def run_epv(
    *,
    reorg: Optional[ReorganizedFinancials] = None,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    wacc: float,
    shares_outstanding: float,
    stage: Optional[str] = None,
    # ---- Back-compat kwargs ----
    tax_rate: Optional[float] = None,                # legacy override; ignored when reorg provided
    normalization_years: int = 3,                    # legacy default was 5; new default 3 matches spec
) -> EPVResult:
    """Compute Earnings Power Value per share.

    Stage-aware normalization:
      - ``cyclical``: through-cycle winsorized margin x latest revenue
      - else: ``normalization_years`` average of NOPAT
    """
    if wacc is None or not np.isfinite(wacc) or wacc <= 0:
        raise ValuationError("WACC must be positive for EPV perpetuity")
    if shares_outstanding is None or shares_outstanding <= 0:
        raise InsufficientDataError("Shares outstanding required for EPV")

    # ---- Build the reorg if the caller didn't pass one (legacy callers) ----
    if reorg is None:
        from analysis.koller_reorg import reorganize as _reorganize
        try:
            reorg = _reorganize(
                income, balance, pd.DataFrame(),  # cash not strictly needed
                wacc=wacc,
            )
        except Exception as exc:
            raise InsufficientDataError(f"Cannot reorganize for EPV: {exc}")

    # ---- Normalize NOPAT by stage ----
    nopat_normalized: Optional[float] = None
    if stage == "cyclical":
        nopat_normalized = _through_cycle_nopat_epv(income, reorg)
        if nopat_normalized is None:
            # Fallback to mean if through-cycle math fails (thin history).
            nopat_clean = reorg.nopat.replace([np.inf, -np.inf], np.nan).dropna()
            if not nopat_clean.empty:
                nopat_normalized = float(nopat_clean.tail(normalization_years).mean())
    else:
        nopat_clean = reorg.nopat.replace([np.inf, -np.inf], np.nan).dropna()
        if nopat_clean.empty:
            raise InsufficientDataError("No NOPAT history for EPV normalization")
        n = min(int(normalization_years), len(nopat_clean))
        nopat_normalized = float(nopat_clean.tail(n).mean())

    if nopat_normalized is None or not np.isfinite(nopat_normalized) or nopat_normalized <= 0:
        raise InsufficientDataError(
            f"Normalized NOPAT non-positive ({nopat_normalized!r}) - EPV not "
            "applicable. Company in distress or restructuring."
        )

    # ---- Enterprise value (Koller's no-growth case) ----
    enterprise_value = nopat_normalized / wacc

    # ---- Net debt + equity ----
    cash_s = _get(balance, "cash_eq")
    net_debt = _resolve_total_debt(balance) - _last_or_zero(cash_s)
    equity_value = enterprise_value - net_debt
    per_share = equity_value / float(shares_outstanding)

    # ---- Diagnostics ----
    current_nopat = float(reorg.latest_nopat) if np.isfinite(reorg.latest_nopat) else 0.0
    normalization_factor = (current_nopat / nopat_normalized
                            if nopat_normalized != 0 else float("nan"))

    # Legacy back-compat fields
    avg_tax = (float(reorg.avg_tax_rate_3y)
               if np.isfinite(reorg.avg_tax_rate_3y) else 0.21)
    avg_ebit = nopat_normalized / (1.0 - avg_tax) if avg_tax < 1.0 else nopat_normalized

    return EPVResult(
        intrinsic_value_per_share=float(per_share),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        nopat_normalized=float(nopat_normalized),
        wacc=float(wacc),
        net_debt=float(net_debt),
        current_nopat=float(current_nopat),
        normalization_factor=float(normalization_factor)
                              if np.isfinite(normalization_factor) else float("nan"),
        avg_ebit=float(avg_ebit),
        tax_rate=float(avg_tax),
        normalization_years=int(min(normalization_years, len(reorg.nopat.dropna()))),
    )
