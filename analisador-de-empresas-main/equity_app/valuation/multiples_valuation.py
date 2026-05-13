"""
Multiples valuation — Damodaran intrinsic edition.

Replaces the sector-median-multiple approach with *intrinsic* multiples
derived from the company's own fundamentals (growth, ROE, payout, cost
of equity, RONIC). Three multiples are blended:

    P/E       = (payout * (1+g)) / (Ke - g)
    P/B       = (ROE - g) / (Ke - g)
    EV/EBITDA = ((1-t)(1 - D&A/EBITDA) - CapEx/EBITDA - dWC/EBITDA)
                * (1+g) / (WACC - g)

The sector medians are still surfaced — as a *diagnostic*, not as the
valuation anchor — so the UI can show the story-to-numbers gap (sector
trades at premium / discount vs the company's intrinsic multiples).

Ref: Damodaran, "Investment Valuation", Part IV, Ch 17-19
("Fundamentals-driven Relative Valuation").
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get
from analysis.koller_reorg import ReorganizedFinancials
from valuation.fundamental_growth import FundamentalGrowth
from core.exceptions import InsufficientDataError
from data.industry_benchmarks import get_benchmark, normalise_sector


@dataclass
class MultiplesValuationResult:
    """Damodaran intrinsic multiples + sector-actual diagnostics."""
    intrinsic_pe: Optional[float]
    intrinsic_pb: Optional[float]
    intrinsic_evebitda: Optional[float]

    implied_per_share_pe: Optional[float]
    implied_per_share_pb: Optional[float]
    implied_per_share_evebitda: Optional[float]

    implied_per_share_median: float

    sector_actual_pe: Optional[float]
    sector_actual_pb: Optional[float]
    sector_actual_evebitda: Optional[float]

    pe_premium_or_discount: Optional[float]   # (actual - intrinsic) / intrinsic
    notes: list[str] = field(default_factory=list)

    # Back-compat alias for code expecting the old MultiplesResult shape.
    @property
    def sector_used(self) -> str:
        return "intrinsic"


# Back-compat alias — pipeline/UI imports may still reference the old name.
MultiplesResult = MultiplesValuationResult


# ============================================================
# Internals
# ============================================================
def _last_finite(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) else None


def _resolve_total_debt(balance: pd.DataFrame) -> float:
    debt = _get(balance, "total_debt")
    last = _last_finite(debt) if debt is not None else None
    if last is not None:
        return last
    ltd = _last_finite(_get(balance, "long_term_debt")) or 0.0
    std = _last_finite(_get(balance, "short_term_debt")) or 0.0
    return ltd + std


def _resolve_ebitda(income: pd.DataFrame, cash: pd.DataFrame) -> Optional[float]:
    direct = _last_finite(_get(income, "ebitda"))
    if direct is not None:
        return direct
    ebit_last = _last_finite(_get(income, "ebit"))
    da_last = _last_finite(_get(cash, "depreciation_cf"))
    if da_last is None:
        da_last = _last_finite(_get(income, "depreciation_inc"))
    if ebit_last is None or da_last is None:
        return None
    return ebit_last + da_last


def _smoothed_roe(income: pd.DataFrame, balance: pd.DataFrame) -> Optional[float]:
    ni = _get(income, "net_income")
    eq = _get(balance, "total_equity")
    if ni is None or eq is None:
        return None
    avg_eq = eq.rolling(window=2, min_periods=1).mean()
    roe = (ni / avg_eq.where(avg_eq != 0)).replace([np.inf, -np.inf], np.nan).dropna()
    if roe.empty:
        return None
    val = float(roe.tail(3).mean())
    if not np.isfinite(val):
        return None
    return float(np.clip(val, 0.0, 0.60))


def _delta_working_capital(balance: pd.DataFrame) -> float:
    """Latest YoY change in non-cash working capital. Returns 0 when the
    series is too thin to compute — a safe neutral assumption for the
    EV/EBITDA numerator."""
    ca = _get(balance, "current_assets")
    cl = _get(balance, "current_liabilities")
    cash_eq = _get(balance, "cash_eq")
    if ca is None or cl is None:
        return 0.0
    wc = ca.sub(cl, fill_value=0.0)
    if cash_eq is not None:
        wc = wc.sub(cash_eq, fill_value=0.0)
    wc = wc.dropna()
    if len(wc) < 2:
        return 0.0
    return float(wc.iloc[-1] - wc.iloc[-2])


# ============================================================
# Public API
# ============================================================
def run_multiples_valuation(
    *,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    reorg: ReorganizedFinancials,
    growth: FundamentalGrowth,
    stage: str,
    wacc: float,
    cost_of_equity: Optional[float] = None,
    sector: Optional[str] = None,
    shares_outstanding: Optional[float] = None,
) -> MultiplesValuationResult:
    """Damodaran intrinsic multiples valuation.

    Uses ``growth.recommended_g_stable`` because the closed-form
    Gordon/Damodaran formulas assume stable growth in perpetuity. The
    *explicit* growth would inflate the multiples beyond what the
    formula's constant-state assumptions justify. For growth-stage
    companies, this is an acknowledged simplification (the proper
    Damodaran two-stage multiple model lives in his texts but adds
    little for stable inference). We log a note when stage is
    growth-stage so downstream can flag reduced accuracy.
    """
    if shares_outstanding is None or shares_outstanding <= 0:
        raise InsufficientDataError("Shares outstanding required for multiples")

    notes: list[str] = []
    if stage in ("high_growth", "young_growth"):
        notes.append(
            "Intrinsic multiples assume stable growth — accuracy reduced "
            "for growth-stage companies; use as a lower-bound diagnostic."
        )

    # ---- Ke + g + t + roe ----
    ke = float(cost_of_equity) if cost_of_equity is not None else float(wacc) + 0.01
    g = float(growth.recommended_g_stable)
    if not np.isfinite(g):
        g = 0.025
    t = float(reorg.avg_tax_rate_3y) if np.isfinite(reorg.avg_tax_rate_3y) else 0.21

    roe = _smoothed_roe(income, balance)
    if roe is None or roe <= g + 0.01 or not np.isfinite(roe):
        # Fallback: floor at slightly above g so payout math is well-defined.
        # `<=` (not `<`) catches the edge where _smoothed_roe clips a negative
        # mean to exactly 0.0 (loss-making companies) and g_stable is -0.01
        # (declining lifecycle) — then both sides equal 0 and the divisor
        # `max(roe, g + 0.01)` below collapses to 0.0 → ZeroDivisionError.
        roe = max(g + 0.02, 0.08)

    # ---- Intrinsic P/E ----
    intrinsic_pe: Optional[float] = None
    if ke > g:
        payout = max(0.0, min(1.0, 1.0 - g / max(roe, g + 0.01)))
        pe_raw = (payout * (1.0 + g)) / (ke - g)
        if np.isfinite(pe_raw) and pe_raw > 0:
            intrinsic_pe = float(np.clip(pe_raw, 5.0, 60.0))

    # ---- Intrinsic P/B ----
    intrinsic_pb: Optional[float] = None
    if ke > g and roe > g:
        pb_raw = (roe - g) / (ke - g)
        if np.isfinite(pb_raw) and pb_raw > 0:
            intrinsic_pb = float(np.clip(pb_raw, 0.5, 15.0))

    # ---- Intrinsic EV/EBITDA ----
    intrinsic_evebitda: Optional[float] = None
    ebitda_last = _resolve_ebitda(income, cash)
    if ebitda_last is not None and ebitda_last > 0 and wacc > g:
        ebit_last = _last_finite(_get(income, "ebit")) or 0.0
        da_last = ebitda_last - ebit_last
        capex_last = abs(_last_finite(_get(cash, "capex")) or 0.0)
        dwc_last = _delta_working_capital(balance)
        da_share = da_last / ebitda_last
        capex_share = capex_last / ebitda_last
        dwc_share = dwc_last / ebitda_last
        numerator = (1.0 - t) * (1.0 - da_share) - capex_share - dwc_share
        if numerator > 0:
            evebitda_raw = numerator * (1.0 + g) / (wacc - g)
            if np.isfinite(evebitda_raw) and evebitda_raw > 0:
                intrinsic_evebitda = float(np.clip(evebitda_raw, 3.0, 30.0))

    # ---- Implied per-share prices ----
    ni_last = _last_finite(_get(income, "net_income"))
    eq_last = _last_finite(_get(balance, "total_equity"))
    eps = (ni_last / shares_outstanding) if ni_last is not None else None
    bvps = (eq_last / shares_outstanding) if eq_last is not None else None

    implied_pe = (eps * intrinsic_pe) if (eps is not None and eps > 0 and intrinsic_pe) else None
    implied_pb = (bvps * intrinsic_pb) if (bvps is not None and bvps > 0 and intrinsic_pb) else None

    implied_evebitda_per_share: Optional[float] = None
    if intrinsic_evebitda is not None and ebitda_last is not None and ebitda_last > 0:
        implied_ev = ebitda_last * intrinsic_evebitda
        net_debt = _resolve_total_debt(balance) - (_last_finite(_get(balance, "cash_eq")) or 0.0)
        implied_equity = implied_ev - net_debt
        if implied_equity > 0:
            implied_evebitda_per_share = implied_equity / shares_outstanding

    # ---- Triangulated median ----
    prices = [p for p in (implied_pe, implied_pb, implied_evebitda_per_share)
              if p is not None and np.isfinite(p) and p > 0]
    if not prices:
        raise InsufficientDataError(
            "All intrinsic multiples non-applicable (no positive EPS/book/EBITDA "
            "or Ke/WACC <= g)."
        )
    implied_median = float(np.median(prices))

    # ---- Sector diagnostics (actual multiples) ----
    canonical = normalise_sector(sector) if sector else None
    sector_actual_pe = get_benchmark(canonical, "pe_ratio") if canonical else None
    sector_actual_pb = get_benchmark(canonical, "pb_ratio") if canonical else None
    sector_actual_evebitda = get_benchmark(canonical, "ev_to_ebitda") if canonical else None

    pe_gap: Optional[float] = None
    if sector_actual_pe is not None and intrinsic_pe is not None and intrinsic_pe > 0:
        pe_gap = (sector_actual_pe - intrinsic_pe) / intrinsic_pe

    # ---- Banks/insurance: prioritise P/B; skip EV/EBITDA ----
    # (The pipeline already routes banks/insurance to RI/DDM-heavy weights,
    # but if a caller forces this model we still produce a sane median.)
    if stage in ("bank", "insurance") and implied_pb is not None:
        # Bias the median toward P/B by replacing the EV/EBITDA leg.
        prices_bp = [p for p in (implied_pe, implied_pb) if p and p > 0]
        if prices_bp:
            implied_median = float(np.median(prices_bp + [implied_pb]))

    return MultiplesValuationResult(
        intrinsic_pe=intrinsic_pe,
        intrinsic_pb=intrinsic_pb,
        intrinsic_evebitda=intrinsic_evebitda,
        implied_per_share_pe=implied_pe,
        implied_per_share_pb=implied_pb,
        implied_per_share_evebitda=implied_evebitda_per_share,
        implied_per_share_median=implied_median,
        sector_actual_pe=(float(sector_actual_pe) if sector_actual_pe else None),
        sector_actual_pb=(float(sector_actual_pb) if sector_actual_pb else None),
        sector_actual_evebitda=(float(sector_actual_evebitda) if sector_actual_evebitda else None),
        pe_premium_or_discount=(float(pe_gap) if pe_gap is not None else None),
        notes=notes,
    )
