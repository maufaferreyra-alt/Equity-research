"""
Three-stage DCF on free cash flow to the firm (FCFF) — Damodaran-Koller
edition.

This module replaced the historical-FCF extrapolation DCF with one
anchored in business fundamentals (Koller value-driver formula +
Damodaran lifecycle-aware growth):

    NOPAT_0           starting point reorganized via analysis/koller_reorg
    g_explicit / g_s  fundamental growth via valuation/fundamental_growth
    RR_t = g_t / ROIC reinvestment tied to value-creation economics
    FCFF_t            = NOPAT_t * (1 - RR_t)
    CV                = NOPAT_{T+1} * (1 - g_s / RONIC) / (WACC - g_s)

The model also decomposes intrinsic value into:
    - asset-in-place value (NOPAT_0 / WACC, zero-growth steady state)
    - value of growth     (the residual)
which is Koller's central diagnostic — most "expensive" stocks owe their
price to the growth component, not the in-place earnings.

The skip gate (banks / insurance / REITs) short-circuits with a
``skipped_reason`` so the aggregator can pick equity-side models (RI,
DDM, FFO multiples) instead.

Back-compat: the public ``run_dcf`` accepts the same keyword args the
old call sites use (``stage1_growth``, ``stage1_years``,
``stage2_years``, ``terminal_growth``, ``fade_curve``). When provided
they override the corresponding Damodaran-derived values; when omitted
the new fundamentals-driven path runs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

from analysis.ratios import _get
from core.constants import DCF_DEFAULTS
from core.exceptions import InsufficientDataError, ValuationError

if TYPE_CHECKING:
    # Late-bind to avoid circular imports — these modules import nothing
    # from us, but listing them at top-level would couple module loading.
    from analysis.koller_reorg import ReorganizedFinancials
    from valuation.fundamental_growth import FundamentalGrowth


FadeCurve = Literal["linear", "logistic"]


# ============================================================
# Result dataclass — existing fields preserved, new ones appended
# ============================================================
@dataclass
class DCFResult:
    """Output of run_dcf — every component a downstream UI may need."""
    # ---- Legacy fields (callers / UI rely on these names) ----
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    pv_explicit: float
    pv_terminal: float
    terminal_value: float
    base_fcf: float
    wacc: float
    terminal_growth: float
    stage1_growth: float
    stage1_years: int
    stage2_years: int
    growth_path: list[float] = field(default_factory=list)
    projected_fcf: list[float] = field(default_factory=list)
    discount_factors: list[float] = field(default_factory=list)
    pv_per_year: list[float] = field(default_factory=list)

    # ---- Damodaran-Koller decomposition (new) ----
    asset_in_place_value: float = 0.0            # per-share, zero-growth steady state
    value_of_growth: float = 0.0                 # per-share, intrinsic - in-place
    pv_explicit_period: float = 0.0              # PV of explicit forecast years
    pv_transition_period: float = 0.0            # PV of transition years
    pv_continuing_value: float = 0.0             # PV of Koller CV at end
    g_explicit: float = 0.0                      # growth rate explicit phase
    g_stable: float = 0.0                        # growth rate perpetuity
    ronic_terminal: float = 0.0                  # RONIC used in CV
    forecast_horizon_explicit: int = 0           # years
    forecast_horizon_transition: int = 0         # years
    lifecycle_stage: str = ""                    # for UI context
    nopat_projections: list[float] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def _skipped(reason: str, *, wacc: float, lifecycle_stage: str = "skipped") -> DCFResult:
    """Build an all-NaN result with a populated ``skipped_reason``. The
    aggregator and UI already tolerate NaN intrinsic values; this keeps
    the skip path uniform. ``lifecycle_stage`` is also populated (with
    the business profile when we skip pre-classification, or the
    classified stage when we skip post-classification) so the UI never
    has to render an empty string."""
    nan = float("nan")
    return DCFResult(
        intrinsic_value_per_share=nan,
        enterprise_value=nan,
        equity_value=nan,
        pv_explicit=nan,
        pv_terminal=nan,
        terminal_value=nan,
        base_fcf=nan,
        wacc=float(wacc),
        terminal_growth=nan,
        stage1_growth=nan,
        stage1_years=0,
        stage2_years=0,
        lifecycle_stage=lifecycle_stage,
        skipped_reason=reason,
    )


# ============================================================
# Internals
# ============================================================
def _net_cash(balance: pd.DataFrame) -> tuple[float, float]:
    """Returns (cash, total_debt) from the most recent balance sheet row."""
    cash_s = _get(balance, "cash_eq")
    debt_s = _get(balance, "total_debt")
    if debt_s is None:
        ltd = _get(balance, "long_term_debt")
        std = _get(balance, "short_term_debt")
        if ltd is not None and std is not None:
            debt_s = ltd.add(std, fill_value=0.0)
        elif ltd is not None:
            debt_s = ltd
        elif std is not None:
            debt_s = std
    cash = (float(cash_s.dropna().iloc[-1])
            if cash_s is not None and not cash_s.dropna().empty else 0.0)
    debt = (float(debt_s.dropna().iloc[-1])
            if debt_s is not None and not debt_s.dropna().empty else 0.0)
    return cash, debt


def _resolve_shares(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    override: Optional[float],
) -> float:
    if override is not None and np.isfinite(override) and override > 0:
        return float(override)
    sh = _get(income, "weighted_avg_shares")
    if sh is not None and not sh.dropna().empty:
        return float(sh.dropna().iloc[-1])
    sh = _get(balance, "common_shares_outstanding")
    if sh is not None and not sh.dropna().empty:
        return float(sh.dropna().iloc[-1])
    raise InsufficientDataError("Cannot find share count in income or balance")


def _book_equity(balance: pd.DataFrame) -> float:
    eq = _get(balance, "total_equity")
    if eq is None or eq.dropna().empty:
        return 0.0
    return float(eq.dropna().iloc[-1])


def _through_cycle_nopat(
    income: pd.DataFrame,
    reorg,                                       # ReorganizedFinancials
) -> float:
    """Damodaran's cyclical normalization (Investment Valuation, Ch 22).

    A simple mean over a full cycle dilutes — trough and peak years
    cancel, hiding both the structural margin and the volatility. The
    correct estimator is a *winsorized* mean: drop the single best and
    worst observation over a 10y window. Falls back to median (3-4y),
    then to latest margin, then to latest NOPAT — in that order — as
    the window gets thinner.

    The historical EBIT is back-derived using the full-history mean
    effective rate; the forward tax assumption uses the 3y-smoothed
    rate so we capture the current tax regime (post-TCJA, etc) rather
    than blending in pre-reform years.
    """
    revenue = _get(income, "revenue")
    if revenue is None or revenue.dropna().empty:
        return float(reorg.latest_nopat)
    avg_tax_full = float(reorg.effective_tax_rate.mean())
    if not np.isfinite(avg_tax_full) or avg_tax_full <= 0:
        avg_tax_full = 0.21
    ebit_series = reorg.nopat / (1.0 - avg_tax_full)

    valid_idx = revenue.index.intersection(ebit_series.index)
    if len(valid_idx) < 2:
        return float(reorg.latest_nopat)
    n_years = min(10, len(valid_idx))
    rev_window = revenue.loc[valid_idx].tail(n_years)
    ebit_window = ebit_series.loc[valid_idx].tail(n_years)
    margins = (ebit_window / rev_window.where(rev_window > 0)).dropna()

    if len(margins) >= 5:
        # Winsorize: drop top 1 and bottom 1.
        sorted_m = margins.sort_values()
        through_cycle_margin = float(sorted_m.iloc[1:-1].mean())
    elif len(margins) >= 3:
        # Too few obs to trim — median is more robust than mean.
        through_cycle_margin = float(margins.median())
    elif not margins.empty:
        through_cycle_margin = float(margins.iloc[-1])
    else:
        return float(reorg.latest_nopat)

    latest_revenue = float(revenue.dropna().iloc[-1])
    fwd_tax = (reorg.avg_tax_rate_3y
               if np.isfinite(reorg.avg_tax_rate_3y) and reorg.avg_tax_rate_3y > 0
               else avg_tax_full)
    return float(latest_revenue * through_cycle_margin * (1.0 - fwd_tax))


# ============================================================
# Public API
# ============================================================
def run_dcf(
    income: Optional[pd.DataFrame] = None,
    balance: Optional[pd.DataFrame] = None,
    cash: Optional[pd.DataFrame] = None,
    *,
    wacc: float,
    shares_outstanding: Optional[float] = None,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    risk_free_rate: float = 0.04,
    # ---- Precomputed invariants (perf hoist for Monte Carlo / pipeline) ----
    # These three quantities depend only on the financial statements, not on
    # WACC/growth perturbations. Callers running many DCFs over the same
    # statements (Monte Carlo, sensitivity) should compute them ONCE and
    # pass them in — avoids ~43ms/call of redundant pandas work.
    reorg: Optional[ReorganizedFinancials] = None,
    lifecycle: Optional[dict] = None,
    growth: Optional[FundamentalGrowth] = None,
    # ---- Legacy kwargs (override the Damodaran defaults when provided) ----
    stage1_growth: Optional[float] = None,
    stage1_years: Optional[int] = None,
    stage2_years: Optional[int] = None,
    terminal_growth: Optional[float] = None,
    fade_curve: Optional[FadeCurve] = None,
) -> DCFResult:
    """Damodaran-Koller three-stage DCF.

    The shape is: explicit period (high growth) -> linear transition ->
    Koller value-driver continuing value. Growth, reinvestment rate and
    terminal RONIC are all tied to fundamentals (Koller's value drivers)
    rather than extrapolated from historical FCF.

    Legacy callers that still pass ``stage1_growth`` / ``stage1_years``
    / ``stage2_years`` / ``terminal_growth`` keep working: those args
    override the corresponding Damodaran-derived values.
    """
    # ---- Late-bound imports break a circular dependency (these modules
    # import valuation/__init__ which historically re-exported run_dcf).
    from analysis.koller_reorg import reorganize
    from analysis.lifecycle_classifier import classify_lifecycle
    from valuation.fundamental_growth import estimate_fundamental_growth
    from analysis.industry_classifier import classify_business_profile

    if income is None or balance is None or cash is None:
        raise InsufficientDataError(
            "run_dcf requires income / balance / cash dataframes."
        )

    # ---- A) Skip gate ----
    if ticker:
        profile = classify_business_profile(ticker, sector)
        if profile in ("bank", "insurance", "reit"):
            return _skipped(
                f"DCF not applicable for {profile} - use RI / DDM / FFO multiples instead",
                wacc=wacc,
                lifecycle_stage=profile,
            )

    # ---- B) Reorganize + classify + growth (use caller-provided when present) ----
    # Monte Carlo / sensitivity callers pass these in to avoid recomputing
    # them ~1000 times — they depend only on the statements, not WACC/g.
    if reorg is None:
        try:
            reorg = reorganize(income, balance, cash, wacc=wacc)
        except ValueError as exc:
            return _skipped(f"Cannot reorganize statements: {exc}", wacc=wacc)

    if lifecycle is None:
        lifecycle = classify_lifecycle(income, cash, ticker=ticker or "", sector=sector)
    stage = lifecycle["stage"]

    if growth is None:
        growth = estimate_fundamental_growth(
            reorg, income, balance,
            stage=stage, risk_free_rate=risk_free_rate, cash=cash,
        )

    # ---- C) Forecast horizon per stage (overridable) ----
    default_horizons = {
        "young_growth":   (10, 5),
        "high_growth":    (10, 5),
        "mature_growth":  (5, 5),
        "mature_stable":  (5, 3),
        "cyclical":       (5, 5),
        "declining":      (3, 0),
    }
    T_exp, T_trans = default_horizons.get(stage, (5, 5))
    if stage1_years is not None:
        T_exp = int(stage1_years)
    if stage2_years is not None:
        T_trans = int(stage2_years)

    # ---- Growth knobs (overridable) ----
    g_e = (float(stage1_growth) if stage1_growth is not None
           else float(growth.recommended_g_explicit))
    g_s = (float(terminal_growth) if terminal_growth is not None
           else float(growth.recommended_g_stable))

    if not np.isfinite(g_e):
        g_e = g_s if np.isfinite(g_s) else float(DCF_DEFAULTS["terminal_growth"])
    if not np.isfinite(g_s):
        g_s = float(DCF_DEFAULTS["terminal_growth"])

    spread = float(DCF_DEFAULTS.get("min_wacc_terminal_spread", 0.005))
    if wacc - g_s < spread:
        raise ValuationError(
            f"WACC ({wacc:.2%}) must exceed stable growth ({g_s:.2%}) "
            f"by at least {spread:.0%}"
        )

    # ---- D) Cyclical normalization / NOPAT_0 ----
    if stage == "cyclical":
        nopat_0 = _through_cycle_nopat(income, reorg)
    else:
        nopat_0 = float(reorg.latest_nopat)

    if not np.isfinite(nopat_0) or nopat_0 <= 0:
        return _skipped(
            "Negative or zero NOPAT - DCF not applicable; try multiples valuation.",
            wacc=wacc,
            lifecycle_stage=stage,
        )

    # ---- E) Explicit-period NOPAT projection ----
    nopat_proj: list[float] = []
    nopat_t = nopat_0
    for _ in range(T_exp):
        nopat_t = nopat_t * (1.0 + g_e)
        nopat_proj.append(nopat_t)

    # ---- F) Transition NOPAT projection (linear decline) ----
    if T_trans > 0:
        for t in range(1, T_trans + 1):
            g_interp = g_e - (g_e - g_s) * (t / T_trans)
            nopat_t = nopat_t * (1.0 + g_interp)
            nopat_proj.append(nopat_t)

    # ---- G) Reinvestment / FCFF ----
    # Damodaran: g = RR x ROIC -> RR = g / ROIC.
    roic_explicit = float(reorg.latest_roic) if np.isfinite(reorg.latest_roic) else wacc
    # Terminal RONIC: Koller convention. Mature companies converge to a
    # small (0-200bp) spread over WACC, not the spot ROIC. The clamp
    # both floors (WACC) and caps (current ROIC) the assumption.
    ronic_terminal = float(max(wacc, min(roic_explicit, wacc + 0.02)))

    rr_explicit = g_e / max(roic_explicit, wacc)
    rr_explicit = float(max(0.0, min(rr_explicit, 0.80)))
    rr_stable = g_s / max(ronic_terminal, wacc)
    rr_stable = float(max(0.0, min(rr_stable, 0.80)))

    fcff_proj: list[float] = []
    for idx, npt in enumerate(nopat_proj, start=1):
        if idx <= T_exp:
            rr_t = rr_explicit
        else:
            progress = (idx - T_exp) / T_trans if T_trans > 0 else 1.0
            rr_t = rr_explicit - (rr_explicit - rr_stable) * progress
        fcff_proj.append(npt * (1.0 - rr_t))

    # ---- H) Continuing value (Koller value-driver) ----
    if T_trans + T_exp > 0:
        nopat_T_plus_1 = nopat_proj[-1] * (1.0 + g_s)
        cv = (nopat_T_plus_1 * (1.0 - g_s / max(ronic_terminal, 1e-6))) / (wacc - g_s)
    else:
        cv = 0.0

    # Declining stage: no perpetuity, use liquidation-style floor.
    if stage == "declining":
        book_eq = _book_equity(balance)
        shares_for_floor = _resolve_shares(income, balance, shares_outstanding)
        floor_equity_per_share = book_eq / shares_for_floor if shares_for_floor > 0 else 0.0
        in_place_per_share_provisional = (nopat_0 / wacc) / max(shares_for_floor, 1.0)
        # Replace CV with a floor (book value or 70% of in-place equity)
        cv = 0.0
        # Final intrinsic will be max(book, 0.7 * in-place) per share — applied below.
        _decline_floor = max(floor_equity_per_share, 0.7 * in_place_per_share_provisional)
    else:
        _decline_floor = None

    # ---- I) Discount everything to PV ----
    T_total = T_exp + T_trans
    pv_fcffs = [
        fcff / (1.0 + wacc) ** t for t, fcff in enumerate(fcff_proj, start=1)
    ]
    pv_cv = cv / (1.0 + wacc) ** T_total if T_total > 0 else 0.0
    pv_explicit_period = float(sum(pv_fcffs[:T_exp]))
    pv_transition_period = float(sum(pv_fcffs[T_exp:]))

    enterprise_value = float(sum(pv_fcffs) + pv_cv)

    # ---- J) Equity value ----
    cash_bs, debt_bs = _net_cash(balance)
    net_debt = debt_bs - cash_bs
    equity_value = enterprise_value - net_debt
    shares = _resolve_shares(income, balance, shares_outstanding)
    if shares <= 0:
        raise InsufficientDataError("Share count must be positive for per-share intrinsic.")

    intrinsic_per_share = equity_value / shares

    # Declining floor override: liquidation/book floor caps how low the
    # per-share intrinsic can go.
    if _decline_floor is not None:
        intrinsic_per_share = max(intrinsic_per_share, _decline_floor)
        equity_value = intrinsic_per_share * shares
        enterprise_value = equity_value + net_debt

    # ---- K) Decomposition: asset-in-place vs value of growth ----
    asset_in_place_ev = nopat_0 / wacc if wacc > 0 else 0.0
    asset_in_place_equity = asset_in_place_ev - net_debt
    asset_in_place_per_share = asset_in_place_equity / shares
    value_of_growth_per_share = intrinsic_per_share - asset_in_place_per_share

    # ---- L) Build legacy-compatible fields too ----
    # Build a "growth_path" series that mirrors the projection (year-by-year g_t)
    growth_path: list[float] = []
    for idx in range(1, T_total + 1):
        if idx <= T_exp:
            growth_path.append(g_e)
        elif T_trans > 0:
            growth_path.append(g_e - (g_e - g_s) * ((idx - T_exp) / T_trans))
    discount_factors = [1.0 / (1.0 + wacc) ** (t + 1) for t in range(len(fcff_proj))]

    return DCFResult(
        intrinsic_value_per_share=float(intrinsic_per_share),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        pv_explicit=pv_explicit_period + pv_transition_period,
        pv_terminal=float(pv_cv),
        terminal_value=float(cv),
        base_fcf=float(nopat_0),
        wacc=float(wacc),
        terminal_growth=float(g_s),
        stage1_growth=float(g_e),
        stage1_years=int(T_exp),
        stage2_years=int(T_trans),
        growth_path=[float(x) for x in growth_path],
        projected_fcf=[float(x) for x in fcff_proj],
        discount_factors=[float(x) for x in discount_factors],
        pv_per_year=[float(x) for x in pv_fcffs],
        # ---- New Damodaran-Koller fields ----
        asset_in_place_value=float(asset_in_place_per_share),
        value_of_growth=float(value_of_growth_per_share),
        pv_explicit_period=float(pv_explicit_period),
        pv_transition_period=float(pv_transition_period),
        pv_continuing_value=float(pv_cv),
        g_explicit=float(g_e),
        g_stable=float(g_s),
        ronic_terminal=float(ronic_terminal),
        forecast_horizon_explicit=int(T_exp),
        forecast_horizon_transition=int(T_trans),
        lifecycle_stage=str(stage),
        nopat_projections=[float(x) for x in nopat_proj],
    )


def sensitivity_table(
    *,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    wacc_grid: list[float],
    g_grid: list[float],
    stage1_growth: Optional[float] = None,
) -> pd.DataFrame:
    """Two-way sensitivity of intrinsic per-share value over WACC x terminal-g.

    Failed cells (WACC <= g, etc.) are returned as NaN.
    Index = WACC, columns = terminal growth.
    """
    out = pd.DataFrame(index=wacc_grid, columns=g_grid, dtype=float)
    for w in wacc_grid:
        for g in g_grid:
            try:
                r = run_dcf(
                    income=income, balance=balance, cash=cash,
                    wacc=w, terminal_growth=g, stage1_growth=stage1_growth,
                )
                out.loc[w, g] = r.intrinsic_value_per_share
            except (ValuationError, InsufficientDataError):
                out.loc[w, g] = float("nan")
    return out
