"""
Single orchestrator that turns ``Assumptions`` + the company's financials
into a complete ``ValuationResults`` bundle.

It runs the five valuation models (DCF, comparables, Monte Carlo, DDM,
Residual Income), aggregates them with sector weights, computes the
sub-score breakdown, and produces a final analyst rating.

Pipeline contract:
    inputs  → Assumptions, financials (income/balance/cash), peers,
              earnings-quality result, current price, sector
    output  → ValuationResults (a single dataclass with everything the
              UI needs to render)

Models that fail return None for that slot — the aggregator and the
scorer both tolerate sparse inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analysis.assumptions import Assumptions
from analysis.earnings_quality import EarningsQuality
from analysis.wacc import calculate_wacc, WACCResult
from core.exceptions import InsufficientDataError, ValuationError
from valuation.dcf_three_stage import run_dcf, DCFResult
from valuation.comparables import (
    PeerSnapshot, TargetFundamentals,
    value_by_comparables, ComparablesResult,
)
from valuation.monte_carlo import run_monte_carlo, MonteCarloResult
from valuation.ddm import (
    is_applicable as ddm_is_applicable,
    two_stage as ddm_two_stage,
    DDMResult,
)
from valuation.residual_income import run_residual_income, RIResult
from valuation.epv import run_epv, EPVResult
from valuation.multiples_valuation import (
    run_multiples_valuation, MultiplesResult,
)
from valuation.valuation_aggregator import aggregate, AggregatedValuation
from scoring.scorer import compute_score, ScoreBreakdown
from scoring.rating import rate, Rating
from analysis.industry_classifier import (
    classify_industry, classify_business_profile,
)
from analysis.koller_reorg import reorganize, ReorganizedFinancials
from analysis.lifecycle_classifier import classify_lifecycle
from valuation.fundamental_growth import (
    estimate_fundamental_growth, FundamentalGrowth,
)


def _should_skip_fcff_dcf(
    ticker: str, sector: Optional[str], industry: Optional[str] = None,
) -> tuple[bool, str]:
    """FCFF DCF doesn't apply cleanly to financials or REITs.

    Returns (skip, reason). Reason is a short human-readable string
    surfaced in the Valuation tab so the user knows why DCF wasn't run
    and what model fits instead.
    """
    cls = classify_industry(ticker, sector, industry)
    if cls.is_bank:
        return True, "Bank — use Residual Income or DDM, not FCFF DCF"
    if cls.is_insurance:
        return True, "Insurance — use Embedded Value or DDM"
    if cls.is_reit:
        return True, "REIT — use FFO / AFFO multiples or DDM"
    return False, ""


@dataclass
class ValuationResults:
    """One dataclass with every artefact the page renders."""
    ticker: str
    sector: Optional[str]
    current_price: Optional[float]

    wacc: WACCResult
    dcf: Optional[DCFResult] = None
    dcf_error: Optional[str] = None
    comparables: Optional[ComparablesResult] = None
    comparables_error: Optional[str] = None
    monte_carlo: Optional[MonteCarloResult] = None
    monte_carlo_error: Optional[str] = None
    ddm: Optional[DDMResult] = None
    ddm_error: Optional[str] = None
    residual_income: Optional[RIResult] = None
    ri_error: Optional[str] = None
    epv: Optional[EPVResult] = None
    epv_error: Optional[str] = None
    multiples: Optional[MultiplesResult] = None
    multiples_error: Optional[str] = None
    profile: str = "default"

    aggregator: AggregatedValuation = field(default=None)  # type: ignore[assignment]
    score: ScoreBreakdown = field(default=None)            # type: ignore[assignment]
    rating: Rating = field(default=None)                   # type: ignore[assignment]


# ============================================================
# Internals
# ============================================================
_BALANCE_SHARE_ALIASES = (
    "commonStockSharesOutstanding", "CommonStockSharesOutstanding",
    "commonStockSharesIssued",      "CommonStockSharesIssued",
    "sharesOutstanding",            "commonStock", "Share Issued",
)
_INCOME_SHARE_ALIASES = (
    "weightedAverageShsOutDil",     "weightedAverageShsOut",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "dilutedAverageShares",         "basicAverageShares",
    "Diluted Average Shares",
)
_INFO_SHARE_KEYS = (
    "sharesOutstanding", "shares_outstanding", "SharesOutstanding",
    "impliedSharesOutstanding",
)
_INFO_MCAP_KEYS = (
    "mktCap", "marketCap", "market_cap",
)


def _resolve_shares_outstanding(
    income: pd.DataFrame, balance: pd.DataFrame,
    *,
    info: Optional[dict] = None,
    quote: Optional[dict] = None,
) -> Optional[float]:
    """Resolve share count via a 4-tier cascade:

      1. Balance sheet — point-in-time count (preferred).
      2. Income statement — weighted-average (diluted first, then basic).
      3. ``info`` dict — provider-level shares outstanding.
      4. Computed market_cap / current_price (last resort).

    Returns None only when every tier fails. Each tier has multiple
    aliases because the raw XBRL concept names ("CommonStockSharesOutstanding")
    diverge from the camelCase FMP variants ("commonStockSharesOutstanding")
    and from the descriptive forms ("sharesOutstanding"). The cascade is
    necessary because some thin-XBRL companies (Visa, Mastercard, etc.)
    have no share count in their income/balance frames at all — only in
    the provider profile / quote dict.
    """
    def _try_df(df: Optional[pd.DataFrame], aliases) -> Optional[float]:
        if df is None or df.empty:
            return None
        for alias in aliases:
            if alias in df.columns:
                series = df[alias].dropna() if hasattr(df[alias], "dropna") else df[alias]
                if hasattr(series, "iloc") and len(series) > 0:
                    try:
                        v = float(series.iloc[-1])
                    except (TypeError, ValueError):
                        continue
                    if v > 0 and np.isfinite(v):
                        return v
        return None

    # Tier 1: balance sheet
    v = _try_df(balance, _BALANCE_SHARE_ALIASES)
    if v is not None:
        return v

    # Tier 2: income statement
    v = _try_df(income, _INCOME_SHARE_ALIASES)
    if v is not None:
        return v

    # Tier 3: info dict
    if info:
        for k in _INFO_SHARE_KEYS:
            raw = info.get(k)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > 0 and np.isfinite(val):
                return val

    # Tier 4: market_cap / price
    if info and quote:
        mc = None
        for k in _INFO_MCAP_KEYS:
            raw = info.get(k)
            if raw is None:
                continue
            try:
                mc = float(raw)
                if mc > 0 and np.isfinite(mc):
                    break
            except (TypeError, ValueError):
                mc = None
        price = quote.get("price") if hasattr(quote, "get") else None
        try:
            price_f = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_f = None
        if mc and price_f and price_f > 0:
            return mc / price_f

    return None


def _build_target_fundamentals(
    income: pd.DataFrame, balance: pd.DataFrame
) -> Optional[TargetFundamentals]:
    """Construct TargetFundamentals from the latest year of statements."""
    if income.empty or balance.empty:
        return None
    last_inc = income.iloc[-1]
    last_bal = balance.iloc[-1]

    def _pick(row: pd.Series, *keys: str) -> Optional[float]:
        for k in keys:
            if k in row and pd.notna(row[k]):
                return float(row[k])
        return None

    shares = _pick(last_inc, "weightedAverageShsOut", "weightedAverageShsOutDil")
    if not shares or shares <= 0:
        return None
    return TargetFundamentals(
        net_income=_pick(last_inc, "netIncome"),
        revenue=_pick(last_inc, "revenue"),
        ebitda=_pick(last_inc, "ebitda"),
        book_value=_pick(last_bal, "totalStockholdersEquity", "totalEquity"),
        shares_outstanding=shares,
        cash=_pick(last_bal, "cashAndCashEquivalents") or 0.0,
        debt=_pick(last_bal, "totalDebt") or 0.0,
    )


# ============================================================
# Public API
# ============================================================
def run_valuation(
    *,
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    assumptions: Assumptions,
    peers: Optional[list[PeerSnapshot]] = None,
    earnings_quality: Optional[EarningsQuality] = None,
    current_price: Optional[float] = None,
    sector: Optional[str] = None,
    info: Optional[dict] = None,
    quote: Optional[dict] = None,
    run_monte_carlo_now: bool = True,
    mc_seed: int = 42,
) -> ValuationResults:
    """
    Run every model the spec calls for and return a single bundle.

    ``run_monte_carlo_now=False`` lets the caller skip the slowest model
    when the user is mid-typing — the page can debounce and re-call with
    True once the inputs settle.
    """
    # ---- WACC (always — every other model needs it) ----
    wacc_res = calculate_wacc(
        risk_free=assumptions.risk_free,
        equity_risk_premium=assumptions.equity_risk_premium,
        beta=assumptions.beta,
        cost_of_debt_pretax=assumptions.cost_of_debt,
        tax_rate=assumptions.tax_rate,
        weight_equity=assumptions.weight_equity,
        weight_debt=assumptions.weight_debt,
    )
    out = ValuationResults(
        ticker=ticker, sector=sector, current_price=current_price, wacc=wacc_res,
    )

    # override_growth is Optional[float]: None ⇒ historical CAGR, 0.0 ⇒
    # explicit zero growth (e.g. user pinning a no-growth assumption).
    g_override: Optional[float] = assumptions.override_growth

    # ---- Resolve share count once (used by DCF / EPV / Multiples) ----
    # The 4-tier cascade (balance → income → info → mcap/price) needs
    # to run BEFORE DCF so thin-XBRL companies (V, MA, …) don't fail
    # on the DCF's own narrower resolver.
    shares = _resolve_shares_outstanding(income, balance, info=info, quote=quote)

    # ---- Reorganization + lifecycle + fundamental growth (shared by DCF/EPV/Multiples) ----
    reorg: Optional[ReorganizedFinancials] = None
    growth: Optional[FundamentalGrowth] = None
    lifecycle: Optional[dict] = None
    stage: str = "default"
    try:
        reorg = reorganize(income, balance, cash, wacc=wacc_res.wacc)
        lifecycle = classify_lifecycle(income, cash, ticker=ticker, sector=sector)
        stage = lifecycle["stage"]
        growth = estimate_fundamental_growth(
            reorg, income, balance,
            stage=stage,
            risk_free_rate=assumptions.risk_free,
            cash=cash,
        )
    except Exception as exc:
        # Reorg/lifecycle failures are non-fatal — downstream models will
        # use their own fallbacks. Surface via dcf_error only if DCF would
        # have been the primary affected model; EPV/multiples raise their
        # own errors.
        if reorg is None:
            out.dcf_error = f"Reorganization failed: {exc}"

    # ---- DCF ----
    skip_dcf, skip_reason = _should_skip_fcff_dcf(ticker, sector)
    if not skip_dcf:
        try:
            out.dcf = run_dcf(
                income=income, balance=balance, cash=cash,
                wacc=wacc_res.wacc,
                shares_outstanding=shares,
                ticker=ticker, sector=sector,
                risk_free_rate=assumptions.risk_free,
                # Reuse the pipeline-level reorg/lifecycle/growth (perf)
                reorg=reorg,
                lifecycle=lifecycle,
                growth=growth,
                stage1_growth=g_override,
                stage1_years=assumptions.stage1_years,
                stage2_years=assumptions.stage2_years,
                terminal_growth=assumptions.terminal_growth,
            )
            # DCF may return a "skipped" result (e.g. bank profile, negative NOPAT).
            if out.dcf is not None and out.dcf.skipped_reason:
                out.dcf_error = out.dcf.skipped_reason
                out.dcf = None
        except (ValuationError, InsufficientDataError) as exc:
            out.dcf_error = str(exc)
    else:
        out.dcf_error = skip_reason

    # ---- Comparables ----
    if peers:
        target = _build_target_fundamentals(income, balance)
        if target is None:
            out.comparables_error = "Could not build target fundamentals."
        else:
            try:
                out.comparables = value_by_comparables(
                    peers=peers, target=target,
                )
            except (ValuationError, InsufficientDataError) as exc:
                out.comparables_error = str(exc)
    else:
        out.comparables_error = "No peers configured."

    # ---- Monte Carlo (slowest — wraps DCF) ----
    if run_monte_carlo_now and out.dcf is not None:
        try:
            out.monte_carlo = run_monte_carlo(
                income=income, balance=balance, cash=cash,
                wacc=wacc_res.wacc,
                n_simulations=int(assumptions.mc_n_simulations),
                wacc_std=assumptions.mc_wacc_std,
                terminal_low=assumptions.mc_terminal_low,
                terminal_high=assumptions.mc_terminal_high,
                stage1_years=assumptions.stage1_years,
                stage2_years=assumptions.stage2_years,
                growth_std=assumptions.mc_rev_growth_std,
                current_price=current_price,
                seed=mc_seed,
                # Enable MC's invariants hoist (and the DCF skip-gate
                # for banks/insurance/reits) by passing the same context
                # the main DCF run got.
                ticker=ticker, sector=sector,
                risk_free_rate=assumptions.risk_free,
            )
        except (ValuationError, InsufficientDataError) as exc:
            out.monte_carlo_error = str(exc)
    elif out.dcf is None:
        out.monte_carlo_error = "Requires a successful DCF run."
    else:
        out.monte_carlo_error = "Monte Carlo skipped (toggle to enable)."

    # ---- DDM ----
    # Pass the pipeline-resolved share count so the per-share dividend
    # floor in is_applicable AND the DPS calc in two_stage both work
    # for thin-XBRL filers (V, MA, …) whose income/balance frames
    # don't expose a share series.
    if ddm_is_applicable(cash, income, shares_outstanding=shares):
        try:
            out.ddm = ddm_two_stage(
                income=income, balance=balance, cash=cash,
                cost_of_equity=wacc_res.cost_of_equity,
                stage1_years=assumptions.stage1_years,
                terminal_growth=assumptions.terminal_growth,
                shares_outstanding=shares,
            )
        except (ValuationError, InsufficientDataError) as exc:
            out.ddm_error = str(exc)
    else:
        out.ddm_error = "Company does not pay material dividends."

    # ---- Residual Income ----
    try:
        out.residual_income = run_residual_income(
            income=income, balance=balance,
            cost_of_equity=wacc_res.cost_of_equity,
            stage1_years=assumptions.stage1_years,
            stage1_growth=g_override,
            terminal_growth=assumptions.terminal_growth,
        )
    except (ValuationError, InsufficientDataError) as exc:
        out.ri_error = str(exc)

    # ---- Business profile + shares (used by EPV / Multiples) ----
    business_profile = classify_business_profile(ticker, sector)
    # Aggregator profile: financial-sector overrides take precedence over
    # lifecycle stage (banks/insurance/reits route to RI/DDM-heavy weights).
    if business_profile in ("bank", "insurance", "reit"):
        aggregator_profile = business_profile
    elif stage and stage != "default":
        aggregator_profile = stage
    else:
        aggregator_profile = business_profile
    out.profile = aggregator_profile

    # ---- EPV ----
    if shares and shares > 0 and reorg is not None:
        try:
            out.epv = run_epv(
                reorg=reorg, income=income, balance=balance,
                wacc=wacc_res.wacc,
                shares_outstanding=shares,
                stage=stage,
            )
        except (ValuationError, InsufficientDataError) as exc:
            out.epv_error = str(exc)
    elif reorg is None:
        out.epv_error = "Reorganization unavailable."
    else:
        out.epv_error = "Share count unavailable."

    # ---- Multiples (Damodaran intrinsic) ----
    if shares and shares > 0 and reorg is not None and growth is not None:
        try:
            out.multiples = run_multiples_valuation(
                income=income, balance=balance, cash=cash,
                reorg=reorg, growth=growth, stage=stage,
                wacc=wacc_res.wacc,
                cost_of_equity=wacc_res.cost_of_equity,
                sector=sector,
                shares_outstanding=shares,
            )
        except (ValuationError, InsufficientDataError) as exc:
            out.multiples_error = str(exc)
    elif reorg is None or growth is None:
        out.multiples_error = "Reorganization / fundamental growth unavailable."
    else:
        out.multiples_error = "Share count unavailable."

    # ---- Aggregator ----
    out.aggregator = aggregate(
        dcf=(out.dcf.intrinsic_value_per_share if out.dcf else None),
        comparables=(
            out.comparables.implied_per_share_median
            if out.comparables and out.comparables.implied_per_share_median
            else None
        ),
        monte_carlo=(out.monte_carlo.median if out.monte_carlo else None),
        ddm=(out.ddm.intrinsic_value_per_share if out.ddm else None),
        residual_income=(
            out.residual_income.intrinsic_value_per_share
            if out.residual_income else None
        ),
        epv=(out.epv.intrinsic_value_per_share if out.epv else None),
        multiples=(out.multiples.implied_per_share_median
                   if out.multiples else None),
        profile=aggregator_profile,
        current_price=current_price,
        sector=sector,
        epv_normalization_factor=(out.epv.normalization_factor
                                  if out.epv is not None else None),
    )

    # ---- Scoring + Rating ----
    upside: Optional[float] = None
    if (np.isfinite(out.aggregator.intrinsic_per_share)
            and current_price and current_price > 0):
        upside = (out.aggregator.intrinsic_per_share - current_price) / current_price

    out.score = compute_score(
        income=income, balance=balance, cash=cash,
        earnings_quality=earnings_quality,
        intrinsic=(out.aggregator.intrinsic_per_share
                   if np.isfinite(out.aggregator.intrinsic_per_share) else None),
        current_price=current_price,
    )
    out.rating = rate(
        composite=out.score.composite,
        upside=upside,
        confidence=out.aggregator.confidence,
    )
    return out
