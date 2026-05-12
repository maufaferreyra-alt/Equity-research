"""
Multi-source news aggregator with relevance scoring and deduplication.

Sources (all already integrated in the codebase):
- ``analysis.news_sentiment.fetch_news``           — yfinance Ticker.news
- ``data.finnhub_provider.fetch_company_news``      — Finnhub /company-news
- ``data.marketaux_provider.fetch_news``            — Marketaux /news/all

Each source returns a pandas DataFrame; we adapt them to a common
``NewsItem`` dataclass, dedupe across sources (URL exact + title
Jaccard), enrich with VADER sentiment if the item didn't already
carry one (Marketaux does), then rank by source authority + recency
+ ticker prominence.

Standalone-usable: the public API (``fetch_news_for_ticker`` /
``fetch_market_news``) returns plain Python objects, no Streamlit
context required at the caller. The ``@st.cache_data`` decorator
falls back to an in-memory cache when no runtime is attached.
"""
from __future__ import annotations
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

# Source authority weights for relevance scoring.
# Higher = more authoritative.
SOURCE_WEIGHTS: dict[str, float] = {
    "bloomberg": 1.0, "reuters": 1.0, "wsj": 1.0,
    "wall street journal": 1.0, "financial times": 0.95, "ft": 0.95,
    "barron's": 0.9, "barrons": 0.9, "barrons.com": 0.9,
    "cnbc": 0.85, "associated press": 0.85, "ap": 0.85,
    "marketwatch": 0.75, "yahoo finance": 0.7, "yahoo": 0.7,
    "investor's business daily": 0.7, "investors business daily": 0.7,
    "kiplinger": 0.6, "seeking alpha": 0.65, "forbes": 0.65,
    "benzinga": 0.55, "zacks": 0.55, "thestreet": 0.55,
    "the motley fool": 0.5, "motley fool": 0.5, "fool.com": 0.5,
    "investing.com": 0.5,
}
DEFAULT_SOURCE_WEIGHT = 0.5


@dataclass
class NewsItem:
    title: str
    url: str
    source: str                                  # display name
    source_normalized: str                       # lowercased for matching
    published_at: datetime                       # UTC
    snippet: Optional[str] = None
    tickers: list[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None      # [-1, +1]
    sentiment_label: Optional[str] = None        # positive/neutral/negative
    relevance_score: float = 0.0
    image_url: Optional[str] = None
    provider: str = "—"                          # yfinance / finnhub / marketaux


def _normalize_source(s: Optional[str]) -> str:
    if not s:
        return ""
    norm = s.lower().strip()
    for suf in (" - bloomberg", " - reuters", " - cnbc",
                " | bloomberg", " | reuters"):
        norm = norm.replace(suf, "")
    return norm.strip()


def _to_utc(ts) -> Optional[datetime]:
    """Coerce a timestamp (datetime, pandas Timestamp, int, str) to UTC."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return None
        return ts.to_pydatetime().astimezone(timezone.utc)
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


# ============================================================
# Source adapters — each defensive: returns [] on ANY failure
# ============================================================
def _fetch_yfinance_news(
    ticker: str, lookback_days: int = 7, limit: int = 30,
) -> list[NewsItem]:
    """Wraps ``analysis.news_sentiment.fetch_news`` which already handles
    yfinance's two schema variants defensively."""
    try:
        from analysis.news_sentiment import fetch_news
        df = fetch_news(ticker, limit=limit)
    except Exception as e:
        log.warning("yfinance news fetch failed for %s: %s", ticker, e)
        return []
    if df is None or df.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[NewsItem] = []
    for _, row in df.iterrows():
        try:
            title = str(row.get("title") or "").strip()
            url = str(row.get("link") or "").strip()
            if not title or not url:
                continue
            published = _to_utc(row.get("published"))
            if published is None or published < cutoff:
                continue
            src = str(row.get("publisher") or "Yahoo Finance").strip() or "Yahoo Finance"
            items.append(NewsItem(
                title=title,
                url=url,
                source=src,
                source_normalized=_normalize_source(src),
                published_at=published,
                tickers=[ticker.upper()],
                provider="yfinance",
            ))
        except Exception:
            continue
    return items


def _fetch_finnhub_news(ticker: str, lookback_days: int = 7) -> list[NewsItem]:
    try:
        from data.finnhub_provider import fetch_company_news
        df = fetch_company_news(ticker, days_back=lookback_days)
    except Exception as e:
        log.warning("Finnhub news fetch failed for %s: %s", ticker, e)
        return []
    if df is None or df.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[NewsItem] = []
    for _, row in df.iterrows():
        try:
            title = str(row.get("headline") or "").strip()
            url = str(row.get("url") or "").strip()
            if not title or not url:
                continue
            published = _to_utc(row.get("published"))
            if published is None or published < cutoff:
                continue
            src = str(row.get("source") or "Finnhub").strip() or "Finnhub"
            snippet = row.get("summary")
            image = row.get("image")
            items.append(NewsItem(
                title=title,
                url=url,
                source=src,
                source_normalized=_normalize_source(src),
                published_at=published,
                snippet=str(snippet).strip() if snippet else None,
                tickers=[ticker.upper()],
                image_url=str(image).strip() if image else None,
                provider="finnhub",
            ))
        except Exception:
            continue
    return items


def _fetch_marketaux_news(
    ticker: str, lookback_days: int = 7, limit: int = 20,
) -> list[NewsItem]:
    try:
        from data.marketaux_provider import fetch_news as ma_fetch
        df = ma_fetch(ticker, limit=limit)
    except Exception as e:
        log.warning("Marketaux fetch failed for %s: %s", ticker, e)
        return []
    if df is None or df.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: list[NewsItem] = []
    for _, row in df.iterrows():
        try:
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            if not title or not url:
                continue
            published = _to_utc(row.get("published"))
            if published is None or published < cutoff:
                continue
            src = str(row.get("source") or "Marketaux").strip() or "Marketaux"

            # Marketaux already supplies per-ticker sentiment in the row.
            sscore = row.get("sentiment_score")
            try:
                sscore_val = (float(sscore)
                              if sscore is not None and not pd.isna(sscore)
                              else None)
            except (TypeError, ValueError):
                sscore_val = None
            slabel = row.get("sentiment_label")

            items.append(NewsItem(
                title=title,
                url=url,
                source=src,
                source_normalized=_normalize_source(src),
                published_at=published,
                snippet=(str(row.get("description")).strip()
                         if row.get("description") else None),
                tickers=[ticker.upper()],
                sentiment_score=sscore_val,
                sentiment_label=(str(slabel) if slabel else None),
                provider="marketaux",
            ))
        except Exception:
            continue
    return items


# ============================================================
# Dedupe + scoring
# ============================================================
def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    """Dedupe by URL exact + title fuzzy (Jaccard > 0.75)."""
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    out: list[NewsItem] = []
    for it in items:
        if it.url and it.url in seen_urls:
            continue
        title_key = re.sub(r"[^a-z0-9 ]", "", it.title.lower())[:80]
        if any(_jaccard(title_key, prev) > 0.75 for prev in seen_titles):
            continue
        seen_urls.add(it.url)
        seen_titles.append(title_key)
        out.append(it)
    return out


def _score_relevance(
    item: NewsItem,
    *,
    query_ticker: Optional[str] = None,
    now: Optional[datetime] = None,
) -> float:
    """Source authority (50%) + recency decay (35%) + sentiment
    strength (15%) + ticker prominence bonuses."""
    now = now or datetime.now(timezone.utc)
    src_w = SOURCE_WEIGHTS.get(item.source_normalized, DEFAULT_SOURCE_WEIGHT)
    hours_old = max((now - item.published_at).total_seconds() / 3600, 0.0)
    recency = math.exp(-hours_old / 24)            # half-life ≈ 24h
    sent_strength = (abs(item.sentiment_score)
                     if item.sentiment_score is not None else 0.3)
    score = src_w * 0.5 + recency * 0.35 + sent_strength * 0.15
    if query_ticker:
        qt = query_ticker.upper()
        if qt in item.title.upper():
            score += 0.25
        if item.tickers and item.tickers[0] == qt:
            score += 0.10
    return min(score, 2.0)


def _enrich_sentiment(items: list[NewsItem]) -> list[NewsItem]:
    """Score sentiment via VADER for items that lack one. Silently no-ops
    if the sentiment module isn't available or returns an unexpected
    shape — sentiment is a nice-to-have, not a hard requirement."""
    needs = [(i, it) for i, it in enumerate(items)
             if it.sentiment_score is None]
    if not needs:
        return items
    try:
        from analysis.news_sentiment import score_sentiment
        texts = [(it.title or "") + ". " + (it.snippet or "")
                 for _, it in needs]
        scored = score_sentiment(texts, engine="vader")
        if scored is None or scored.empty or "value" not in scored.columns:
            return items
        for (_, it), (_, row) in zip(needs, scored.iterrows()):
            val = row.get("value")
            try:
                v = float(val) if val is not None else None
            except (TypeError, ValueError):
                v = None
            it.sentiment_score = v
            label = row.get("label")
            it.sentiment_label = str(label) if label else None
    except Exception as e:
        log.debug("sentiment enrichment skipped: %s", e)
    return items


# ============================================================
# Public API
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_for_ticker(
    ticker: str,
    *,
    lookback_days: int = 7,
    max_items: int = 50,
) -> list[NewsItem]:
    """Aggregated, deduped, scored news for one ticker. All 3 sources
    fetched in parallel; any source failure is silent (returns [])."""
    if not ticker:
        return []
    fetchers = [
        (_fetch_yfinance_news,  (ticker, lookback_days, 30)),
        (_fetch_finnhub_news,   (ticker, lookback_days)),
        (_fetch_marketaux_news, (ticker, lookback_days, 20)),
    ]
    all_items: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="news") as pool:
        futures = [pool.submit(fn, *args) for fn, args in fetchers]
        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result() or [])
            except Exception as e:
                log.warning("news fetcher failed: %s", e)

    all_items = _enrich_sentiment(all_items)
    all_items = _dedupe(all_items)
    for it in all_items:
        it.relevance_score = _score_relevance(it, query_ticker=ticker)
    all_items.sort(key=lambda x: x.relevance_score, reverse=True)
    return all_items[:max_items]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_news(
    *,
    seed_tickers: Optional[tuple[str, ...]] = None,
    lookback_days: int = 2,
    max_items: int = 60,
) -> list[NewsItem]:
    """Market-wide feed: yfinance news across major tickers, deduped
    and ranked. Only yfinance is hit (the per-ticker Finnhub / Marketaux
    endpoints would burn paid quota fast across the seed list).

    Seed defaults to ``data.popular_lists.SP500_LEADERS``."""
    if seed_tickers is None:
        try:
            from data.popular_lists import SP500_LEADERS
            seed_tickers = SP500_LEADERS
        except Exception:
            seed_tickers = (
                "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                "BRK-B", "TSLA", "AVGO", "LLY",
            )

    all_items: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="mkt-news") as pool:
        futures = [pool.submit(_fetch_yfinance_news, t, lookback_days, 15)
                   for t in seed_tickers]
        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result() or [])
            except Exception:
                continue

    all_items = _enrich_sentiment(all_items)
    all_items = _dedupe(all_items)
    for it in all_items:
        it.relevance_score = _score_relevance(it, query_ticker=None)
    all_items.sort(key=lambda x: x.relevance_score, reverse=True)
    return all_items[:max_items]
