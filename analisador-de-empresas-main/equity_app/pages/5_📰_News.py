import streamlit as st

# Header
st.markdown(
    "<div style='text-align:center;color:#94a3b8;"
    "font-size:12px;letter-spacing:0.1em;margin:8px 0;'>NEWS</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='text-align:center;margin-bottom:24px;'>Market News</h1>",
    unsafe_allow_html=True,
)

# Search bar + clear
col_search, col_clear = st.columns([6, 1])
with col_search:
    ticker_query = st.text_input(
        "Search by ticker",
        value="",
        placeholder="e.g., AAPL, NVDA, WMT",
        label_visibility="collapsed",
    ).strip().upper()
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# Filters
col_time, col_count, _ = st.columns([2, 2, 6])
with col_time:
    period_options = [
        ("Last 24h", 1),
        ("Last 3 days", 3),
        ("Last week", 7),
        ("Last 2 weeks", 14),
    ]
    lookback = st.selectbox(
        "Period",
        options=period_options,
        format_func=lambda x: x[0],
        index=2,
    )[1]
with col_count:
    items_to_show = st.selectbox(
        "Show",
        options=[20, 30, 50, 100],
        index=1,
    )

st.markdown("---")

# Lazy import — defensive: if aggregator breaks, page still loads
try:
    from analysis.news_aggregator import (
        fetch_news_for_ticker,
        fetch_market_news,
    )
    from ui.components.news_card import render_news_card_with_modal
    _IMPORT_OK = True
except Exception as e:
    st.error(f"News module failed to load: {e}")
    _IMPORT_OK = False


def _render_news_grid(items, n_cols: int = 3):
    """Render news items in a multi-column layout.

    Uses st.columns(n_cols) and distributes cards round-robin across
    columns. n_cols=3 is a sensible default for desktop viewports
    (1280px+). Each card renders via render_news_card in its own
    column context — Streamlit handles the layout.
    """
    if not items:
        return

    # Iterate in chunks of n_cols. Each chunk becomes a row.
    for i in range(0, len(items), n_cols):
        cols = st.columns(n_cols, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(items):
                with col:
                    render_news_card_with_modal(
                        items[idx],
                        key=f"newscard_{idx}_{i}_{j}",
                    )


if _IMPORT_OK:
    if ticker_query:
        st.caption(
            f"Showing news for **{ticker_query}** — sorted by relevance"
        )
        try:
            with st.spinner("Fetching news…"):
                items = fetch_news_for_ticker(
                    ticker_query,
                    lookback_days=lookback,
                    max_items=items_to_show,
                )
        except Exception as e:
            st.error(f"Failed to fetch news for {ticker_query}: {e}")
            items = []

        if not items:
            st.info(
                f"No news found for {ticker_query} in the last "
                f"{lookback} days. Try a different ticker or extend "
                f"the time range."
            )
        else:
            _render_news_grid(items)
    else:
        st.caption("Top market news across major tickers — sorted by relevance")
        try:
            with st.spinner("Loading market news…"):
                items = fetch_market_news(
                    lookback_days=lookback,
                    max_items=items_to_show,
                )
        except Exception as e:
            st.error(f"Failed to fetch market news: {e}")
            items = []

        if not items:
            st.warning(
                "No news fetched. Yahoo Finance may be temporarily "
                "throttled. Configure FINNHUB_API_KEY and "
                "MARKETAUX_API_KEY for fuller coverage."
            )
        else:
            _render_news_grid(items)
