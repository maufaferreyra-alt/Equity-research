"""Twitter-style news card for the News page.

CSS is scoped with the 'news-card-' class prefix to prevent style
leakage to other pages.
"""

from datetime import datetime, timezone
import streamlit as st
from analysis.news_aggregator import NewsItem


def _relative_time(ts: datetime) -> str:
    now = datetime.now(timezone.utc)
    secs = (now - ts).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs / 60)}m ago"
    if secs < 86400:
        return f"{int(secs / 3600)}h ago"
    if secs < 604800:
        return f"{int(secs / 86400)}d ago"
    return ts.strftime("%b %d")


_SENTIMENT_COLOR = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
}


def build_card_html(item: NewsItem) -> str:
    """Build the card HTML as a string (for batch grid rendering).

    Use render_news_card() instead when rendering a single card inline;
    use this when wrapping multiple cards in a CSS grid container."""
    sent_color = _SENTIMENT_COLOR.get(item.sentiment_label or "", "transparent")
    sent_dot = (
        f'<span class="news-card-sentiment-dot" '
        f'style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{sent_color};margin-right:8px;'
        f'vertical-align:middle;"></span>'
        if item.sentiment_label else ""
    )

    tickers_html = ""
    if item.tickers:
        tags = " ".join(
            f'<span class="news-card-ticker-tag" '
            f'style="display:inline-block;padding:2px 8px;'
            f'background:rgba(148,163,184,0.15);border-radius:4px;'
            f'font-size:11px;color:#94a3b8;margin-right:4px;">'
            f'${t}</span>'
            for t in item.tickers[:5]
        )
        tickers_html = f'<div class="news-card-tags" style="margin-top:8px;">{tags}</div>'

    snippet_html = ""
    if item.snippet:
        snippet_trunc = item.snippet[:200] + ("…" if len(item.snippet) > 200 else "")
        snippet_html = (
            f'<div class="news-card-snippet" style="font-size:13px;'
            f'color:#94a3b8;margin-top:6px;line-height:1.4;">'
            f'{snippet_trunc}</div>'
        )

    # max-width removed (grid item is bounded by column);
    # height:100% + flex so all cards in a row align to the tallest.
    return f"""
    <div class="news-card-container" style="border:1px solid rgba(148,163,184,0.15);
                border-radius:8px;padding:14px 16px;
                background:rgba(15,23,42,0.4);
                height:100%;display:flex;flex-direction:column;">
      <div class="news-card-header" style="display:flex;
                  justify-content:space-between;align-items:center;
                  font-size:12px;color:#94a3b8;margin-bottom:6px;">
        <span>{sent_dot}<strong style="color:#e2e8f0;">{item.source}</strong></span>
        <span>{_relative_time(item.published_at)}</span>
      </div>
      <a class="news-card-link" href="{item.url}" target="_blank"
         style="text-decoration:none;color:inherit;">
        <div class="news-card-title" style="font-size:15px;font-weight:600;
                    color:#e2e8f0;line-height:1.35;">
          {item.title}
        </div>
      </a>
      {snippet_html}
      {tickers_html}
    </div>
    """


def render_news_card(item: NewsItem) -> None:
    """Render a single news card inline. For batch grid rendering use
    build_card_html() + a single st.markdown() wrap."""
    st.markdown(build_card_html(item), unsafe_allow_html=True)


@st.dialog("Article preview", width="large")
def _show_article_modal(item: NewsItem) -> None:
    """Streamlit modal dialog showing extended article preview.

    We can't embed the publisher page in an iframe — Bloomberg / Reuters
    / WSJ / Yahoo / Barron's all set X-Frame-Options to deny. So the
    modal shows whatever metadata the aggregator collected (snippet,
    tickers, sentiment) and offers a button to open the article in a
    new tab if the user wants to read the full piece."""
    sent_color = _SENTIMENT_COLOR.get(item.sentiment_label or "", "transparent")
    sent_dot = (
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{sent_color};margin-right:8px;'
        f'vertical-align:middle;"></span>'
        if item.sentiment_label else ""
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;font-size:13px;color:#94a3b8;'
        f'margin-bottom:12px;">'
        f'<span>{sent_dot}<strong style="color:#e2e8f0;">{item.source}</strong></span>'
        f'<span>{_relative_time(item.published_at)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<h2 style="font-size:22px;line-height:1.3;color:#e2e8f0;'
        f'margin-bottom:16px;">{item.title}</h2>',
        unsafe_allow_html=True,
    )

    if item.snippet:
        st.markdown(
            f'<div style="font-size:14px;color:#cbd5e1;line-height:1.6;'
            f'margin-bottom:16px;">{item.snippet}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:13px;color:#94a3b8;font-style:italic;'
            'margin-bottom:16px;">No preview text available for this article.</div>',
            unsafe_allow_html=True,
        )

    if item.tickers:
        tags_html = " ".join(
            f'<span style="display:inline-block;padding:3px 10px;'
            f'background:rgba(148,163,184,0.15);border-radius:4px;'
            f'font-size:12px;color:#94a3b8;margin-right:6px;">'
            f'${t}</span>'
            for t in item.tickers
        )
        st.markdown(
            f'<div style="margin-bottom:20px;">{tags_html}</div>',
            unsafe_allow_html=True,
        )

    col_open, col_close = st.columns([3, 1])
    with col_open:
        st.link_button(
            "Read full article →",
            item.url,
            use_container_width=True,
            type="primary",
        )
    with col_close:
        if st.button("Close", use_container_width=True):
            st.rerun()


def render_news_card_with_modal(item: NewsItem, key: str) -> None:
    """Render card HTML + a 'Preview' button that opens an in-app
    article modal on click. ``key`` must be unique per card on the
    page (typical: ``f"newscard_{idx}"``)."""
    st.markdown(build_card_html(item), unsafe_allow_html=True)
    if st.button("Preview", key=key, use_container_width=True,
                 help="Open article preview"):
        _show_article_modal(item)
