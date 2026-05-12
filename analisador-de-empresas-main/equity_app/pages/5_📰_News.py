import streamlit as st

# NO se llama a st.set_page_config — heredamos del app.py global.

st.markdown(
    "<div style='text-align:center;color:#94a3b8;"
    "font-size:12px;letter-spacing:0.1em;margin:8px 0;'>NEWS</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='text-align:center;margin-bottom:24px;'>Market News</h1>",
    unsafe_allow_html=True,
)

st.text_input(
    "Search by ticker",
    placeholder="e.g., AAPL, NVDA, WMT",
    label_visibility="collapsed",
)

st.info("📰 News feed module under construction. Coming in next iteration.")
