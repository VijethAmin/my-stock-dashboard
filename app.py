import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import warnings
import nltk
 
warnings.filterwarnings('ignore')
 
# --- Conditional Imports for Optional Libraries ---
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

from analysis.forecasting import (
    HAS_LSTM,
    Prophet,
    compare_all_models,
    plot_forecast,
    run_forecast,
)
from analysis.indicators import add_technical_indicators
from analysis.metrics import compute_metrics
from services.market_data import load_data, parse_tickers, to_nse_symbol
from services.news import SentimentIntensityAnalyzer, get_news
from services.s3 import build_s3_client, get_s3_config
from ui.downloads import create_download_buttons, s3_upload_section
 
st.markdown("""
<style>
.main > div {padding-top: 2rem;}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.stTabs [data-baseweb="tab-list"] {gap: 2px;}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #f0f2f6;
    border-radius: 4px 4px 0 0;
    padding: 0 20px;
}
.stTabs [aria-selected="true"] {background-color: #1f77b4; color: white;}
.s3-success {
    padding: 0.5rem 1rem;
    background: #d1fae5;
    border-left: 4px solid #10b981;
    border-radius: 4px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.s3-error {
    padding: 0.5rem 1rem;
    background: #fee2e2;
    border-left: 4px solid #ef4444;
    border-radius: 4px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("# 📊 Indian Stock Market Dashboard")
st.markdown("### Advanced Analysis • Forecasting • Sentiment • Portfolio")
 
# ─────────────────────────────────────────────────────────────────────────────
# NLTK VADER CHECK
# ─────────────────────────────────────────────────────────────────────────────
if SentimentIntensityAnalyzer:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        with st.spinner("Downloading sentiment data..."):
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception as e:
                st.error(f"Failed to download NLTK data: {e}")
                SentimentIntensityAnalyzer = None
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
S3_CFG = get_s3_config()
 
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("### 📈 Stock Selection")
 
    tickers_input = st.text_input(
        "Enter NSE stock symbols (comma separated)", "RELIANCE,TCS,INFY,HDFCBANK"
    )
 
    popular_stocks = {
        "FAANG of India":  "RELIANCE,TCS,INFY,HDFCBANK,ITC",
        "Banking Stocks":  "HDFCBANK,ICICIBANK,SBIN,KOTAKBANK,AXISBANK",
        "IT Stocks":       "TCS,INFY,WIPRO,HCLTECH,TECHM",
        "Auto Stocks":     "MARUTI,TATAMOTORS,M&M,BAJAJ-AUTO,HEROMOTOCO",
    }
    preset = st.selectbox("Or choose a preset:", ["Custom"] + list(popular_stocks.keys()))
    if preset != "Custom":
        tickers_input = popular_stocks[preset]
        st.info(f"Selected: {preset}")
 
    tickers = parse_tickers(tickers_input)
    if not tickers:
        st.error("Enter at least one valid NSE ticker.")
        st.stop()
    symbols = [to_nse_symbol(t) for t in tickers]
 
    st.markdown("### 📊 Analysis Mode")
    mode = st.radio(
        "Select mode",
        ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis"],
        index=1 if len(symbols) > 1 else 0,
    )
 
    st.markdown("### 🔮 Forecasting")
    forecast_options = ["None", "ARIMA", "SARIMA"]
 
    if Prophet:
        forecast_options.append("Prophet")
    if HAS_LSTM:
        forecast_options.append("LSTM")
 
    forecast_options.append("Compare All Models")
 
    forecast_method = st.selectbox("Forecast method", forecast_options)
    forecast_days   = st.slider("Forecast horizon (days)", 5, 365, 30, step=5)
 
    st.markdown("### ⚙️ Settings")
    period_options = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max",
    }
    period      = period_options[st.selectbox("Data period", list(period_options.keys()), index=4)]
    show_volume = st.checkbox("Show volume data", value=True)
    show_news   = st.checkbox("Show news & sentiment", value=True)
 
    refresh_rate = st.slider("Auto-refresh (seconds, 0 = off)", 0, 300, 0)
    if refresh_rate > 0:
        if HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_rate * 1000, key="refresh")
        else:
            st.warning("Install `streamlit-autorefresh` for auto-refresh.")
 
    st.markdown("---")
    st.markdown("### ☁️ AWS S3")
    if S3_CFG:
        st.success(f"✅ Connected\n\nBucket: `{S3_CFG['bucket']}`\nRegion: `{S3_CFG['region']}`")
    else:
        st.warning("Not configured.\nAdd `[aws]` block to `.streamlit/secrets.toml`")
 
    st.markdown("---")
    st.markdown("### 📋 Current Selection")
    for i, ticker in enumerate(tickers, 1):
        st.markdown(f"{i}. **{ticker}**")
 
st.session_state["current_tickers"] = tickers
 
# ─────────────────────────────────────────────────────────────────────────────
# EARLY-EXIT IF NO TICKERS
# ─────────────────────────────────────────────────────────────────────────────
if not tickers:
    st.info("👉 Enter stock symbols in the sidebar to begin.")
    st.stop()
 
# ═════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Single Stock Analysis":
    if len(symbols) > 1:
        st.warning(
            f"Single Stock mode: using **{tickers[0]}** only. "
            "Switch to *Multi-Stock Comparison* to analyse all."
        )
    ticker, symbol = tickers[0], symbols[0]
 
    with st.spinner(f"Loading **{ticker}**…"):
        df, info = load_data(symbol, period)
 
    if df is None:
        st.error(f"Could not load **{ticker}**.")
        st.stop()
 
    df = add_technical_indicators(df)
    st.markdown(f"## 📈 {ticker} — Detailed Analysis")
 
    if info:
        cur   = info.get("currentPrice",  df["Close"].iloc[-1])
        prev  = info.get("previousClose", df["Close"].iloc[-2] if len(df) >= 2 else None)
        cap   = info.get("marketCap",     None)
        pe    = info.get("trailingPE",    None)
        wk52h = info.get("fiftyTwoWeekHigh", None)
        wk52l = info.get("fiftyTwoWeekLow",  None)
 
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if prev and prev != 0:
                pct = (cur - prev) / prev * 100
                st.metric("💰 Price", f"₹{cur:.2f}", f"{pct:+.2f}%")
            else:
                st.metric("💰 Price", f"₹{cur:.2f}")
        with col2:
            st.metric("📊 Prev Close", f"₹{prev:.2f}" if isinstance(prev, float) else "N/A")
        with col3:
            st.metric("🏢 Mkt Cap",
                      f"₹{cap/1e7:.0f} Cr" if isinstance(cap, (int, float)) and cap > 1e7 else "N/A")
        with col4:
            st.metric("📈 P/E", f"{pe:.2f}" if isinstance(pe, float) else "N/A")
        with col5:
            st.metric("📅 52W High", f"₹{wk52h:.2f}" if isinstance(wk52h, float) else "N/A")
        with col6:
            st.metric("📅 52W Low",  f"₹{wk52l:.2f}" if isinstance(wk52l, float) else "N/A")
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Price Chart", "📈 Indicators & Signals",
        "🔮 Forecasting", "📰 News & Sentiment",
        "📋 Data Export", "ℹ️ Stock Info",
    ])
 
    # ── Tab 1: Price Chart ─────────────────────────────────────────────
    with tab1:
        st.subheader("Candlestick Price Chart")
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            subplot_titles=("Price", "Volume") if show_volume else ("Price",),
            row_width=[0.25, 0.75] if show_volume else [1],
        )
        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
        ), row=1, col=1)
        for ma, color in [("SMA20", "orange"), ("SMA50", "royalblue"), ("EMA20", "tomato")]:
            if df[ma].dropna().shape[0] > 0:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], name=ma,
                                         line=dict(color=color, width=1.2)), row=1, col=1)
        if df["BB_Upper"].dropna().shape[0] > 0:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper",
                                     line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower",
                                     line=dict(color="gray", dash="dash", width=1),
                                     fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)
        if show_volume and rows == 2:
            colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                                  marker_color=colors, opacity=0.7), row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(height=700, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")
 
    # ── Tab 2: Indicators & Signals ────────────────────────────────────
    with tab2:
        st.subheader("Technical Indicators")
        ind_tabs = st.tabs(["RSI", "MACD", "Stochastic", "ATR", "🚦 Signals"])
        with ind_tabs[0]:
            if df["RSI"].dropna().shape[0] > 0:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                                             line=dict(color="purple")))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                                  annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                                  annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI (14)", template="plotly_white", height=350)
                st.plotly_chart(fig_rsi, width="stretch")
            else:
                st.info("RSI unavailable — need ≥ 14 data points.")
        with ind_tabs[1]:
            if df["MACD"].dropna().shape[0] > 0:
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.1, row_heights=[0.6, 0.4])
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
                                              name="MACD", line=dict(color="blue")), row=1, col=1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"],
                                              name="Signal", line=dict(color="red")), row=1, col=1)
                colors_hist = ["green" if v >= 0 else "red" for v in df["MACD_Histogram"].fillna(0)]
                fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Histogram"],
                                          name="Histogram", marker_color=colors_hist), row=2, col=1)
                fig_macd.update_layout(title="MACD", template="plotly_white", height=450)
                st.plotly_chart(fig_macd, width="stretch")
            else:
                st.info("MACD unavailable — need ≥ 26 data points.")
        with ind_tabs[2]:
            if df["Stoch_K"].dropna().shape[0] > 0:
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"],
                                               name="%K", line=dict(color="blue")))
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"],
                                               name="%D", line=dict(color="red")))
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red",
                                    annotation_text="Overbought (80)")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green",
                                    annotation_text="Oversold (20)")
                fig_stoch.update_layout(title="Stochastic Oscillator",
                                        template="plotly_white", height=350)
                st.plotly_chart(fig_stoch, width="stretch")
            else:
                st.info("Stochastic unavailable — need ≥ 14 data points.")
        with ind_tabs[3]:
            if df["ATR"].dropna().shape[0] > 0:
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(x=df["Date"], y=df["ATR"],
                                             name="ATR", line=dict(color="teal")))
                fig_atr.update_layout(title="ATR", template="plotly_white", height=350)
                st.plotly_chart(fig_atr, width="stretch")
            else:
                st.info("ATR unavailable — need ≥ 14 data points.")
        with ind_tabs[4]:
            st.subheader("🚦 Buy / Sell Signals")
            buy_df  = df[df["Signal_Flag"] == 1]
            sell_df = df[df["Signal_Flag"] == -1]
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                                         name="Close", line=dict(color="#1f77b4")))
            fig_sig.add_trace(go.Scatter(x=buy_df["Date"], y=buy_df["Close"], mode="markers",
                                         name="Buy", marker=dict(color="green",
                                         symbol="triangle-up", size=10)))
            fig_sig.add_trace(go.Scatter(x=sell_df["Date"], y=sell_df["Close"], mode="markers",
                                         name="Sell", marker=dict(color="red",
                                         symbol="triangle-down", size=10)))
            fig_sig.update_layout(title="Buy/Sell Signals", template="plotly_white", height=450)
            st.plotly_chart(fig_sig, width="stretch")
            col_b, col_s = st.columns(2)
            col_b.metric("🟢 Buy Signals",  len(buy_df))
            col_s.metric("🔴 Sell Signals", len(sell_df))
            st.caption("⚠️ For educational purposes only — not financial advice.")
 
    # ── Tab 3: Forecasting ─────────────────────────────────────────────
    # FIX 3: tab3/tab4/tab5/tab6 all at the same indentation level (were nested before)
    with tab3:
        if forecast_method == "None":
            st.info("Select a forecasting method in the sidebar.")
        elif forecast_method == "Compare All Models":
            st.subheader("📊 Model Comparison")
            with st.spinner("Running all models…"):
                comp_df, comp_forecasts = compare_all_models(df, forecast_days)
            if not comp_df.empty:
                st.dataframe(comp_df.set_index("Model"), width="stretch")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=df["Date"], y=df["Close"], name="Historical",
                    line=dict(color="#1f77b4")))
                for mdl, res in comp_forecasts.items():
                    fig_comp.add_trace(go.Scatter(
                        x=res.index, y=res["Forecast"],
                        name=mdl, line=dict(dash="dash")))
                fig_comp.update_layout(
                    title=f"{ticker} — All Model Forecasts",
                    xaxis_title="Date", yaxis_title="Price (₹)",
                    height=500, template="plotly_white")
                st.plotly_chart(fig_comp, width="stretch")
            else:
                st.warning("No models produced results.")
        else:
            with st.spinner(f"Running {forecast_method}…"):
                result = run_forecast(df, forecast_method, forecast_days)
 
            if result is not None and not result.empty:
                st.plotly_chart(plot_forecast(df, result, ticker, forecast_method),
                                width="stretch")
                cur_p  = df["Close"].iloc[-1]
                fwd_p  = result["Forecast"].iloc[-1]
                pct_ch = (fwd_p - cur_p) / cur_p * 100
                c1, c2, c3 = st.columns(3)
                c1.metric(f"In {forecast_days} days", f"₹{fwd_p:.2f}", f"{pct_ch:+.2f}%")
                c2.metric("Forecast High", f"₹{result['Forecast'].max():.2f}")
                c3.metric("Forecast Low",  f"₹{result['Forecast'].min():.2f}")
 
                df_idx  = df.set_index("Date")
                overlap = result.join(df_idx[["Close"]], how="inner")
                if len(overlap) > 1:
                    st.subheader("🎯 Model Accuracy")
                    metrics  = compute_metrics(overlap["Close"], overlap["Forecast"])
                    met_cols = st.columns(len(metrics))
                    for i, (name, val) in enumerate(metrics.items()):
                        with met_cols[i]:
                            if isinstance(val, float) and not np.isnan(val):
                                fmt = (f"{val:.4f}" if name == "R²" else
                                       f"{val:.2f}%" if "%" in name else f"{val:.2f}")
                                st.metric(name, fmt)
                            else:
                                st.metric(name, "N/A")
 
                st.markdown("#### 📥 Export Forecast")
                create_download_buttons(
                    result,
                    filename_prefix=f"{ticker}_forecast_{forecast_method.lower()}",
                    ticker=ticker,
                    label=f"forecast_{forecast_method.lower()}",
                    s3_cfg=S3_CFG,
                )
            else:
                st.error(
                    f"The {forecast_method} model could not generate a forecast. "
                    "This usually happens if there is insufficient historical data."
                )
 
    # ── Tab 4: News & Sentiment ────────────────────────────────────────
    with tab4:
        if not show_news:
            st.info("Enable 'Show news & sentiment' in sidebar.")
        elif not SentimentIntensityAnalyzer:
            st.error("NLTK VADER not available — run `pip install nltk`.")
        else:
            st.subheader(f"📰 Latest News — {ticker}")
            with st.spinner("Fetching news…"):
                news_items = get_news(symbol)
            if not news_items:
                st.info("No recent news found.")
            else:
                sid = SentimentIntensityAnalyzer()
                sentiment_data = []
                for title, link in news_items:
                    s = sid.polarity_scores(title)
                    c = s["compound"]
                    lbl = ("🟢 Positive" if c > 0.05 else
                           "🔴 Negative" if c < -0.05 else "⚪ Neutral")
                    sentiment_data.append({"Title": title, "Link": link,
                                           "Sentiment": lbl, "Score": c})
                for item in sentiment_data:
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"**[{item['Title']}]({item['Link']})**")
                    c2.markdown(f"{item['Sentiment']} ({item['Score']:.2f})")
                    st.markdown("---")
                scores  = [d["Score"] for d in sentiment_data]
                avg     = np.mean(scores)
                lbl_avg = ("Very Positive" if avg > 0.3 else
                           "Positive"      if avg > 0.05 else
                           "Neutral"       if avg >= -0.05 else
                           "Negative"      if avg > -0.3 else "Very Negative")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🟢 Positive", sum(s > 0.05       for s in scores))
                sc2.metric("⚪ Neutral",  sum(-0.05 <= s <= 0.05 for s in scores))
                sc3.metric("🔴 Negative", sum(s < -0.05      for s in scores))
                fig_sent = go.Figure(go.Bar(
                    x=["Positive", "Neutral", "Negative"],
                    y=[sum(s > 0.05 for s in scores),
                       sum(-0.05 <= s <= 0.05 for s in scores),
                       sum(s < -0.05 for s in scores)],
                    marker_color=["green", "gray", "red"],
                ))
                fig_sent.update_layout(title="Sentiment Distribution",
                                       template="plotly_white", height=300)
                st.plotly_chart(fig_sent, width="stretch")
                st.info(f"**Average Sentiment**: {avg:.3f} → {lbl_avg}")
 
    # ── Tab 5: Data Export ─────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Data Export")
        st.dataframe(df.tail(10), width="stretch")
 
        export_opts = st.multiselect(
            "Select data to export:",
            ["Price Data", "Technical Indicators"],
            default=["Price Data"],
        )
 
        if "Price Data" in export_opts:
            st.markdown("#### 📊 Price Data")
            price_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")
            create_download_buttons(
                price_df,
                filename_prefix=f"{ticker}_price",
                ticker=ticker,
                label="price_data",
                s3_cfg=S3_CFG,
            )
            if S3_CFG:
                st.markdown("**☁️ Upload as JSON to S3:**")
                s3_upload_section(price_df, ticker, "price_data_json", S3_CFG)
 
        if "Technical Indicators" in export_opts:
            st.markdown("#### 📈 Technical Indicators")
            excl = {"Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"}
            tech_cols = [c for c in df.columns if c not in excl]
            tech_df   = df.set_index("Date")[tech_cols]
            create_download_buttons(
                tech_df,
                filename_prefix=f"{ticker}_indicators",
                ticker=ticker,
                label="technical_indicators",
                s3_cfg=S3_CFG,
            )
 
        if S3_CFG:
            with st.expander("🗂️ View files already in S3 for this ticker"):
                try:
                    s3_client = build_s3_client(S3_CFG)
                    prefix    = f"stocks/{ticker}/"
                    response  = s3_client.list_objects_v2(
                        Bucket=S3_CFG["bucket"], Prefix=prefix
                    )
                    objects = response.get("Contents", [])
                    if objects:
                        rows = [{"Key": o["Key"],
                                 "Size (KB)": round(o["Size"] / 1024, 2),
                                 "Last Modified": o["LastModified"].strftime("%Y-%m-%d %H:%M")}
                                for o in objects]
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                    else:
                        st.info(f"No files yet at `s3://{S3_CFG['bucket']}/{prefix}`")
                except Exception as e:
                    st.warning(f"Could not list S3 objects: {e}")
 
    # ── Tab 6: Stock Info ──────────────────────────────────────────────
    with tab6:
        st.subheader("ℹ️ Company Information")
        if info:
            display_keys = [
                "longName", "sector", "industry", "country", "website",
                "fullTimeEmployees", "longBusinessSummary",
                "beta", "forwardPE", "priceToBook", "returnOnEquity",
                "debtToEquity", "currentRatio", "revenuePerShare",
            ]
            for k in display_keys:
                v = info.get(k)
                if v is not None:
                    label = k.replace("long", "").replace("full", "").replace("Time", "").strip()
                    if k == "longBusinessSummary":
                        st.markdown(f"**Description**: {v}")
                    else:
                        st.markdown(f"**{label}**: {v}")
        else:
            st.info("Company info unavailable.")


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-STOCK COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Multi-Stock Comparison":
    if len(symbols) == 1:
        st.warning("Multi-Stock mode requires ≥ 2 tickers.")
        st.stop()

    st.markdown(f"## 📈 Multi-Stock Comparison — {', '.join(tickers)}")

    with st.spinner("Loading stocks…"):
        all_data = {}
        for t, s in zip(tickers, symbols):
            df_tmp, _ = load_data(s, period)
            if df_tmp is not None:
                all_data[t] = add_technical_indicators(df_tmp)

    if not all_data:
        st.error("Could not load any stock data.")
        st.stop()

    forecast_results = {}

    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
        "📊 Price Comparison", "📈 Performance Metrics",
        "🔮 Forecast Comparison", "📋 Export Data",
    ])

    with comp_tab1:
        st.subheader("Normalised Price Comparison")
        normalize = st.checkbox("Normalise prices (% change from start)", value=True)
        fig_cmp   = go.Figure()
        for t, df_t in all_data.items():
            y = (df_t["Close"] / df_t["Close"].iloc[0] - 1) * 100 if normalize else df_t["Close"]
            fig_cmp.add_trace(go.Scatter(x=df_t["Date"], y=y, mode="lines", name=t))
        fig_cmp.update_layout(title="Price Comparison",
                               yaxis_title="% Change" if normalize else "Price (₹)",
                               template="plotly_white", height=500)
        st.plotly_chart(fig_cmp, width="stretch")
        if show_volume:
            fig_vol = go.Figure()
            for t, df_t in all_data.items():
                fig_vol.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Volume"],
                                             mode="lines", name=t))
            fig_vol.update_layout(title="Volume", template="plotly_white", height=350)
            st.plotly_chart(fig_vol, width="stretch")
        close_df = pd.DataFrame({t: df_t.set_index("Date")["Close"] for t, df_t in all_data.items()})
        corr = close_df.pct_change().dropna().corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Return Correlation Matrix")
        st.plotly_chart(fig_corr, width="stretch")

    with comp_tab2:
        st.subheader("Performance Metrics")
        perf_rows = []
        rf_daily  = 0.065 / 252
        for t, df_t in all_data.items():
            if len(df_t) < 2:
                continue
            rets      = df_t["Close"].pct_change().dropna()
            cur_p     = df_t["Close"].iloc[-1]
            first_p   = df_t["Close"].iloc[0]
            total_ret = (cur_p / first_p - 1) * 100
            vol       = rets.std() * np.sqrt(252) * 100
            sharpe    = (rets.mean() - rf_daily) / rets.std() * np.sqrt(252) if rets.std() else 0
            max_dd    = ((df_t["Close"] / df_t["Close"].cummax()) - 1).min() * 100
            perf_rows.append({
                "Stock": t, "Current (₹)": f"{cur_p:.2f}",
                "Total Return (%)": f"{total_ret:.2f}", "Ann. Vol (%)": f"{vol:.2f}",
                "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown (%)": f"{max_dd:.2f}",
                "Avg Volume": f"{df_t['Volume'].mean():,.0f}",
            })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows).set_index("Stock")
            st.dataframe(perf_df, width="stretch")
            stocks    = [r["Stock"] for r in perf_rows]
            returns   = [float(r["Total Return (%)"]) for r in perf_rows]
            vols      = [float(r["Ann. Vol (%)"]) for r in perf_rows]
            sharpes   = [float(r["Sharpe Ratio"]) for r in perf_rows]
            drawdowns = [float(r["Max Drawdown (%)"]) for r in perf_rows]
            fig_p = make_subplots(rows=2, cols=2,
                                  subplot_titles=("Total Return (%)", "Volatility (%)",
                                                  "Sharpe Ratio", "Max Drawdown (%)"))
            fig_p.add_trace(go.Bar(x=stocks, y=returns,   marker_color="seagreen"), row=1, col=1)
            fig_p.add_trace(go.Bar(x=stocks, y=vols,      marker_color="tomato"),   row=1, col=2)
            fig_p.add_trace(go.Bar(x=stocks, y=sharpes,   marker_color="steelblue"),row=2, col=1)
            fig_p.add_trace(go.Bar(x=stocks, y=drawdowns, marker_color="darkorange"),row=2, col=2)
            fig_p.update_layout(height=600, showlegend=False,
                                 title_text="Performance Overview", template="plotly_white")
            st.plotly_chart(fig_p, width="stretch")

    with comp_tab3:
        if forecast_method == "None":
            st.info("Select a forecasting method in the sidebar.")
        else:
            st.subheader(f"🔮 Forecast Comparison — {forecast_method} ({forecast_days} days)")
            fcast_perf = []
            with st.spinner(f"Generating {forecast_method} forecasts…"):
                for t, df_t in all_data.items():
                    res = run_forecast(df_t, forecast_method, forecast_days)
                    if res is not None and not res.empty:
                        forecast_results[t] = res
                        cur_p = df_t["Close"].iloc[-1]
                        fwd_p = res["Forecast"].iloc[-1]
                        fcast_perf.append({
                            "Stock": t, "Last Close (₹)": f"{cur_p:.2f}",
                            f"{forecast_days}d Forecast (₹)": f"{fwd_p:.2f}",
                            "Change (%)": f"{(fwd_p - cur_p) / cur_p * 100:+.2f}",
                            "Fcast High (₹)": f"{res['Forecast'].max():.2f}",
                            "Fcast Low (₹)":  f"{res['Forecast'].min():.2f}",
                        })
            if forecast_results:
                st.dataframe(pd.DataFrame(fcast_perf).set_index("Stock"), width="stretch")
                fig_cf = go.Figure()
                for t, res in forecast_results.items():
                    df_t = all_data[t]
                    fig_cf.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Close"],
                                                name=f"{t} (Hist)", opacity=0.7))
                    fig_cf.add_trace(go.Scatter(x=res.index, y=res["Forecast"],
                                                name=f"{t} (Forecast)",
                                                line=dict(dash="dash")))
                fig_cf.update_layout(title=f"Multi-Stock {forecast_method} Forecast",
                                     xaxis_title="Date", yaxis_title="Price (₹)",
                                     height=600, template="plotly_white")
                st.plotly_chart(fig_cf, width="stretch")
            else:
                st.warning("No forecasts generated.")

    # ── Export tab (multi-stock) ───────────────────────────────────────
    with comp_tab4:
        st.subheader("📋 Combined Data Export")
        available = [t for t in tickers if t in all_data]
        if not available:
            st.error("No data available to export.")
        else:
            merged = all_data[available[0]].set_index("Date")[["Close", "Volume"]].rename(
                columns={"Close": f"{available[0]}_Close", "Volume": f"{available[0]}_Volume"})
            for t in available[1:]:
                tmp = all_data[t].set_index("Date")[["Close", "Volume"]].rename(
                    columns={"Close": f"{t}_Close", "Volume": f"{t}_Volume"})
                merged = merged.join(tmp, how="outer")

            st.dataframe(merged.tail(10), width="stretch")
            st.markdown("#### 📥 Combined Historical Data")
            create_download_buttons(
                merged,
                filename_prefix="multi_stock_data",
                ticker="MULTI",
                label="combined_price_data",
                s3_cfg=S3_CFG,
            )

            if forecast_results:
                st.markdown("#### 📥 Combined Forecast Data")
                fdf = pd.DataFrame()
                for t, res in forecast_results.items():
                    col_df = res[["Forecast"]].rename(columns={"Forecast": f"{t}_Forecast"})
                    fdf = col_df if fdf.empty else fdf.join(col_df, how="outer")
                st.dataframe(fdf.tail(10), width="stretch")
                create_download_buttons(
                    fdf,
                    filename_prefix=f"multi_forecast_{forecast_method.lower()}",
                    ticker="MULTI",
                    label=f"combined_forecast_{forecast_method.lower()}",
                    s3_cfg=S3_CFG,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Portfolio Analysis":
    if len(symbols) < 2:
        st.warning("Portfolio Analysis requires ≥ 2 stocks.")
        st.stop()

    st.markdown(f"## 💼 Portfolio Analysis — {', '.join(tickers)}")

    with st.spinner("Loading portfolio data…"):
        port_data = {}
        for t, s in zip(tickers, symbols):
            df_tmp, _ = load_data(s, period)
            if df_tmp is not None:
                port_data[t] = df_tmp.set_index("Date")["Close"]

    if len(port_data) < 2:
        st.error("Could not load enough stocks.")
        st.stop()

    close_df = pd.DataFrame(port_data).dropna()
    ret_df   = close_df.pct_change().dropna()
    n        = len(port_data)
    weights  = np.array([1 / n] * n)
    port_ret = ret_df @ weights

    port_tab1, port_tab2, port_tab3 = st.tabs(["📊 Overview", "📈 Risk & Return", "🔗 Correlations"])

    with port_tab1:
        st.subheader("Equal-Weight Portfolio Overview")
        port_value = (1 + port_ret).cumprod()
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(x=port_value.index, y=port_value, name="Portfolio",
                                    fill="tozeroy", line=dict(color="steelblue")))
        for t, ser in close_df.items():
            norm = ser / ser.iloc[0]
            fig_pv.add_trace(go.Scatter(x=norm.index, y=norm, name=t,
                                        line=dict(dash="dot"), opacity=0.7))
        fig_pv.update_layout(title="Portfolio vs Individual Stocks (Normalised)",
                              yaxis_title="Growth of ₹1", template="plotly_white", height=450)
        st.plotly_chart(fig_pv, width="stretch")
        fig_pie = px.pie(values=weights * 100, names=list(port_data.keys()),
                         title="Portfolio Allocation (Equal Weight)")
        st.plotly_chart(fig_pie, width="stretch")

        # ── Export portfolio returns with S3 ──────────────────────────
        st.markdown("#### 📥 Export Portfolio Returns")
        create_download_buttons(
            ret_df,
            filename_prefix="portfolio_returns",
            ticker="PORTFOLIO",
            label="daily_returns",
            s3_cfg=S3_CFG,
        )

    with port_tab2:
        st.subheader("Risk & Return Metrics")
        rf_daily = 0.065 / 252
        ann_ret  = port_ret.mean() * 252 * 100
        ann_vol  = port_ret.std()  * np.sqrt(252) * 100
        sharpe   = (port_ret.mean() - rf_daily) / port_ret.std() * np.sqrt(252)
        max_dd   = ((port_value / port_value.cummax()) - 1).min() * 100
        var_95   = np.percentile(port_ret, 5) * 100
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Ann. Return",     f"{ann_ret:.2f}%")
        mc2.metric("Ann. Volatility", f"{ann_vol:.2f}%")
        mc3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
        mc4.metric("Max Drawdown",    f"{max_dd:.2f}%")
        mc5.metric("VaR (95%, 1d)",   f"{var_95:.2f}%")
        ind_stats = []
        for t, ser in close_df.items():
            r = ser.pct_change().dropna()
            ind_stats.append({"Stock": t,
                               "Return": r.mean() * 252 * 100,
                               "Volatility": r.std() * np.sqrt(252) * 100,
                               "Sharpe": (r.mean() - rf_daily) / r.std() * np.sqrt(252)})
        stats_df = pd.DataFrame(ind_stats)
        fig_scatter = px.scatter(
            stats_df, x="Volatility", y="Return", text="Stock",
            size=[abs(s) + 0.1 for s in stats_df["Sharpe"]],
            color="Sharpe", color_continuous_scale="RdYlGn",
            title="Risk vs Return",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, width="stretch")

    with port_tab3:
        st.subheader("Correlation Analysis")
        corr = ret_df.corr()
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Return Correlation Matrix")
        st.plotly_chart(fig_heat, width="stretch")
        if len(tickers) >= 2:
            roll_corr = ret_df[tickers[0]].rolling(30).corr(ret_df[tickers[1]])
            fig_roll  = go.Figure()
            fig_roll.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                          name=f"{tickers[0]} vs {tickers[1]}",
                                          line=dict(color="purple")))
            fig_roll.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_roll.update_layout(
                title=f"Rolling 30-Day Correlation: {tickers[0]} vs {tickers[1]}",
                template="plotly_white", height=350)
            st.plotly_chart(fig_roll, width="stretch")
        create_download_buttons(ret_df, "portfolio_returns", "PORTFOLIO", "returns", S3_CFG)
 
st.markdown("""
<style>
.main > div {padding-top: 2rem;}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.stTabs [data-baseweb="tab-list"] {gap: 2px;}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #f0f2f6;
    border-radius: 4px 4px 0 0;
    padding: 0 20px;
}
.stTabs [aria-selected="true"] {background-color: #1f77b4; color: white;}
.s3-success {
    padding: 0.5rem 1rem;
    background: #d1fae5;
    border-left: 4px solid #10b981;
    border-radius: 4px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.s3-error {
    padding: 0.5rem 1rem;
    background: #fee2e2;
    border-left: 4px solid #ef4444;
    border-radius: 4px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("# 📊 Indian Stock Market Dashboard")
st.markdown("### Advanced Analysis • Forecasting • Sentiment • Portfolio")
 
# ─────────────────────────────────────────────────────────────────────────────
# NLTK VADER CHECK
# ─────────────────────────────────────────────────────────────────────────────
if SentimentIntensityAnalyzer:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        with st.spinner("Downloading sentiment data..."):
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception as e:
                st.error(f"Failed to download NLTK data: {e}")
                SentimentIntensityAnalyzer = None
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
S3_CFG = get_s3_config()
 
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("### 📈 Stock Selection")
 
    tickers_input = st.text_input(
        "Enter NSE stock symbols (comma separated)", "RELIANCE,TCS,INFY,HDFCBANK"
    )
 
    popular_stocks = {
        "FAANG of India":  "RELIANCE,TCS,INFY,HDFCBANK,ITC",
        "Banking Stocks":  "HDFCBANK,ICICIBANK,SBIN,KOTAKBANK,AXISBANK",
        "IT Stocks":       "TCS,INFY,WIPRO,HCLTECH,TECHM",
        "Auto Stocks":     "MARUTI,TATAMOTORS,M&M,BAJAJ-AUTO,HEROMOTOCO",
    }
    preset = st.selectbox("Or choose a preset:", ["Custom"] + list(popular_stocks.keys()))
    if preset != "Custom":
        tickers_input = popular_stocks[preset]
        st.info(f"Selected: {preset}")
 
    tickers = parse_tickers(tickers_input)
    if not tickers:
        st.error("Enter at least one valid NSE ticker.")
        st.stop()
    symbols = [to_nse_symbol(t) for t in tickers]
 
    st.markdown("### 📊 Analysis Mode")
    mode = st.radio(
        "Select mode",
        ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis"],
        index=1 if len(symbols) > 1 else 0,
    )
 
    st.markdown("### 🔮 Forecasting")
    forecast_options = ["None", "ARIMA", "SARIMA"]
 
    if Prophet:
        forecast_options.append("Prophet")
    if HAS_LSTM:
        forecast_options.append("LSTM")
 
    forecast_options.append("Compare All Models")
 
    forecast_method = st.selectbox("Forecast method", forecast_options)
    forecast_days   = st.slider("Forecast horizon (days)", 5, 365, 30, step=5)
 
    st.markdown("### ⚙️ Settings")
    period_options = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max",
    }
    period      = period_options[st.selectbox("Data period", list(period_options.keys()), index=4)]
    show_volume = st.checkbox("Show volume data", value=True)
    show_news   = st.checkbox("Show news & sentiment", value=True)
 
    refresh_rate = st.slider("Auto-refresh (seconds, 0 = off)", 0, 300, 0)
    if refresh_rate > 0:
        if HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_rate * 1000, key="refresh")
        else:
            st.warning("Install `streamlit-autorefresh` for auto-refresh.")
 
    st.markdown("---")
    st.markdown("### ☁️ AWS S3")
    if S3_CFG:
        st.success(f"✅ Connected\n\nBucket: `{S3_CFG['bucket']}`\nRegion: `{S3_CFG['region']}`")
    else:
        st.warning("Not configured.\nAdd `[aws]` block to `.streamlit/secrets.toml`")
 
    st.markdown("---")
    st.markdown("### 📋 Current Selection")
    for i, ticker in enumerate(tickers, 1):
        st.markdown(f"{i}. **{ticker}**")
 
st.session_state["current_tickers"] = tickers
 
# ─────────────────────────────────────────────────────────────────────────────
# EARLY-EXIT IF NO TICKERS
# ─────────────────────────────────────────────────────────────────────────────
if not tickers:
    st.info("👉 Enter stock symbols in the sidebar to begin.")
    st.stop()
 
# ═════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Single Stock Analysis":
    if len(symbols) > 1:
        st.warning(
            f"Single Stock mode: using **{tickers[0]}** only. "
            "Switch to *Multi-Stock Comparison* to analyse all."
        )
    ticker, symbol = tickers[0], symbols[0]
 
    with st.spinner(f"Loading **{ticker}**…"):
        df, info = load_data(symbol, period)
 
    if df is None:
        st.error(f"Could not load **{ticker}**.")
        st.stop()
 
    df = add_technical_indicators(df)
    st.markdown(f"## 📈 {ticker} — Detailed Analysis")
 
    if info:
        cur   = info.get("currentPrice",  df["Close"].iloc[-1])
        prev  = info.get("previousClose", df["Close"].iloc[-2] if len(df) >= 2 else None)
        cap   = info.get("marketCap",     None)
        pe    = info.get("trailingPE",    None)
        wk52h = info.get("fiftyTwoWeekHigh", None)
        wk52l = info.get("fiftyTwoWeekLow",  None)
 
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if prev and prev != 0:
                pct = (cur - prev) / prev * 100
                st.metric("💰 Price", f"₹{cur:.2f}", f"{pct:+.2f}%")
            else:
                st.metric("💰 Price", f"₹{cur:.2f}")
        with col2:
            st.metric("📊 Prev Close", f"₹{prev:.2f}" if isinstance(prev, float) else "N/A")
        with col3:
            st.metric("🏢 Mkt Cap",
                      f"₹{cap/1e7:.0f} Cr" if isinstance(cap, (int, float)) and cap > 1e7 else "N/A")
        with col4:
            st.metric("📈 P/E", f"{pe:.2f}" if isinstance(pe, float) else "N/A")
        with col5:
            st.metric("📅 52W High", f"₹{wk52h:.2f}" if isinstance(wk52h, float) else "N/A")
        with col6:
            st.metric("📅 52W Low",  f"₹{wk52l:.2f}" if isinstance(wk52l, float) else "N/A")
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Price Chart", "📈 Indicators & Signals",
        "🔮 Forecasting", "📰 News & Sentiment",
        "📋 Data Export", "ℹ️ Stock Info",
    ])
 
    # ── Tab 1: Price Chart ─────────────────────────────────────────────
    with tab1:
        st.subheader("Candlestick Price Chart")
        rows = 2 if show_volume else 1
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            subplot_titles=("Price", "Volume") if show_volume else ("Price",),
            row_width=[0.25, 0.75] if show_volume else [1],
        )
        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
        ), row=1, col=1)
        for ma, color in [("SMA20", "orange"), ("SMA50", "royalblue"), ("EMA20", "tomato")]:
            if df[ma].dropna().shape[0] > 0:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], name=ma,
                                         line=dict(color=color, width=1.2)), row=1, col=1)
        if df["BB_Upper"].dropna().shape[0] > 0:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper",
                                     line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower",
                                     line=dict(color="gray", dash="dash", width=1),
                                     fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)
        if show_volume and rows == 2:
            colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                                  marker_color=colors, opacity=0.7), row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(height=700, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")
 
    # ── Tab 2: Indicators & Signals ────────────────────────────────────
    with tab2:
        st.subheader("Technical Indicators")
        ind_tabs = st.tabs(["RSI", "MACD", "Stochastic", "ATR", "🚦 Signals"])
        with ind_tabs[0]:
            if df["RSI"].dropna().shape[0] > 0:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                                             line=dict(color="purple")))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                                  annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                                  annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI (14)", template="plotly_white", height=350)
                st.plotly_chart(fig_rsi, width="stretch")
            else:
                st.info("RSI unavailable — need ≥ 14 data points.")
        with ind_tabs[1]:
            if df["MACD"].dropna().shape[0] > 0:
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.1, row_heights=[0.6, 0.4])
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
                                              name="MACD", line=dict(color="blue")), row=1, col=1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"],
                                              name="Signal", line=dict(color="red")), row=1, col=1)
                colors_hist = ["green" if v >= 0 else "red" for v in df["MACD_Histogram"].fillna(0)]
                fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Histogram"],
                                          name="Histogram", marker_color=colors_hist), row=2, col=1)
                fig_macd.update_layout(title="MACD", template="plotly_white", height=450)
                st.plotly_chart(fig_macd, width="stretch")
            else:
                st.info("MACD unavailable — need ≥ 26 data points.")
        with ind_tabs[2]:
            if df["Stoch_K"].dropna().shape[0] > 0:
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"],
                                               name="%K", line=dict(color="blue")))
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"],
                                               name="%D", line=dict(color="red")))
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red",
                                    annotation_text="Overbought (80)")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green",
                                    annotation_text="Oversold (20)")
                fig_stoch.update_layout(title="Stochastic Oscillator",
                                        template="plotly_white", height=350)
                st.plotly_chart(fig_stoch, width="stretch")
            else:
                st.info("Stochastic unavailable — need ≥ 14 data points.")
        with ind_tabs[3]:
            if df["ATR"].dropna().shape[0] > 0:
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(x=df["Date"], y=df["ATR"],
                                             name="ATR", line=dict(color="teal")))
                fig_atr.update_layout(title="ATR", template="plotly_white", height=350)
                st.plotly_chart(fig_atr, width="stretch")
            else:
                st.info("ATR unavailable — need ≥ 14 data points.")
        with ind_tabs[4]:
            st.subheader("🚦 Buy / Sell Signals")
            buy_df  = df[df["Signal_Flag"] == 1]
            sell_df = df[df["Signal_Flag"] == -1]
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                                         name="Close", line=dict(color="#1f77b4")))
            fig_sig.add_trace(go.Scatter(x=buy_df["Date"], y=buy_df["Close"], mode="markers",
                                         name="Buy", marker=dict(color="green",
                                         symbol="triangle-up", size=10)))
            fig_sig.add_trace(go.Scatter(x=sell_df["Date"], y=sell_df["Close"], mode="markers",
                                         name="Sell", marker=dict(color="red",
                                         symbol="triangle-down", size=10)))
            fig_sig.update_layout(title="Buy/Sell Signals", template="plotly_white", height=450)
            st.plotly_chart(fig_sig, width="stretch")
            col_b, col_s = st.columns(2)
            col_b.metric("🟢 Buy Signals",  len(buy_df))
            col_s.metric("🔴 Sell Signals", len(sell_df))
            st.caption("⚠️ For educational purposes only — not financial advice.")
 
    # ── Tab 3: Forecasting ─────────────────────────────────────────────
    # FIX 3: tab3/tab4/tab5/tab6 all at the same indentation level (were nested before)
    with tab3:
        if forecast_method == "None":
            st.info("Select a forecasting method in the sidebar.")
        elif forecast_method == "Compare All Models":
            st.subheader("📊 Model Comparison")
            with st.spinner("Running all models…"):
                comp_df, comp_forecasts = compare_all_models(df, forecast_days)
            if not comp_df.empty:
                st.dataframe(comp_df.set_index("Model"), width="stretch")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=df["Date"], y=df["Close"], name="Historical",
                    line=dict(color="#1f77b4")))
                for mdl, res in comp_forecasts.items():
                    fig_comp.add_trace(go.Scatter(
                        x=res.index, y=res["Forecast"],
                        name=mdl, line=dict(dash="dash")))
                fig_comp.update_layout(
                    title=f"{ticker} — All Model Forecasts",
                    xaxis_title="Date", yaxis_title="Price (₹)",
                    height=500, template="plotly_white")
                st.plotly_chart(fig_comp, width="stretch")
            else:
                st.warning("No models produced results.")
        else:
            with st.spinner(f"Running {forecast_method}…"):
                result = run_forecast(df, forecast_method, forecast_days)
 
            if result is not None and not result.empty:
                st.plotly_chart(plot_forecast(df, result, ticker, forecast_method),
                                width="stretch")
                cur_p  = df["Close"].iloc[-1]
                fwd_p  = result["Forecast"].iloc[-1]
                pct_ch = (fwd_p - cur_p) / cur_p * 100
                c1, c2, c3 = st.columns(3)
                c1.metric(f"In {forecast_days} days", f"₹{fwd_p:.2f}", f"{pct_ch:+.2f}%")
                c2.metric("Forecast High", f"₹{result['Forecast'].max():.2f}")
                c3.metric("Forecast Low",  f"₹{result['Forecast'].min():.2f}")
 
                df_idx  = df.set_index("Date")
                overlap = result.join(df_idx[["Close"]], how="inner")
                if len(overlap) > 1:
                    st.subheader("🎯 Model Accuracy")
                    metrics  = compute_metrics(overlap["Close"], overlap["Forecast"])
                    met_cols = st.columns(len(metrics))
                    for i, (name, val) in enumerate(metrics.items()):
                        with met_cols[i]:
                            if isinstance(val, float) and not np.isnan(val):
                                fmt = (f"{val:.4f}" if name == "R²" else
                                       f"{val:.2f}%" if "%" in name else f"{val:.2f}")
                                st.metric(name, fmt)
                            else:
                                st.metric(name, "N/A")
 
                st.markdown("#### 📥 Export Forecast")
                create_download_buttons(
                    result,
                    filename_prefix=f"{ticker}_forecast_{forecast_method.lower()}",
                    ticker=ticker,
                    label=f"forecast_{forecast_method.lower()}",
                    s3_cfg=S3_CFG,
                )
            else:
                st.error(
                    f"The {forecast_method} model could not generate a forecast. "
                    "This usually happens if there is insufficient historical data."
                )
 
    # ── Tab 4: News & Sentiment ────────────────────────────────────────
    with tab4:
        if not show_news:
            st.info("Enable 'Show news & sentiment' in sidebar.")
        elif not SentimentIntensityAnalyzer:
            st.error("NLTK VADER not available — run `pip install nltk`.")
        else:
            st.subheader(f"📰 Latest News — {ticker}")
            with st.spinner("Fetching news…"):
                news_items = get_news(symbol)
            if not news_items:
                st.info("No recent news found.")
            else:
                sid = SentimentIntensityAnalyzer()
                sentiment_data = []
                for title, link in news_items:
                    s = sid.polarity_scores(title)
                    c = s["compound"]
                    lbl = ("🟢 Positive" if c > 0.05 else
                           "🔴 Negative" if c < -0.05 else "⚪ Neutral")
                    sentiment_data.append({"Title": title, "Link": link,
                                           "Sentiment": lbl, "Score": c})
                for item in sentiment_data:
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"**[{item['Title']}]({item['Link']})**")
                    c2.markdown(f"{item['Sentiment']} ({item['Score']:.2f})")
                    st.markdown("---")
                scores  = [d["Score"] for d in sentiment_data]
                avg     = np.mean(scores)
                lbl_avg = ("Very Positive" if avg > 0.3 else
                           "Positive"      if avg > 0.05 else
                           "Neutral"       if avg >= -0.05 else
                           "Negative"      if avg > -0.3 else "Very Negative")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🟢 Positive", sum(s > 0.05       for s in scores))
                sc2.metric("⚪ Neutral",  sum(-0.05 <= s <= 0.05 for s in scores))
                sc3.metric("🔴 Negative", sum(s < -0.05      for s in scores))
                fig_sent = go.Figure(go.Bar(
                    x=["Positive", "Neutral", "Negative"],
                    y=[sum(s > 0.05 for s in scores),
                       sum(-0.05 <= s <= 0.05 for s in scores),
                       sum(s < -0.05 for s in scores)],
                    marker_color=["green", "gray", "red"],
                ))
                fig_sent.update_layout(title="Sentiment Distribution",
                                       template="plotly_white", height=300)
                st.plotly_chart(fig_sent, width="stretch")
                st.info(f"**Average Sentiment**: {avg:.3f} → {lbl_avg}")
 
    # ── Tab 5: Data Export ─────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Data Export")
        st.dataframe(df.tail(10), width="stretch")
 
        export_opts = st.multiselect(
            "Select data to export:",
            ["Price Data", "Technical Indicators"],
            default=["Price Data"],
        )
 
        if "Price Data" in export_opts:
            st.markdown("#### 📊 Price Data")
            price_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")
            create_download_buttons(
                price_df,
                filename_prefix=f"{ticker}_price",
                ticker=ticker,
                label="price_data",
                s3_cfg=S3_CFG,
            )
            if S3_CFG:
                st.markdown("**☁️ Upload as JSON to S3:**")
                s3_upload_section(price_df, ticker, "price_data_json", S3_CFG)
 
        if "Technical Indicators" in export_opts:
            st.markdown("#### 📈 Technical Indicators")
            excl = {"Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"}
            tech_cols = [c for c in df.columns if c not in excl]
            tech_df   = df.set_index("Date")[tech_cols]
            create_download_buttons(
                tech_df,
                filename_prefix=f"{ticker}_indicators",
                ticker=ticker,
                label="technical_indicators",
                s3_cfg=S3_CFG,
            )
 
        if S3_CFG:
            with st.expander("🗂️ View files already in S3 for this ticker"):
                try:
                    s3_client = build_s3_client(S3_CFG)
                    prefix    = f"stocks/{ticker}/"
                    response  = s3_client.list_objects_v2(
                        Bucket=S3_CFG["bucket"], Prefix=prefix
                    )
                    objects = response.get("Contents", [])
                    if objects:
                        rows = [{"Key": o["Key"],
                                 "Size (KB)": round(o["Size"] / 1024, 2),
                                 "Last Modified": o["LastModified"].strftime("%Y-%m-%d %H:%M")}
                                for o in objects]
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                    else:
                        st.info(f"No files yet at `s3://{S3_CFG['bucket']}/{prefix}`")
                except Exception as e:
                    st.warning(f"Could not list S3 objects: {e}")
 
    # ── Tab 6: Stock Info ──────────────────────────────────────────────
    with tab6:
        st.subheader("ℹ️ Company Information")
        if info:
            display_keys = [
                "longName", "sector", "industry", "country", "website",
                "fullTimeEmployees", "longBusinessSummary",
                "beta", "forwardPE", "priceToBook", "returnOnEquity",
                "debtToEquity", "currentRatio", "revenuePerShare",
            ]
            for k in display_keys:
                v = info.get(k)
                if v is not None:
                    label = k.replace("long", "").replace("full", "").replace("Time", "").strip()
                    if k == "longBusinessSummary":
                        st.markdown(f"**Description**: {v}")
                    else:
                        st.markdown(f"**{label}**: {v}")
        else:
            st.info("Company info unavailable.")


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-STOCK COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Multi-Stock Comparison":
    if len(symbols) == 1:
        st.warning("Multi-Stock mode requires ≥ 2 tickers.")
        st.stop()

    st.markdown(f"## 📈 Multi-Stock Comparison — {', '.join(tickers)}")

    with st.spinner("Loading stocks…"):
        all_data = {}
        for t, s in zip(tickers, symbols):
            df_tmp, _ = load_data(s, period)
            if df_tmp is not None:
                all_data[t] = add_technical_indicators(df_tmp)

    if not all_data:
        st.error("Could not load any stock data.")
        st.stop()

    forecast_results = {}

    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
        "📊 Price Comparison", "📈 Performance Metrics",
        "🔮 Forecast Comparison", "📋 Export Data",
    ])

    with comp_tab1:
        st.subheader("Normalised Price Comparison")
        normalize = st.checkbox("Normalise prices (% change from start)", value=True)
        fig_cmp   = go.Figure()
        for t, df_t in all_data.items():
            y = (df_t["Close"] / df_t["Close"].iloc[0] - 1) * 100 if normalize else df_t["Close"]
            fig_cmp.add_trace(go.Scatter(x=df_t["Date"], y=y, mode="lines", name=t))
        fig_cmp.update_layout(title="Price Comparison",
                               yaxis_title="% Change" if normalize else "Price (₹)",
                               template="plotly_white", height=500)
        st.plotly_chart(fig_cmp, width="stretch")
        if show_volume:
            fig_vol = go.Figure()
            for t, df_t in all_data.items():
                fig_vol.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Volume"],
                                             mode="lines", name=t))
            fig_vol.update_layout(title="Volume", template="plotly_white", height=350)
            st.plotly_chart(fig_vol, width="stretch")
        close_df = pd.DataFrame({t: df_t.set_index("Date")["Close"] for t, df_t in all_data.items()})
        corr = close_df.pct_change().dropna().corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Return Correlation Matrix")
        st.plotly_chart(fig_corr, width="stretch")

    with comp_tab2:
        st.subheader("Performance Metrics")
        perf_rows = []
        rf_daily  = 0.065 / 252
        for t, df_t in all_data.items():
            if len(df_t) < 2:
                continue
            rets      = df_t["Close"].pct_change().dropna()
            cur_p     = df_t["Close"].iloc[-1]
            first_p   = df_t["Close"].iloc[0]
            total_ret = (cur_p / first_p - 1) * 100
            vol       = rets.std() * np.sqrt(252) * 100
            sharpe    = (rets.mean() - rf_daily) / rets.std() * np.sqrt(252) if rets.std() else 0
            max_dd    = ((df_t["Close"] / df_t["Close"].cummax()) - 1).min() * 100
            perf_rows.append({
                "Stock": t, "Current (₹)": f"{cur_p:.2f}",
                "Total Return (%)": f"{total_ret:.2f}", "Ann. Vol (%)": f"{vol:.2f}",
                "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown (%)": f"{max_dd:.2f}",
                "Avg Volume": f"{df_t['Volume'].mean():,.0f}",
            })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows).set_index("Stock")
            st.dataframe(perf_df, width="stretch")
            stocks    = [r["Stock"] for r in perf_rows]
            returns   = [float(r["Total Return (%)"]) for r in perf_rows]
            vols      = [float(r["Ann. Vol (%)"]) for r in perf_rows]
            sharpes   = [float(r["Sharpe Ratio"]) for r in perf_rows]
            drawdowns = [float(r["Max Drawdown (%)"]) for r in perf_rows]
            fig_p = make_subplots(rows=2, cols=2,
                                  subplot_titles=("Total Return (%)", "Volatility (%)",
                                                  "Sharpe Ratio", "Max Drawdown (%)"))
            fig_p.add_trace(go.Bar(x=stocks, y=returns,   marker_color="seagreen"), row=1, col=1)
            fig_p.add_trace(go.Bar(x=stocks, y=vols,      marker_color="tomato"),   row=1, col=2)
            fig_p.add_trace(go.Bar(x=stocks, y=sharpes,   marker_color="steelblue"),row=2, col=1)
            fig_p.add_trace(go.Bar(x=stocks, y=drawdowns, marker_color="darkorange"),row=2, col=2)
            fig_p.update_layout(height=600, showlegend=False,
                                 title_text="Performance Overview", template="plotly_white")
            st.plotly_chart(fig_p, width="stretch")

    with comp_tab3:
        if forecast_method == "None":
            st.info("Select a forecasting method in the sidebar.")
        else:
            st.subheader(f"🔮 Forecast Comparison — {forecast_method} ({forecast_days} days)")
            fcast_perf = []
            with st.spinner(f"Generating {forecast_method} forecasts…"):
                for t, df_t in all_data.items():
                    res = run_forecast(df_t, forecast_method, forecast_days)
                    if res is not None and not res.empty:
                        forecast_results[t] = res
                        cur_p = df_t["Close"].iloc[-1]
                        fwd_p = res["Forecast"].iloc[-1]
                        fcast_perf.append({
                            "Stock": t, "Last Close (₹)": f"{cur_p:.2f}",
                            f"{forecast_days}d Forecast (₹)": f"{fwd_p:.2f}",
                            "Change (%)": f"{(fwd_p - cur_p) / cur_p * 100:+.2f}",
                            "Fcast High (₹)": f"{res['Forecast'].max():.2f}",
                            "Fcast Low (₹)":  f"{res['Forecast'].min():.2f}",
                        })
            if forecast_results:
                st.dataframe(pd.DataFrame(fcast_perf).set_index("Stock"), width="stretch")
                fig_cf = go.Figure()
                for t, res in forecast_results.items():
                    df_t = all_data[t]
                    fig_cf.add_trace(go.Scatter(x=df_t["Date"], y=df_t["Close"],
                                                name=f"{t} (Hist)", opacity=0.7))
                    fig_cf.add_trace(go.Scatter(x=res.index, y=res["Forecast"],
                                                name=f"{t} (Forecast)",
                                                line=dict(dash="dash")))
                fig_cf.update_layout(title=f"Multi-Stock {forecast_method} Forecast",
                                     xaxis_title="Date", yaxis_title="Price (₹)",
                                     height=600, template="plotly_white")
                st.plotly_chart(fig_cf, width="stretch")
            else:
                st.warning("No forecasts generated.")

    # ── Export tab (multi-stock) ───────────────────────────────────────
    with comp_tab4:
        st.subheader("📋 Combined Data Export")
        available = [t for t in tickers if t in all_data]
        if not available:
            st.error("No data available to export.")
        else:
            merged = all_data[available[0]].set_index("Date")[["Close", "Volume"]].rename(
                columns={"Close": f"{available[0]}_Close", "Volume": f"{available[0]}_Volume"})
            for t in available[1:]:
                tmp = all_data[t].set_index("Date")[["Close", "Volume"]].rename(
                    columns={"Close": f"{t}_Close", "Volume": f"{t}_Volume"})
                merged = merged.join(tmp, how="outer")

            st.dataframe(merged.tail(10), width="stretch")
            st.markdown("#### 📥 Combined Historical Data")
            create_download_buttons(
                merged,
                filename_prefix="multi_stock_data",
                ticker="MULTI",
                label="combined_price_data",
                s3_cfg=S3_CFG,
            )

            if forecast_results:
                st.markdown("#### 📥 Combined Forecast Data")
                fdf = pd.DataFrame()
                for t, res in forecast_results.items():
                    col_df = res[["Forecast"]].rename(columns={"Forecast": f"{t}_Forecast"})
                    fdf = col_df if fdf.empty else fdf.join(col_df, how="outer")
                st.dataframe(fdf.tail(10), width="stretch")
                create_download_buttons(
                    fdf,
                    filename_prefix=f"multi_forecast_{forecast_method.lower()}",
                    ticker="MULTI",
                    label=f"combined_forecast_{forecast_method.lower()}",
                    s3_cfg=S3_CFG,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Portfolio Analysis":
    if len(symbols) < 2:
        st.warning("Portfolio Analysis requires ≥ 2 stocks.")
        st.stop()

    st.markdown(f"## 💼 Portfolio Analysis — {', '.join(tickers)}")

    with st.spinner("Loading portfolio data…"):
        port_data = {}
        for t, s in zip(tickers, symbols):
            df_tmp, _ = load_data(s, period)
            if df_tmp is not None:
                port_data[t] = df_tmp.set_index("Date")["Close"]

    if len(port_data) < 2:
        st.error("Could not load enough stocks.")
        st.stop()

    close_df = pd.DataFrame(port_data).dropna()
    ret_df   = close_df.pct_change().dropna()
    n        = len(port_data)
    weights  = np.array([1 / n] * n)
    port_ret = ret_df @ weights

    port_tab1, port_tab2, port_tab3 = st.tabs(["📊 Overview", "📈 Risk & Return", "🔗 Correlations"])

    with port_tab1:
        st.subheader("Equal-Weight Portfolio Overview")
        port_value = (1 + port_ret).cumprod()
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(x=port_value.index, y=port_value, name="Portfolio",
                                    fill="tozeroy", line=dict(color="steelblue")))
        for t, ser in close_df.items():
            norm = ser / ser.iloc[0]
            fig_pv.add_trace(go.Scatter(x=norm.index, y=norm, name=t,
                                        line=dict(dash="dot"), opacity=0.7))
        fig_pv.update_layout(title="Portfolio vs Individual Stocks (Normalised)",
                              yaxis_title="Growth of ₹1", template="plotly_white", height=450)
        st.plotly_chart(fig_pv, width="stretch")
        fig_pie = px.pie(values=weights * 100, names=list(port_data.keys()),
                         title="Portfolio Allocation (Equal Weight)")
        st.plotly_chart(fig_pie, width="stretch")

        # ── Export portfolio returns with S3 ──────────────────────────
        st.markdown("#### 📥 Export Portfolio Returns")
        create_download_buttons(
            ret_df,
            filename_prefix="portfolio_returns",
            ticker="PORTFOLIO",
            label="daily_returns",
            s3_cfg=S3_CFG,
        )

    with port_tab2:
        st.subheader("Risk & Return Metrics")
        rf_daily = 0.065 / 252
        ann_ret  = port_ret.mean() * 252 * 100
        ann_vol  = port_ret.std()  * np.sqrt(252) * 100
        sharpe   = (port_ret.mean() - rf_daily) / port_ret.std() * np.sqrt(252)
        max_dd   = ((port_value / port_value.cummax()) - 1).min() * 100
        var_95   = np.percentile(port_ret, 5) * 100
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Ann. Return",     f"{ann_ret:.2f}%")
        mc2.metric("Ann. Volatility", f"{ann_vol:.2f}%")
        mc3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
        mc4.metric("Max Drawdown",    f"{max_dd:.2f}%")
        mc5.metric("VaR (95%, 1d)",   f"{var_95:.2f}%")
        ind_stats = []
        for t, ser in close_df.items():
            r = ser.pct_change().dropna()
            ind_stats.append({"Stock": t,
                               "Return": r.mean() * 252 * 100,
                               "Volatility": r.std() * np.sqrt(252) * 100,
                               "Sharpe": (r.mean() - rf_daily) / r.std() * np.sqrt(252)})
        stats_df = pd.DataFrame(ind_stats)
        fig_scatter = px.scatter(
            stats_df, x="Volatility", y="Return", text="Stock",
            size=[abs(s) + 0.1 for s in stats_df["Sharpe"]],
            color="Sharpe", color_continuous_scale="RdYlGn",
            title="Risk vs Return",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, width="stretch")

    with port_tab3:
        st.subheader("Correlation Analysis")
        corr = ret_df.corr()
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Return Correlation Matrix")
        st.plotly_chart(fig_heat, width="stretch")
        if len(tickers) >= 2:
            roll_corr = ret_df[tickers[0]].rolling(30).corr(ret_df[tickers[1]])
            fig_roll  = go.Figure()
            fig_roll.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                          name=f"{tickers[0]} vs {tickers[1]}",
                                          line=dict(color="purple")))
            fig_roll.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_roll.update_layout(
                title=f"Rolling 30-Day Correlation: {tickers[0]} vs {tickers[1]}",
                template="plotly_white", height=350)
            st.plotly_chart(fig_roll, width="stretch")
        create_download_buttons(ret_df, "portfolio_returns", "PORTFOLIO", "returns", S3_CFG)
