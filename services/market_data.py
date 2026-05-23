import pandas as pd
import streamlit as st
import yfinance as yf


def parse_tickers(tickers_input: str) -> list[str]:
    seen = set()
    parsed = []
    for raw_ticker in tickers_input.split(","):
        ticker = raw_ticker.strip().upper()
        if not ticker or ticker in seen:
            continue
        if not all(ch.isalnum() or ch in {".", "-", "&"} for ch in ticker):
            st.warning(f"Skipping invalid ticker: {ticker}")
            continue
        seen.add(ticker)
        parsed.append(ticker)
    return parsed


def to_nse_symbol(ticker: str) -> str:
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"


@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker: str, period: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        if df.empty:
            return None, None
        df.reset_index(inplace=True)
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None, None
