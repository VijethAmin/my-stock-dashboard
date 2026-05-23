import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    rm = df["Close"].rolling(20).mean()
    rs = df["Close"].rolling(20).std()
    df["BB_Upper"] = rm + rs * 2
    df["BB_Lower"] = rm - rs * 2
    df["BB_Middle"] = rm
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs_ratio = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs_ratio))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]
    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / denom
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()
    conditions_buy = (df["RSI"] < 35) & (df["MACD"] > df["Signal"])
    conditions_sell = (df["RSI"] > 65) & (df["MACD"] < df["Signal"])
    df["Signal_Flag"] = np.where(conditions_buy, 1, np.where(conditions_sell, -1, 0))
    return df
