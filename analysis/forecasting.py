from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.metrics import compute_metrics

# ── Prophet Optional ─────────────────────────────
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# ── TensorFlow / LSTM Optional ──────────────────
HAS_LSTM = False

try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential

    HAS_LSTM = True

except ImportError:
    tf = Sequential = LSTM = Dense = Dropout = MinMaxScaler = None
    HAS_LSTM = False


# ────────────────────────────────────────────────
# PROPHET FORECAST
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_prophet(df: pd.DataFrame, periods: int):

    if not Prophet:
        st.error("Prophet is not installed.")
        return None

    try:
        fdf = df[["ds", "y"]].dropna().copy()

        fdf["ds"] = pd.to_datetime(fdf["ds"]).dt.tz_localize(None)

        if len(fdf) < 10:
            st.warning("Insufficient data for Prophet.")
            return None

        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            growth="linear"
        )

        model.fit(fdf)

        future = model.make_future_dataframe(periods=periods)

        forecast = model.predict(future)

        return (
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .rename(columns={
                "ds": "Date",
                "yhat": "Forecast",
                "yhat_lower": "Lower_CI",
                "yhat_upper": "Upper_CI"
            })
            .set_index("Date")
        )

    except Exception as e:
        st.error(f"Prophet error: {e}")
        return None


# ────────────────────────────────────────────────
# ARIMA FORECAST
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_arima(df_indexed, periods, order=(5,1,0)):

    try:
        series = df_indexed["Close"].dropna()

        if len(series) < 50:
            st.warning("ARIMA needs at least 50 data points.")
            return None

        model = ARIMA(series, order=order)
        fit = model.fit()

        forecast = fit.get_forecast(steps=periods)

        pred = forecast.predicted_mean
        conf = forecast.conf_int()

        dates = pd.date_range(
            start=series.index[-1] + timedelta(days=1),
            periods=periods
        )

        return pd.DataFrame({
            "Forecast": pred.values,
            "Lower_CI": conf.iloc[:, 0].values,
            "Upper_CI": conf.iloc[:, 1].values,
        }, index=dates)

    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return None


# ────────────────────────────────────────────────
# SARIMA FORECAST
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_sarima(
    df_indexed,
    periods,
    order=(2,1,2),
    seasonal_order=(1,1,1,12)
):

    try:
        series = df_indexed["Close"].dropna()

        if len(series) < 100:
            st.warning("SARIMA needs at least 100 data points.")
            return None

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order
        )

        fit = model.fit(disp=False)

        forecast = fit.get_forecast(steps=periods)

        pred = forecast.predicted_mean
        conf = forecast.conf_int()

        dates = pd.date_range(
            start=series.index[-1] + timedelta(days=1),
            periods=periods
        )

        return pd.DataFrame({
            "Forecast": pred.values,
            "Lower_CI": conf.iloc[:, 0].values,
            "Upper_CI": conf.iloc[:, 1].values,
        }, index=dates)

    except Exception as e:
        st.error(f"SARIMA error: {e}")
        return None


# ────────────────────────────────────────────────
# LSTM FORECAST
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_lstm(df, periods):

    if not HAS_LSTM:
        st.warning("LSTM unavailable.")
        return None

    try:
        series = df["Close"].values.reshape(-1, 1)

        if len(series) < 60:
            st.warning(f"LSTM needs ≥ 60 data points (got {len(series)}).")
            return None

        scaler = MinMaxScaler((0,1))

        scaled = scaler.fit_transform(series)

        seq_len = 60

        x = []
        y = []

        for i in range(seq_len, len(scaled)):
            x.append(scaled[i-seq_len:i, 0])
            y.append(scaled[i,0])

        x = np.array(x)
        y = np.array(y)

        x = x.reshape((x.shape[0], x.shape[1], 1))

        model = Sequential()

        model.add(LSTM(50, return_sequences=True,
                       input_shape=(x.shape[1],1)))

        model.add(Dropout(0.2))

        model.add(LSTM(50))

        model.add(Dropout(0.2))

        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        model.fit(x, y, epochs=10, batch_size=32, verbose=0)

        current = scaled[-seq_len:].reshape(1, seq_len, 1)

        preds = []

        for _ in range(periods):

            pred = model.predict(current, verbose=0)[0,0]

            preds.append(pred)

            current = np.roll(current, -1, axis=1)

            current[0,-1,0] = pred

        preds = scaler.inverse_transform(
            np.array(preds).reshape(-1,1)
        )

        dates = pd.date_range(
            start=df["Date"].max() + timedelta(days=1),
            periods=periods
        )

        return pd.DataFrame(
            preds.flatten(),
            index=dates,
            columns=["Forecast"]
        )

    except Exception as e:
        st.error(f"LSTM error: {e}")
        return None


# ────────────────────────────────────────────────
# FORECAST ROUTER
# ────────────────────────────────────────────────
def run_forecast(df, method, days):

    df2 = df.set_index("Date").copy()

    if method == "Prophet":

        return forecast_prophet(
            df.rename(columns={
                "Date": "ds",
                "Close": "y"
            }),
            days
        )

    elif method == "LSTM":

        return forecast_lstm(df, days)

    elif method == "ARIMA":

        return forecast_arima(df2, days)

    elif method == "SARIMA":

        return forecast_sarima(df2, days)

    return None


# ────────────────────────────────────────────────
# COMPARE ALL MODELS
# ────────────────────────────────────────────────
def compare_all_models(df, forecast_days):

    models = ["ARIMA", "SARIMA"]

    if Prophet:
        models.append("Prophet")

    if HAS_LSTM:
        models.append("LSTM")

    results = []
    forecasts = {}

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    actual = test_df["Close"].values

    for model in models:

        try:

            forecast = run_forecast(
                train_df,
                model,
                len(test_df)
            )

            if forecast is not None and not forecast.empty:

                pred = forecast["Forecast"].values

                min_len = min(len(actual), len(pred))

                metrics = compute_metrics(
                    actual[:min_len],
                    pred[:min_len]
                )

                forecasts[model] = forecast

                results.append({
                    "Model": model,
                    "MAE": round(metrics["MAE"], 2),
                    "RMSE": round(metrics["RMSE"], 2),
                    "MAPE": round(metrics["MAPE"], 2),
                    "R²": round(metrics["R²"], 4),
                    "Direction Accuracy (%)": round(
                        metrics["Direction Accuracy (%)"], 2
                    ),
                })

        except Exception as e:
            st.warning(f"{model} failed: {e}")

    comparison_df = pd.DataFrame(results)

    return comparison_df, forecasts


# ────────────────────────────────────────────────
# FORECAST PLOT
# ────────────────────────────────────────────────
def plot_forecast(df, result, ticker, method):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Historical",
        line=dict(color="#1f77b4")
    ))

    fig.add_trace(go.Scatter(
        x=result.index,
        y=result["Forecast"],
        name=f"{method} Forecast",
        line=dict(color="crimson", dash="dash")
    ))

    if "Lower_CI" in result.columns and "Upper_CI" in result.columns:

        dates_fwd = result.index.tolist()

        fig.add_trace(go.Scatter(
            x=dates_fwd + dates_fwd[::-1],
            y=result["Upper_CI"].tolist()
            + result["Lower_CI"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(220,20,60,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    fig.update_layout(
        title=f"{ticker} — {method} Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        template="plotly_white",
    )

    return fig