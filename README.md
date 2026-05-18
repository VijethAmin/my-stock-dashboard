# Indian Stock Market Dashboard

An interactive Streamlit dashboard for NSE stock analysis, forecasting, sentiment analysis, portfolio comparison, and data export.

## Overview

This project provides a single web dashboard for investors, traders, and analysts to monitor Indian stock market data. It uses Yahoo Finance data through `yfinance`, interactive Plotly charts, technical indicators, news sentiment analysis, and multiple forecasting models.

## Features

- Real-time and historical NSE stock data from Yahoo Finance
- Single-stock detailed analysis
- Multi-stock comparison
- Equal-weight portfolio analysis
- Candlestick and line charts
- Technical indicators:
  - SMA
  - EMA
  - RSI
  - MACD
  - Bollinger Bands
  - Stochastic Oscillator
  - ATR
- Forecasting models:
  - ARIMA
  - SARIMA
  - Prophet
  - LSTM
  - Compare All Models
- Model evaluation metrics:
  - MAE
  - RMSE
  - MAPE
  - R²
  - Direction Accuracy
- News and sentiment analysis using NLTK VADER
- CSV and Excel downloads
- Optional AWS S3 export
- Optional auto-refresh

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── etl.py
├── test_db.py
├── analysis/
│   ├── forecasting.py
│   ├── indicators.py
│   └── metrics.py
├── services/
│   ├── market_data.py
│   ├── news.py
│   └── s3.py
├── ui/
│   └── downloads.py
├── csv_data/
└── .streamlit/
```

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- NumPy
- Plotly
- scikit-learn
- statsmodels
- Prophet
- TensorFlow / Keras
- NLTK
- BeautifulSoup
- requests
- xlsxwriter
- boto3

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Sidebar Options

The dashboard sidebar lets you:

- Enter comma-separated NSE ticker symbols
- Choose presets such as banking, IT, auto, and large-cap stocks
- Select analysis mode:
  - Single Stock Analysis
  - Multi-Stock Comparison
  - Portfolio Analysis
- Select a forecasting method
- Set forecast horizon from 5 to 365 days
- Choose historical data period
- Enable or disable volume charts
- Enable or disable news and sentiment
- Configure optional auto-refresh

## AWS S3 Export

S3 export is optional. Add AWS settings to `.streamlit/secrets.toml`:

```toml
[aws]
access_key_id = "YOUR_ACCESS_KEY"
secret_access_key = "YOUR_SECRET_KEY"
region = "ap-south-1"
bucket_name = "your-bucket-name"
```

The app also supports `bucket` instead of `bucket_name`.

## Methodology

1. Fetch historical data using Yahoo Finance.
2. Clean and normalize dates.
3. Calculate technical indicators.
4. Visualize historical price, volume, momentum, and volatility.
5. Run selected forecasting model.
6. Compare predicted values with actual data where possible.
7. Fetch news headlines and classify sentiment.
8. Export selected datasets locally or to S3.

## Forecasting Models

- **ARIMA:** Classical time-series model for short-term trend and autocorrelation.
- **SARIMA:** Seasonal ARIMA model for data with recurring seasonal patterns.
- **Prophet:** Additive forecasting model with trend and seasonality support.
- **LSTM:** Deep learning sequence model for non-linear price patterns.

## Sentiment Analysis

News headlines are scored using NLTK VADER:

- Score greater than `0.05`: Positive
- Score less than `-0.05`: Negative
- Score between `-0.05` and `0.05`: Neutral

## Notes

- Forecasts are for educational and analytical use only.
- The dashboard is not financial advice.
- LSTM and Prophet can be slower than ARIMA/SARIMA because they require heavier model setup.
- Internet access is required for Yahoo Finance data and news.

## Future Enhancements

- Add more data sources for improved news reliability
- Add cryptocurrency and commodities support
- Add custom portfolio weights
- Add background scheduled ETL jobs
- Add user authentication
- Add model persistence for faster forecasting
- Deploy to AWS EC2, Streamlit Cloud, or another cloud platform

## References

- Yahoo Finance
- Streamlit documentation
- Plotly documentation
- Prophet documentation
- statsmodels ARIMA/SARIMA documentation
- TensorFlow/Keras documentation
- NLTK VADER sentiment documentation
