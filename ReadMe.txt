Project Report: Indian Stock Dashboard with Forecast & Sentiment Analysis
1. Project Overview

The Indian Stock Dashboard project is an interactive web application built using Streamlit, designed to provide real-time stock data, technical analysis, forecasting, and news sentiment analysis for NSE-listed stocks. It allows investors, traders, and analysts to monitor stock trends, visualize key metrics, and make informed decisions.

Key features include:

Real-time price metrics and candlestick charts.

Technical indicators like SMA, EMA, RSI, and MACD.

Stock price forecasting using Prophet, LSTM, ARIMA, and SARIMA.

Latest news scraping with sentiment analysis (positive, neutral, negative).

Data download options in CSV and Excel formats.

Single stock view or comparison mode for multiple stocks.

Auto-refresh functionality for live data monitoring.

2. Technologies & Libraries

Python 3.11+

Streamlit – For interactive web UI

yfinance – To fetch historical stock data from Yahoo Finance

Plotly – For dynamic charts (Candlestick, Line plots)

pandas & numpy – Data manipulation and computation

scikit-learn – Metrics evaluation (MAE, RMSE, MAPE)

statsmodels – ARIMA and SARIMA forecasting

Prophet – Time series forecasting

TensorFlow / Keras – LSTM-based forecasting model

nltk – Sentiment analysis using VADER

BeautifulSoup / requests – Scraping latest stock news

xlsxwriter – Export data to Excel

3. Application Features
3.1 Sidebar Input

Enter stock symbols (comma-separated).

Select mode: Single Stock or Compare Multiple.

Choose forecast method: None, Prophet, LSTM, ARIMA, SARIMA.

Set forecast horizon in days (5–365).

Auto-refresh interval in seconds (optional).

3.2 Single Stock Mode

Metrics displayed: Current price, Previous close.

Candlestick chart over 2 years with SMA20, SMA50, EMA20 overlays.

RSI and MACD charts for momentum analysis.

Forecast visualization with selected model and evaluation metrics (MAE, RMSE, MAPE).

Latest news with sentiment classification using VADER (Positive, Neutral, Negative).

Download options: CSV and Excel.

3.3 Compare Multiple Stocks

Display line chart comparing historical closing prices.

Forecasting for each stock with performance evaluation.

Data export for comparison analysis.

4. Technical Implementation
4.1 Data Acquisition

Stock data retrieved using yfinance.Ticker(symbol).history(period="2y").

Latest news scraped from Yahoo Finance using BeautifulSoup.

4.2 Technical Indicators

SMA (Simple Moving Average) 20 & 50 days.

EMA (Exponential Moving Average) 20 days.

RSI (Relative Strength Index) for overbought/oversold conditions.

MACD (Moving Average Convergence Divergence) with Signal line.

4.3 Forecast Models

Prophet – Additive time series forecasting, handles seasonality & trends.

LSTM – Deep learning sequence model for predicting future stock prices.

ARIMA – Classical time series model capturing trend and autocorrelation.

SARIMA – Seasonal ARIMA for datasets with seasonal trends.

4.4 Sentiment Analysis

VADER (Valence Aware Dictionary for Sentiment Reasoning) scores news headlines.

Sentiment thresholds:

> 0.05: Positive

< -0.05: Negative

Between -0.05 and 0.05: Neutral

4.5 Metrics Evaluation

MAE (Mean Absolute Error) – Average magnitude of prediction errors.

RMSE (Root Mean Squared Error) – Penalizes larger errors.

MAPE (%) – Mean Absolute Percentage Error, for normalized error evaluation.

5. User Interface

Built entirely with Streamlit components.

Interactive sidebar for user inputs.

Dynamic charts for historical and forecast data.

News with clickable links and sentiment badges.

Responsive layout for both single and multi-stock views.

6. Challenges & Solutions
Challenge	Solution
Handling missing data in historical stock prices	Applied .dropna() and rolling windows for indicators
Forecasting with small datasets	Used LSTM with MinMax scaling and Prophet with additive model
News scraping reliability	Handled HTTP errors and missing elements in BeautifulSoup
Multiple stock comparison & forecast	Joined dataframes, evaluated metrics individually
7. Conclusion

The Indian Stock Dashboard successfully integrates stock market analysis, forecasting, and sentiment analysis in a single interactive platform. Users can:

Monitor real-time stock performance.

Visualize technical indicators for better trading decisions.

Forecast future price trends using multiple models.

Stay updated with news sentiment affecting the stock.

This project demonstrates end-to-end data acquisition, analysis, forecasting, visualization, and reporting, making it a strong tool for traders and analysts.