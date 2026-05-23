import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker: str, max_articles: int = 10) -> list:
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
        news = []
        for item in raw_news[:max_articles]:
            title = item.get("title", "")
            link = item.get("link", "#")
            if title:
                news.append((title, link))
        if news:
            return news
    except Exception:
        pass

    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            news = []
            for a in soup.find_all("a", href=True):
                title = a.get_text(strip=True)
                href = a["href"]
                if len(title) > 30 and ("/news/" in href or "finance.yahoo" in href):
                    if not href.startswith("http"):
                        href = "https://finance.yahoo.com" + href
                    news.append((title, href))
                    if len(news) >= max_articles:
                        break
            return news
    except Exception:
        return []
    return []
