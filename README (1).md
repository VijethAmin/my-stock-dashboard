# 📊 Cloud-Based Stock Market Dashboard

## 🚀 Overview

A **cloud-deployed stock analytics dashboard** built using Streamlit, integrated with AWS services for scalable data processing and storage.

This project enables users to:

* Analyze Indian stock market data 📈
* Perform forecasting using ML models 🤖
* Export and store results directly to AWS S3 ☁️

---

## 🌐 Live Demo

👉 http://65.2.34.254:8501

---

## 🏗️ Architecture

User → Streamlit App (EC2) → AWS S3 (Storage)

* **Frontend & Backend**: Streamlit
* **Cloud Hosting**: AWS EC2
* **Storage**: AWS S3
* **Authentication**: IAM Role-based access (no credentials exposed)

---

## ✨ Features

### 📈 Stock Analysis

* Real-time stock data using yfinance
* Candlestick charts and technical indicators
* Multi-stock comparison

### 🔮 Forecasting Models

* ARIMA
* SARIMA
* Prophet
* LSTM

### 📰 Sentiment Analysis

* News scraping from Yahoo Finance
* Sentiment scoring using NLTK

### 📊 Data Export

* Download data as CSV / Excel
* Upload results directly to AWS S3

### ☁️ Cloud Integration

* Secure S3 upload using IAM role
* No hardcoded credentials

---

## 🧰 Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Plotly
* Scikit-learn
* Statsmodels
* TensorFlow (optional)
* AWS (EC2, S3, IAM)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/VijethAmin/my-stock-dashboard.git
cd my-stock-dashboard
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## ☁️ AWS Setup (Important)

* Create S3 bucket
* Create IAM role with `AmazonS3FullAccess`
* Attach IAM role to EC2 instance

👉 The app automatically uses IAM (no keys required)

---

## 📂 Project Structure

```
my-stock-dashboard/
│
├── app.py
├── venv/
├── requirements.txt
├── log.txt
└── README.md
```

---

## 📈 Future Enhancements

* Add user authentication
* Deploy using Docker
* Integrate RDS for database storage
* Add real-time streaming data

---

## 👨‍💻 Author

**Vijeth Amin**

* GitHub: https://github.com/VijethAmin

---

## ⭐ Acknowledgements

* Yahoo Finance API (yfinance)
* AWS Cloud Services
* Open-source Python libraries

---

## ⚠️ Disclaimer

This project is for educational purposes only and not financial advice.
