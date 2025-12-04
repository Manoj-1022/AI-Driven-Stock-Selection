
# AI-Driven Stock Selection & Portfolio Optimization System
<p align="center">
  <img src="https://img.shields.io/badge/Project-AI_Driven_Stock_Selection-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
</p>


## 1. Project Overview

Core Features:
- Load historical stock CSVs
- Compute technical indicators
- Compare stock performance with gold (10-year)
- Train Attention-BiLSTM accuracy model
- Train LSTM model for 24-month forecasting
- Generate Past / Accuracy / Future charts
- Load fundamentals from fundamentals.pkl
- Warning system (High PE, Low Growth, Low Market Cap)
- Portfolio allocation (2–6 stocks)
- Trade calculator tool
- HTML / CSS / JavaScript frontend

## 2. Technologies Used

Backend:
- Python
- TensorFlow
- Keras
- pandas
- numpy
- scikit-learn
- matplotlib
- Flask

Frontend:
- HTML5
- CSS3
- JavaScript

## 3. Installation

git clone https://github.com/Manoj-1022/AI-Driven-Stock-Selection 
cd <project-folder>

pip install -r requirements.txt
python app.py

## 4. Project Folder Structure

AI-Driven-Stock-Selection/
├── app.py
├── model.py
├── fundamentals.pkl
├── requirements.txt
├── datasets/
│   ├── STOCK.csv
│   ├── STOCK_clean.csv
│   ├── portfolio2.csv
│   └── gold.csv
├── templates/
│   ├── home.html
│   ├── test.html
│   ├── portfolio.html
│   └── trade_calculator.html
└── static/
    ├── css/
    │   └── style.css
    ├── images/
    │   └── Logo.png
    └── pdfs/

## 5. Dataset Formats

Stock CSV Format:
Date,Open,High,Low,Close,Volume
2015-01-01,800,820,795,815,1200000

Required Columns:
Open, High, Low, Close, Volume

Gold CSV Format:
Date,Close
02-01-2015,1186
05-01-2015,1203.90

Notes:
- dd-mm-YYYY supported
- Parsed using dayfirst=True
- Reindexed to match stock dates
- Monthly-smoothed for chart use

## 6. System Workflow

1. Load stock CSV
2. Calculate indicators (EMA, SMA, RSI, MACD, Volatility)
3. Generate historical performance chart
4. Load & align gold data
5. Train Attention-BiLSTM accuracy model
6. Train LSTM 24-month forecast model
7. Inverse-transform predictions
8. Convert charts to Base64
9. Load fundamentals & generate warnings
10. Portfolio scoring & allocation

## 7. Flask Endpoints

GET    /                  
POST   /predict               
GET    /portfolio             
POST   /simulate_portfolio    
GET    /trade-calculator      

## 8. Accuracy Metrics

RMSE:
- Measures absolute error
- Lower is better

MAPE:
- Measures percentage error
- < 5% excellent
- 5–10% good
- 10–20% moderate
- > 20% poor

Notes:
- RMSE increases with stock price
- MAPE better for comparing different stocks
- Accuracy only applies to test data

## 9. Limitations

- Requires at least 200–300 rows
- Not suitable for intraday/minute data
- LSTM struggles during high volatility
- Not financial advice

## 10. Future Improvements

- Transformer-based model
- News sentiment scoring
- Live NSE/MCX data feed
- AWS deployment
- Explainability using SHAP/LIME

## 11. Author Information

*** Duggina Manoj Kumar ***
contact: manojduggina39@gmail.com

Project:
AI-Driven Stock Selection & Portfolio Optimization

