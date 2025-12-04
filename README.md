
<p align="center">
  <img src="https://img.shields.io/badge/Project-AI_Driven_Stock_Selection-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask">
</p>

# **AI-Driven Stock Selection & Portfolio Optimization System**

A complete machine learning + Flask system for stock analysis, forecasting, fundamentals evaluation, and portfolio optimization.

---

## **📌 1. Project Overview**

### **Core Features**
- Load historical stock CSVs  
- Compute technical indicators  
- Compare stock performance with gold (10-year)  
- Attention-BiLSTM accuracy model  
- LSTM 24-month forecasting  
- Generate Past / Accuracy / Future charts  
- Load fundamentals from fundamentals.pkl  
- Warning system (High PE, Low Growth, Low Market Cap)  
- Portfolio allocation (2–6 stocks)  
- Trade calculator  
- Integrated HTML / CSS / JavaScript frontend  

---

## **🛠️ 2. Technologies Used**

### Backend
- Python  
- TensorFlow  
- Keras  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- Flask  

### Frontend
- HTML5  
- CSS3  
- JavaScript  

---

## **⚙️ 3. Installation**

```bash
git clone https://github.com/Manoj-1022/AI-Driven-Stock-Selection
cd AI-Driven-Stock-Selection

pip install -r requirements.txt
python app.py
````

---

## 📂 **4. Project Folder Structure**

```
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
```

---

## 📑 **5. Dataset Formats**

### Stock CSV Format

```
Date,Open,High,Low,Close,Volume
2015-01-01,800,820,795,815,1200000
```

### Gold CSV Format

```
Date,Close
02-01-2015,1186
05-01-2015,1203.90
```

---

## 🔄 **6. System Workflow**

* Load stock CSV
* Compute indicators (EMA, SMA, RSI, MACD, Volatility)
* Generate 10-year historical chart
* Load & align gold data
* Train Attention-BiLSTM (accuracy)
* Train LSTM (24-month forecast)
* Inverse-transform predictions
* Convert charts to Base64
* Load fundamentals → warnings
* Score stocks → allocate portfolio weights

---

## 🌐 **7. Flask Endpoints**

```bash
GET    /                   # Home page
POST   /predict            # Run ML + return charts + fundamentals
GET    /portfolio          # Portfolio UI
POST   /simulate_portfolio # Portfolio weight allocation engine
GET    /trade-calculator   # Profit/Loss calculator
```

---

## 📊 **8. Accuracy Metrics**

### RMSE

* Measures absolute error
* Lower = better

### MAPE

* < 5% excellent
* 5–10% good
* 10–20% moderate
* > 20% poor

Notes:

* RMSE increases with stock price
* MAPE is better for comparing stocks
* Accuracy applies only to test data

---

## ⚠️ **9. Limitations**

* Requires 200–300+ rows
* Not suitable for intraday/minute data
* LSTM struggles during extreme volatility
* Not financial advice

---

## 👤 **10. Author Information**

**Duggina Manoj Kumar**
📧 Email: **[manojduggina39@gmail.com](mailto:manojduggina39@gmail.com)**

Project: **AI-Driven Stock Selection & Portfolio Optimization**

