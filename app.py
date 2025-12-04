import os
import pickle
import logging

import pandas as pd
from flask import Flask, render_template, request, jsonify

from model import analyze_stock_from_csv

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# ---------------------- Fundamentals Loader ----------------------

def load_fundamentals(path="fundamentals.pkl"):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


FUNDAMENTALS = load_fundamentals()


def get_fundamentals_for_symbol(symbol: str):
    if not symbol:
        return {}

    symbol = symbol.strip().upper()
    base = symbol.split(".")[0]

    if symbol in FUNDAMENTALS:
        return FUNDAMENTALS[symbol]
    if base in FUNDAMENTALS:
        return FUNDAMENTALS[base]

    for k in FUNDAMENTALS.keys():
        ks = str(k).strip().upper()
        if ks == symbol or ks == base or base in ks:
            return FUNDAMENTALS[k]

    return {}


def safe_float(val, default=None):
    try:
        return float(val) if val is not None else default
    except:
        return default


def build_warnings(fund):
    w = []
    if not fund:
        return ["⚠ Fundamentals not available."]

    mc = safe_float(fund.get("Market Cap"))
    tpe = safe_float(fund.get("Trailing P/E Ratio"))
    ipe = safe_float(fund.get("Industry P/E"))
    sg = safe_float(fund.get("Sales Growth"))

    if mc and mc < 5_000_000_000:
        w.append("⚠ Low market cap.")
    if tpe and ipe and tpe > ipe:
        w.append("⚠ High P/E vs Industry.")
    if sg is not None and sg < 0:
        w.append("⚠ Negative sales growth.")

    return w


# ---------------------- Routes ----------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    stock = request.form.get("stock", "").strip().upper()
    if not stock:
        return jsonify({"error": "No stock symbol provided"}), 200

    try:
        result = analyze_stock_from_csv(stock)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 200
    except Exception:
        return jsonify({"error": "Internal error. Try again later."}), 200

    resolved = result.get("stock", stock)
    fund = get_fundamentals_for_symbol(resolved)
    warnings = build_warnings(fund)

    return jsonify({
        "stock": resolved,
        "warnings": warnings,
        "fundamentals": fund,
        "past_chart": result["past_chart"],
        "accuracy_chart": result["accuracy_chart"],
        "future_chart": result["future_chart"],
    }), 200


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")


@app.route("/trade-calculator")
def trade_calculator():
    return render_template("trade_calculator.html")


# ---------------------- Portfolio Simulator ----------------------

@app.route("/simulate_portfolio", methods=["POST"])
def simulate_portfolio():
    data = request.get_json()
    total = float(data.get("total", 0))
    stocks = [s.strip().upper() for s in data.get("stocks", [])]

    if not (2 <= len(stocks) <= 6):
        return jsonify({"error": "Enter between 2–6 stocks."}), 200
    if total <= 5000:
        return jsonify({"error": "Invalid investment amount."}), 200

    try:
        df = pd.read_csv("datasets/portfolio2.csv")
    except Exception as e:
        return jsonify({"error": f"CSV error: {e}"}), 200

    df = df[df["Stock"].isin(stocks)]
    if df.empty:
        return jsonify({"error": "Stocks not found in CSV."}), 200

    df["Score"] = 0
    df.loc[df["Market Cap Category"].str.lower() == "largecap", "Score"] += 3
    df.loc[df["Market Cap Category"].str.lower() == "midcap", "Score"] += 2
    df.loc[df["Market Cap Category"].str.lower() == "smallcap", "Score"] += 1
    df.loc[df["PE"] < df["Industry PE"], "Score"] += 1
    df.loc[df["Promoter Holding"] > 50, "Score"] += 1

    if "Debt/Equity" in df.columns:
        df.loc[df["Debt/Equity"] < 1, "Score"] += 1
    if "Profit Margin" in df.columns:
        df.loc[df["Profit Margin"] > 10, "Score"] += 1
    if "EPS Growth" in df.columns:
        df.loc[df["EPS Growth"] > 0, "Score"] += 1

    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    weight_map = {
        2: [70, 30],
        3: [50, 30, 20],
        4: [40, 30, 20, 10],
        5: [35, 25, 20, 15, 5],
        6: [35, 25, 20, 13, 5, 2],
    }

    if len(df) not in weight_map:
        return jsonify({"error": "Weight mapping missing."}), 200

    df["Weight"] = weight_map[len(df)]
    df["Amount"] = (df["Weight"] / 100) * total

    return jsonify({
        "allocations": df[["Stock", "Score", "Weight", "Amount"]]
            .round(2)
            .to_dict(orient="records")
    }), 200


# ---------------------- Run ----------------------

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
