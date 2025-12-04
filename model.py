import os
import io
import base64
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib

# Use headless backend only when imported by Flask
if __name__ != "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    Input as KInput,
    Dense as KDense,
    Dropout as KDropout,
    LSTM as KLSTM,
    Bidirectional,
    Layer,
)

ATT_EPOCHS = 25
FORECAST_EPOCHS = 10

GOLD_DF = None

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "datasets"


# -------------------- Attention Layer --------------------

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            name="att_weight",
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            name="att_bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        e = tf.squeeze(tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b), -1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, -1)
        context = tf.reduce_sum(inputs * alpha, axis=1)
        return context


# -------------------- Utils --------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    return base64.b64encode(img_bytes).decode("utf-8")


def make_sequences(X, y, seq_len=60):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# -------------------- Indicators --------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].copy()

    df["EMA10"] = close.ewm(span=10).mean()
    df["EMA20"] = close.ewm(span=20).mean()
    df["EMA50"] = close.ewm(span=50).mean()

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / 14).mean()
    roll_down = down.ewm(alpha=1 / 14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["MACD"] = macd
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = macd - signal

    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std()
    df["Volatility_20"] = df["Returns"].rolling(20).std()

    return df


# -------------------- Models --------------------

def build_attention_model(seq_len, features):
    inp = KInput(shape=(seq_len, features))
    x = Bidirectional(KLSTM(64, return_sequences=True))(inp)
    x = Bidirectional(KLSTM(32, return_sequences=True))(x)
    x = Attention()(x)
    x = KDense(64, activation="relu")(x)
    x = KDropout(0.2)(x)
    out = KDense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_univariate_lstm(seq_len):
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64),
        Dropout(0.1),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# -------------------- CSV Handling --------------------

def find_csv_for_stock(stock: str):
    sym = stock.upper().strip()

    base_syms = [sym]
    if not sym.endswith(".NS") and not sym.endswith(".BO"):
        base_syms.append(sym + ".NS")
        base_syms.append(sym + ".BO")

    candidates = []
    for s in base_syms:
        candidates.append((s, os.path.join(DATA_DIR, f"{s}_clean.csv")))
        candidates.append((s, os.path.join(DATA_DIR, f"{s}.csv")))

    for resolved_sym, path in candidates:
        if os.path.exists(path):
            print(f"[CSV] Using {path} for symbol {resolved_sym}")
            return resolved_sym, path

    raise FileNotFoundError(
        f"No CSV found for {stock}. Expected something like "
        f"'datasets/{sym}.csv' or 'datasets/{sym}_clean.csv'."
    )


def load_stock_csv(resolved_symbol: str, csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{resolved_symbol}: missing columns {missing} in {csv_path}")

    df = df[required].copy()
    df = df.sort_index()
    df = df.dropna(how="any")

    if len(df) < 200:
        raise RuntimeError(f"{resolved_symbol}: not enough rows in CSV ({len(df)})")

    return df


# -------------------- Gold Handling --------------------

def load_gold_df():
    global GOLD_DF

    if GOLD_DF is not None:
        try:
            print(
                f"[GOLD] Using cached df, rows={len(GOLD_DF)}, "
                f"range {GOLD_DF.index.min().date()} -> {GOLD_DF.index.max().date()}"
            )
        except Exception:
            print(f"[GOLD] Using cached df, rows={len(GOLD_DF)}")
        return GOLD_DF

    candidates = [
        os.path.join(DATA_DIR, "gold_clean.csv"),
        os.path.join(DATA_DIR, "gold.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"[GOLD] Reading {path}")
            df = pd.read_csv(path)

            df.columns = df.columns.str.strip()

            if "Date" not in df.columns or "Close" not in df.columns:
                print(f"[GOLD] Missing Date/Close in {path}, columns={df.columns}")
                continue

            df["Date"] = df["Date"].astype(str).str.strip()
            df["Close"] = df["Close"].astype(str).str.strip()

            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

            df = df.dropna(subset=["Date", "Close"])
            if df.empty:
                print(f"[GOLD] After cleaning, no valid rows in {path}")
                continue

            df = df.set_index("Date")
            df = df[["Close"]].sort_index()

            GOLD_DF = df
            print(
                f"[GOLD] Loaded gold data from {path}, rows={len(df)}, "
                f"range {df.index.min().date()} -> {df.index.max().date()}"
            )
            return GOLD_DF

    print("[GOLD] No usable gold.csv found in datasets/. Skipping gold overlay.")
    GOLD_DF = None
    return GOLD_DF


# -------------------- Attention Accuracy Model --------------------

def train_attention_accuracy_model(
    resolved_symbol: str,
    price_df: pd.DataFrame,
    debug_show: bool = False,
):
    ind_df = price_df[["Close"]].copy()
    ind_df = compute_indicators(ind_df).dropna()

    feature_cols = [
        "Close", "EMA10", "EMA20", "EMA50", "RSI14",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "Volatility_10", "Volatility_20",
    ]

    X = ind_df[feature_cols].values.astype(np.float32)
    y = ind_df["Close"].values.astype(np.float32).reshape(-1, 1)

    if len(ind_df) < 200:
        raise RuntimeError(f"{resolved_symbol}: not enough indicator rows ({len(ind_df)})")

    split = int(len(ind_df) * 0.8)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_scaler.fit(X[:split])
    Y_scaler.fit(y[:split])

    X_scaled = X_scaler.transform(X)
    y_scaled = Y_scaler.transform(y)

    seq_len = 60
    X_seq, y_seq = make_sequences(X_scaled, y_scaled, seq_len=seq_len)
    if len(X_seq) < 100:
        raise RuntimeError(f"{resolved_symbol}: not enough sequences ({len(X_seq)}) for attention model")

    s2 = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:s2], X_seq[s2:]
    y_train, y_test = y_seq[:s2], y_seq[s2:]

    model = build_attention_model(seq_len, X_train.shape[-1])
    model.fit(X_train, y_train, epochs=ATT_EPOCHS, batch_size=32, verbose=True)

    preds = model.predict(X_test)
    y_test_inv = Y_scaler.inverse_transform(y_test)
    preds_inv = Y_scaler.inverse_transform(preds)

    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    mape = mean_absolute_percentage_error(y_test_inv, preds_inv) * 100.0

    test_dates = ind_df.index[-len(y_test_inv):]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(test_dates, y_test_inv, label="Actual", linewidth=1)
    ax.plot(test_dates, preds_inv, label="Predicted", linewidth=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.title(f"{resolved_symbol} - Actual vs Predicted | RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    plt.xlabel("Date (Month-Year)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    if debug_show:
        fig.show()

    accuracy_b64 = fig_to_base64(fig)
    return accuracy_b64, float(rmse), float(mape)


# -------------------- Main Analysis --------------------

def analyze_stock_from_csv(stock: str, debug_show: bool = False):
    resolved_symbol, csv_path = find_csv_for_stock(stock)
    df = load_stock_csv(resolved_symbol, csv_path)

    last_n = 10 * 252  # approx 10 years of trading days
    if len(df) > last_n:
        df_recent = df.iloc[-last_n:].copy()
    else:
        df_recent = df.copy()

    close = df_recent["Close"]
    df_recent["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df_recent["EMA_200"] = close.ewm(span=200, adjust=False).mean()
    df_recent["SMA_100"] = close.rolling(window=100).mean()

    fig_past, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_recent.index, df_recent["Close"], label=f"{resolved_symbol} Close", linewidth=1)
    ax.plot(df_recent.index, df_recent["EMA_50"], label="EMA 50", linewidth=1)
    ax.plot(df_recent.index, df_recent["EMA_200"], label="EMA 200", linewidth=1)
    ax.plot(df_recent.index, df_recent["SMA_100"], label="SMA 100", linewidth=1)

    gold_df = load_gold_df()
    print("[GOLD] load_gold_df() returned:",
          "None" if gold_df is None else f"{len(gold_df)} rows")

    if gold_df is not None and not gold_df.empty:
        start_date = df_recent.index.min().normalize()
        end_date = df_recent.index.max().normalize()

        gold_recent = gold_df[(gold_df.index >= start_date) & (gold_df.index <= end_date)].copy()

        print(
            f"[GOLD] stock range {start_date.date()} -> {end_date.date()}, "
            f"gold rows in range = {len(gold_recent)}"
        )

        if not gold_recent.empty:
            gold_monthly = gold_recent["Close"].resample("M").mean()
            idx = df_recent.index
            gold_aligned = gold_monthly.reindex(idx).interpolate("time").ffill().bfill()

            ax2 = ax.twinx()
            ax2.plot(
                idx,
                gold_aligned,
                color="#FFD700",
                linestyle="-",
                linewidth=2.0,
                label="Gold Price",
            )
            ax2.set_ylabel("Gold Price")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            print("[GOLD] gold_recent is EMPTY after date filter – no overlay.")
            ax.legend(loc="upper left")
    else:
        print("[GOLD] gold_df is None or empty – skipping overlay.")
        ax.legend(loc="upper left")

    ax.set_title(f"{resolved_symbol} vs Gold - Last 10 Years")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)

    if debug_show:
        fig_past.show()

    past_b64 = fig_to_base64(fig_past)

    accuracy_b64, rmse, mape = train_attention_accuracy_model(
        resolved_symbol, df, debug_show=debug_show
    )

    prices = df["Close"].values.astype("float32").reshape(-1, 1)

    seq_len = 60
    if len(prices) < seq_len + 50:
        raise RuntimeError(
            f"{resolved_symbol}: not enough data for seq_len={seq_len} (len={len(prices)})"
        )

    scaler_close = MinMaxScaler()
    prices_scaled = scaler_close.fit_transform(prices)

    X_seq, y_seq = make_sequences(prices_scaled, prices_scaled, seq_len)
    if len(X_seq) < 100:
        raise RuntimeError(
            f"{resolved_symbol}: not enough sequences ({len(X_seq)}) for forecast model"
        )

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    forecast_model = build_univariate_lstm(seq_len)
    forecast_model.fit(
        X_train,
        y_train,
        epochs=FORECAST_EPOCHS,
        batch_size=32,
        verbose=True,
    )

    last_seq = prices_scaled[-seq_len:].copy()
    future_scaled = []
    for _ in range(24):
        pred_scaled = forecast_model.predict(last_seq.reshape(1, seq_len, 1), verbose=True)[0, 0]
        future_scaled.append(pred_scaled)
        last_seq = np.vstack([last_seq[1:], [[pred_scaled]]])

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_prices = scaler_close.inverse_transform(future_scaled).flatten()

    future_dates = pd.date_range(start=df.index[-1], periods=24, freq="ME")

    fig_future, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(future_dates, future_prices, marker="o", linestyle="-", label="Predicted Price")
    plt.setp(ax3.get_xticklabels(), rotation=45)
    ax3.set_title(f"{resolved_symbol} - 24-Month Forecast")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.grid(True)
    ax3.legend()

    if debug_show:
        fig_future.show()

    future_b64 = fig_to_base64(fig_future)

    return {
        "stock": resolved_symbol,
        "past_chart": past_b64,
        "accuracy_chart": accuracy_b64,
        "future_chart": future_b64,
    }


if __name__ == "__main__":
    out = analyze_stock_from_csv("HDFCBANK.NS", debug_show=True)
    print("OK keys:", out.keys())
