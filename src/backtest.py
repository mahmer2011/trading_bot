# File: src/backtest.py

import pandas as pd
import numpy as np
import talib
import pickle
from tensorflow.keras.models import load_model


def load_trained_model(model_path: str, scaler_path: str):
    """
    Load the saved Keras LSTM model and the fitted scaler (pickle).
    """
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def build_features(df: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw M5 df and H1 df, compute all training features:
      - M5 RSI, EMA_20, EMA_50, ATR
      - Candlestick patterns
      - VSA signals
      - Order‐block flags (OB_bullish, OB_bearish, Distance_to_OB)
      - H1 RSI_14, EMA_50, ATR
      - 3-class Direction3 (only for analysis/metrics, not used as an input feature)
    Returns a DataFrame with all new columns and no NaNs (drops where necessary).
    """

    # 1) M5 indicator functions should be imported from your features module
    from src.features import (
        add_RSI_EMA,
        add_ATR,
        add_candlestick_patterns,
        add_VSA_signals_refined,
        detect_order_blocks,
    )

    # Compute M5 indicators
    df = add_RSI_EMA(df, rsi_period=14, ema_periods=[20, 50])
    df = add_ATR(df, atr_period=14)
    df = add_candlestick_patterns(df)
    df = add_VSA_signals_refined(df)
    df = detect_order_blocks(df)

    # Build OB features
    df["OB_bullish"]     = (df["OB_type"] == "bullish").astype(int)
    df["OB_bearish"]     = (df["OB_type"] == "bearish").astype(int)
    df["Distance_to_OB"] = (df["Close"] - df["OB_price"]).abs() / df["Close"]
    df["Distance_to_OB"].fillna(1.0, inplace=True)

    # (Optional) Create 3-class labels—useful if you want to track model performance on “Hold”
    df["ATR_14"]          = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["next_pct_change"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["threshold"]       = 0.5 * (df["ATR_14"] / df["Close"])

    def label_with_hold(row):
        c   = row["next_pct_change"]
        thr = row["threshold"]
        if abs(c) <= thr:
            return 2
        elif c > thr:
            return 1
        else:
            return 0

    df["Direction3"] = df.apply(label_with_hold, axis=1)

    # Now merge H1 features via merge_asof (ensuring both DFs are sorted)
    df_h1 = df_h1.sort_values("time").reset_index(drop=True)
    df_h1 = add_RSI_EMA(df_h1, rsi_period=14, ema_periods=[50, 100])
    df_h1 = add_ATR(df_h1, atr_period=14)
    df_h1 = df_h1[["time", "RSI", "EMA_50", "ATR"]].rename(
        columns={"RSI": "H1_RSI", "EMA_50": "H1_EMA_50", "ATR": "H1_ATR"}
    )

    df = df.sort_values("time").reset_index(drop=True)

    df = pd.merge_asof(
        left=df,
        right=df_h1,
        left_on="time",
        right_on="time",
        direction="backward",
    )

    # Drop rows missing any essential feature
    essential = [
        "RSI", "EMA_20", "EMA_50", "ATR",
        "H1_RSI", "H1_EMA_50", "H1_ATR",
        "OB_bullish", "OB_bearish", "Distance_to_OB",
        "ATR_14", "next_pct_change", "threshold", "Direction3"
    ]
    df.dropna(subset=essential, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def generate_signals(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
    lookback: int = 120,
    thr_rsi_m5: float = 60,
    thr_rsi_h1: float = 50,
    ob_dist_threshold: float = 0.05
):
    """
    Given a fully‐featured DataFrame (as returned by build_features), generate filtered signals:
      - Scale features using the provided `scaler`
      - Build lookback sequences of length `lookback`
      - Get raw predictions from `model` (3-class or 2-class)
      - Apply technical filters (RSI + VSA + OB proximity) to produce df["Signal"] = ±1 or 0
    Returns df with two added columns: "lstm_raw_pred" and "Signal".
    """

    # 1) Build the scaled feature‐tensor X_all of shape (n_bars - lookback, lookback, num_features)
    data = df[feature_cols].values.astype(float)
    data_scaled = scaler.transform(data)
    num_samples = len(data_scaled) - lookback
    X_all = np.zeros((num_samples, lookback, len(feature_cols)), dtype=np.float32)

    for i in range(lookback, len(data_scaled)):
        X_all[i - lookback] = data_scaled[i - lookback : i, :]

    # 2) Raw predictions
    probs = model.predict(X_all, batch_size=64, verbose=0)
    if probs.ndim == 2 and probs.shape[1] == 3:
        # 3-class: take argmax
        raw_preds = np.argmax(probs, axis=1)
        conf_scores = np.max(probs, axis=1)
    else:
        # 2-class (sigmoid): threshold at 0.5
        raw_preds = (probs[:, 0] > 0.5).astype(int)
        conf_scores = probs[:, 0]

    # 3) Map raw_preds/conf_scores into the DataFrame
    df["lstm_raw_pred"]    = np.nan
    df["lstm_confidence"]  = np.nan
    df.loc[lookback:, "lstm_raw_pred"]   = raw_preds
    df.loc[lookback:, "lstm_confidence"] = conf_scores

    # 4) Initialize “Signal” column
    df["Signal"] = 0

    # 5) Loop over each index and apply filters:
    for i in range(lookback, len(df)):
        pred = int(df.at[i, "lstm_raw_pred"])
        rsi  = df.at[i, "RSI"]
        h1r  = df.at[i, "H1_RSI"]
        prev1_ns = df.at[i-1, "VSA_No_Supply"]
        prev2_ns = df.at[i-2, "VSA_No_Supply"]
        vsa_ns   = df.at[i, "VSA_No_Supply"]
        vsa_sv   = df.at[i, "VSA_Stopping_Volume"]
        vsa_nd   = df.at[i, "VSA_No_Demand"]
        vsa_bc   = df.at[i, "VSA_Buying_Climax"]
        ob_dist  = df.at[i, "Distance_to_OB"]
        is_bull  = df.at[i, "OB_bullish"] == 1
        is_bear  = df.at[i, "OB_bearish"] == 1

        # If 3-class, pred=2 means “Hold”—skip
        if pred == 1:  # LSTM says “Up”
            cond_rsi = (rsi < thr_rsi_m5) #and (h1r < thr_rsi_h1)
            cond_vsa = (vsa_ns == 1) or (vsa_sv == 1) or (prev1_ns == 1) or (prev2_ns == 1)
            cond_ob  = is_bull and (ob_dist < ob_dist_threshold)
            if cond_rsi and cond_vsa and cond_ob:
                df.at[i, "Signal"] = 1

        elif pred == 0:  # LSTM says “Down”
            cond_rsi = (rsi > thr_rsi_m5) #and (h1r > thr_rsi_h1)
            cond_vsa = (vsa_nd == 1) or (vsa_bc == 1)
            cond_ob  = is_bear and (ob_dist < ob_dist_threshold)
            if cond_rsi and cond_vsa and cond_ob:
                df.at[i, "Signal"] = -1

        # If pred == 2 or any filter fails → Signal remains 0 (no trade)

    return df


def backtest_strategy(
    df: pd.DataFrame,
    lot_size: float = 0.01,
    risk_atr_multiplier: float = 1.0
):
    """
    Given a DataFrame `df` containing at least columns:
      ["time", "Open", "High", "Low", "Close", "ATR", "Signal"]
    this function simulates trades as follows:
      - Whenever df["Signal"] shifts from 0 → +1, open a Long at the next bar’s open
      - Whenever df["Signal"] shifts from 0 → -1, open a Short at the next bar’s open
      - For each trade, SL = entry_price − (ATR * risk_atr_multiplier) if Long,
                      or entry_price + (ATR * risk_atr_multiplier) if Short
      - TP  = entry_price + 2×(distance_to_SL) if Long,
              or entry_price − 2×(distance_to_SL) if Short
      - Once SL or TP is hit, exit. If neither is hit by the close of the next 10 bars,
        exit at market close of the 10th bar.
    Returns:
      - trades_df: DataFrame of individual trades with columns
           ["entry_time","exit_time","direction","entry_price","exit_price","profit"]
      - metrics: dict with “total_trades”, “wins”, “losses”, “win_rate_%”,
                 “total_profit”, “max_drawdown”, “profit_factor”.
    """

    trades = []
    position = 0      # +1=Long open, -1=Short open, 0=no position
    entry_idx = None  # row index where the position was opened
    entry_price = None
    sl_price = None
    tp_price = None

    for i in range(1, len(df)):
        sig_today = df.at[i, "Signal"]
        sig_yest  = df.at[i - 1, "Signal"]

        # 1) Check if we need to open a new position (0→±1)
        if position == 0 and sig_yest == 0 and sig_today != 0:
            position = sig_today
            entry_idx   = i + 1 if (i + 1 < len(df)) else i  # enter at next bar’s open
            entry_time  = df.at[entry_idx, "time"]
            entry_price = df.at[entry_idx, "Open"]
            atr_val     = df.at[entry_idx, "ATR"]
            dist_SL     = atr_val * risk_atr_multiplier

            if position == 1:
                sl_price = entry_price - dist_SL
                tp_price = entry_price + 2 * dist_SL
            else:  # position == -1
                sl_price = entry_price + dist_SL
                tp_price = entry_price - 2 * dist_SL

            # We’ll track this trade until exit, so skip to next bar
            continue

        # 2) If a position is open, check for SL/TP or forced exit after 10 bars
        if position != 0:
            idx = i
            low  = df.at[idx, "Low"]
            high = df.at[idx, "High"]
            close = df.at[idx, "Close"]
            time_now = df.at[idx, "time"]

            exit_trade = False
            exit_price = None

            # Long: check TP first (more aggressive), then SL
            if position == 1:
                if high >= tp_price:
                    exit_price = tp_price
                    exit_trade = True
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_trade = True

            # Short: check TP first (price goes down), then SL
            else:  # position == -1
                if low <= tp_price:
                    exit_price = tp_price
                    exit_trade = True
                elif high >= sl_price:
                    exit_price = sl_price
                    exit_trade = True

            # 3) Forced exit after 10 bars: i ≥ entry_idx + 10
            if (not exit_trade) and (idx >= entry_idx + 10):
                exit_price = close
                exit_trade = True

            if exit_trade:
                profit = (exit_price - entry_price) * position * lot_size * 100  # e.g. XAUUSD points
                trades.append({
                    "entry_time": df.at[entry_idx, "time"],
                    "exit_time":  time_now,
                    "direction":  position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit":     profit
                })
                # Reset for next trade
                position = 0
                entry_idx = None
                entry_price = None
                sl_price = None
                tp_price = None

    trades_df = pd.DataFrame(trades)

    # Compute metrics
    total_trades = len(trades_df)
    wins         = (trades_df["profit"] > 0).sum()
    losses       = (trades_df["profit"] <= 0).sum()
    win_rate     = wins / total_trades * 100 if total_trades > 0 else 0
    total_profit = trades_df["profit"].sum()
    gross_profit = trades_df.loc[trades_df["profit"] > 0, "profit"].sum()
    gross_loss   = -trades_df.loc[trades_df["profit"] <= 0, "profit"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Compute max drawdown on equity curve
    trades_df["equity_curve"] = trades_df["profit"].cumsum()
    rolling_max = trades_df["equity_curve"].cummax()
    drawdown   = rolling_max - trades_df["equity_curve"]
    max_drawdown = drawdown.max()

    metrics = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_%": win_rate,
        "total_profit": total_profit,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown
    }

    return trades_df, metrics
