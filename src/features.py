# src/features.py
import pandas as pd
import talib

def add_RSI_EMA(df: pd.DataFrame, rsi_period: int = 14, ema_periods: list = [20, 50]):
    """
    Given a DataFrame with columns [time, Open, High, Low, Close, Volume],
    compute RSI and a list of EMAs. Return DataFrame with new columns:
       'RSI', 'EMA_{period}' for each period in ema_periods.
    """
    close = df["Close"].values
    df["RSI"] = talib.RSI(close, timeperiod=rsi_period)
    for p in ema_periods:
        df[f"EMA_{p}"] = talib.EMA(close, timeperiod=p)
    return df

def add_ATR(df: pd.DataFrame, atr_period: int = 14):
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=atr_period)
    return df

def add_candlestick_patterns(df: pd.DataFrame):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    # Example: Hammer, Engulfing, Doji
    df["HAMMER"] = talib.CDLHAMMER(o, h, l, c)
    df["ENGULFING"] = talib.CDLENGULFING(o, h, l, c)
    df["DOJI"] = talib.CDLDOJI(o, h, l, c)
    # ... add more as needed
    return df

def add_VSA_signals(df: pd.DataFrame):
    """
    Add simplistic VSA signals:
      - 'Climactic Spread': when volume is > 2x average volume, and spread (High−Low) is large.
      - 'No Demand': when volume is below moving average of volume.
      - 'No Supply': similar concept, but you decide thresholds.
    For serious VSA, consult detailed sources. Below is a toy illustration.
    """
    # Calculate average volume over last N bars (e.g., 20)
    df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
    # Spread: high-low
    df["Spread"] = df["High"] - df["Low"]

    # Climactic Spread: huge volume & large spread
    df["Climactic_Spread"] = ((df["Volume"] > 2 * df["Vol_MA20"]) & (df["Spread"] > df["Spread"].rolling(20).mean())).astype(int)

    # No Demand: low volume, small spread, closes near low
    df["No_Demand"] = ((df["Volume"] < 0.5 * df["Vol_MA20"]) & (df["Close"] < df["High"] * 0.995)).astype(int)

    # No Supply: low volume, small spread, closes near high
    df["No_Supply"] = ((df["Volume"] < 0.5 * df["Vol_MA20"]) & (df["Close"] > df["Low"] * 1.005)).astype(int)

    # Clean up
    df.drop(columns=["Vol_MA20", "Spread"], inplace=True)
    return df


def add_order_blocks(df: pd.DataFrame):
    """
    Very naive: mark any candle that precedes a 3-bar consecutive close-up move,
    and whose close is the lowest of the prior 5 bars, as a bullish order block.
    Mirror for bearish. This is illustrative only.
    """
    n = len(df)
    df["Bullish_OB"] = 0
    df["Bearish_OB"] = 0
    for i in range(5, n - 3):
        window5 = df.iloc[i-5:i]["Close"]
        # If the current close is the lowest of the past 5
        if df.at[i, "Close"] == window5.min():
            # Check next 3 bars are bullish (close > open)
            if all(df.at[j, "Close"] > df.at[j, "Open"] for j in range(i+1, i+4)):
                df.at[i, "Bullish_OB"] = 1
        # Similarly for bearish
        window5_high = df.iloc[i-5:i]["Close"]
        if df.at[i, "Close"] == window5_high.max():
            if all(df.at[j, "Close"] < df.at[j, "Open"] for j in range(i+1, i+4)):
                df.at[i, "Bearish_OB"] = 1
    return df


from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_sequences(df: pd.DataFrame, feature_cols: list, label_col: str,
                     lookback: int = 60) -> (np.ndarray, np.ndarray, MinMaxScaler):
    """
    - feature_cols: list of column names to use as features (e.g., ["Close", "RSI", "EMA_20", ...]).
    - label_col: column name for target (e.g., "Direction" = 1 if next bar up else 0).
    - lookback: number of past bars to include in each sequence.
    Returns: (X, y, scaler), where
      X.shape = (num_samples, lookback, len(feature_cols)),
      y.shape = (num_samples,).
    """
    # 1. Extract feature matrix and label vector
    data = df[feature_cols].values.astype(float)
    labels = df[label_col].values.astype(int)

    # 2. Scale features to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(data_scaled) - 1):
        X.append(data_scaled[i - lookback:i, :])
        y.append(labels[i + 1])  # next bar’s label
    X, y = np.array(X), np.array(y)
    return X, y, scaler

