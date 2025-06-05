# src/features.py
import pandas as pd
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 1. Basic Indicator Functions
# -------------------------

def add_RSI_EMA(df: pd.DataFrame, rsi_period: int = 14, ema_periods: list = [20, 50]) -> pd.DataFrame:
    """
    Compute RSI and multiple EMAs using TA-Lib.
    """
    close = df["Close"].values
    df["RSI"] = talib.RSI(close, timeperiod=rsi_period)
    for p in ema_periods:
        df[f"EMA_{p}"] = talib.EMA(close, timeperiod=p)
    return df


def add_ATR(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Compute ATR using TA-Lib.
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    df["ATR"] = talib.ATR(high, low, close, timeperiod=atr_period)
    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute selected candlestick pattern signals using TA-Lib.
    Each output is an integer: >0 bullish, <0 bearish, 0 none.
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    df["HAMMER"] = talib.CDLHAMMER(o, h, l, c)
    df["ENGULFING"] = talib.CDLENGULFING(o, h, l, c)
    df["DOJI"] = talib.CDLDOJI(o, h, l, c)
    # Add more patterns if desired
    return df

# -------------------------
# 2. Refined Volume Spread Analysis (VSA)
# -------------------------

def classify_vsa(row: pd.Series) -> str:
    """
    Classify a single bar into VSA categories:
      - 'No Demand': narrow spread up bar on low volume
      - 'No Supply': narrow spread down bar on low volume
      - 'Buying Climax': wide spread up bar on very high volume closing off high
      - 'Selling Climax': wide spread down bar on very high volume closing off low
      - 'Stopping Volume': high-volume down bar that closes near its high
      - 'Normal': none of the above
    """
    spread = row['spread']
    open_ = row['Open']
    high = row['High']
    low = row['Low']
    close = row['Close']
    volume = row['Volume']
    volume_avg = row['volume_avg']
    spread_avg = row['spread_avg']

    is_up_bar = close > open_
    is_down_bar = close < open_

    # No Demand / No Supply: narrow spread & low volume
    if spread < 0.8 * spread_avg and volume < 0.8 * volume_avg:
        if is_up_bar:
            return 'No Demand'
        elif is_down_bar:
            return 'No Supply'

    # Buying Climax / Selling Climax: wide spread & very high volume
    if spread > 1.5 * spread_avg and volume > 1.5 * volume_avg:
        # bullish candle but close is off high by some margin
        if is_up_bar and (close < high * 0.995):
            return 'Buying Climax'
        # bearish candle but close is off low by some margin
        if is_down_bar and (close > low * 1.005):
            return 'Selling Climax'

    # Stopping Volume: high-volume down bar closing near high
    if is_down_bar and volume > 1.3 * volume_avg and (close > open_ * 1.0):
        return 'Stopping Volume'

    return 'Normal'


def add_VSA_signals_refined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add refined VSA signals to DataFrame. Columns added:
      - spread, volume_avg, spread_avg
      - vsa_signal (categorical string)
      - One-hot columns for each VSA type
    """
    df = df.copy()
    # 1. Compute spread and rolling averages
    df['spread'] = df['High'] - df['Low']
    df['volume_avg'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['spread_avg'] = df['spread'].rolling(window=20, min_periods=1).mean()

    # 2. Classify each bar
    df['vsa_signal'] = df.apply(classify_vsa, axis=1)

    # 3. One-hot encode categories
    vsa_types = ['No Demand', 'No Supply', 'Buying Climax', 'Selling Climax', 'Stopping Volume']
    for vt in vsa_types:
        df[f"VSA_{vt.replace(' ', '_')}"] = (df['vsa_signal'] == vt).astype(int)

    # 4. Drop intermediate columns if not needed
    df.drop(columns=['spread', 'volume_avg', 'spread_avg'], inplace=True)
    return df

# -------------------------
# 3. Refined Order Block Detection
# -------------------------

def is_strong_bearish_move(df: pd.DataFrame, i: int, body_factor: float = 1.5) -> bool:
    """
    Detect if the candle at index i is a strong bearish move:
      - Body size > body_factor * avg spread
      - Next candle is also bearish and strong
    """
    spread_avg = df['High'].rolling(window=20, min_periods=1).mean()
    avg_spread = spread_avg.iloc[i] if i < len(spread_avg) else spread_avg.iloc[-1]

    open_i = df.at[i, 'Open']
    close_i = df.at[i, 'Close']
    body_i = open_i - close_i  # positive if bearish

    open_next = df.at[i+1, 'Open']
    close_next = df.at[i+1, 'Close']
    body_next = open_next - close_next  # positive if bearish

    return (body_i > body_factor * avg_spread) and (body_next > body_factor * avg_spread)


def is_strong_bullish_move(df: pd.DataFrame, i: int, body_factor: float = 1.5) -> bool:
    """
    Detect if the candle at index i is a strong bullish move:
      - Body size > body_factor * avg spread
      - Next candle is also bullish and strong
    """
    spread_avg = df['High'].rolling(window=20, min_periods=1).mean()
    avg_spread = spread_avg.iloc[i] if i < len(spread_avg) else spread_avg.iloc[-1]

    open_i = df.at[i, 'Open']
    close_i = df.at[i, 'Close']
    body_i = close_i - open_i  # positive if bullish

    open_next = df.at[i+1, 'Open']
    close_next = df.at[i+1, 'Close']
    body_next = close_next - open_next  # positive if bullish

    return (body_i > body_factor * avg_spread) and (body_next > body_factor * avg_spread)


def detect_order_blocks(df: pd.DataFrame, body_factor: float = 1.5) -> pd.DataFrame:
    """
    Identify bullish and bearish order blocks:
      - Record the high (for bearish OB) or low (for bullish OB) of the candle at i
      - Only if bar i and i+1 form a strong directional move
      - Returns DataFrame with columns:
          OB_type: 'bullish' or 'bearish' (NaN if none)
          OB_price: price level of order block (NaN if none)
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    df['OB_type'] = np.nan
    df['OB_price'] = np.nan

    for i in range(n - 1):
        # Make sure we have enough future data
        if i + 1 >= n:
            break
        if is_strong_bearish_move(df, i, body_factor):
            # Bearish OB: high of candle i
            df.at[i, 'OB_type'] = 'bearish'
            df.at[i, 'OB_price'] = df.at[i, 'High']
        elif is_strong_bullish_move(df, i, body_factor):
            # Bullish OB: low of candle i
            df.at[i, 'OB_type'] = 'bullish'
            df.at[i, 'OB_price'] = df.at[i, 'Low']
    return df

# -------------------------
# 4. Sequence Creation for LSTM
# -------------------------

def create_sequences(df: pd.DataFrame, feature_cols: list, label_col: str, lookback: int = 60):
    """
    Build normalized sequences for LSTM input.
    - feature_cols: list of feature column names
    - label_col: name of target column (binary or categorical integer)
    - lookback: number of past bars per sequence
    Returns: X (num_samples, lookback, num_features), y (num_samples,), scaler (fitted MinMaxScaler)
    """
    # Extract feature matrix and labels
    data = df[feature_cols].values.astype(float)
    labels = df[label_col].values.astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(data_scaled) - 1):
        X.append(data_scaled[i - lookback:i, :])
        y.append(labels[i + 1])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler
