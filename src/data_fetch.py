import MetaTrader5 as mt5
from datetime import datetime

def connect_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
    # Optionally print account info
    account_info = mt5.account_info()
    if account_info:
        print("Logged in as:", account_info.login, "@", account_info.server)
        print("Balance:", account_info.balance)
    else:
        print("Failed to retrieve account info:", mt5.last_error())
    return True

def disconnect_mt5():
    mt5.shutdown()

if __name__ == "__main__":
    if connect_mt5():
        # Fetch latest 5 bars for XAUUSD M5 as a sanity check
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 5)
        if rates is not None:
            print("Recent XAUUSD M5 bars:")
            for r in rates:
                print(datetime.fromtimestamp(r[0]), r)
        else:
            print("Failed to fetch rates:", mt5.last_error())
        disconnect_mt5()


import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# 1. Connect to MT5 (reuse connect_mt5)
def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed, code={mt5.last_error()}")
    print("MT5 initialized.")

# 2. Map string timeframes to mt5 constants
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}

def fetch_symbol_data(symbol: str, timeframe: str, n_bars: int, save_csv: bool = True, folder: str = "data"):
    """
    Fetch the last n_bars of historical OHLCV for a given symbol/timeframe.
    Save to CSV (if save_csv=True) in folder/{symbol}_{timeframe}.csv.
    """
    tf_const = TF_MAP.get(timeframe)
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    # Copy rates from current time
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {symbol} {timeframe}: {mt5.last_error()}")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"
    })[["time", "Open", "High", "Low", "Close", "Volume"]]

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)
    if save_csv:
        file_path = os.path.join(folder, f"{symbol}_{timeframe}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {symbol} {timeframe} to {file_path}")
    return df

if __name__ == "__main__":
    connect_mt5()
    symbols = ["XAUUSD", "BTCUSD", "AAPL.OQ", "TSLA.OQ", "MSFT.OQ"]
    timeframes = ["M1", "M5", "M15", "H1", "D1"]
    bars_each = 20_000   # e.g., last 20k bars (~14 days of M5)
    for sym in symbols:
        for tf in timeframes:
            try:
                fetch_symbol_data(sym, tf, bars_each)
            except Exception as e:
                print("Error fetching", sym, tf, ":", e)
    mt5.shutdown()
