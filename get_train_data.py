import pandas as pd
import numpy as np
import requests
import time
import joblib
from sklearn.preprocessing import StandardScaler
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

market = config.get("market", {}) or {}
provider = str(market.get("provider", "okx")).lower()
symbol = config.get("symbol", "BTCUSDT")
interval = config.get("interval", "1m")
lookback_hours = int(config.get("lookback_hours", 48))
symbol_map = market.get("symbol_map", {}) or {}


def _ema(s: pd.Series, span: int = 14):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14):
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(s: pd.Series, fast: int = 12, slow: int = 26):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    return ef - es


def fetch_klines_binance(sym: str):
    print("📥 从 Binance 拉取历史K线...")
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_hours * 60 * 60 * 1000
    data = []
    while start_time < end_time:
        params = {"symbol": sym.upper(), "interval": interval, "startTime": start_time, "endTime": end_time, "limit": 1000}
        batch = requests.get(url, params=params, timeout=15).json()
        if not batch:
            break
        data.extend(batch)
        start_time = int(batch[-1][0]) + 1
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "tb_base"]:
        df[c] = df[c].astype(float)
    return df


def fetch_klines_okx(inst_id: str):
    print("📥 从 OKX 拉取历史K线...")
    # docs: /api/v5/market/history-candles
    bar_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H"}
    bar = bar_map.get(interval, "1m")
    url = "https://www.okx.com/api/v5/market/history-candles"
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - lookback_hours * 60 * 60 * 1000

    rows = []
    after = None
    for _ in range(120):  # 防无限循环
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)
        r = requests.get(url, params=params, timeout=15).json()
        data = (r or {}).get("data", [])
        if not data:
            break
        # OKX data: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
        rows.extend(data)
        oldest = int(data[-1][0])
        if oldest <= start_ts:
            break
        after = oldest

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    # OKX 没有 taker_buy_base，用0占位
    df["tb_base"] = 0.0
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def resolve_symbol():
    s = str(symbol)
    if provider == "okx":
        if s.lower() in symbol_map:
            return str(symbol_map[s.lower()])
        if s in symbol_map:
            return str(symbol_map[s])
        if s.lower().endswith("usdt"):
            base = s[:-4].upper()
            return f"{base}-USDT-SWAP"
    return s.upper()


resolved = resolve_symbol()
if provider == "okx":
    df = fetch_klines_okx(resolved)
else:
    df = fetch_klines_binance(resolved)

if df.empty:
    raise SystemExit("❌ 未拉取到K线数据")

# 指标（纯 pandas，无 ta-lib 依赖）
df["ema"] = _ema(df["close"], 14)
df["rsi"] = _rsi(df["close"], 14)
df["macd"] = _macd(df["close"], 12, 26)

# 近似 spread / imbalance
df["spread_approx"] = df["high"] - df["low"]
df["imbalance_approx"] = (df["tb_base"] - (df["volume"] - df["tb_base"])) / (df["volume"] + 1e-6)

future_shift = int(config.get("future_shift", 5))
df["future_return"] = (df["close"].shift(-future_shift) - df["close"]) / df["close"]

train_df = df[["close", "volume", "rsi", "macd", "ema", "spread_approx", "imbalance_approx", "future_return"]].dropna().copy()

scaler = StandardScaler()
train_df.iloc[:, :-1] = scaler.fit_transform(train_df.iloc[:, :-1])
joblib.dump(scaler, "scaler.pkl")
train_df.to_csv("train_data.csv", index=False)
print(f"✅ 已生成 train_data.csv, 共 {len(train_df)} 条数据 | provider={provider} | symbol={resolved}")
