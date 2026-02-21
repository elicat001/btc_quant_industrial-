#!/usr/bin/env python3
import sys
from pathlib import Path
import requests
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.features import FeatureBuilder
from modules.model import ModelManager


def fetch_okx(inst_id: str, bar: str = "1m", limit: int = 300):
    url = "https://www.okx.com/api/v5/market/candles"
    r = requests.get(url, params={"instId": inst_id, "bar": bar, "limit": str(limit)}, timeout=15).json()
    data = (r or {}).get("data", [])
    data = sorted(data, key=lambda x: int(x[0]))
    out = []
    for row in data:
        ts, o, h, l, c, vol = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        out.append((ts, o, h, l, c, vol))
    return out


def main():
    cfg = yaml.safe_load(open("config.yaml")) or {}
    market = (cfg.get("market") or {})
    sym_map = market.get("symbol_map") or {}
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    inst = sym_map.get(symbol.lower(), "BTC-USDT-SWAP")

    kl = fetch_okx(inst, bar="1m", limit=500)
    fb = FeatureBuilder(seq_len=int(cfg.get("seq_len", 30)), k_levels=3)
    mm = ModelManager(symbol.lower())

    preds = []
    future_n = 3
    closes = [x[4] for x in kl]

    for i, (_, o, h, l, c, v) in enumerate(kl):
        # 伪造与线上一致事件结构（无L2时用近似）
        spread = max((h - l) * 0.05, c * 0.00015)
        bid = c - spread / 2
        ask = c + spread / 2
        depth_evt = {"b": [[str(bid), str(max(v * 0.5, 1.0))]], "a": [[str(ask), str(max(v * 0.5, 1.0))]]}
        trade_evt = {"p": str(c), "q": str(max(v * 0.1, 1.0)), "m": False}
        seq = fb.build(trade_evt, depth_evt)
        if seq is None:
            continue

        _, p = mm.predict(seq)
        if p >= 0.58:
            sig = "BUY"
        elif p <= 0.42:
            sig = "SELL"
        else:
            sig = "HOLD"

        hit = None
        if i + future_n < len(closes):
            fr = (closes[i + future_n] - c) / max(c, 1e-9)
            if sig == "BUY":
                hit = 1 if fr > 0 else 0
            elif sig == "SELL":
                hit = 1 if fr < 0 else 0
        preds.append((sig, p, hit))

    total = len(preds)
    c_buy = sum(1 for x in preds if x[0] == "BUY")
    c_sell = sum(1 for x in preds if x[0] == "SELL")
    c_hold = sum(1 for x in preds if x[0] == "HOLD")
    active = [x for x in preds if x[0] != "HOLD" and x[2] is not None]
    hit = (sum(x[2] for x in active) / len(active) * 100) if active else 0.0

    print("Replay Backtest (OKX)")
    print(f"- samples: {total}")
    print(f"- BUY/SELL/HOLD: {c_buy}/{c_sell}/{c_hold}")
    print(f"- active ratio: {(c_buy+c_sell)/max(total,1)*100:.1f}%")
    print(f"- directional hit@+3bars: {hit:.1f}%")


if __name__ == "__main__":
    main()
