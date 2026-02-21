#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import requests
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.features import FeatureBuilder
from modules.model import ModelManager


def fetch_okx(inst_id: str, bar: str = "1m", limit: int = 800):
    url = "https://www.okx.com/api/v5/market/candles"
    r = requests.get(url, params={"instId": inst_id, "bar": bar, "limit": str(limit)}, timeout=20).json()
    data = (r or {}).get("data", [])
    data = sorted(data, key=lambda x: int(x[0]))
    out = []
    for row in data:
        ts, o, h, l, c, vol = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        out.append((ts, o, h, l, c, vol))
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml")) or {}
    market = cfg.get("market", {}) or {}
    sym_map = market.get("symbol_map", {}) or {}
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    inst = sym_map.get(symbol.lower(), "BTC-USDT-SWAP")

    mm = ModelManager(symbol.lower())
    fb = FeatureBuilder(seq_len=int(cfg.get("seq_len", 30)), k_levels=3)

    kl = fetch_okx(inst, bar="1m", limit=900)
    probs = []
    for _, o, h, l, c, v in kl:
        spread = max((h - l) * 0.05, c * 0.00015)
        bid = c - spread / 2
        ask = c + spread / 2
        depth_evt = {"b": [[str(bid), str(max(v * 0.5, 1.0))]], "a": [[str(ask), str(max(v * 0.5, 1.0))]]}
        trade_evt = {"p": str(c), "q": str(max(v * 0.1, 1.0)), "m": False}
        seq = fb.build(trade_evt, depth_evt)
        if seq is None:
            continue
        _, p = mm.predict(seq)
        probs.append(float(p))

    if len(probs) < 100:
        raise SystemExit("not enough probability samples to calibrate")

    arr = np.asarray(probs)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    # center median -> 0.5 by blend bias in logit domain
    b_shift = float(-logit(med))

    # soften/expand probabilities mildly via temperature
    temp = 1.2 if std < 0.08 else 1.0

    # practical trading thresholds by quantiles
    buy_thr = float(np.quantile(arr, 0.70))
    sell_thr = float(np.quantile(arr, 0.30))

    out = {
        "temp": {"tft": temp, "nbt": temp},
        "smooth_alpha": 0.08,
        "blend": {"w": [0.6, 0.4], "b": b_shift},
        "runtime_thresholds": {
            "buy": buy_thr,
            "sell": sell_thr,
            "median_before": med,
            "mean_before": mean,
            "std_before": std
        },
        "risk": {
            "fee_bp": 0.0005,
            "tp_mult": 2.0,
            "sl_mult": 2.0,
            "tp_min": 0.0025,
            "sl_min": 0.0025,
            "future_holding": 30
        }
    }

    (ROOT / "thresholds.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("calibration done")
    print(json.dumps(out["runtime_thresholds"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
