import asyncio
import json
import os
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import torch.optim as optim
import websockets
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model_definitions import EnhancedNBeats, EnhancedTFT
from modules.features import FeatureBuilder

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

market = (config.get("market") or {})
provider = str(market.get("provider", "okx")).lower()
symbol = config.get("symbol", "BTCUSDT")
seq_len = int(config.get("seq_len", 30))
input_size = int(config.get("input_size", 12))
batch_size = int(config.get("batch_size", 32))
save_interval = int(config.get("save_interval", 50))
future_shift = int(config.get("future_shift", 5))


def resolve_symbol_for_provider(sym: str):
    m = (market.get("symbol_map") or {})
    if provider == "okx":
        if sym.lower() in m:
            return str(m[sym.lower()])
        if sym in m:
            return str(m[sym])
        if sym.lower().endswith("usdt"):
            return f"{sym[:-4].upper()}-USDT-SWAP"
    return sym.lower()


symbol_resolved = resolve_symbol_for_provider(str(symbol))

# models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tft_model = EnhancedTFT(input_size=input_size).to(device)
nbeats_model = EnhancedNBeats(input_size=input_size).to(device)
criterion = torch.nn.MSELoss()
tft_optimizer = optim.Adam(tft_model.parameters(), lr=2e-5)
nbeats_optimizer = optim.Adam(nbeats_model.parameters(), lr=2e-5)

# shared state
latest_depth_evt = None
trade_queue = asyncio.Queue(maxsize=5000)
feature_builder = FeatureBuilder(seq_len=seq_len, k_levels=3)
scaler_path = "scaler.pkl"
# 训练阶段强制重建 scaler，避免历史坏缩放污染
scaler = None


# buffers
seq_buffer = []  # list[(seq_np[T,F], price)]
X_buffer = []
y_buffer = []


def train_batch(seqs, labels):
    x = torch.tensor(seqs, dtype=torch.float32).to(device)      # (B,T,F)
    y = torch.tensor(labels, dtype=torch.float32).to(device)    # (B,)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    tft_loss_avg = 0.0
    nbeats_loss_avg = 0.0
    n = 0
    for xb, yb in dl:
        # TFT uses full sequence
        tft_optimizer.zero_grad()
        tft_pred = tft_model(xb).squeeze(-1)
        tft_loss = criterion(tft_pred, yb)
        tft_loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), max_norm=1.0)
        tft_optimizer.step()

        # NBeats keeps inference semantics: mean pooling over time
        nbeats_optimizer.zero_grad()
        x_pool = xb.mean(dim=1)
        nbeats_pred = nbeats_model(x_pool).squeeze(-1)
        nbeats_loss = criterion(nbeats_pred, yb)
        nbeats_loss.backward()
        torch.nn.utils.clip_grad_norm_(nbeats_model.parameters(), max_norm=1.0)
        nbeats_optimizer.step()

        tft_loss_avg += float(tft_loss.item())
        nbeats_loss_avg += float(nbeats_loss.item())
        n += 1

    return tft_loss_avg / max(1, n), nbeats_loss_avg / max(1, n)


async def trade_handler_okx():
    url = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(url, ping_interval=20) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "trades", "instId": symbol_resolved}]}))
        async for msg in ws:
            j = json.loads(msg)
            if "data" not in j:
                continue
            row = (j.get("data") or [None])[0]
            if not row:
                continue
            px = float(row.get("px", 0) or 0)
            if px <= 0:
                continue
            t_evt = {"p": str(px), "q": str(row.get("sz", "0")), "m": False}
            await trade_queue.put(t_evt)


async def depth_handler_okx():
    global latest_depth_evt
    url = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(url, ping_interval=20) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "books5", "instId": symbol_resolved}]}))
        async for msg in ws:
            j = json.loads(msg)
            if "data" not in j:
                continue
            row = (j.get("data") or [None])[0]
            if not row:
                continue
            bids = row.get("bids", [])
            asks = row.get("asks", [])
            latest_depth_evt = {"b": [[b[0], b[1]] for b in bids], "a": [[a[0], a[1]] for a in asks]}


async def training_loop():
    global scaler
    steps = 0
    warmup_feats = []

    print(f"📡 等待数据流开始... provider={provider} symbol={symbol_resolved}")
    while True:
        trade_evt = await trade_queue.get()
        seq = feature_builder.build(trade_evt, latest_depth_evt)
        if seq is None:
            continue

        # align dim
        if seq.shape[1] > input_size:
            seq = seq[:, :input_size]
        elif seq.shape[1] < input_size:
            pad = np.zeros((seq.shape[0], input_size - seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)

        last_feat = seq[-1]
        if scaler is None:
            warmup_feats.append(last_feat)
            if len(warmup_feats) >= 100:
                scaler = StandardScaler()
                scaler.fit(np.array(warmup_feats, dtype=np.float32))
                joblib.dump(scaler, scaler_path)
                print(f"💾 已生成新的 scaler.pkl ({len(warmup_feats)} 条特征)")
            else:
                continue

        seq = scaler.transform(seq)
        seq = np.clip(seq, -10.0, 10.0)

        price = float(trade_evt.get("p", 0) or 0)
        seq_buffer.append((seq.astype(np.float32), price))

        if len(seq_buffer) > future_shift:
            seq_old, p_old = seq_buffer[-future_shift - 1]
            p_new = seq_buffer[-1][1]
            if p_old <= 0 or p_new <= 0:
                continue
            y = (p_new - p_old) / max(p_old, 1e-9)
            y = float(np.clip(y, -0.03, 0.03))
            X_buffer.append(seq_old)
            y_buffer.append(y)

        if len(y_buffer) >= batch_size:
            tft_loss, nbeats_loss = train_batch(np.array(X_buffer[-batch_size:]), np.array(y_buffer[-batch_size:]))
            steps += 1
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ {now} 训练 | Step: {steps} | TFT Loss: {tft_loss:.6f} | NBeats Loss: {nbeats_loss:.6f}")
            if steps % save_interval == 0:
                torch.save(tft_model.state_dict(), "tft_model.pth")
                torch.save(nbeats_model.state_dict(), "nbeats_model.pth")
                print("💾 已保存模型")


async def status_monitor():
    while True:
        qn = trade_queue.qsize()
        if latest_depth_evt is None:
            print("⏳ 等待 books5 盘口数据...")
        elif qn == 0:
            print("⏳ 等待 trades 成交数据...")
        else:
            print(f"✅ 数据流正常，queue={qn}，已生成样本={len(y_buffer)}")
        await asyncio.sleep(3)


async def main():
    if provider != "okx":
        raise SystemExit("当前训练脚本仅启用 okx provider")
    print(f"🚀 实时训练启动 provider={provider}")
    await asyncio.gather(trade_handler_okx(), depth_handler_okx(), training_loop(), status_monitor())


if __name__ == "__main__":
    asyncio.run(main())
