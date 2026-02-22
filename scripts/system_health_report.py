#!/usr/bin/env python3
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "run.log"
JOURNAL = ROOT / "logs" / "paper_trades.jsonl"


def parse_ts(line):
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def load_journal(since):
    rows = []
    if not JOURNAL.exists():
        return rows
    for ln in JOURNAL.read_text(errors="ignore").splitlines()[-200000:]:
        try:
            j = json.loads(ln)
            ts = datetime.fromtimestamp(float(j.get("ts", 0)), tz=timezone.utc).replace(tzinfo=None)
            if ts >= since:
                rows.append(j)
        except Exception:
            continue
    return rows


def main():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    since = now - timedelta(hours=24)

    sig = Counter()
    reasons = Counter()
    probs = []

    if LOG.exists():
        for line in LOG.read_text(errors="ignore").splitlines()[-300000:]:
            ts = parse_ts(line)
            if not ts or ts < since:
                continue
            m = re.search(r"fused_signal=(BUY|SELL|HOLD)", line)
            if m:
                sig[m.group(1)] += 1
            p = re.search(r"\bp=([0-9]*\.?[0-9]+)", line)
            if p:
                probs.append(float(p.group(1)))
            if "决策:" in line:
                if "止盈" in line or "take_profit" in line:
                    reasons["take_profit"] += 1
                if "止损" in line or "stop_loss" in line:
                    reasons["stop_loss"] += 1
                if "trail" in line:
                    reasons["trailing"] += 1

    rows = load_journal(since)
    closes = [r for r in rows if str(r.get("event", "")).startswith("CLOSE")]
    opens = [r for r in rows if str(r.get("event", "")).startswith("OPEN")]

    pnl_total = sum(float(r.get("pnl_net", r.get("pnl", 0.0))) for r in closes)
    pnl_gross_total = sum(float(r.get("pnl_gross", r.get("pnl", 0.0))) for r in closes)
    fee_total = sum(float(r.get("fee_total", 0.0)) for r in closes)
    wins = sum(1 for r in closes if float(r.get("pnl_net", r.get("pnl", 0.0))) > 0)
    losses = sum(1 for r in closes if float(r.get("pnl_net", r.get("pnl", 0.0))) < 0)
    win_rate = (wins / max(1, wins + losses)) * 100

    close_reason = Counter(str(r.get("reason", "")) for r in closes)

    total_sig = sum(sig.values())
    active_ratio = (sig["BUY"] + sig["SELL"]) / max(1, total_sig) * 100
    avg_prob = sum(probs) / len(probs) if probs else 0

    print("量化系统日报（近24h, UTC）")
    print(f"- 信号: BUY/SELL/HOLD = {sig['BUY']}/{sig['SELL']}/{sig['HOLD']} (active {active_ratio:.1f}%)")
    print(f"- 平均prob: {avg_prob:.3f}")
    print(f"- 订单: OPEN={len(opens)} CLOSE={len(closes)}")
    print(f"- 已平仓PnL(net): {pnl_total:.2f} USD")
    print(f"- 已平仓PnL(gross): {pnl_gross_total:.2f} USD | fee: {fee_total:.2f} USD")
    print(f"- 胜率: {win_rate:.1f}% (win={wins}, loss={losses})")
    if close_reason:
        top = ", ".join([f"{k}:{v}" for k, v in close_reason.most_common(5)])
        print(f"- 平仓原因Top: {top}")
    if reasons:
        top2 = ", ".join([f"{k}:{v}" for k, v in reasons.most_common()])
        print(f"- 日志风控线索: {top2}")
    print(f"- 动态止盈止损: {'已启用(由risk模块触发)' if (close_reason or reasons) else '未在近24h观察到触发'}")


if __name__ == "__main__":
    main()
