#!/usr/bin/env python3
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = Path("logs/run.log")


def parse_dt(line: str):
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def main():
    if not LOG.exists():
        print("策略小时报：暂无日志")
        return

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    since = now - timedelta(hours=1)

    c = {"BUY": 0, "SELL": 0, "HOLD": 0}
    probs = []

    for line in LOG.read_text(errors="ignore").splitlines()[-20000:]:
        ts = parse_dt(line)
        if not ts or ts < since:
            continue
        m = re.search(r"fused_signal=(BUY|SELL|HOLD)", line)
        if m:
            c[m.group(1)] += 1
        p = re.search(r"\bprob=([0-9]*\.?[0-9]+)", line)
        if p:
            probs.append(float(p.group(1)))

    total = sum(c.values())
    avgp = sum(probs) / len(probs) if probs else 0.0
    hold_ratio = (c['HOLD'] / total * 100) if total else 0.0

    print("策略小时报（UTC）")
    print(f"- 时间窗: {since.strftime('%H:%M')} ~ {now.strftime('%H:%M')}")
    print(f"- 信号总数: {total}")
    print(f"- BUY/SELL/HOLD: {c['BUY']}/{c['SELL']}/{c['HOLD']}")
    print(f"- HOLD占比: {hold_ratio:.1f}%")
    print(f"- 平均prob: {avgp:.3f}")


if __name__ == "__main__":
    main()
