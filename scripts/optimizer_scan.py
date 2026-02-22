#!/usr/bin/env python3
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "run.log"
JOURNAL = ROOT / "logs" / "paper_trades.jsonl"
CFG = ROOT / "config.yaml"
THR = ROOT / "thresholds.json"


def ts_from_line(line: str):
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def analyze(window_min=20):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    since = now - timedelta(minutes=window_min)

    sig = Counter()
    blockers = Counter()
    probs = []

    if LOG.exists():
        lines = LOG.read_text(errors="ignore").splitlines()[-200000:]
        for ln in lines:
            ts = ts_from_line(ln)
            if not ts or ts < since:
                continue
            m = re.search(r"fused_signal=(BUY|SELL|HOLD)", ln)
            if m:
                sig[m.group(1)] += 1
            p = re.search(r"\bp=([0-9]*\.?[0-9]+)", ln)
            if p:
                probs.append(float(p.group(1)))
            for k in ["dir_margin", "p_min=", "low_vol=True", "mm_block"]:
                if k in ln:
                    blockers[k] += 1

    closes = 0
    pnl = 0.0
    if JOURNAL.exists():
        for ln in JOURNAL.read_text(errors="ignore").splitlines()[-200000:]:
            try:
                j = json.loads(ln)
                ts = datetime.fromtimestamp(float(j.get("ts", 0)), tz=timezone.utc).replace(tzinfo=None)
                if ts < since:
                    continue
                if str(j.get("event", "")).startswith("CLOSE"):
                    closes += 1
                    pnl += float(j.get("pnl", 0.0))
            except Exception:
                continue

    total = sum(sig.values())
    active = sig["BUY"] + sig["SELL"]
    hold_ratio = (sig["HOLD"] / total * 100) if total else 100.0
    active_ratio = (active / total * 100) if total else 0.0
    avg_prob = sum(probs) / len(probs) if probs else 0.5

    return {
        "window_min": window_min,
        "since_utc": since.isoformat(sep=" "),
        "signals": dict(sig),
        "hold_ratio": hold_ratio,
        "active_ratio": active_ratio,
        "avg_prob": avg_prob,
        "blockers": dict(blockers),
        "closes": closes,
        "pnl": pnl,
    }


def apply_tuning(stats: dict):
    cfg = yaml.safe_load(CFG.read_text()) or {}
    changed = []

    cfg.setdefault("confirm", {})
    cfg.setdefault("gate", {})
    cfg.setdefault("mm", {})
    cfg.setdefault("low_vol", {})
    cfg.setdefault("strategy", {})

    # ===== 实盘优先的保守边界（防止自动优化一路下压） =====
    MIN_PMIN_CAP = 0.54
    MIN_LOW_VOL_OVERRIDE = 0.009

    # 若当前已经低于安全边界，先拉回
    cur_pmin = float(cfg["strategy"].get("pmin_cap", 0.58))
    if cur_pmin < MIN_PMIN_CAP:
        cfg["strategy"]["pmin_cap"] = MIN_PMIN_CAP
        changed.append(f"strategy.pmin_cap {cur_pmin:.3f}->{MIN_PMIN_CAP:.3f} (rebound)")

    cur_lvo = float(cfg["low_vol"].get("override_margin", 0.015))
    if cur_lvo < MIN_LOW_VOL_OVERRIDE:
        cfg["low_vol"]["override_margin"] = MIN_LOW_VOL_OVERRIDE
        changed.append(f"low_vol.override_margin {cur_lvo:.3f}->{MIN_LOW_VOL_OVERRIDE:.3f} (rebound)")

    # 质量门：当模型均值仍接近 0.5 时，禁止继续放松阈值
    weak_signal = abs(float(stats.get("avg_prob", 0.5)) - 0.5) < 0.004

    # bounded micro-tuning rules
    if stats["hold_ratio"] > 75 and stats["blockers"].get("dir_margin", 0) > 200 and not weak_signal:
        old = float(cfg["gate"].get("min_dir_margin", 0.01))
        new = max(0.0, old - 0.002)
        if new != old:
            cfg["gate"]["min_dir_margin"] = new
            changed.append(f"gate.min_dir_margin {old:.4f}->{new:.4f}")

    if stats["blockers"].get("p_min=", 0) > 200 and not weak_signal:
        old = float(cfg["strategy"].get("pmin_cap", 0.58))
        new = max(MIN_PMIN_CAP, old - 0.01)
        if new != old:
            cfg["strategy"]["pmin_cap"] = new
            changed.append(f"strategy.pmin_cap {old:.3f}->{new:.3f}")

    if stats["blockers"].get("low_vol=True", 0) > 200 and not weak_signal:
        old = float(cfg["low_vol"].get("override_margin", 0.015))
        new = max(MIN_LOW_VOL_OVERRIDE, old - 0.002)
        if new != old:
            cfg["low_vol"]["override_margin"] = new
            changed.append(f"low_vol.override_margin {old:.3f}->{new:.3f}")

    if stats["active_ratio"] > 45 and stats["closes"] == 0:
        # too active but not closing: reduce churn
        old = int(cfg["mm"].get("cooldown_s", 12))
        new = min(60, old + 5)
        if new != old:
            cfg["mm"]["cooldown_s"] = new
            changed.append(f"mm.cooldown_s {old}->{new}")

    if changed:
        CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    return changed


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    stats = analyze(args.window)
    changed = []
    if args.apply:
        changed = apply_tuning(stats)

    out = {
        "stats": stats,
        "changed": changed,
        "action": "restart_required" if changed else "no_change",
    }

    (ROOT / "optimizer_scan.latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )

    print("优化扫描结果")
    print(f"- 窗口: {stats['window_min']} min")
    print(f"- HOLD占比: {stats['hold_ratio']:.1f}% | active: {stats['active_ratio']:.1f}% | avg_prob: {stats['avg_prob']:.3f}")
    print(f"- blockers: {stats['blockers']}")
    print(f"- 平仓数: {stats['closes']} | PnL: {stats['pnl']:.4f}")
    if changed:
        print("- 已自动微调:")
        for c in changed:
            print(f"  * {c}")
    else:
        print("- 未触发自动微调")


if __name__ == "__main__":
    main()
