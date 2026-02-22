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
STATE = ROOT / "optimizer_state.json"


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
            # 仅统计明确阻塞项，避免把诊断字段误算成阻塞
            if "mm_block(" in ln:
                blockers["mm_block"] += 1
            if "low_vol<" in ln:
                blockers["low_vol_block"] += 1

    closes = 0
    pnl = 0.0
    wins = 0
    losses = 0
    win_sum = 0.0
    loss_sum = 0.0
    if JOURNAL.exists():
        for ln in JOURNAL.read_text(errors="ignore").splitlines()[-200000:]:
            try:
                j = json.loads(ln)
                ts = datetime.fromtimestamp(float(j.get("ts", 0)), tz=timezone.utc).replace(tzinfo=None)
                if ts < since:
                    continue
                if str(j.get("event", "")).startswith("CLOSE"):
                    closes += 1
                    p0 = float(j.get("pnl", 0.0))
                    pnl += p0
                    if p0 > 0:
                        wins += 1
                        win_sum += p0
                    elif p0 < 0:
                        losses += 1
                        loss_sum += abs(p0)
            except Exception:
                continue

    total = sum(sig.values())
    active = sig["BUY"] + sig["SELL"]
    hold_ratio = (sig["HOLD"] / total * 100) if total else 100.0
    active_ratio = (active / total * 100) if total else 0.0
    avg_prob = sum(probs) / len(probs) if probs else 0.5

    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0
    avg_win = (win_sum / wins) if wins else 0.0
    avg_loss = (loss_sum / losses) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)

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
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr": rr,
    }


def _load_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def _save_state(state: dict):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


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

    # ===== 二阶优化：连续窗口确认（抗噪声） =====
    # 条件：连续 3 个窗口同时满足“高阻塞 + HOLD 偏高”才允许进一步放松阈值
    state = _load_state()
    streak = int(state.get("relax_streak", 0))
    relax_candidate = (
        stats.get("hold_ratio", 100) > 60
        and stats.get("blockers", {}).get("dir_margin", 0) > 200
        and stats.get("blockers", {}).get("p_min=", 0) > 200
    )
    streak = streak + 1 if relax_candidate else 0
    state["relax_streak"] = streak

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
    # 业绩门：目标是高胜率(>=90%) + 较好盈亏比(>=1.3)，未达标时只允许收紧，不允许放松
    perf_ok = (stats.get("win_rate", 0.0) >= 90.0) and (stats.get("rr", 0.0) >= 1.3)
    allow_relax = (not weak_signal) and (streak >= 3) and perf_ok

    # bounded micro-tuning rules
    if stats["hold_ratio"] > 75 and stats["blockers"].get("dir_margin", 0) > 200 and allow_relax:
        old = float(cfg["gate"].get("min_dir_margin", 0.01))
        new = max(0.0, old - 0.001)  # 单次限幅，避免激进
        if new != old:
            cfg["gate"]["min_dir_margin"] = new
            changed.append(f"gate.min_dir_margin {old:.4f}->{new:.4f}")

    if stats["blockers"].get("p_min=", 0) > 200 and allow_relax:
        old = float(cfg["strategy"].get("pmin_cap", 0.58))
        new = max(MIN_PMIN_CAP, old - 0.005)  # 单次限幅，避免激进
        if new != old:
            cfg["strategy"]["pmin_cap"] = new
            changed.append(f"strategy.pmin_cap {old:.3f}->{new:.3f}")

    if stats["blockers"].get("low_vol=True", 0) > 200 and allow_relax:
        old = float(cfg["low_vol"].get("override_margin", 0.015))
        new = max(MIN_LOW_VOL_OVERRIDE, old - 0.001)  # 单次限幅，避免激进
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

    # 当业绩未达目标时，优先“收紧质量”，而不是盲目放松
    if not perf_ok:
        old = float(cfg["strategy"].get("pmin_cap", 0.56))
        new = min(0.58, old + 0.005)
        if new != old:
            cfg["strategy"]["pmin_cap"] = new
            changed.append(f"strategy.pmin_cap {old:.3f}->{new:.3f} (tighten)")

        old2 = float(cfg["low_vol"].get("override_margin", 0.011))
        new2 = min(0.015, old2 + 0.001)
        if new2 != old2:
            cfg["low_vol"]["override_margin"] = new2
            changed.append(f"low_vol.override_margin {old2:.3f}->{new2:.3f} (tighten)")

    if changed:
        CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    _save_state(state)
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

    st = _load_state()
    out = {
        "stats": stats,
        "state": {"relax_streak": int(st.get("relax_streak", 0))},
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
    print(f"- 胜率: {stats['win_rate']:.1f}% (w={stats['wins']}, l={stats['losses']}) | RR: {stats['rr']:.2f}")
    if changed:
        print("- 已自动微调:")
        for c in changed:
            print(f"  * {c}")
    else:
        print("- 未触发自动微调")


if __name__ == "__main__":
    main()
