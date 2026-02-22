# main.py
import os
import sys
import asyncio
import logging
import yaml
import traceback
import time
import signal
import re
import math
import torch
import numpy as np
from datetime import datetime, timezone
from collections import deque
from logging.handlers import TimedRotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple

from modules.collector import BinanceCollector
from modules.features import FeatureBuilder
from modules.model import ModelManager
from modules.push import PushManager
from modules.signal import SignalFusion
from modules.risk import RiskController
from modules.executor import TradeExecutor  # 需要 ob_getter
from modules.midtrend import MidTrend  # 中期模块

# ===== 统一日志：仅写文件，杜绝重复/半行 =====
def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler], force=True)
    logging.getLogger().propagate = False


setup_logging()
logger = logging.getLogger("main")


# ===== 读取配置 =====
def load_config():
    cfg = {}
    if os.path.exists("config.yaml"):
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"读取 config.yaml 失败：{e}")
    return cfg


CONFIG = load_config()
SYMBOLS = [s.lower() for s in CONFIG.get("symbols", ["btcusdt", "ethusdt"])]
SEQ_LEN = int(CONFIG.get("seq_len", 30))
PUSH_ENABLED = bool(CONFIG.get("push_enabled", True))
MIN_PUSH_INTERVAL = int(CONFIG.get("min_push_interval_sec", 120))
LOW_GPU_MODE = bool(CONFIG.get("low_gpu_mode", True))
INFER_COOLDOWN = float(CONFIG.get("infer_cooldown_sec", 0.0 if not LOW_GPU_MODE else 0.5))

# 交易相关（给 MM 传费率等）
TRADING_CFG = CONFIG.get("trading", {}) or {}
TAKER_FEE_BP = float(TRADING_CFG.get("taker_fee_bp", 2.0))  # 默认 2bps

# ---- 交易所合约别名修正（例如 renderusdt -> RNDRUSDT）----
ALIASES_EXCHANGE = {"renderusdt": "RNDRUSDT"}
sym_map = (TRADING_CFG.get("symbol_map", {}) or {}).copy()
changed = False
for k in list(sym_map.keys()):
    kk = k.lower()
    if kk in ALIASES_EXCHANGE and sym_map[k] != ALIASES_EXCHANGE[kk]:
        sym_map[k] = ALIASES_EXCHANGE[kk]
        changed = True
if changed:
    TRADING_CFG["symbol_map"] = sym_map

# 双确认（减少抖动）
CONFIRM_CFG = CONFIG.get("confirm", {}) or {}
CONFIRM_NEED_PROB = float(CONFIRM_CFG.get("need_prob", 0.60))
CONFIRM_MAX_GAP = int(CONFIRM_CFG.get("max_gap_sec", 30))
CONFIRM_NEED_PROB_LOW_VOL = float(CONFIRM_CFG.get("need_prob_low_vol", CONFIRM_NEED_PROB))

# 门控（方向边际）
GATE_CFG = CONFIG.get("gate", {}) or {}
MIN_DIR_MARGIN = float(GATE_CFG.get("min_dir_margin", 0.06))
MIN_DIR_MARGIN_LOW_VOL = float(GATE_CFG.get("min_dir_margin_low_vol", max(0.10, MIN_DIR_MARGIN)))

# 模拟交易配置
SIM_CFG = CONFIG.get("sim", {}) or {
    "enable": True,
    "use_phase_signals": True,
    "use_model_decisions": False,
    "partials": [0.4, 0.4, 0.2],
    "resend_updates_sec": 60,
    "max_holding_min": 240
}

pusher = PushManager(config=CONFIG) if PUSH_ENABLED else None

# ===== 全局缓存（给执行器/模拟器用） =====
PRICE_CACHE: Dict[str, float] = {}     # symbol -> last trade price (float)
OB_CACHE: Dict[str, Tuple[float, float]] = {}  # symbol -> (best_bid, best_ask) (floats)


def _extract_best_from_depth(depth_msg):
    """从 depth20@100ms 消息中提取买一/卖一。"""
    if not depth_msg:
        return None, None
    d = depth_msg.get("data", depth_msg)
    bids = d.get("b") or d.get("bids") or []
    asks = d.get("a") or d.get("asks") or []
    try:
        best_bid = max(float(x[0]) for x in bids) if bids else None
        best_ask = min(float(x[0]) for x in asks) if asks else None
        return best_bid, best_ask
    except Exception:
        return None, None


# =========================
# 工具函数：诊断解析与门控
# =========================
def safe_fmt(x, nd=2, na="—"):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return na


def _bool_from(diag: str, key: str) -> Optional[bool]:
    m = re.search(rf"{key}\s*=\s*(True|False)", diag)
    if not m:
        return None
    return True if m.group(1) == "True" else False


def parse_diag(diag: str) -> Dict[str, Any]:
    """
    从 SignalFusion 的 diag 字符串中提取关键信息（鲁棒解析）
    """
    out: Dict[str, Any] = {}
    if not isinstance(diag, str):
        return out

    def fnum(pattern, default=None, cast=float):
        m = re.search(pattern, diag)
        try:
            return cast(m.group(1)) if m else default
        except Exception:
            return default

    out["p"] = fnum(r"p=([0-9.]+)")
    out["p_min"] = fnum(r"p_min=([0-9.]+)")
    out["atr_pct"] = fnum(r"atr%=\s*([0-9.]+)")
    out["macd"] = fnum(r"macd=\s*([+\-0-9.]+)")
    out["rsi"] = fnum(r"rsi=\s*([0-9.]+)")
    out["mm_score"] = fnum(r"mm_score=\s*([0-9.]+)")
    out["dir_margin"] = fnum(r"dir_margin=\s*([+\-0-9.]+)")
    out["spread_bp"] = fnum(r"spread_bp=\s*([0-9.]+)")
    out["imb"] = fnum(r"imb=\s*([+\-0-9.]+)")
    out["age_ms"] = fnum(r"age_ms=\s*([0-9]+)", cast=int)
    out["mm_ok"] = _bool_from(diag, "mm_ok")
    out["force_ok"] = _bool_from(diag, "force_ok")
    out["low_vol"] = ("low_vol=True" in diag) or ("low_vol<" in diag)
    m = re.search(r"mm_side\s*=\s*(buy|sell)", diag)
    out["mm_side"] = m.group(1) if m else None
    out["mm_block"] = "mm_block" in diag
    return out


def build_env_and_gate(diag: str, cfg: dict) -> Tuple[str, bool]:
    """
    返回 (环境摘要, 是否拒绝执行)
    """
    info = parse_diag(diag or "")
    reasons = []
    reject = False

    low_vol = bool(info.get("low_vol"))
    mm_score = float(info.get("mm_score") or 0.0)
    dir_margin = float(info.get("dir_margin") or 0.0)
    p = float(info.get("p") or 0.0)
    p_min = float(info.get("p_min") or 1.0)

    # 选择阈值（低波动档优先）
    score_open = cfg.get("score_open_low_vol", cfg.get("score_open", 0.62)) if low_vol else cfg.get("score_open", 0.62)
    min_dm = MIN_DIR_MARGIN_LOW_VOL if low_vol else MIN_DIR_MARGIN
    need_prob = CONFIRM_NEED_PROB_LOW_VOL if low_vol else CONFIRM_NEED_PROB

    # 逐项检查
    if p < max(p_min, need_prob):
        reject = True
        reasons.append("置信不足")
    if abs(dir_margin) < min_dm:
        reject = True
        reasons.append("方向边际不足")
    if mm_score < score_open:
        reject = True
        reasons.append("MM分低")
    if info.get("mm_block") or (info.get("mm_ok") is False) or (info.get("force_ok") is False):
        reject = True
        if "MM阻拦" not in reasons:
            reasons.append("MM阻拦")
    if low_vol:
        if "低波动" not in reasons:
            reasons.append("低波动")

    env_str = "/".join(reasons) if reasons else "环境:正常"
    return env_str, reject


def execution_gate_state(diag: str, cfg: dict) -> Tuple[bool, List[str], str]:
    """
    读取模型诊断行，结合配置返回 (allow: bool, reasons: list[str], view: 'BUY'|'SELL')
    view 仅是“观点”，不代表可执行。
    """
    info = parse_diag(diag or "")
    p = float(info.get("p") or 0.0)
    p_min = float(info.get("p_min") or 1.0)
    mm_score = float(info.get("mm_score") or 0.0)
    dir_margin = float(info.get("dir_margin") or 0.0)
    low_vol = bool(info.get("low_vol"))

    # 观点：优先 mm_side，其次 dir_margin 正负，否则根据 p 相对 p_min 粗判
    if info.get("mm_side") in ("buy", "sell"):
        view = "BUY" if info["mm_side"] == "buy" else "SELL"
    elif dir_margin is not None and abs(dir_margin) > 1e-9:
        view = "BUY" if dir_margin > 0 else "SELL"
    else:
        view = "BUY" if p >= p_min else "SELL"

    # 阈值选择
    score_open = cfg.get("score_open_low_vol", cfg.get("score_open", 0.62)) if low_vol else cfg.get("score_open", 0.62)
    need_prob = CONFIRM_NEED_PROB_LOW_VOL if low_vol else CONFIRM_NEED_PROB
    min_dm = MIN_DIR_MARGIN_LOW_VOL if low_vol else MIN_DIR_MARGIN

    reasons = []
    allow = True

    if p < max(p_min, need_prob):
        allow = False; reasons.append(f"置信不足(p={p:.3f} < {max(p_min, need_prob):.3f})")
    if abs(dir_margin) < min_dm:
        allow = False; reasons.append(f"方向边际不足(|Δp|={abs(dir_margin):.3f} < {min_dm:.3f})")
    if mm_score < score_open:
        allow = False; reasons.append(f"MM分不足({mm_score:.3f} < {score_open:.3f})")
    if low_vol:
        reasons.append("低波动")
    if info.get("mm_block") or (info.get("mm_ok") is False) or (info.get("force_ok") is False):
        allow = False
        if "MM阻拦" not in reasons:
            reasons.append("MM阻拦")

    return allow, reasons, view


def format_model_confirm(diag: str, cfg: dict) -> str:
    allow, reasons, view = execution_gate_state(diag, cfg)
    if not allow:
        why = "/".join(reasons) if reasons else "过滤"
        return f"模型观点:{'看多' if view=='BUY' else '看空'}（未执行）\n环境:{why}"
    return f"模型确认:{'BUY' if view=='BUY' else 'SELL'}（可执行）"


# =========================
# 模拟器：PhaseSim（保持不变）
# =========================
class PhaseSim:
    """
    规则（与推送一致）：
      - bottom(做多)：到 T1 实现 40%，SL 抬至保本；到 T2 再 40%，启用追踪止盈（k×ATR%）；余下 20% 走 T3 或追踪止损。
      - top(做空)：镜像逻辑。
      - 费用：用 trading.taker_fee_bp 与 slippage_bp 粗估（双边）。
    """
    def __init__(self, cfg: dict, pusher_obj=None):
        self.cfg = cfg
        self.ps: List[dict] = []
        self.closed: List[dict] = []
        self.pusher = pusher_obj
        tcfg = cfg.get("trading", {}) or {}
        self.fee_bp = float(tcfg.get("taker_fee_bp", 0.5))
        self.slip_bp = float(tcfg.get("slippage_bp", 0.0))
        self.partials = cfg.get("sim", {}).get("partials", [0.4, 0.4, 0.2])
        self.reverse_guard_s = int(cfg.get("risk", {}).get("reverse_guard_seconds", 240))

        self._last_open_side: Dict[str, str] = {}
        self._last_open_ts: Dict[str, float] = {}

        self._daily_ymd = None
        self._daily = {"n": 0, "win": 0, "gross_bp": 0.0, "net_bp": 0.0}

    @staticmethod
    def _now_ts():
        return time.time()

    def _fees_roundtrip_bp(self):
        return 2.0 * (self.fee_bp + self.slip_bp)

    def _push(self, text: str):
        logging.getLogger("SIM").info(text)
        if self.pusher:
            try:
                if hasattr(self.pusher, "push_text"):
                    self.pusher.push_text(text)
                else:
                    self.pusher.send_text(text)
            except Exception as e:
                logging.getLogger("SIM").error(f"[SIM] 推送失败: {e}")

    def open_from_phase(self, symbol: str, phase: dict):
        try:
            side = "LONG" if str(phase.get("kind")).lower() == "bottom" else "SHORT"
            px = float(phase["price"])
            t1 = float(phase["t1"]); t2 = float(phase["t2"]); t3 = float(phase["t3"])
            sl = float(phase["sl"])
            conf = float(phase.get("conf", 0.0))
            atr_pct = float(phase.get("atr_pct", 0.006))
            trail_k = float(phase.get("trail_k", self.cfg.get("phase", {}).get("trail_k_atr", 0.6)))

            sym = symbol.lower()
            last_side = self._last_open_side.get(sym)
            last_ts = float(self._last_open_ts.get(sym, 0.0))
            now_ts = self._now_ts()

            if last_side and last_side != side and (now_ts - last_ts) < self.reverse_guard_s:
                self._push(f"【SIM】{symbol.upper()} | 忽略反手信号（{int(self.reverse_guard_s)}s 护栏）")
                return
            if last_side == side and (now_ts - last_ts) < 30:
                self._push(f"【SIM】{symbol.upper()} | 忽略 30s 内同向重复开仓")
                return

            trade = {
                "id": f"{symbol}-{int(now_ts*1000)}",
                "symbol": symbol,
                "side": side,
                "entry": px,
                "t1": t1, "t2": t2, "t3": t3,
                "sl": sl, "trail_k": trail_k, "atr_pct": atr_pct,
                "conf": conf,
                "opened_at": now_ts,
                "status": "OPEN",
                "hit": {"t1": False, "t2": False, "t3": False},
                "trail_stop": None,
                "realized_bp": 0.0,
                "realized_frac": 0.0,
                "last_report_ts": 0.0
            }
            self.ps.append(trade)
            self._last_open_side[sym] = side
            self._last_open_ts[sym] = now_ts

            text = (f"【SIM】开仓 | {symbol.upper()} | {side} | 入场:{px:g} | "
                    f"T1:{t1:g} T2:{t2:g} T3:{t3:g} | SL:{sl:g} | 置信:{conf:.2f} | trail≈{trail_k}×ATR")
            self._push(text)
        except Exception as e:
            logging.getLogger("SIM").error(f"[SIM] open_from_phase 异常: {e}")

    @staticmethod
    def _bp_from(px_a: float, px_b: float) -> float:
        return (px_a - px_b) / max(px_b, 1e-12) * 1e4

    def _update_one(self, tr: dict, price: float, now_ts: float):
        if tr["status"] != "OPEN":
            return
        side = tr["side"]
        entry = tr["entry"]
        atr_pct = tr["atr_pct"]
        k = tr["trail_k"]

        def is_up(x):   return price >= x
        def is_dn(x):   return price <= x

        hit_T1 = hit_T2 = hit_T3 = hit_SL = hit_Trail = False
        fill_msgs = []

        if side == "LONG":
            if (not tr["hit"]["t1"]) and is_up(tr["t1"]):
                tr["hit"]["t1"] = True; hit_T1 = True
                frac = SIM_CFG.get("partials", [0.4, 0.4, 0.2])[0]
                pnl_bp = self._bp_from(price, entry) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] += frac
                tr["sl"] = max(tr["sl"], entry)
                fill_msgs.append(f"T1 触发 | 成交:{price:g} (+{self._bp_from(price, entry):.2f}bp) | 实现 {int(frac*100)}% | SL抬至保本")
            if (not tr["hit"]["t2"]) and is_up(tr["t2"]):
                tr["hit"]["t2"] = True; hit_T2 = True
                frac = SIM_CFG.get("partials", [0.4, 0.4, 0.2])[1]
                pnl_bp = self._bp_from(price, entry) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] += frac
                trail = price * (1.0 - k * atr_pct)
                tr["trail_stop"] = max(tr.get("trail_stop") or -1e18, trail)
                fill_msgs.append(f"T2 触发 | 成交:{price:g} (+{self._bp_from(price, entry):.2f}bp) | 实现 {int(frac*100)}% | 启用追踪止盈≈{k}×ATR")
            if tr["hit"]["t2"]:
                trail = price * (1.0 - k * atr_pct)
                tr["trail_stop"] = max(tr.get("trail_stop") or -1e18, trail)

            if (not tr["hit"]["t3"]) and is_up(tr["t3"]):
                tr["hit"]["t3"] = True; hit_T3 = True
                frac = 1.0 - tr["realized_frac"]
                pnl_bp = self._bp_from(price, entry) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] = 1.0
                fill_msgs.append(f"T3 触发 | 成交:{price:g} (+{self._bp_from(price, entry):.2f}bp) | 全部平仓")
            else:
                if tr.get("trail_stop") is not None and price <= tr["trail_stop"] and tr["realized_frac"] < 1.0:
                    frac = 1.0 - tr["realized_frac"]
                    pnl_bp = self._bp_from(tr["trail_stop"], entry) * frac
                    tr["realized_bp"] += pnl_bp
                    tr["realized_frac"] = 1.0
                    hit_Trail = True
                    fill_msgs.append(f"追踪止盈触发 | 成交:{tr['trail_stop']:g} (+{self._bp_from(tr['trail_stop'], entry):.2f}bp) | 全部平仓")
                if price <= tr["sl"] and tr["realized_frac"] < 1.0:
                    frac = 1.0 - tr["realized_frac"]
                    pnl_bp = self._bp_from(tr["sl"], entry) * frac
                    tr["realized_bp"] += pnl_bp
                    tr["realized_frac"] = 1.0
                    hit_SL = True
                    fill_msgs.append(f"止损触发 | 成交:{tr['sl']:g} ({self._bp_from(tr['sl'], entry):.2f}bp) | 全部平仓")
        else:
            if (not tr["hit"]["t1"]) and is_dn(tr["t1"]):
                tr["hit"]["t1"] = True; hit_T1 = True
                frac = SIM_CFG.get("partials", [0.4, 0.4, 0.2])[0]
                pnl_bp = self._bp_from(entry, price) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] += frac
                tr["sl"] = min(tr["sl"], entry)
                fill_msgs.append(f"T1 触发 | 成交:{price:g} (+{self._bp_from(entry, price):.2f}bp) | 实现 {int(frac*100)}% | SL下调至保本")
            if (not tr["hit"]["t2"]) and is_dn(tr["t2"]):
                tr["hit"]["t2"] = True; hit_T2 = True
                frac = SIM_CFG.get("partials", [0.4, 0.4, 0.2])[1]
                pnl_bp = self._bp_from(entry, price) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] += frac
                trail = price * (1.0 + k * atr_pct)
                tr["trail_stop"] = min(tr.get("trail_stop") or 1e18, trail)
                fill_msgs.append(f"T2 触发 | 成交:{price:g} (+{self._bp_from(entry, price):.2f}bp) | 实现 {int(frac*100)}% | 启用追踪止盈≈{k}×ATR")
            if tr["hit"]["t2"]:
                trail = price * (1.0 + k * atr_pct)
                tr["trail_stop"] = min(tr.get("trail_stop") or 1e18, trail)

            if (not tr["hit"]["t3"]) and is_dn(tr["t3"]):
                tr["hit"]["t3"] = True; hit_T3 = True
                frac = 1.0 - tr["realized_frac"]
                pnl_bp = self._bp_from(entry, price) * frac
                tr["realized_bp"] += pnl_bp
                tr["realized_frac"] = 1.0
                fill_msgs.append(f"T3 触发 | 成交:{price:g} (+{self._bp_from(entry, price):.2f}bp) | 全部平仓")
            else:
                if tr.get("trail_stop") is not None and price >= tr["trail_stop"] and tr["realized_frac"] < 1.0:
                    frac = 1.0 - tr["realized_frac"]
                    pnl_bp = self._bp_from(entry, tr["trail_stop"]) * frac
                    tr["realized_bp"] += pnl_bp
                    tr["realized_frac"] = 1.0
                    hit_Trail = True
                    fill_msgs.append(f"追踪止盈触发 | 成交:{tr['trail_stop']:g} (+{self._bp_from(entry, tr['trail_stop']):.2f}bp) | 全部平仓")
                if price >= tr["sl"] and tr["realized_frac"] < 1.0:
                    frac = 1.0 - tr["realized_frac"]
                    pnl_bp = self._bp_from(entry, tr["sl"]) * frac
                    tr["realized_bp"] += pnl_bp
                    tr["realized_frac"] = 1.0
                    hit_SL = True
                    fill_msgs.append(f"止损触发 | 成交:{tr['sl']:g} ({self._bp_from(entry, tr['sl']):.2f}bp) | 全部平仓")

        if fill_msgs:
            self._push("【SIM】" + tr["symbol"].upper() + " | " + " / ".join(fill_msgs))

        if tr["realized_frac"] >= 1.0:
            fees_bp = self._fees_roundtrip_bp()
            net_bp = tr["realized_bp"] - fees_bp
            tr["status"] = "CLOSED"
            tr["closed_at"] = now_ts
            tr["gross_bp"] = tr["realized_bp"]
            tr["net_bp"] = net_bp
            self.closed.append(tr)
            self.ps = [x for x in self.ps if x["id"] != tr["id"]]

            win = 1 if net_bp > 0 else 0
            ymd = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            if self._daily_ymd != ymd:
                self._daily_ymd = ymd
                self._daily = {"n": 0, "win": 0, "gross_bp": 0.0, "net_bp": 0.0}
            self._daily["n"] += 1
            self._daily["win"] += win
            self._daily["gross_bp"] += tr["gross_bp"]
            self._daily["net_bp"] += tr["net_bp"]

            wr = (self._daily["win"] / max(1, self._daily["n"])) * 100.0
            self._push(
                f"【SIM】平仓总结 | {tr['symbol'].upper()} | {tr['side']} | 毛:{tr['gross_bp']:.2f}bp "
                f"净:{tr['net_bp']:.2f}bp(扣费≈{fees_bp:.2f}bp) | 当日：{self._daily['n']}笔 胜率:{wr:.1f}% 净:{self._daily['net_bp']:.2f}bp"
            )

    def on_tick(self, symbol: str, price: float, ts: float = None):
        if not SIM_CFG.get("enable", True):
            return
        ts = ts or self._now_ts()
        for tr in list(self.ps):
            if tr["symbol"].lower() != symbol.lower():
                continue
            self._update_one(tr, float(price), ts)


SIM = PhaseSim(CONFIG, pusher_obj=pusher)


class SymbolRunner:
    _push_seen: Dict[str, float] = {}
    _push_ttl_sec = 5.0

    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.collector = BinanceCollector(self.symbol, config=CONFIG)
        self.fb = FeatureBuilder(seq_len=SEQ_LEN)
        self.mm = ModelManager(self.symbol)
        self.sf = SignalFusion()
        self.risk = RiskController(self.symbol)

        self._last_depth = None
        self._last_push_ts = 0.0
        self._last_infer_ts = 0.0

        self._closes = deque(maxlen=max(SEQ_LEN * 4, 200))
        self._sig_buf = deque(maxlen=3)
        self.logger = logging.getLogger(f"runner.{self.symbol}")

        self.executor = None  # 由 main() 注入

        # 阶段性事件去重
        self._last_phase_ts: float = 0.0
        self._phase_dedup_sec = int(CONFIG.get("phase", {}).get("dedup_sec", 90))

        # === 中期模块 ===
        self.mid = MidTrend(CONFIG)
        self._swing_task: Optional[asyncio.Task] = None
        self._last_swing_bar: Optional[int] = None
        self._last_swing_side: Optional[str] = None

    def _append_close(self, price: float):
        try:
            p = float(price)
            if p > 0:
                self._closes.append(p)
        except Exception:
            pass

    def _confirm_signal(self, signal: str, prob: float,
                        max_gap_sec: int = CONFIRM_MAX_GAP,
                        need_prob: float = CONFIRM_NEED_PROB) -> bool:
        if signal not in ("BUY", "SELL"):
            return False
        now = time.time()
        self._sig_buf.append((now, signal, float(prob)))
        if len(self._sig_buf) >= 2:
            t2, s2, p2 = self._sig_buf[-1]
            t1, s1, p1 = self._sig_buf[-2]
            if s1 == s2 == signal and p1 >= need_prob and p2 >= need_prob and (t2 - t1) <= max_gap_sec:
                return True
        return False

    def _push_phase_event(self, phase: dict):
        try:
            now = time.time()
            ts_evt = float(phase.get("ts_ms", now * 1000)) / 1000.0
            if (ts_evt - self._last_phase_ts) < self._phase_dedup_sec:
                return
            self._last_phase_ts = ts_evt

            kind = str(phase.get("kind", "")).lower()
            side_txt = "阶段性低点" if kind == "bottom" else "阶段性高点"
            conf = float(phase.get("conf", 0.0))
            price = float(phase.get("price", 0.0))
            t1 = float(phase.get("t1", price))
            t2 = float(phase.get("t2", price))
            t3 = float(phase.get("t3", price))
            sl = float(phase.get("sl", price))
            atr_pct = float(phase.get("atr_pct", 0.006))
            trail_k = float(phase.get("trail_k", CONFIG.get("phase", {}).get("trail_k_atr", 0.6)))

            ev = phase.get("evidence", {}) or {}
            ev_bits = []
            if "z" in ev and "z_thr" in ev:
                ev_bits.append(f"z={ev['z']:+.2f}(thr={ev['z_thr']:+.2f})")
            if "rsi" in ev:
                ev_bits.append(f"rsi={ev['rsi']:.1f}")
            if "macdH" in ev:
                ev_bits.append(f"macdH={ev['macdH']:+.4f}")
            if "dH" in ev:
                ev_bits.append(f"dH={ev['dH']:+.4f}")
            if "imb" in ev:
                ev_bits.append(f"imb={ev['imb']:+.3f}")
            if "gate" in ev:
                ev_bits.append(f"gate={ev['gate']:.2f}")
            if "lc_score" in ev and "lc_state" in ev and "lc_side" in ev:
                ev_bits.append(f"LC={ev['lc_score']:.2f}/{ev['lc_side']}/{ev['lc_state']}")
            ev_str = " | ".join(ev_bits) if ev_bits else ""

            ph_cfg = CONFIG.get("phase", {}) or {}
            tradeable = bool(phase.get("tradeable", False))
            if "tradeable" not in phase:
                edge_bp = abs((t1 - price) / max(price, 1e-12)) * 1e4
                fees_bp = 2.0 * (TRADING_CFG.get("taker_fee_bp", 0.5) + TRADING_CFG.get("slippage_bp", 0.0))
                r1 = edge_bp / max(fees_bp, 1e-9)
                tradeable = (edge_bp >= float(ph_cfg.get("min_edge_bp", 8))) and \
                            (r1 >= float(ph_cfg.get("min_r1", 1.10))) and \
                            (edge_bp >= float(ph_cfg.get("min_edge_fee_x", 3.0)) * fees_bp)

            header = f"{side_txt} | 置信:{conf:.2f} | 价格:{price:g}"
            band = (f"波段空间估计：T1={t1:g}({((t1/price-1)*100):+.2f}%) | "
                    f"T2={t2:g}({((t2/price-1)*100):+.2f}%) | "
                    f"T3={t3:g}({((t3/price-1)*100):+.2f}%) | "
                    f"SL={sl:g}({((sl/price-1)*100):+.2f}%)")
            tag = "✅可交易" if tradeable else "❎空间不足"
            advice = ("买入→分批止盈 40%/40%/20% 于 T1/T2/T3；到达 T1 后止损抬至保本；到达 T2 后启用追踪止盈（≈0.6×ATR）。"
                      if kind == "bottom"
                      else "做空→分批止盈 40%/40%/20% 于 T1/T2/T3；到达 T1 后止损下调至保本；到达 T2 后启用追踪止盈（≈0.6×ATR）。")
            content = f"{header}\n{band}\n执行建议：{tag}；{advice}\n证据：{ev_str}"

            self.logger.info(content)
            if pusher:
                try:
                    if hasattr(pusher, "push_text"):
                        pusher.push_text(content)
                    else:
                        pusher.send_text(content)
                except Exception as e:
                    self.logger.error(f"[{self.symbol}] 阶段性推送失败: {e}")

            if SIM_CFG.get("enable", True) and SIM_CFG.get("use_phase_signals", True) and tradeable:
                SIM.open_from_phase(self.symbol, {
                    "kind": "bottom" if kind == "bottom" else "top",
                    "conf": conf, "price": price, "t1": t1, "t2": t2, "t3": t3, "sl": sl,
                    "atr_pct": atr_pct, "trail_k": trail_k
                })
        except Exception as e:
            self.logger.error(f"[{self.symbol}] _push_phase_event 异常: {e}")

    async def _swing_loop(self):
        """中期趋势/波段空间评估循环（独立于短线WS）"""
        mt_cfg = CONFIG.get("midtrend", {}) or {}
        if not mt_cfg.get("enable", True):
            return
        poll = max(60, int(mt_cfg.get("poll_seconds", 300)))
        self.logger.info(f"[{self.symbol}] SWING loop启动，每 {poll}s 评估一次")
        while True:
            try:
                res = self.mid.analyze(self.symbol)
                if isinstance(res, dict) and int(res.get("score", 0)) >= int(res.get("score_open", mt_cfg.get("score_open", 65))):
                    bar_time = int(res.get("bar_time", 0))
                    side = res.get("side", "LONG")  # LONG/SHORT
                    if bar_time != self._last_swing_bar or side != self._last_swing_side:
                        # 推送
                        symU = self.symbol.upper()
                        t1 = res.get("t1"); t2 = res.get("t2"); t3 = res.get("t3"); sl = res.get("sl")
                        score = res.get("score"); regime = res.get("regime")
                        adx = res.get("adx"); atrp = res.get("atr_pct"); ema200d = res.get("ema200_d")

                        content = (
                            f"【SWING】{symU} | {side} | 评分 {safe_fmt(score,0)}/100 | Regime:{regime}\n"
                            f"目标：T1={safe_fmt(t1,0)}  T2={safe_fmt(t2,0)}  T3={safe_fmt(t3,0)}  |  SL={safe_fmt(sl,0)}\n"
                            f"证据：ADX={safe_fmt(adx,1)}/{mt_cfg.get('adx_thr', 22)} | ATR%={safe_fmt(atrp*100 if atrp is not None else None,2)}% | EMA200Δ={safe_fmt(ema200d*100 if ema200d is not None else None,2)}%"
                        )
                        self.logger.info(content)
                        if pusher:
                            try:
                                if hasattr(pusher, "push_text"):
                                    pusher.push_text(content)
                                else:
                                    pusher.send_text(content)
                            except Exception as e:
                                self.logger.error(f"[{self.symbol}] SWING 推送失败: {e}")

                        # 可选：用PhaseSim开“观测仓”
                        if SIM_CFG.get("enable", True) and SIM_CFG.get("use_phase_signals", True):
                            price_now = float(PRICE_CACHE.get(self.symbol, t1 if side == "LONG" else t3) or 0.0)
                            SIM.open_from_phase(self.symbol, {
                                "kind": "bottom" if side == "LONG" else "top",
                                "conf": float(score or 0) / 100.0,
                                "price": price_now,
                                "t1": float(t1 or price_now), "t2": float(t2 or price_now), "t3": float(t3 or price_now),
                                "sl": float(sl or price_now),
                                "atr_pct": float(res.get("atr_pct", 0.006) or 0.006),
                                "trail_k": CONFIG.get("phase", {}).get("trail_k_atr", 0.6)
                            })

                        self._last_swing_bar = bar_time
                        self._last_swing_side = side
            except Exception as e:
                self.logger.error(f"[{self.symbol}] SWING 异常: {e}")
            await asyncio.sleep(poll)

    async def start(self):
        self.logger.info(f"[{self.symbol}] Runner 已启动")
        asyncio.create_task(self.collector.start())
        # 启动SWING循环
        self._swing_task = asyncio.create_task(self._swing_loop())

        try:
            async for trade, depth in self.collector.stream():
                if depth:
                    bb, ba = _extract_best_from_depth(depth)
                    if bb and ba:
                        OB_CACHE[self.symbol] = (bb, ba)
                        self._last_depth = depth

                if not trade:
                    continue

                payload = trade.get("data", trade)
                price = payload.get("p") or payload.get("price")
                if price is not None:
                    PRICE_CACHE[self.symbol] = float(price)
                    self._append_close(price)
                    SIM.on_tick(self.symbol, float(price), time.time())

                seq = self.fb.build(trade, self._last_depth)
                if seq is None or (hasattr(seq, "__len__") and len(seq) == 0):
                    continue

                if INFER_COOLDOWN > 0:
                    now_cool = time.time()
                    if now_cool - self._last_infer_ts < INFER_COOLDOWN:
                        continue
                    self._last_infer_ts = now_cool

                label, prob = self.mm.predict(seq)
                self.logger.info(f"[{self.symbol}] prob={prob:.3f} label={label}")

                book_snapshot, last_q_ms, d_spread_dt_bp, now_ms = self.collector.get_orderbook_ctx()

                last_feat = np.asarray(seq)[-1]
                vol_abs = float(last_feat[6]) if len(last_feat) > 6 else 0.0
                vol_regime = max(0.0, min(1.0, vol_abs / 0.2)) if vol_abs > 0 else 0.5

                fused_signal, diag, phase_evt = self.sf.fuse(
                    label, seq, prob,
                    book_snapshot=book_snapshot,
                    now_ts_ms=now_ms,
                    last_quote_change_ms=last_q_ms,
                    fee_taker_bp=TAKER_FEE_BP,
                    d_spread_dt_bp=d_spread_dt_bp,
                    asr_flag=False,
                    vwap_bias=0.0,
                    vol_regime=vol_regime,
                    closes=list(self._closes)
                )
                self.logger.info(f"[{self.symbol}] fused_signal={fused_signal} | {diag}")

                # 阶段性事件推送 & 模拟
                if isinstance(phase_evt, dict):
                    self._push_phase_event(phase_evt)

                # 统一门控：先看是否允许执行（观点与执行分离）
                mm_cfg = CONFIG.get("mm", {}) or {}
                env_str, gate_reject = build_env_and_gate(diag or "", mm_cfg)
                model_line = format_model_confirm(diag or "", {
                    "score_open": mm_cfg.get("score_open", 0.62),
                    "score_force": mm_cfg.get("score_force", 0.52),
                    "score_open_low_vol": mm_cfg.get("score_open_low_vol", mm_cfg.get("score_open", 0.62)),
                    "score_force_low_vol": mm_cfg.get("score_force_low_vol", mm_cfg.get("score_force", 0.52)),
                })

                # 如果门控拒绝：不开新仓，但已持仓时仍允许风控检查平仓（TP/SL/超时等）
                if gate_reject:
                    if prob is None:
                        continue

                    # 若当前已有仓位，优先让风险模块做持仓管理，避免“只开不平”
                    in_pos = getattr(self.risk, "position", "HOLD") in ("BUY", "SELL")
                    if in_pos:
                        trade_for_risk = {
                            "symbol": self.symbol,
                            "p": float(price) if price is not None else None,
                            "close": list(self._closes) if len(self._closes) else None,
                            "p_hat_prob": float(prob),
                            "p_min": getattr(self.sf, "p_min", 0.0),
                            "reason_text": diag
                        }
                        decision, reason = self.risk.judge("HOLD", trade_for_risk)
                        self.logger.info(f"[{self.symbol}] gate_reject risk_check decision={decision} reason={reason}")
                        if decision in ("BUY", "SELL"):
                            await self._maybe_push(decision, trade_for_risk, prob,
                                                   hint=diag, fused_signal=fused_signal, reason=reason,
                                                   env_str=env_str, model_line=model_line)
                            if self.executor:
                                exec_resp = self.executor.execute(self.symbol, decision, reason)
                                self.logger.info(f"[{self.symbol}] EXEC {exec_resp}")
                            continue

                    await self._maybe_push("HOLD@filtered", {
                        "p": float(price) if price is not None else None
                    }, prob, fused_signal=fused_signal, reason="filtered", env_str=env_str, model_line=model_line)
                    continue

                # 双确认
                if not self._confirm_signal(fused_signal, prob, CONFIRM_MAX_GAP,
                                            CONFIRM_NEED_PROB_LOW_VOL if "低波动" in env_str else CONFIRM_NEED_PROB):
                    continue

                # 进入风控裁决
                trade_for_risk = {
                    "symbol": self.symbol,
                    "p": float(price) if price is not None else None,
                    "close": list(self._closes) if len(self._closes) else None,
                    "p_hat_prob": float(prob),
                    "p_min": getattr(self.sf, "p_min", 0.0),
                    "reason_text": diag
                }
                decision, reason = self.risk.judge(fused_signal, trade_for_risk)
                self.logger.info(f"[{self.symbol}] decision={decision} reason={reason}")
                if decision not in ("BUY", "SELL"):
                    # 非买卖也给出“模型观点 + 环境”但不强推
                    await self._maybe_push(f"HOLD@{reason or 'no_change'}", trade_for_risk, prob,
                                           fused_signal=fused_signal, reason=reason, env_str=env_str, model_line=model_line)
                    continue

                # 推送 + 执行
                await self._maybe_push(decision, trade_for_risk, prob,
                                       hint=diag, fused_signal=fused_signal, reason=reason,
                                       env_str=env_str, model_line=model_line)

                if self.executor:
                    exec_resp = self.executor.execute(self.symbol, decision, reason)
                    self.logger.info(f"[{self.symbol}] EXEC {exec_resp}")

        except asyncio.CancelledError:
            self.logger.info(f"[{self.symbol}] 收到取消信号，准备退出 …")
            raise
        except Exception as e:
            self.logger.error(f"[{self.symbol}] 处理异常: {e}")
            traceback.print_exc()
            await asyncio.sleep(0.1)

    async def _maybe_push(self, decision, trade_msg, prob,
                          hint="", fused_signal=None, reason="",
                          env_str: str = "", model_line: str = ""):
        now = time.time()
        if now - self._last_push_ts < MIN_PUSH_INTERVAL:
            return

        price = trade_msg.get("p") or trade_msg.get("price")
        symU = self.symbol.upper()

        # 统一文案
        if decision.startswith("HOLD@"):
            content = (
                f"{symU}\n"
                f"决策:{decision} 价格:{safe_fmt(price, 2)} 置信:{prob:.2f}\n"
                f"{model_line}\n"
                f"环境: {env_str}"
            )
        elif decision in ("BUY", "SELL"):
            content = (
                f"{symU}\n"
                f"决策:{decision}{('@' + reason) if reason else ''} 价格:{safe_fmt(price, 2)} 置信:{prob:.2f}\n"
                f"{model_line}\n"
                f"环境: {env_str}"
            )
        else:
            content = (
                f"{symU}\n"
                f"决策:{decision}{('@' + reason) if reason else ''} 价格:{safe_fmt(price, 2)} 置信:{prob:.2f}\n"
                f"{model_line}\n"
                f"环境: {env_str}"
            )

        key = f"{self.symbol}|{decision}|{round(float(price or 0),2)}|{int(prob*100)}|{reason}|{fused_signal}"
        last_ts = self._push_seen.get(key, 0.0)
        if now - last_ts < self._push_ttl_sec:
            self.logger.debug(f"[{self.symbol}] 推送去重拦截: {key}")
            return
        self._push_seen[key] = now

        self._last_push_ts = now

        self.logger.info(content)
        if pusher:
            try:
                if hasattr(pusher, "push_text"):
                    pusher.push_text(content)
                else:
                    pusher.send_text(content)
            except Exception as e:
                self.logger.error(f"[{self.symbol}] 推送失败: {e}")


async def main():
    if torch.cuda.is_available():
        logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
        logger.info("未检测到 GPU，使用 CPU 多线程")

    logger.info(f"=== 启动 {len(SYMBOLS)} 个交易对: {', '.join(SYMBOLS)} ===")
    logger.info(f"低算力模式: {'ON' if LOW_GPU_MODE else 'OFF'}，推理冷却: {INFER_COOLDOWN:.2f}s")
    logger.info(f"最小推送间隔: {MIN_PUSH_INTERVAL}s, 双确认: need_prob={CONFIRM_NEED_PROB}, max_gap={CONFIRM_MAX_GAP}s")

    # 打印 MM Gate 配置
    try:
        mm_cfg = (CONFIG.get("mm", {}) or {})
        logger.info("[MM Gate] enable=%s k_levels=%s gate=%s",
                    mm_cfg.get("enable", True),
                    mm_cfg.get("k_levels", 3),
                    str(mm_cfg.get("gate_mode", "basic")))
        logger.info("[MM Gate] min_spread=%.4f imb_th=%.2f score_open=%.2f score_force=%.2f ml_margin=%.2f",
                    float(mm_cfg.get("min_spread", 0.0004)),
                    float(mm_cfg.get("imb_th", 0.18)),
                    float(mm_cfg.get("score_open", 0.75)),
                    float(mm_cfg.get("score_force", 0.55)),
                    float(mm_cfg.get("ml_margin", 0.08)))
        lc = (mm_cfg.get("lc") or {})
        if lc:
            logger.info("[MM Gate.LC] cancel_build=%.2f cancel_cool=%.2f add_multi=%.2f",
                        float(lc.get("cancel_build", 0.65)),
                        float(lc.get("cancel_cool", 0.50)),
                        float(lc.get("add_multi", 2.0)))
    except Exception as e:
        logger.warning(f"[MM Gate] 配置打印失败: {e}")

    # 打印 SWING 配置
    try:
        mt = CONFIG.get("midtrend", {}) or {}
        logger.info("[SWING] enable=%s poll=%ss score_open=%s hold_days=%s vol_method=%s",
                    bool(mt.get("enable", True)),
                    int(mt.get("poll_seconds", 300)),
                    int(mt.get("score_open", 65)),
                    int(mt.get("hold_days", 5)),
                    str(mt.get("vol_method", "yz")))
    except Exception as e:
        logger.warning(f"[SWING] 配置打印失败: {e}")

    runners = [SymbolRunner(sym) for sym in SYMBOLS]

    executor = TradeExecutor(
        CONFIG,
        price_getter=lambda s: PRICE_CACHE.get(s),
        ob_getter=lambda s: OB_CACHE.get(s)
    )
    for r in runners:
        r.executor = executor

    if pusher:
        try:
            test_msg = f"✅ 系统启动测试推送 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            logger.info(test_msg)
            if hasattr(pusher, "push_text"):
                pusher.push_text(test_msg)
            else:
                pusher.send_text(test_msg)
            logger.info("推送测试消息已发送")
        except Exception as e:
            logger.error(f"测试推送失败: {e}")

    tasks = [asyncio.create_task(r.start()) for r in runners]

    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()

    def _ask_exit(signame):
        logger.info(f"收到信号 {signame}，准备优雅退出 …")
        stop_evt.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _ask_exit, sig.name)
        except NotImplementedError:
            pass

    await stop_evt.wait()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("所有子任务已结束，主循环退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("进程结束（Ctrl+C）")
