# modules/signal.py
# 决策逻辑（稳健版）：
# - 温度缩放 + 融合校准；基于 ATR% 的期望值阈值 p_min 入场；
# - 低波动拦截（强信号可越过）；可选逆动量 boost；
# - 优先 MM 狙击（含 LC：撤单/堆墙 加分），并修复方向边际校验；
# - 阶段事件（Phase）：冷启动限幅（ATR/MAD 地板与上限）、只推“可交易 + 过置信阈”的事件。
import json
import os
import yaml
import numpy as np
import time
from collections import deque
from typing import Any, Dict, List, Tuple

try:
    from reversal import ReversalDetector
except Exception:
    ReversalDetector = None

class SignalFusion:
    def __init__(self):
        # ===== 默认参数（可被 thresholds.json 与 config.yaml 覆盖）=====
        self.vol_threshold = 0.01  # 仅用于逆动量 boost 缩放
        self.rev_enable = True
        self.rev_base_boost = 0.08
        self.rev_gate = 0.35
        self.rev_hard = False

        # 低波拦截与兜底
        self.low_vol_bp = 0.0002              # 低于该 ATR% 视为低波（2bp）
        self.low_vol_override_margin = 0.07   # Δp ≥ 0.07 可越过
        self.low_vol_macd_gate = 0.30
        self.low_vol_rsi_lo = 35.0
        self.low_vol_rsi_hi = 65.0

        # 做市商狙击（MM）
        self.mm_cfg = {
            "enable": True,
            "k_levels": 3,
            "min_spread": 0.0004,   # 4 bps = 0.04%
            "fee_mult": 1.6,
            "slip_bps": 1.0,
            "imb_th": 0.18,
            "quote_age_ms": 120,
            "cooldown_s": 90,
            "score_open": 0.75,
            "score_force": 0.55,    # 强直通门槛
            "gate_mode": "basic",   # basic | hybrid | relaxed | force
            "lc": {"cancel_build": 0.65, "cancel_cool": 0.50, "add_multi": 2.0},
            "ml_margin": 0.08,
            "vwap_nudge": 0.03
        }

        # 读取 config.yaml
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.vol_threshold = float(cfg.get("vol_threshold", self.vol_threshold))
        rev_cfg = cfg.get("reversal", {}) or {}
        self.rev_enable = bool(rev_cfg.get("enable", self.rev_enable))
        self.rev_base_boost = float(rev_cfg.get("boost", self.rev_base_boost))
        self.rev_gate = float(rev_cfg.get("gate", self.rev_gate))
        self.rev_hard = bool(rev_cfg.get("hard_override", self.rev_hard))

        lowv = cfg.get("low_vol", {}) or {}
        self.low_vol_bp = float(lowv.get("bp", self.low_vol_bp))
        self.low_vol_override_margin = float(lowv.get("override_margin", self.low_vol_override_margin))
        self.low_vol_macd_gate = float(lowv.get("macd_gate", self.low_vol_macd_gate))
        self.low_vol_rsi_lo = float(lowv.get("rsi_lo", self.low_vol_rsi_lo))
        self.low_vol_rsi_hi = float(lowv.get("rsi_hi", self.low_vol_rsi_hi))

        # 读 fusion.atr 的 floor/ceil（用于冷启动限幅）
        fusion_atr = (cfg.get("fusion", {}) or {}).get("atr", {}) or {}
        self.atr_floor = float(fusion_atr.get("floor", 5e-6))
        self.atr_ceil  = float(fusion_atr.get("ceil", 7.5e-3))
        self.pmin_low_vol_floor = float((cfg.get("strategy", {}) or {}).get("pmin_low_vol_floor", 0.54))
        self.pmin_cap = float((cfg.get("strategy", {}) or {}).get("pmin_cap", 0.62))

        mm_user = (cfg.get("mm", {}) or {})
        # shallow keys
        for k in [kk for kk in self.mm_cfg.keys() if kk != "lc"]:
            if k in mm_user:
                self.mm_cfg[k] = mm_user[k]
        # nested lc
        if isinstance(mm_user.get("lc"), dict):
            self.mm_cfg.setdefault("lc", {})
            self.mm_cfg["lc"].update(mm_user["lc"])

        # thresholds.json
        self.T_tft = 1.0
        self.T_nbt = 1.0
        self.blend_w = [0.6, 0.4]
        self.blend_b = 0.0
        self.risk_params = {
            "fee_bp": 0.0005,
            "tp_mult": 2.0,
            "sl_mult": 2.0,
            "tp_min": 0.003,
            "sl_min": 0.003,
            "future_holding": 30
        }
        if os.path.exists("thresholds.json"):
            try:
                j = json.load(open("thresholds.json", "r"))
                t = j.get("temp", 1.0)
                if isinstance(t, dict):
                    self.T_tft = float(t.get("tft", 1.0))
                    self.T_nbt = float(t.get("nbt", 1.0))
                else:
                    self.T_tft = float(t); self.T_nbt = float(t)
                b = j.get("blend", {})
                if isinstance(b, dict):
                    self.blend_w = b.get("w", self.blend_w)
                    self.blend_b = float(b.get("b", self.blend_b))
                self.risk_params = j.get("risk", self.risk_params)
            except Exception as e:
                print(f"[SignalFusion] thresholds.json 读取失败: {e}")

        # 逆动量
        self.reversal = None
        if self.rev_enable and ReversalDetector is not None:
            try:
                self.reversal = ReversalDetector()
            except Exception as e:
                print(f"[SignalFusion] Reversal init failed: {e}")
                self.rev_enable = False
                self.reversal = None

        # 对外暴露
        self.p_min = 0.5

        # MM 冷却与撤单速率
        self._mm_last_open_ts = {"buy": 0.0, "sell": 0.0}
        self._mm_recent_adds = deque(maxlen=50)
        self._mm_recent_cancels = deque(maxlen=50)

        # Phase 配置（用于阶段性顶/底）
        self.phase_cfg = (cfg.get("phase", {}) or {})
        self.conf_gate = float(self.phase_cfg.get("conf_gate", 0.6))
        self.only_push_tradeable = bool(self.phase_cfg.get("only_push_tradeable", True))

        self.cfg = cfg  # 备份整体 cfg 供内部使用

    # ========= 工具 =========
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _blend_prob(self, lg_tft: float, lg_nbt: float) -> float:
        z = self.blend_w[0] * (lg_tft / self.T_tft) + self.blend_w[1] * (lg_nbt / self.T_nbt) + self.blend_b
        return self._sigmoid(z)

    def _p_min(self, atr_pct: float, fee: float, tp_mult: float, sl_mult: float, tp_min: float, sl_min: float) -> float:
        # E = p*G - (1-p)*L - fee > 0  =>  p > (L+fee)/(G+L)
        G = max(tp_mult * atr_pct, tp_min)
        L = max(sl_mult * atr_pct, sl_min)
        return (L + fee) / (G + L + 1e-12)

    def _atr_pct_estimate(self, feats_seq: Any) -> float:
        """
        估计 ATR%：若 vol<0.2，视为已归一化波动（如 0.004 = 0.4%），直接用；否则 vol/close。
        并做冷启动夹取： [atr_floor, atr_ceil]
        """
        last = np.asarray(feats_seq)[-1]
        close = float(last[0])
        vol = float(last[6]) if len(last) > 6 else 0.0
        if 0 < vol < 0.2:
            atr = max(1e-6, vol)
        else:
            atr = max(1e-6, vol / max(close, 1e-12))
        return float(min(self.atr_ceil, max(self.atr_floor, atr)))

    # ========= MM 工具 =========
    @staticmethod
    def _safe_num(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except Exception:
            return default

    @staticmethod
    def _lvl_val(level: Any, key: str) -> float:
        if isinstance(level, dict):
            return SignalFusion._safe_num(level.get(key, 0.0))
        return SignalFusion._safe_num(getattr(level, key, 0.0))

    def _depth_k(self, side_levels: List[Any], k: int) -> float:
        s = 0.0
        for lvl in side_levels[:max(0, k)]:
            s += self._lvl_val(lvl, "size")
        return s

    def _book_features(self, book_snapshot: Any, k: int, last_quote_change_ms: int, now_ts_ms: int, vol_regime: float) -> Dict[str, float]:
        bid_levels = book_snapshot.get("bid") if isinstance(book_snapshot, dict) else getattr(book_snapshot, "bid", [])
        ask_levels = book_snapshot.get("ask") if isinstance(book_snapshot, dict) else getattr(book_snapshot, "ask", [])
        if not bid_levels or not ask_levels:
            return {}

        bid = self._lvl_val(bid_levels[0], "price")
        ask = self._lvl_val(ask_levels[0], "price")
        if bid <= 0 or ask <= 0 or ask <= bid:
            return {}

        mid = 0.5 * (bid + ask)
        spread = (ask - bid) / max(mid, 1e-12)

        bid_sz_k = self._depth_k(bid_levels, k)
        ask_sz_k = self._depth_k(ask_levels, k)
        denom = max(bid_sz_k + ask_sz_k, 1e-12)
        depth_imb = (ask_sz_k - bid_sz_k) / denom

        micro = (ask * bid_sz_k + bid * ask_sz_k) / denom
        micro_bias = (micro - mid) / max(mid, 1e-12)

        quote_age_ms = max(0, int(now_ts_ms) - int(last_quote_change_ms))

        adds = sum(self._mm_recent_adds) if self._mm_recent_adds else 0
        cancels = sum(self._mm_recent_cancels) if self._mm_recent_cancels else 0
        cancel_rate = cancels / max(adds + cancels, 1e-12)

        stats = {}
        try:
            stats = (book_snapshot.get("stats") if isinstance(book_snapshot, dict) else {}) or {}
        except Exception:
            stats = {}

        adds_b = int(stats.get("adds_b", 0))
        canc_b = int(stats.get("canc_b", 0))
        adds_a = int(stats.get("adds_a", 0))
        canc_a = int(stats.get("canc_a", 0))
        ofi    = float(stats.get("ofi", 0.0))

        return dict(
            spread=spread,
            depth_imb=depth_imb,
            micro_bias=micro_bias,
            quote_age_ms=quote_age_ms,
            cancel_rate=cancel_rate,
            vol_regime=float(vol_regime),
            bid=bid, ask=ask,
            adds_b=adds_b, canc_b=canc_b, adds_a=adds_a, canc_a=canc_a,
            ofi=ofi
        )

    def mm_rate_update(self, adds: int, cancels: int):
        self._mm_recent_adds.append(int(adds))
        self._mm_recent_cancels.append(int(cancels))

    def _mm_opportunity(
        self,
        feat: Dict[str, float],
        fee_taker_bp: float,
        d_spread_dt_bp: float,
        asr_flag: bool,
        p: float,
        p_min: float,
        vwap_bias: float
    ) -> Dict[str, Any]:
        if not feat:
            return {"active": False}

        cfg = self.mm_cfg
        # 点差要求：min_spread 或 费率倍数（含滑点安全边际）
        spread_ok = feat["spread"] >= max((fee_taker_bp / 1e4) * cfg["fee_mult"], cfg["min_spread"]) + (cfg["slip_bps"] / 1e4)
        # 深度失衡 + 微型价偏
        imb_ok = (abs(feat["depth_imb"]) >= cfg["imb_th"]) and (abs(feat["micro_bias"]) >= 0.5 * abs(feat["depth_imb"]))
        # 报价老化：更老更容易被吃
        age_ok = feat["quote_age_ms"] >= cfg["quote_age_ms"] or (d_spread_dt_bp > 0)
        # 模型边际
        ml_ok = (max(p - p_min, (1.0 - p) - p_min) >= cfg["ml_margin"])
        # 波动落在做市甜蜜区
        vol_ok = 0.2 <= feat["vol_regime"] <= 0.85
        # 有毒流过滤
        toxic_ok = not bool(asr_flag)

        # —— LC：撤单占比/堆墙 —— #
        tot_b = feat.get("adds_b", 0) + feat.get("canc_b", 0)
        tot_a = feat.get("adds_a", 0) + feat.get("canc_a", 0)
        cr_b = (feat.get("canc_b", 0) / max(tot_b, 1e-9)) if tot_b > 0 else 0.0
        cr_a = (feat.get("canc_a", 0) / max(tot_a, 1e-9)) if tot_a > 0 else 0.0

        lc = self.mm_cfg.get("lc", {})
        CANCEL_BUILD = float(lc.get("cancel_build", 0.65))
        CANCEL_COOL  = float(lc.get("cancel_cool", 0.50))
        ADD_MULTI    = float(lc.get("add_multi", 2.0))

        up_build   = (cr_a >= CANCEL_BUILD) or (feat.get("adds_b", 0) >= ADD_MULTI * max(1, feat.get("adds_a", 0)))
        up_cooling = (cr_a >= CANCEL_COOL)
        dn_build   = (cr_b >= CANCEL_BUILD) or (feat.get("adds_a", 0) >= ADD_MULTI * max(1, feat.get("adds_b", 0)))
        dn_cooling = (cr_b >= CANCEL_COOL)

        lc_state = "idle"
        if up_build and not dn_build:
            lc_state = "buy/building"
        elif up_cooling and not dn_build:
            lc_state = "buy/cooling"
        elif dn_build and not up_build:
            lc_state = "sell/building"
        elif dn_cooling and not up_build:
            lc_state = "sell/cooling"

        # 组合基础分
        score = 0.0
        for cond, w in [(spread_ok, 0.35), (imb_ok, 0.25), (age_ok, 0.15), (ml_ok, 0.15), (vol_ok, 0.05), (toxic_ok, 0.05)]:
            score += (w if cond else 0.0)

        # 方向与 vwap bias
        side = "sell" if feat["depth_imb"] > 0 else "buy"
        same_dir = (side == "buy" and vwap_bias > 0) or (side == "sell" and vwap_bias < 0)
        score += (cfg["vwap_nudge"] if same_dir else -cfg["vwap_nudge"])

        # LC 方向一致时，温和加分
        if lc_state.startswith("buy") and side == "buy":
            score += 0.08 if "building" in lc_state else 0.05
        if lc_state.startswith("sell") and side == "sell":
            score += 0.08 if "building" in lc_state else 0.05

        score = max(0.0, min(1.0, score))

        return {
            "active": True,
            "score": score,
            "side": side,
            "reasons": {
                "spread_ok": spread_ok, "imb_ok": imb_ok, "age_ok": age_ok,
                "ml_ok": ml_ok, "vol_ok": vol_ok, "toxic_ok": toxic_ok,
                "lc_state": lc_state,
                "cr_b": cr_b, "cr_a": cr_a, "adds_b": feat.get("adds_b", 0), "adds_a": feat.get("adds_a", 0),
                "ofi": feat.get("ofi", 0.0)
            },
            "feat": feat
        }

    def _mm_cooldown_ok(self, side: str, now_ts: float) -> bool:
        return (now_ts - self._mm_last_open_ts.get(side, 0.0)) >= float(self.mm_cfg["cooldown_s"])

    def _mm_mark_open(self, side: str, now_ts: float):
        self._mm_last_open_ts[side] = float(now_ts)

    # ========= 阶段性探测（稳健版） =========
    @staticmethod
    def _simple_rsi(series: List[float], period: int = 14) -> float:
        if not series or len(series) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(-period, -1):
            d = series[i+1] - series[i]
            if d >= 0: gains.append(d)
            else:      losses.append(-d)
        ag = np.mean(gains) if gains else 1e-12
        al = np.mean(losses) if losses else 1e-12
        rs = ag / max(al, 1e-12)
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _atr_pct_from_series(prices: List[float], win: int = 40) -> float:
        if not prices or len(prices) < 2:
            return 0.006
        arr = np.asarray(prices[-win:], dtype=float)
        diff = np.abs(np.diff(arr))
        atr = np.mean(diff) / max(np.mean(arr), 1e-12)
        return max(atr, 1e-6)

    def _phase_detect(
        self,
        feats_seq: Any,
        book_snapshot: Any,
        p: float,
        p_min: float,
        now_ts_ms: int,
        closes: List[float] = None
    ) -> Dict[str, Any] or None:
        """生成阶段性高/低点事件（用于外部推送+模拟）。"""
        try:
            last = np.asarray(feats_seq)[-1]
            close = float(last[0])
            macdH = float(last[3]) if len(last) > 3 else 0.0
            macd = float(last[4]) if len(last) > 4 else 0.0
            rsi = float(last[5]) if len(last) > 5 else 50.0
            vol_abs = float(last[6]) if len(last) > 6 else 0.0

            # ATR 估计（用 closes 更稳）+ 夹取
            atr_pct = self._atr_pct_estimate(feats_seq)
            if closes and len(closes) >= 10:
                win = int(self.cfg.get("fusion", {}).get("atr", {}).get("window", 40))
                atr_pct = self._atr_pct_from_series(closes, win=win)
                atr_pct = float(min(self.atr_ceil, max(self.atr_floor, atr_pct)))

            # 低波过滤（冷启动不刷屏）
            if atr_pct < self.low_vol_bp:
                return None

            # z 分数（MAD 近似）+ 防炸
            z = 0.0
            z_thr = float(self.cfg.get("reversal", {}).get("min_z", 1.0))
            if closes and len(closes) >= 20:
                arr = np.asarray(closes[-60:], dtype=float)
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                if mad <= 1e-9:
                    z = 0.0
                else:
                    z = (close - med) / (1.4826 * mad)
            z = float(np.clip(z, -6.0, 6.0))

            is_bottom = (z <= -z_thr and rsi <= 30.0)
            is_top    = (z >=  z_thr and rsi >= 70.0)
            if not (is_bottom or is_top):
                return None

            ph = self.phase_cfg
            t1m = float(ph.get("t1_mult_atr", 1.5))
            t2m = float(ph.get("t2_mult_atr", 2.5))
            t3m = float(ph.get("t3_mult_atr", 4.0))
            trail_k = float(ph.get("trail_k_atr", 0.6))
            slm = float(self.risk_params.get("sl_mult", 2.0))

            if is_bottom:
                t1 = close * (1.0 + t1m * atr_pct)
                t2 = close * (1.0 + t2m * atr_pct)
                t3 = close * (1.0 + t3m * atr_pct)
                sl = close * (1.0 - max(self.risk_params.get("sl_min", 0.003), slm * atr_pct))
                kind = "bottom"
            else:
                t1 = close * (1.0 - t1m * atr_pct)
                t2 = close * (1.0 - t2m * atr_pct)
                t3 = close * (1.0 - t3m * atr_pct)
                sl = close * (1.0 + max(self.risk_params.get("sl_min", 0.003), slm * atr_pct))
                kind = "top"

            # 可交易性评估（edge vs cost）
            taker_bp = float(self.cfg.get("trading", {}).get("taker_fee_bp", 0.5))
            slip_bp = float(self.cfg.get("trading", {}).get("slippage_bp", 0.0))
            fees_bp = 2.0 * (taker_bp + slip_bp)

            edge_bp = abs((t1 - close) / max(close, 1e-12)) * 1e4
            r1 = edge_bp / max(fees_bp, 1e-9)
            min_edge_bp = float(ph.get("min_edge_bp", 8))
            min_r1 = float(ph.get("min_r1", 1.10))
            min_fee_x = float(ph.get("min_edge_fee_x", 3.0))
            tradeable = (edge_bp >= min_edge_bp) and (r1 >= min_r1) and (edge_bp >= min_fee_x * fees_bp)

            # 置信度：用 max(p,1-p) 与门槛的相对距离粗估
            conf = max(p, 1.0 - p)

            ev = {
                "z": z, "z_thr": z_thr,
                "rsi": rsi,
                "macdH": macdH,
                "dH": float(last[2]) if len(last) > 2 else 0.0,
                "imb": float(last[7]) if len(last) > 7 else 0.0,
                "gate": conf
            }

            # —— 稳健：不在这里过滤 “only_push_tradeable/conf_gate”，让上层 main 决定是否推送；
            # 但把 tradeable/conf 都带上，供上层判断 —— #
            return {
                "kind": kind,
                "conf": float(conf),
                "price": float(close),
                "t1": float(t1),
                "t2": float(t2),
                "t3": float(t3),
                "sl": float(sl),
                "atr_pct": float(atr_pct),
                "trail_k": float(trail_k),
                "tradeable": bool(tradeable),
                "evidence": ev
            }
        except Exception:
            return None

    # ========= 主融合 =========
    def fuse(self, pred_logits_or_label, feats_seq, raw_probs=None, **ctx) -> Tuple[str, str, Any]:
        """
        兼容你的调用：(label:int, prob:float)
        返回: (signal, diag, phase_evt) —— phase_evt 可为 None
        ctx 可包含：
          book_snapshot, now_ts_ms, last_quote_change_ms, fee_taker_bp, d_spread_dt_bp,
          asr_flag(False), vwap_bias(0.0), vol_regime(0.5), regime(str), closes(list[float])
        """
        last = np.asarray(feats_seq)[-1]
        close = float(last[0])
        macd = float(last[4]) if len(last) > 4 else 0.0
        rsi = float(last[5]) if len(last) > 5 else 50.0
        vol_abs = float(last[6]) if len(last) > 6 else 0.0

        # 逆动量（可选）
        rev_sig, rev_conf, rev_reason = "HOLD", 0.0, ""
        if self.rev_enable and self.reversal is not None:
            try:
                rev_sig, rev_conf, rctx = self.reversal.decide(feats_seq)
                rev_reason = rctx.get("reason", "")
            except Exception:
                rev_sig, rev_conf = "HOLD", 0.0

        # 主概率
        p = float(raw_probs) if raw_probs is not None else float(pred_logits_or_label)

        # ATR% 与 p_min
        atr_pct_est = self._atr_pct_estimate(feats_seq)
        fee = float(self.risk_params.get("fee_bp", 0.0005))
        tp_mult = float(self.risk_params.get("tp_mult", 2.0))
        sl_mult = float(self.risk_params.get("sl_mult", 2.0))
        tp_min = float(self.risk_params.get("tp_min", 0.003))
        sl_min = float(self.risk_params.get("sl_min", 0.003))
        p_min_raw = self._p_min(atr_pct_est, fee, tp_mult, sl_mult, tp_min, sl_min)
        p_min = min(float(p_min_raw), float(self.pmin_cap))
        if atr_pct_est < self.low_vol_bp:
            p_min = min(p_min, float(self.pmin_low_vol_floor))
        self.p_min = float(p_min)

        regime = str(ctx.get("regime", "n/a"))
        diag = [
            f"p={p:.3f}", f"p_min={p_min:.3f}", f"p_min_raw={p_min_raw:.3f}", f"atr%={atr_pct_est:.3f}",
            f"macd={macd:.3f}", f"rsi={rsi:.1f}", f"vol={vol_abs:.4f}",
            f"regime={regime}"
        ]

        # MM 上下文标识
        book_snapshot = ctx.get("book_snapshot")
        mm_enable = bool(self.mm_cfg.get("enable", True))
        mm_active = mm_enable and (book_snapshot is not None)
        mm_score_last = 0.0
        diag.append(f"mm_ctx={'ON' if mm_active else 'OFF'}")

        # 逆动量 boost（只影响门槛）
        boost = 0.0
        if rev_sig != "HOLD" and rev_conf >= self.rev_gate:
            vol_ratio = 1.0 if self.vol_threshold <= 0 else min(1.0, vol_abs / self.vol_threshold)
            conf_ratio = min(1.0, rev_conf / max(self.rev_gate, 1e-6))
            boost = self.rev_base_boost * vol_ratio * conf_ratio
            diag.append(f"rev:{rev_sig}@{rev_conf:.2f} boost={boost:.3f}")
            if self.rev_hard:
                return rev_sig, " | ".join(diag), None

        # 原有基础 gating（先算出，供 MM 使用“方向一致性”）
        buy_gate_basic = (p >= max(0.5, p_min - boost)) and (rsi < 70) and (macd >= 0.0)
        sell_gate_basic = ((1.0 - p) >= max(0.5, p_min - boost)) and (rsi > 30) and (macd <= 0.0)

        # ====== 优先：MM 狙击（修复方向性边际 + 可观测性）======
        if mm_active:
            now_ts_ms = int(ctx.get("now_ts_ms", int(time.time() * 1000)))
            last_q_ms = int(ctx.get("last_quote_change_ms", now_ts_ms))
            fee_taker_bp = float(ctx.get("fee_taker_bp", 2.0))
            d_spread_dt_bp = float(ctx.get("d_spread_dt_bp", 0.0))
            asr_flag = bool(ctx.get("asr_flag", False))
            vwap_bias = float(ctx.get("vwap_bias", 0.0))
            vol_regime = float(ctx.get("vol_regime", 0.5))

            feat = self._book_features(book_snapshot, int(self.mm_cfg["k_levels"]), last_q_ms, now_ts_ms, vol_regime)
            opp = self._mm_opportunity(
                feat=feat, fee_taker_bp=fee_taker_bp, d_spread_dt_bp=d_spread_dt_bp,
                asr_flag=asr_flag, p=p, p_min=p_min, vwap_bias=vwap_bias
            )

            # 额外：展示双向还需的边际（便于调参观察）
            try:
                mlm = float(self.mm_cfg.get("ml_margin", 0.0))
                extra_buy = max(0.0, mlm - (p - p_min))
                extra_sell = max(0.0, mlm - ((1.0 - p) - p_min))
                diag.append(f"extra_margin(buy={extra_buy:.3f},sell={extra_sell:.3f})")
            except Exception:
                pass

            if opp.get("active", False):
                mm_score = float(opp["score"]); mm_score_last = mm_score
                mm_side = str(opp["side"])
                cfg = self.mm_cfg
                mm_ok = mm_score >= float(cfg["score_open"])
                cooldown_ok = self._mm_cooldown_ok(mm_side, now_ts_ms / 1000.0)

                # 方向性概率边际
                dir_margin = (p - p_min) if mm_side == "buy" else ((1.0 - p) - p_min)
                prob_ok = dir_margin >= float(cfg["ml_margin"])
                force_ok = (mm_score >= float(cfg["score_force"])) and prob_ok
                gate_mode = str(cfg.get("gate_mode", "basic")).lower()

                # 诊断细化
                try:
                    sp_bp = opp["feat"]["spread"] * 1e4
                except Exception:
                    sp_bp = 0.0
                diag += [
                    f"mm_score={mm_score:.3f}", f"mm_side={mm_side}",
                    f"spread_bp={sp_bp:.3f}",
                    f"imb={opp['feat']['depth_imb']:.3f}",
                    f"micro%={opp['feat']['micro_bias']*100:.3f}",
                    f"age_ms={int(opp['feat']['quote_age_ms'])}",
                    f"cooldown_ok={cooldown_ok}", f"gate_mode={gate_mode}",
                    f"dir_margin={dir_margin:+.3f}", f"mm_ok={mm_ok}", f"force_ok={force_ok}"
                ]
                # 打印 LC 状态
                try:
                    diag.append(f"LC={opp['reasons'].get('lc_state','idle')}")
                except Exception:
                    pass

                # gate 组合逻辑
                basic_dir_ok = (buy_gate_basic if mm_side == "buy" else sell_gate_basic)
                if gate_mode == "basic":
                    dir_ok = basic_dir_ok
                elif gate_mode == "hybrid":
                    dir_ok = (basic_dir_ok and mm_ok) or force_ok
                elif gate_mode == "relaxed":
                    # 低波环境下给做市盘口强分一个快速通道，避免长期全HOLD
                    dir_ok = mm_ok and (prob_ok or (mm_score >= float(cfg["score_force"]) * 0.9))
                elif gate_mode == "force":
                    dir_ok = force_ok
                else:
                    dir_ok = basic_dir_ok

                # 阻断理由
                block_reasons = []
                if not cooldown_ok:
                    block_reasons.append("cooldown")
                if gate_mode == "hybrid" and not ((basic_dir_ok and mm_ok) or force_ok):
                    if not basic_dir_ok: block_reasons.append("basic_dir")
                    if not mm_ok:       block_reasons.append("mm_ok")
                    if not force_ok:    block_reasons.append("force")
                elif gate_mode == "basic" and not basic_dir_ok:
                    block_reasons.append("basic_dir")
                elif gate_mode == "relaxed" and not (mm_ok and (prob_ok or (mm_score >= float(cfg["score_force"]) * 0.9))):
                    if not mm_ok:    block_reasons.append("mm_ok")
                    if not prob_ok and mm_score < float(cfg["score_force"]) * 0.9:  block_reasons.append("prob")
                elif gate_mode == "force" and not force_ok:
                    block_reasons.append("force")
                if block_reasons:
                    diag.append("mm_block(" + ",".join(block_reasons) + ")")

                # 实际执行
                if mm_side == "buy" and dir_ok and cooldown_ok:
                    self._mm_mark_open("buy", now_ts_ms / 1000.0)
                    tag = "mm_gate" if gate_mode in ("basic", "hybrid") else f"mm_gate_{gate_mode}"
                    return "BUY", " | ".join(diag + [f"BUY via {tag}"]), None
                if mm_side == "sell" and dir_ok and cooldown_ok:
                    self._mm_mark_open("sell", now_ts_ms / 1000.0)
                    tag = "mm_gate" if gate_mode in ("basic", "hybrid") else f"mm_gate_{gate_mode}"
                    return "SELL", " | ".join(diag + [f"SELL via {tag}"]), None
            else:
                diag.append("mm_unavailable")

        # ===== 低波拦截（放在 MM 之后；强信号可越过）=====
        atr_pct_est = max(atr_pct_est, 1e-6)
        low_vol = atr_pct_est < self.low_vol_bp
        diag.append(f"low_vol_bp={self.low_vol_bp:.6f} low_vol={'True' if low_vol else 'False'}")
        if low_vol:
            strong_prob = max(p - p_min, (1.0 - p) - p_min) >= self.low_vol_override_margin
            trend_evidence = (abs(macd) >= self.low_vol_macd_gate) or (rsi <= self.low_vol_rsi_lo) or (rsi >= self.low_vol_rsi_hi)
            mm_fastlane = mm_score_last >= float(self.mm_cfg.get("score_force", 0.55)) * 0.9
            if not ((strong_prob and trend_evidence) or mm_fastlane):
                phase_evt = self._phase_detect(feats_seq, book_snapshot, p, p_min, int(ctx.get("now_ts_ms", time.time()*1000)), ctx.get("closes"))
                return "HOLD", " | ".join(diag + [f"low_vol<{self.low_vol_bp:.4f} HOLD"]), phase_evt
            else:
                diag.append(f"low_vol_override (Δp={max(p - p_min, (1.0 - p) - p_min):.3f},mm_fastlane={mm_fastlane})")

        # ===== 常规 gating =====
        if buy_gate_basic:
            phase_evt = self._phase_detect(feats_seq, book_snapshot, p, p_min, int(ctx.get("now_ts_ms", time.time()*1000)), ctx.get("closes"))
            return "BUY", " | ".join(diag + ["BUY via p>p_min"]), phase_evt
        if sell_gate_basic:
            phase_evt = self._phase_detect(feats_seq, book_snapshot, p, p_min, int(ctx.get("now_ts_ms", time.time()*1000)), ctx.get("closes"))
            return "SELL", " | ".join(diag + ["SELL via (1-p)>p_min"]), phase_evt

        # 逆动量探针
        if rev_sig != "HOLD" and rev_conf >= (self.rev_gate + 0.2) and abs(macd) < 0.1:
            phase_evt = self._phase_detect(feats_seq, book_snapshot, p, p_min, int(ctx.get("now_ts_ms", time.time()*1000)), ctx.get("closes"))
            return rev_sig, " | ".join(diag + [f"probe:{rev_sig}@{rev_conf:.2f}"]), phase_evt

        phase_evt = self._phase_detect(feats_seq, book_snapshot, p, p_min, int(ctx.get("now_ts_ms", time.time()*1000)), ctx.get("closes"))
        return "HOLD", " | ".join(diag + ["HOLD"]), phase_evt
