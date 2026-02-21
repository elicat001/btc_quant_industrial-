# modules/midtrend.py
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None  # 由上层日志提示

import numpy as np

logger = logging.getLogger("modules.midtrend")


# ----------------------------
# 工具：安全函数/滚动计算
# ----------------------------
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    """简单 EMA（与 Wilder 略有差异，但足够用于 EMA 通道/Keltner）"""
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    w = alpha
    for i in range(1, len(arr)):
        out[i] = w * arr[i] + (1.0 - w) * out[i - 1]
    return out


def _wilder_rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder RMA，用于 ATR/ADX 计算"""
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = x[:period].mean() if len(x) >= period else x.mean()
    alpha = 1.0 / period
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def _rolling_max(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return a
    out = np.full_like(a, fill_value=np.nan, dtype=float)
    from collections import deque
    dq = deque()
    for i, v in enumerate(a):
        dq.append(v)
        if len(dq) > n:
            dq.popleft()
        out[i] = np.max(dq)
    return out


def _rolling_min(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return a
    out = np.full_like(a, fill_value=np.nan, dtype=float)
    from collections import deque
    dq = deque()
    for i, v in enumerate(a):
        dq.append(v)
        if len(dq) > n:
            dq.popleft()
        out[i] = np.min(dq)
    return out


# ----------------------------
# 波动率估计：YZ / GK / PK
# 输入为 OHLC numpy 数组
# ----------------------------
def realized_vol_pk(high: np.ndarray, low: np.ndarray) -> float:
    """Parkinson：基于高低价；返回“每根bar”的对数收益波动（非年化）"""
    hl = np.log(_safe_div(high, low, 1.0))
    var = (hl ** 2).mean() / (4.0 * math.log(2.0))
    return float(math.sqrt(max(var, 0.0)))


def realized_vol_gk(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Garman–Klass：开收/高低结合"""
    log_hl = np.log(_safe_div(high, low, 1.0))
    log_co = np.log(_safe_div(close, open_, 1.0))
    var = (0.5 * (log_hl ** 2) - (2.0 * (math.log(2) - 1.0)) * (log_co ** 2)).mean()
    return float(math.sqrt(max(var, 0.0)))


def realized_vol_yz(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """
    Yang–Zhang 简化实现（bar 级别），返回“每根bar”的对数收益波动（非年化）。
    适合作为更稳健的 RV，用于仓位/止损/持仓期调节。
    """
    k = 0.34 / (1.34 + (len(open_) + 1) / (len(open_) - 1 + 1e-9))
    log_ho = np.log(_safe_div(high, open_, 1.0))
    log_lo = np.log(_safe_div(low, open_, 1.0))
    log_co = np.log(_safe_div(close, open_, 1.0))
    log_oc = np.log(_safe_div(open_[1:], close[:-1], 1.0))
    sigma_o2 = np.var(log_oc) if len(log_oc) > 1 else 0.0
    sigma_c2 = np.var(log_co) if len(log_co) > 1 else 0.0
    sigma_rs2 = np.mean((log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)))
    var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
    return float(math.sqrt(max(var, 0.0)))


# ----------------------------
# 指标：ATR%、ADX、Donchian、Bollinger、Keltner
# ----------------------------
def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return _wilder_rma(tr, period)


def adx_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    up = high - np.roll(high, 1)
    dn = np.roll(low, 1) - low
    up[0] = 0.0; dn[0] = 0.0
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    tr_rma = _wilder_rma(tr, period)
    plus_di = 100.0 * _wilder_rma(plus_dm, period) / np.maximum(tr_rma, 1e-12)
    minus_di = 100.0 * _wilder_rma(minus_dm, period) / np.maximum(tr_rma, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    adx = _wilder_rma(dx, period)
    return adx


def bollinger(close: np.ndarray, period: int = 20, k: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ma = _ema(close, alpha=2.0 / (period + 1.0))
    # 使用样本标准差
    std = np.array([close[max(0, i - period + 1): i + 1].std(ddof=1) for i in range(len(close))], dtype=float)
    upper = ma + k * std
    lower = ma - k * std
    return lower, ma, upper


def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5):
    ema = _ema(close, alpha=2.0 / (period + 1.0))
    atr = atr_wilder(high, low, close, period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return lower, ema, upper, atr


# ----------------------------
# 配置与类
# ----------------------------
@dataclass
class MidCfg:
    enable: bool = True
    poll_seconds: int = 300
    adx_thr: int = 22
    don_base: int = 20
    don_alt: int = 55
    bb_period: int = 20
    ema_long: int = 200
    keltner_mult: float = 1.5
    atr_period: int = 20
    vol_method: str = "yz"      # yz | gk | pk
    hold_days: int = 5
    t1_atr: float = 1.0
    t2_atr: float = 1.6
    t3_atr: float = 2.5
    sl_atr: float = 1.2
    score_open: int = 65
    cooldown_min: int = 120
    interval: str = "5m"
    limit: int = 500
    # 权重
    w_trend: float = 0.40
    w_carry: float = 0.25
    w_vol: float = 0.15
    w_micro: float = 0.10  # 仅作乘子，默认 1.0
    w_onchain: float = 0.10  # 默认 0（BTC/ETH 可扩展）
    # 端点超时
    http_timeout: int = 6


class MidTrend:
    def __init__(self, config: Dict[str, Any]):
        mt = (config.get("midtrend") or {}).copy()
        self.cfg = MidCfg(
            enable=bool(mt.get("enable", True)),
            poll_seconds=int(mt.get("poll_seconds", 300)),
            adx_thr=int(mt.get("adx_thr", mt.get("adx_thr", 22))),
            don_base=int(mt.get("donchian_base", mt.get("don_base", 20))),
            don_alt=int(mt.get("donchian_alt", 55)),
            bb_period=int(mt.get("bb_period", 20)),
            ema_long=int(mt.get("ema_long", 200)),
            keltner_mult=float(mt.get("keltner_mult", 1.5)),
            atr_period=int(mt.get("atr_period", 20)),
            vol_method=str(mt.get("vol_method", "yz")),
            hold_days=int(mt.get("hold_days", 5)),
            t1_atr=float(mt.get("t1_atr", 1.0)),
            t2_atr=float(mt.get("t2_atr", 1.6)),
            t3_atr=float(mt.get("t3_atr", 2.5)),
            sl_atr=float(mt.get("sl_atr", 1.2)),
            score_open=int(mt.get("score_open", 65)),
            cooldown_min=int(mt.get("cooldown_min", 120)),
            interval=str(mt.get("interval", "5m")),
            limit=int(mt.get("limit", 500)),
            w_trend=float((mt.get("weights", {}) or {}).get("trend", 0.40)),
            w_carry=float((mt.get("weights", {}) or {}).get("carry", 0.25)),
            w_vol=float((mt.get("weights", {}) or {}).get("vol", 0.15)),
            w_micro=float((mt.get("weights", {}) or {}).get("micro", 0.10)),
            w_onchain=float((mt.get("weights", {}) or {}).get("onchain", 0.10)),
            http_timeout=int(mt.get("timeout", mt.get("http_timeout", 6)))
        )
        self.enabled = self.cfg.enable
        self.score_open = self.cfg.score_open
        self._trading = (config.get("trading") or {})
        self._symbol_map = (self._trading.get("symbol_map") or {}).copy()
        self._session = requests.Session() if requests else None
        self._market = (config.get("market") or {})
        self._provider = str(self._market.get("provider", "binance")).lower()
        self._http_451_count = 0
        self._max_451_before_disable = int((config.get("midtrend") or {}).get("max_451_before_disable", 3))

        if self._provider != "binance":
            self.enabled = False
            logger.info(f"[midtrend] provider={self._provider}，自动禁用 midtrend（当前仅支持 binance REST）")

        # 最近错误节流
        self._last_err_ts = 0.0
        self._err_cool_s = 15.0

    # ----------------------------
    # HTTP helpers
    # ----------------------------
    def _res_symbol(self, sym: str) -> str:
        """将本地 symbol 映射成 Binance 交易对（大写），优先使用 config.trading.symbol_map"""
        if not sym:
            return ""
        s = sym
        if s in self._symbol_map:
            s = self._symbol_map[s]
        elif s.lower() in self._symbol_map:
            s = self._symbol_map[s.lower()]
        s = s.upper()
        # 兼容极端大小写
        if s.endswith("usdt".upper()) is False and s.endswith("USDT") is False:
            s = s + "USDT" if not s.endswith("USD") else s
        return s

    def _get(self, base: str, path: str, params: Dict[str, Any]) -> Optional[Any]:
        if not self._session:
            if time.time() - self._last_err_ts > self._err_cool_s:
                logger.error("[midtrend] requests 未安装，无法请求 Binance 接口")
                self._last_err_ts = time.time()
            return None
        url = f"{base}{path}"
        try:
            r = self._session.get(url, params=params, timeout=self.cfg.http_timeout)
            if r.status_code != 200:
                if r.status_code == 451:
                    self._http_451_count += 1
                    if self._http_451_count >= self._max_451_before_disable:
                        self.enabled = False
                        logger.error("[midtrend] 连续触发 HTTP 451，已自动禁用 midtrend，避免日志风暴")
                        return None
                if time.time() - self._last_err_ts > self._err_cool_s:
                    logger.error(f"[midtrend] HTTP {r.status_code}: {r.text}")
                    self._last_err_ts = time.time()
                return None
            return r.json()
        except Exception as e:
            if time.time() - self._last_err_ts > self._err_cool_s:
                logger.error(f"[midtrend] 请求异常: {e}")
                self._last_err_ts = time.time()
            return None

    # ----------------------------
    # 数据拉取
    # ----------------------------
    def _fetch_fut_klines(self, symbol: str, interval: str, limit: int) -> Optional[np.ndarray]:
        """
        Futures USDT-M klines: https://fapi.binance.com/fapi/v1/klines
        返回 ndarray: [open_time, open, high, low, close, volume, close_time]
        """
        sym = self._res_symbol(symbol)
        j = self._get(
            "https://fapi.binance.com",
            "/fapi/v1/klines",
            {"symbol": sym, "interval": interval, "limit": limit}
        )
        if not j or not isinstance(j, list):
            return None
        try:
            out = np.array(j, dtype=float)
            # 列定义见官方：0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close_time, ...
            return out[:, [0, 1, 2, 3, 4, 5, 6]]
        except Exception:
            return None

    def _fetch_funding(self, symbol: str, limit: int = 36) -> Tuple[float, float]:
        """
        fundingRate 列表，返回 (last_funding, cum_72h)
        一般 8h 一次，72h ≈ 9笔（limit 取 36 以防不足）
        """
        sym = self._res_symbol(symbol)
        j = self._get(
            "https://fapi.binance.com",
            "/fapi/v1/fundingRate",
            {"symbol": sym, "limit": limit}
        )
        if not j or not isinstance(j, list):
            # 尝试 premiumIndex 的 lastFundingRate 兜底
            pj = self._get(
                "https://fapi.binance.com",
                "/fapi/v1/premiumIndex",
                {"symbol": sym}
            )
            last = float(pj.get("lastFundingRate", 0.0)) if isinstance(pj, dict) else 0.0
            return last, last
        try:
            rates = [float(x.get("fundingRate", 0.0)) for x in j[-9:]]  # 近72h
            last = float(j[-1].get("fundingRate", 0.0)) if len(j) else 0.0
            return last, float(np.nansum(rates))
        except Exception:
            return 0.0, 0.0

    def _fetch_premium_and_spot(self, symbol: str) -> Tuple[float, float, float]:
        """
        返回 (markPrice, spot_last, premium_ratio) 其中 premium_ratio = (mark-spot)/spot
        失败返回 (0,0,0)
        """
        sym = self._res_symbol(symbol)
        pj = self._get(
            "https://fapi.binance.com",
            "/fapi/v1/premiumIndex",
            {"symbol": sym}
        )
        mark = float(pj.get("markPrice", 0.0)) if isinstance(pj, dict) else 0.0

        # 现货价格
        sj = self._get(
            "https://api.binance.com",
            "/api/v3/ticker/price",
            {"symbol": sym}
        )
        spot = 0.0
        if isinstance(sj, dict):
            try:
                spot = float(sj.get("price", 0.0))
            except Exception:
                spot = 0.0

        prem = _safe_div(mark - spot, spot, 0.0) if (mark > 0 and spot > 0) else 0.0
        return mark, spot, prem

    # ----------------------------
    # 因子与评分
    # ----------------------------
    def _trend_block(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict[str, Any]:
        """返回 trend 因子组件（标准化到 -1..+1 但不做最终加权）"""
        N = len(c)
        p = self.cfg

        # Donchian 20/55 突破（使用上一根通道作为阈，避免包含当根）
        don20_hi = _rolling_max(h, p.don_base)
        don20_lo = _rolling_min(l, p.don_base)
        don55_hi = _rolling_max(h, p.don_alt)
        don55_lo = _rolling_min(l, p.don_alt)

        # 使用前一根的通道阈值
        sig20 = 0.0
        sig55 = 0.0
        if N >= p.don_base + 1 and not np.isnan(don20_hi[-2]) and not np.isnan(don20_lo[-2]):
            sig20 = 1.0 if c[-1] > don20_hi[-2] else (-1.0 if c[-1] < don20_lo[-2] else 0.0)
        if N >= p.don_alt + 1 and not np.isnan(don55_hi[-2]) and not np.isnan(don55_lo[-2]):
            sig55 = 1.0 if c[-1] > don55_hi[-2] else (-1.0 if c[-1] < don55_lo[-2] else 0.0)

        # EMA200 与斜率 z-score
        ema200 = _ema(c, alpha=2.0 / (p.ema_long + 1.0))
        if N >= (p.ema_long // 2 + 5):
            # 最近 10 根的 EMA 斜率序列
            win = min(60, max(20, p.ema_long // 3))
            slope = ema200[1:] - ema200[:-1]
            if len(slope) >= win:
                sl = slope[-win:]
                mu, sd = float(np.mean(sl)), float(np.std(sl, ddof=1)) if win > 1 else 0.0
                z = _safe_div(sl[-1] - mu, sd if sd > 1e-12 else 1.0, 0.0)
                # 标准化到 [-1,1]（2σ 截断）
                slope_z = float(np.clip(z / 2.0, -1.0, 1.0))
            else:
                slope_z = 0.0
        else:
            slope_z = 0.0

        # ADX
        adx = adx_wilder(h, l, c, period=self.cfg.atr_period) if N >= self.cfg.atr_period + 5 else np.zeros_like(c)
        adx_now = float(adx[-1]) if len(adx) else 0.0
        adx_gate = 1.0 if adx_now >= self.cfg.adx_thr else 0.5  # 弱趋势衰减

        # 组合 trend score（-1..+1）
        trend_raw = 0.5 * sig20 + 0.3 * sig55 + 0.2 * slope_z
        trend = float(np.clip(trend_raw * adx_gate, -1.0, 1.0))

        return {
            "sig20": sig20, "sig55": sig55, "slope_z": slope_z,
            "ema200": float(ema200[-1]) if len(ema200) else float('nan'),
            "adx": adx_now,
            "trend": trend
        }

    def _vol_block(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict[str, Any]:
        """返回波动/空间组件：ATR%、Boll/Keltner 宽度、YZ/GK/PK RV 百分位等"""
        p = self.cfg
        N = len(c)

        # ATR%
        atr = atr_wilder(h, l, c, period=p.atr_period) if N >= p.atr_period + 5 else np.zeros_like(c) + 1e-12
        atr_pct = atr / np.maximum(c, 1e-12)
        atr_now = float(atr_pct[-1]) if len(atr_pct) else 0.0

        # Bollinger/Keltner 宽度（相对当前价）
        bb_lo, bb_ma, bb_hi = bollinger(c, period=p.bb_period, k=2.0)
        bb_w = (bb_hi - bb_lo) / np.maximum(bb_ma, 1e-12)
        kel_lo, kel_ma, kel_hi, _ = keltner_channels(h, l, c, period=p.bb_period, mult=p.keltner_mult)
        kel_w = (kel_hi - kel_lo) / np.maximum(kel_ma, 1e-12)

        # 百分位（用最近 200 根）
        W = min(200, len(bb_w))
        if W >= 10:
            bb_pct = float(100.0 * (np.sum(bb_w[-W:] <= bb_w[-1]) / W))
            kel_pct = float(100.0 * (np.sum(kel_w[-W:] <= kel_w[-1]) / W))
        else:
            bb_pct = 50.0
            kel_pct = 50.0

        # RV（YZ/GK/PK）
        if p.vol_method.lower() == "yz":
            rv = realized_vol_yz(o, h, l, c)
        elif p.vol_method.lower() == "gk":
            rv = realized_vol_gk(o, h, l, c)
        else:
            rv = realized_vol_pk(h, l)

        # 以 ATR% 百分位定义 regime
        if W >= 20:
            atr_pct_rank = float(100.0 * (np.sum(atr_pct[-W:] <= atr_pct[-1]) / W))
        else:
            atr_pct_rank = 50.0
        if atr_pct_rank < 33:
            regime = "low"
        elif atr_pct_rank > 66:
            regime = "high"
        else:
            regime = "mid"

        # vol_score（偏门槛，不带方向）
        vol_gate = max(bb_pct, kel_pct) / 100.0  # 0..1
        vol_score = float(2.0 * (vol_gate - 0.5))  # 映射到 -1..+1，但在总分里只作为“空间有无”的贡献

        return {
            "atr_pct": float(atr_now),
            "bb_width_pct": bb_pct,
            "kel_width_pct": kel_pct,
            "rv": float(rv),
            "regime": regime,
            "vol_score": float(np.clip(vol_score, -1.0, 1.0))
        }

    def _carry_block(self, symbol: str) -> Dict[str, Any]:
        """资金费与 perp-spot 溢价作为 carry 代理（方向性），标准化到 -1..+1。"""
        last_f, cum72 = self._fetch_funding(symbol, limit=36)
        mark, spot, prem = self._fetch_premium_and_spot(symbol)

        # funding：winsorize 后线性缩放（±0.03 约等于 ±3% 累积）
        f_scaled = float(np.clip(cum72 / 0.03, -1.0, 1.0))
        # 溢价：±1% 以内映射到 -1..+1
        prem_scaled = float(np.clip(prem / 0.01, -1.0, 1.0))

        # 组合 carry（-1..+1）
        carry = 0.6 * f_scaled + 0.4 * prem_scaled
        carry = float(np.clip(carry, -1.0, 1.0))

        return {
            "funding_last": float(last_f),
            "funding_cum72h": float(cum72),
            "premium_ratio": float(prem),
            "carry": carry
        }

    # ----------------------------
    # 主入口
    # ----------------------------
    def analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        返回：
        {
          "side": "LONG"/"SHORT",
          "score": 0..100,
          "bar_time": <int open_time_ms>,
          "t1": float, "t2": float, "t3": float, "sl": float,
          "atr_pct": float,
          "regime": "low/mid/high",
          "reason": "... 因子拆解 ...",
          "score_open": <门槛（回显给上层日志）>
        }
        """
        if not self.enabled:
            return None

        k = self._fetch_fut_klines(symbol, self.cfg.interval, self.cfg.limit)
        if k is None or len(k) < max(self.cfg.ema_long + 5, 120):
            # 数据不足或请求失败
            return None

        open_t = k[:, 0].astype(np.int64)    # ms
        open_ = k[:, 1].astype(float)
        high = k[:, 2].astype(float)
        low = k[:, 3].astype(float)
        close = k[:, 4].astype(float)
        # volume = k[:, 5]

        # 计算三大块
        trend = self._trend_block(open_, high, low, close)
        vol = self._vol_block(open_, high, low, close)
        carry = self._carry_block(symbol)

        # 方向来自 “趋势 + carry” 的合成符号
        dir_val = self.cfg.w_trend * trend["trend"] + self.cfg.w_carry * carry["carry"]
        side = "LONG" if dir_val >= 0 else "SHORT"

        # 分值大小只取绝对值（不受方向影响），但要乘以空间/微观门槛
        vol_contrib = self.cfg.w_vol * max(0.0, vol["vol_score"])  # 仅正向加分（空间够）
        # micro 门槛（暂无 L2/冲击因子，这里留 1.0，可日后接入 OB 特征）
        micro_gate = 1.0
        total_abs = abs(self.cfg.w_trend * trend["trend"]) + abs(self.cfg.w_carry * carry["carry"]) + abs(vol_contrib) + self.cfg.w_onchain * 0.0
        score = int(np.clip(100.0 * total_abs * micro_gate, 0.0, 100.0))

        # 目标/止损（ATR%）
        px = float(close[-1])
        atrp = float(max(vol["atr_pct"], 1e-6))
        if side == "LONG":
            t1 = px * (1.0 + self.cfg.t1_atr * atrp)
            t2 = px * (1.0 + self.cfg.t2_atr * atrp)
            t3 = px * (1.0 + self.cfg.t3_atr * atrp)
            sl = px * (1.0 - self.cfg.sl_atr * atrp)
        else:
            t1 = px * (1.0 - self.cfg.t1_atr * atrp)
            t2 = px * (1.0 - self.cfg.t2_atr * atrp)
            t3 = px * (1.0 - self.cfg.t3_atr * atrp)
            sl = px * (1.0 + self.cfg.sl_atr * atrp)

        # 证据字符串
        ev_bits: List[str] = []
        ev_bits.append(f"trend={trend['trend']:+.2f}(20={trend['sig20']:+.0f},55={trend['sig55']:+.0f},slopeZ={trend['slope_z']:+.2f},ADX={trend['adx']:.1f})")
        ev_bits.append(f"carry={carry['carry']:+.2f}(fund72h={carry['funding_cum72h']:+.3f},prem={carry['premium_ratio']:+.3%})")
        ev_bits.append(f"ATR%={atrp:.4f} | BB%={vol['bb_width_pct']:.0f} Kel%={vol['kel_width_pct']:.0f} | RV({self.cfg.vol_method})={vol['rv']:.5f} | regime={vol['regime']}")
        ev_bits.append(f"w={self.cfg.w_trend:.2f}/{self.cfg.w_carry:.2f}/{self.cfg.w_vol:.2f}/{self.cfg.w_micro:.2f}/{self.cfg.w_onchain:.2f}")
        reason = " | ".join(ev_bits)

        return {
            "side": side,
            "score": score,
            "bar_time": int(open_t[-1]),
            "t1": float(t1), "t2": float(t2), "t3": float(t3), "sl": float(sl),
            "atr_pct": atrp,
            "regime": vol["regime"],
            "reason": reason,
            "score_open": self.cfg.score_open
        }
