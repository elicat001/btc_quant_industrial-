# modules/executor.py
import time
import threading
import json
from pathlib import Path
from collections import deque, defaultdict

__all__ = ["TradeExecutor", "PaperBroker"]


class PaperBroker:
    """
    极简纸面撮合：
    - 若提供 best bid/ask，用近似真实的吃单价成交（买按 ask，卖按 bid）
    - 否则退化为用 last price 并叠加滑点基点（可自适应 spread）
    - 维护持仓/权益/当日回撤
    """
    def __init__(self, journal_path: str = "logs/paper_trades.jsonl"):
        self.position = defaultdict(lambda: {"side": "FLAT", "qty": 0.0, "entry": 0.0})
        self.equity = 100000.0
        self.daily_eq_hi = self.equity
        self.daily_eq_lo = self.equity
        self.trades = []
        self.day = time.strftime("%Y-%m-%d")
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _roll_day(self):
        d = time.strftime("%Y-%m-%d")
        if d != self.day:
            self.day = d
            self.daily_eq_hi = self.equity
            self.daily_eq_lo = self.equity

    def equity_drawdown_bp(self):
        self._roll_day()
        if self.daily_eq_hi <= 0:
            return 0.0
        dd = (self.daily_eq_hi - self.equity) / self.daily_eq_hi
        return max(0.0, dd) * 10000.0

    def mark_to_market(self, sym: str, price: float) -> float:
        pos = self.position[sym]
        if pos["side"] == "LONG":
            return (price - pos["entry"]) * pos["qty"]
        if pos["side"] == "SHORT":
            return (pos["entry"] - price) * pos["qty"]
        return 0.0

    def _update_daily_extrema(self):
        self.daily_eq_hi = max(self.daily_eq_hi, self.equity)
        self.daily_eq_lo = min(self.daily_eq_lo, self.equity)

    def _append_journal(self, rec: dict):
        try:
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _fill_price_with_slippage(self, side: str, last_px: float,
                                  best_bid: float, best_ask: float,
                                  slippage_bp: float):
        """
        有盘口：买吃 ask，卖吃 bid；无盘口：last_px*(1±slippage_bp)
        """
        if best_bid and best_ask and best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            return best_ask if side == "BUY" else best_bid
        # fallback：仅 last
        if last_px <= 0:
            return 0.0
        slip = slippage_bp / 10000.0
        return last_px * (1.0 + slip if side == "BUY" else 1.0 - slip)

    def place(self, sym: str, side: str, last_px: float, notional_quote: float,
              best_bid: float = None, best_ask: float = None, reduce_only: bool = False,
              slippage_bp: float = 5.0, reason: str = ""):
        """返回 {status, price, qty, equity, event, pnl?}"""
        self._roll_day()
        ts = time.time()
        px = float(self._fill_price_with_slippage(side, last_px, best_bid, best_ask, slippage_bp))
        if px <= 0:
            return {"status": "ERR", "info": "bad_price"}

        qty = max(0.0001, notional_quote / px)
        pos = self.position[sym]
        event = "OPEN"
        pnl = 0.0

        if side == "BUY":
            if pos["side"] == "SHORT" and reduce_only:
                event = "CLOSE_SHORT"
                close_qty = min(pos["qty"], qty)
                pnl = (pos["entry"] - px) * close_qty
                self.equity += pnl
                pos["qty"] -= close_qty
                if pos["qty"] <= 1e-10:
                    pos.update({"side": "FLAT", "qty": 0.0, "entry": 0.0})
            else:
                pos["side"] = "LONG"
                pos["qty"] = qty
                pos["entry"] = px
                event = "OPEN_LONG"

        elif side == "SELL":
            if pos["side"] == "LONG" and reduce_only:
                event = "CLOSE_LONG"
                close_qty = min(pos["qty"], qty)
                pnl = (px - pos["entry"]) * close_qty
                self.equity += pnl
                pos["qty"] -= close_qty
                if pos["qty"] <= 1e-10:
                    pos.update({"side": "FLAT", "qty": 0.0, "entry": 0.0})
            else:
                pos["side"] = "SHORT"
                pos["qty"] = qty
                pos["entry"] = px
                event = "OPEN_SHORT"

        rec = {
            "ts": ts,
            "symbol": sym,
            "side": side,
            "event": event,
            "qty": qty,
            "price": px,
            "pnl": float(pnl),
            "equity": float(self.equity),
            "reduce_only": bool(reduce_only),
            "reason": reason,
        }
        self._append_journal(rec)

        self.trades.append((ts, sym, side, qty, px, event, pnl))
        self._update_daily_extrema()
        return {"status": "FILLED", "price": px, "qty": qty, "equity": self.equity, "event": event, "pnl": float(pnl)}


class TradeExecutor:
    """
    交易执行器
    - 支持 paper / live（live 预留，未接交易所 SDK 时返回未实现）
    - 限流与护栏：每小时限单、当日回撤熔断、幂等 3 秒窗口
    - reduce_only：当风控 reason 属于平仓类时仅允许减仓
    - 支持从外部注入 price_getter（last）与 ob_getter（best bid/ask）
    """
    def __init__(self, cfg, price_getter, ob_getter=None):
        tcfg = (cfg.get("trading") or {})
        self.enabled = bool(tcfg.get("enabled", False))
        self.mode = str(tcfg.get("mode", "paper")).lower()     # paper | live
        self.venue = str(tcfg.get("venue", "futures")).lower() # futures | spot（live时用）
        self.symbol_map = {str(k).lower(): str(v) for k, v in (tcfg.get("symbol_map") or {}).items()}

        self.base_notional = float(tcfg.get("base_notional_usd", 200))
        self.max_pos = int(tcfg.get("max_pos_per_symbol", 1))
        self.leverage = int(tcfg.get("leverage", 2))
        self.reduce_only_on_exit = bool(tcfg.get("reduce_only_on_exit", True))
        self.max_trades_per_hour = int(tcfg.get("max_trades_per_hour", 6))
        self.max_daily_loss_bp = float(tcfg.get("max_daily_loss_bp", 150))
        self.slippage_bp = float(tcfg.get("slippage_bp", 5))
        self.order_type = str(tcfg.get("order_type", "MARKET")).upper()
        self.testnet = bool(tcfg.get("testnet", True))

        # 依赖注入
        self.price_getter = price_getter          # fn(symbol)->last_price
        self.ob_getter = ob_getter                # fn(symbol)->(best_bid, best_ask)

        # 节流/幂等
        self.hour_buckets = defaultdict(deque)    # symbol -> timestamps within 1h
        self.idem_guard = {}                      # symbol -> last_exec_ts
        self.lock = threading.Lock()

        # 纸面经纪商
        self.paper = PaperBroker()

        # TODO: live 接入位（Binance/OKX 等）
        self._live_ready = False  # 未接 SDK 前默认 False

    # ---------- 限流与护栏 ----------
    def _rate_limited(self, sym: str) -> bool:
        now = time.time()
        q = self.hour_buckets[sym]
        while q and now - q[0] > 3600:
            q.popleft()
        return len(q) >= self.max_trades_per_hour

    def _bump_rate(self, sym: str):
        self.hour_buckets[sym].append(time.time())

    def _daily_circuit_break(self) -> bool:
        # 使用纸面经纪商的权益与当日回撤估算
        return self.paper.equity_drawdown_bp() >= self.max_daily_loss_bp

    # ---------- 主执行 ----------
    def execute(self, symbol: str, decision: str, reason: str):
        """
        仅在风控判定为 BUY/SELL 时调用
        返回 dict：{status, fill_px?, qty?, info}
        """
        if not self.enabled:
            return {"status": "SKIP", "info": "trading disabled"}

        sym = str(symbol).lower()
        mapped = self.symbol_map.get(sym, sym.upper())

        with self.lock:
            # 幂等：3 秒窗口不重复
            last_ts = self.idem_guard.get(sym, 0)
            if time.time() - last_ts < 3.0:
                return {"status": "SKIP", "info": "idem_guard"}

            # 限速
            if self._rate_limited(sym):
                return {"status": "SKIP", "info": "rate_limited"}

            # 当日熔断
            if self._daily_circuit_break():
                return {"status": "SKIP", "info": "daily_circuit_break"}

            # 价格与盘口
            last_px = float(self.price_getter(sym) or 0.0)
            if last_px <= 0:
                return {"status": "ERR", "info": "no_price"}

            best_bid, best_ask = (None, None)
            if self.ob_getter:
                try:
                    ob = self.ob_getter(sym)
                    if ob and len(ob) == 2:
                        best_bid, best_ask = float(ob[0] or 0.0), float(ob[1] or 0.0)
                except Exception:
                    best_bid, best_ask = (None, None)

            # 自适应滑点：若无盘口可用，则按最近的 spread 动态放大滑点
            dynamic_slip_bp = float(self.slippage_bp)
            if (not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0 or best_ask < best_bid):
                try:
                    if self.ob_getter:
                        ob2 = self.ob_getter(sym)
                        if ob2 and ob2[0] and ob2[1] and ob2[1] > 0 and ob2[0] > 0 and ob2[1] >= ob2[0]:
                            mid = 0.5 * (float(ob2[0]) + float(ob2[1]))
                            spread_bp = (float(ob2[1]) - float(ob2[0])) / max(mid, 1e-12) * 1e4
                            dynamic_slip_bp = max(dynamic_slip_bp, spread_bp)
                except Exception:
                    pass

            # notional（名义美元）
            notional = float(self.base_notional)
            if self.leverage > 1:
                notional = self.base_notional * float(self.leverage)

            # reduce_only：风控给出的平仓类原因
            reduce_only_reasons = {"take_profit", "stop_loss", "trail_take", "timeout", "breakeven", "fast_take"}
            reduce_only = self.reduce_only_on_exit and (reason in reduce_only_reasons)

            # === 执行 ===
            if self.mode == "paper":
                resp = self.paper.place(
                    mapped, decision, last_px, notional,
                    best_bid=best_bid, best_ask=best_ask,
                    reduce_only=reduce_only,
                    slippage_bp=dynamic_slip_bp,
                    reason=reason
                )
                if resp.get("status") == "FILLED":
                    self._bump_rate(sym)
                    self.idem_guard[sym] = time.time()
                    return {"status": "FILLED", "fill_px": resp["price"], "qty": resp["qty"], "info": "paper"}
                return {"status": resp.get("status", "ERR"), "info": resp.get("info", "paper_error")}

            # ==== live（预留位）：未接交易所 SDK 时返回未实现 ====
            return {"status": "ERR", "info": "live_not_implemented_yet"}
