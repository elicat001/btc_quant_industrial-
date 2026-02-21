# modules/model.py
import os
import json
import logging
import numpy as np
import torch
import yaml

try:
    import joblib
except Exception:
    joblib = None  # 不是强依赖

# 可选导入：存在就用；没有就自动回退
try:
    from model_definitions import EnhancedTFT, EnhancedNBeats  # 放根目录
except Exception:
    try:
        from .model_definitions import EnhancedTFT, EnhancedNBeats  # 放 modules/
    except Exception:
        EnhancedTFT = None
        EnhancedNBeats = None

logger = logging.getLogger(__name__)


# ========= 简易 scaler（与 sklearn.StandardScaler 接口对齐）=========
class _SimpleScaler:
    """
    - 若提供了已训练 scaler（带 mean_/scale_），则用之；
    - 否则构造恒等 scaler；
    - transform 过程做维度自适应：多的截断，少的补零。
    """
    def __init__(self, n_features: int, loaded=None):
        self.n_features = int(n_features)
        self.mean_ = None
        self.scale_ = None

        if loaded is not None:
            # 尽量从 sklearn 对象/字典里取出 mean_ / scale_
            try:
                m = getattr(loaded, "mean_", None)
                s = getattr(loaded, "scale_", None)
                if m is None and isinstance(loaded, dict):
                    m = loaded.get("mean_")
                    s = loaded.get("scale_")
                if m is not None and s is not None:
                    self.mean_ = np.asarray(m, dtype=float).ravel()
                    self.scale_ = np.asarray(s, dtype=float).ravel()
            except Exception:
                pass

        # 恒等
        if self.mean_ is None or self.scale_ is None:
            self.mean_ = np.zeros(self.n_features, dtype=float)
            self.scale_ = np.ones(self.n_features, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        nf = X.shape[1]
        ms = len(self.mean_)
        if nf != ms:
            if nf > ms:
                X = X[:, :ms]
            else:
                pad = np.zeros((X.shape[0], ms - nf), dtype=float)
                X = np.concatenate([X, pad], axis=1)

        scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return (X - self.mean_) / scale


# ========= 无权重时的轻量回退模型 =========
def _sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

class _FallbackModel:
    """
    用少量技术因子给出保守概率：
      期望特征索引（与 FeatureBuilder 对齐）：
        0 close, 1 ret, 3 macdH, 4 macd, 5 rsi, 6 vol(abs≈ATR%), 7 imb
    """
    def __init__(self, T=1.0):
        self.T = float(T)
        self.w = np.array([  # 对应 [ret, macdH, macd, rsi_z, vol, imb]
            2.2,   # ret
            1.6,   # macdH
            0.9,   # macd
           -0.8,   # (rsi-50)/20
            0.4,   # vol
            0.7    # imb
        ], dtype=float)
        self.b = 0.0

    def predict_proba(self, raw_last_row: np.ndarray) -> float:
        r   = float(raw_last_row[1]) if len(raw_last_row) > 1 else 0.0
        h   = float(raw_last_row[3]) if len(raw_last_row) > 3 else 0.0
        m   = float(raw_last_row[4]) if len(raw_last_row) > 4 else 0.0
        rsi = float(raw_last_row[5]) if len(raw_last_row) > 5 else 50.0
        v   = float(raw_last_row[6]) if len(raw_last_row) > 6 else 0.0
        imb = float(raw_last_row[7]) if len(raw_last_row) > 7 else 0.0
        rsi_z = (rsi - 50.0) / 20.0
        z = float(np.dot(self.w, np.array([r, h, m, rsi_z, v, imb], dtype=float)) + self.b)
        return float(_sigmoid(z / max(1e-6, self.T)))


class ModelManager:
    """
    负责加载 scaler/温度/模型，并提供 predict(features_seq) -> (label, prob)
    - 任一权重/文件缺失自动回退
    - 特征维度不一致自动截断/补零
    - 打统一运行平均日志
    """
    def __init__(self, symbol: str):
        self.symbol = str(symbol).lower()

        # === 读取配置 ===
        cfg = {}
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
        self.input_size = int(cfg.get("input_size", 12))
        self.seq_len = int(cfg.get("seq_len", 30))

        # === 阈值与平滑 ===
        self.T_tft = 1.0
        self.T_nbt = 1.0
        self.ema_alpha = 0.08
        self.blend_w = [0.6, 0.4]
        self.blend_b = 0.0

        if os.path.exists("thresholds.json"):
            try:
                j = json.load(open("thresholds.json", "r"))
                t = j.get("temp", 1.0)
                if isinstance(t, dict):
                    self.T_tft = float(t.get("tft", 1.0))
                    self.T_nbt = float(t.get("nbt", 1.0))
                else:
                    self.T_tft = self.T_nbt = float(t)
                self.ema_alpha = float(j.get("smooth_alpha", self.ema_alpha))
                b = j.get("blend", {})
                if isinstance(b, dict):
                    self.blend_w = list(b.get("w", self.blend_w))
                    self.blend_b = float(b.get("b", self.blend_b))
            except Exception as e:
                logger.warning(f"[ModelMgr] thresholds.json 读取失败: {e}")

        # === Scaler ===
        self.scaler = None
        loaded_scaler = None
        if os.path.exists("scaler.pkl") and joblib is not None:
            try:
                loaded_scaler = joblib.load("scaler.pkl")
                logger.info("[ModelMgr] scaler.pkl 已加载")
            except Exception as e:
                logger.warning(f"[ModelMgr] 读取 scaler.pkl 失败，将使用恒等 scaler: {e}")
        else:
            logger.warning("[ModelMgr] 未找到 scaler.pkl 或 joblib 不可用，将使用恒等 scaler")

        self.scaler = _SimpleScaler(n_features=self.input_size, loaded=loaded_scaler)

        # === 模型 ===
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tft = None
        self.nbeats = None

        # 加载 TFT
        if EnhancedTFT is not None and os.path.exists("tft_model.pth"):
            try:
                self.tft = EnhancedTFT(self.input_size).to(self.device)
                state = torch.load("tft_model.pth", map_location=self.device)
                if isinstance(state, dict):
                    self.tft.load_state_dict(state, strict=False)
                else:
                    # 兼容 TorchScript
                    self.tft = torch.jit.load("tft_model.pth", map_location=self.device)
                self.tft.eval()
                logger.info("[ModelMgr] tft_model.pth 已加载")
            except Exception as e:
                logger.warning(f"[ModelMgr] 加载 tft_model.pth 失败，跳过 TFT: {e}")
                self.tft = None
        else:
            if EnhancedTFT is None:
                logger.warning("[ModelMgr] 未找到 EnhancedTFT 定义，跳过 TFT")
            else:
                logger.warning("[ModelMgr] 未找到 tft_model.pth，跳过 TFT")

        # 加载 NBeats
        if EnhancedNBeats is not None and os.path.exists("nbeats_model.pth"):
            try:
                self.nbeats = EnhancedNBeats(self.input_size).to(self.device)
                state = torch.load("nbeats_model.pth", map_location=self.device)
                if isinstance(state, dict):
                    self.nbeats.load_state_dict(state, strict=False)
                else:
                    self.nbeats = torch.jit.load("nbeats_model.pth", map_location=self.device)
                self.nbeats.eval()
                logger.info("[ModelMgr] nbeats_model.pth 已加载")
            except Exception as e:
                logger.warning(f"[ModelMgr] 加载 nbeats_model.pth 失败，跳过 NBeats: {e}")
                self.nbeats = None
        else:
            if EnhancedNBeats is None:
                logger.warning("[ModelMgr] 未找到 EnhancedNBeats 定义，跳过 NBeats")
            else:
                logger.warning("[ModelMgr] 未找到 nbeats_model.pth，跳过 NBeats")

        # 全缺则启用回退
        self.fallback = None
        if self.tft is None and self.nbeats is None:
            logger.warning("[ModelMgr] 未加载到任何模型，启用回退模型")
            self.fallback = _FallbackModel(T=self.T_tft)

        # 运行平均日志
        self._ema = None

    # —— 工具：准备输入（自适应维度）——
    def _prep(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (T,F) 或 (F,)
        返回：缩放后的 (T, F_expected)
        """
        x = np.asarray(x_seq, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        T, F = x.shape
        if F > self.input_size:
            x = x[:, :self.input_size]
            logger.debug(f"[ModelMgr] 特征过宽，截断到 F={self.input_size}")
        elif F < self.input_size:
            pad = np.zeros((T, self.input_size - F), dtype=float)
            x = np.concatenate([x, pad], axis=1)
            logger.debug(f"[ModelMgr] 特征过窄，右侧补零到 F={self.input_size}")

        Xs = self.scaler.transform(x)  # (T, F_expected)
        return Xs

    @torch.no_grad()
    def _infer_tft(self, Xs: np.ndarray) -> float | None:
        if self.tft is None:
            return None
        x = torch.tensor(Xs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,F)
        y = self.tft(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        z = float(np.array(y.detach().cpu()).ravel()[-1])
        p = float(_sigmoid(z / max(1e-6, self.T_tft)))
        return p

    @torch.no_grad()
    def _infer_nbeats(self, Xs: np.ndarray) -> float | None:
        if self.nbeats is None:
            return None
        x = torch.tensor(Xs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,F)
        x_pool = x.mean(dim=1)  # (1,F)
        y = self.nbeats(x_pool)
        if isinstance(y, (tuple, list)):
            y = y[0]
        z = float(np.array(y.detach().cpu()).ravel()[-1])
        p = float(_sigmoid(z / max(1e-6, self.T_nbt)))
        return p

    def predict(self, features_seq: np.ndarray):
        """
        输入：features_seq (T,F)
        输出：(label:int{0,1}, prob:float)
        """
        try:
            Xs = self._prep(features_seq)         # (T, F)
            raw_last = Xs * 0.0 + features_seq[-1][:Xs.shape[1]] if np.asarray(features_seq).ndim == 2 else Xs[-1]

            probs = []
            w = []

            p_tft = self._infer_tft(Xs)
            if p_tft is not None:
                probs.append(p_tft); w.append(float(self.blend_w[0]))

            p_nbt = self._infer_nbeats(Xs)
            if p_nbt is not None:
                # 如果 TFT 缺失，就把 NBeats 权重归一到 1
                default_w = float(self.blend_w[1] if len(self.blend_w) > 1 else 0.5)
                probs.append(p_nbt); w.append(default_w if p_tft is not None else 1.0)

            if not probs:
                # 回退模型只需要最后一帧原始特征
                last_row = np.asarray(features_seq, dtype=float)
                if last_row.ndim == 2:
                    last_row = last_row[-1]
                # 补齐/截断到 input_size
                if last_row.shape[0] > self.input_size:
                    last_row = last_row[:self.input_size]
                elif last_row.shape[0] < self.input_size:
                    last_row = np.concatenate([last_row, np.zeros(self.input_size - last_row.shape[0])])
                if self.fallback is None:
                    self.fallback = _FallbackModel(T=self.T_tft)
                p = float(self.fallback.predict_proba(last_row))
            else:
                wsum = sum(w)
                w = [wi / wsum for wi in w]
                p = float(np.dot(np.array(probs, dtype=float), np.array(w, dtype=float)) + self.blend_b)
                p = float(np.clip(p, 1e-6, 1 - 1e-6))

            # 运行平均日志
            if self._ema is None:
                self._ema = p
            else:
                self._ema = self.ema_alpha * p + (1 - self.ema_alpha) * self._ema
            logger.info(f"[ModelMgr] running avg prob: {self._ema:.3f} (T={max(self.T_tft,self.T_nbt):.2f})")

            label = 1 if p >= 0.5 else 0
            return int(label), float(p)

        except Exception as e:
            logger.error(f"[ModelMgr] 预测异常: {e}")
            # 极端兜底
            return 0, 0.5
