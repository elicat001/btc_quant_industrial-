# btc_quant_industrial

一个面向 **实盘前验证** 的 BTC 量化系统：
- 实时行情（当前默认 OKX）
- 特征 → 模型 → 信号融合 → 风控 → 执行（paper）
- 可观测性（小时报、阻塞归因、系统健康报）

> 当前仓库主打 **paper 交易闭环 + 可观测优化**，用于持续迭代策略，不建议直接实盘。

---

## 1. 当前能力（2026-02 版本）

### 数据与模型
- 行情源：`market.provider: okx`（已兼容受限地区场景）
- 特征：`modules/features.py` 统一 12 维特征
- 模型：TFT + NBeats（`model_definitions.py`）
- 推理管理：`modules/model.py`
  - 支持 scaler 加载
  - 支持概率融合
  - 含 fallback 兜底

### 信号与风控
- `modules/signal.py`
  - `p_min` 期望值门槛
  - MM Gate（basic/hybrid/relaxed/force）
  - 低波动过滤与快速通道
- `modules/risk.py`
  - 冷却、反手保护、止盈止损、超时平仓等

### 执行与记录
- `modules/executor.py`
  - 默认 `paper` 执行
  - 限速、日内回撤熔断、reduce-only
  - 新增成交日志：`logs/paper_trades.jsonl`

### 可观测性脚本（重点）
- `scripts/strategy_hourly_report.py`：小时信号分布
- `scripts/strategy_blockers_report.py`：阻塞原因归因
- `scripts/system_health_report.py`：系统健康报告（信号、成交、PnL、胜率、平仓原因）
- `scripts/replay_backtest.py`：离线回放评估
- `scripts/calibrate_thresholds.py`：阈值/温度校准

---

## 2. 项目结构

```text
.
├── main.py
├── config.yaml
├── thresholds.json
├── modules/
│   ├── collector.py
│   ├── features.py
│   ├── model.py
│   ├── signal.py
│   ├── risk.py
│   ├── executor.py
│   ├── midtrend.py
│   └── push.py
└── scripts/
    ├── train_models.py
    ├── replay_backtest.py
    ├── calibrate_thresholds.py
    ├── strategy_hourly_report.py
    ├── strategy_blockers_report.py
    └── system_health_report.py
```

---

## 3. 启动方式（推荐 Docker）

```bash
cd /home/huahuapanda183/apps/btc_quant_industrial-
sudo docker compose up -d btc-quant
sudo docker compose logs -f btc-quant
```

停止：
```bash
sudo docker compose stop btc-quant
```

---

## 4. 训练（干净重训流程）

```bash
cd /home/huahuapanda183/apps/btc_quant_industrial-
rm -f tft_model.pth nbeats_model.pth scaler.pkl thresholds.json
sudo docker compose run --rm btc-quant python train_models.py
```

训练稳定后会生成：
- `tft_model.pth`
- `nbeats_model.pth`
- `scaler.pkl`

---

## 5. 校准与回测

### 5.1 阈值校准
```bash
sudo docker compose run --rm btc-quant python scripts/calibrate_thresholds.py
```
生成/更新 `thresholds.json`。

### 5.2 回放评估
```bash
sudo docker compose run --rm btc-quant python scripts/replay_backtest.py
```

---

## 6. 结果验证（你最关心的）

### 6.1 小时报表（信号层）
```bash
python3 scripts/strategy_hourly_report.py
```

### 6.2 阻塞归因（为什么 HOLD）
```bash
python3 scripts/strategy_blockers_report.py
```

### 6.3 系统健康报告（交易层）
```bash
python3 scripts/system_health_report.py
```
该报告会给出：
- BUY/SELL/HOLD
- active ratio
- OPEN/CLOSE 数
- 已平仓 PnL
- 胜率
- 平仓原因（含止盈/止损）

---

## 7. 关键配置说明

### `config.yaml`
重点项：
- `market.provider`: `okx` / `binance`
- `confirm.need_prob`
- `gate.min_dir_margin`
- `low_vol.*`
- `mm.*`（score_open/score_force/gate_mode）
- `trading.mode`: 建议长期 `paper`

### `thresholds.json`
- `temp.tft / temp.nbt`
- `blend.w / blend.b`
- `runtime_thresholds.buy / sell`
- `risk.*`

---

## 8. 当前已知边界

1. `live` 执行仍未接交易所真实下单 SDK（当前主流程是 paper）。
2. 低波动行情下，策略可能仍偏保守，需要持续调优 `mm + p_min + low_vol`。
3. 胜率/PNL 依赖 `paper_trades.jsonl` 的成交闭环，不应只看信号数量。

---

## 9. 建议运营流程（每天）

1. 看 `strategy_hourly_report`（信号是否过稀/过密）
2. 看 `strategy_blockers_report`（主要被谁挡）
3. 看 `system_health_report`（是否真的赚钱、胜率如何）
4. 调整参数后：小步上线 -> 观察 2~4 小时 -> 再调

---

## 10. 风险声明

本项目用于研究与工程验证，不构成投资建议。高波动资产风险极高，请仅在可承受损失范围内操作，并优先 paper 验证。
