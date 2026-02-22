# Challenger Checklist（每次看日志/报告都要跑）

- [ ] 指标是否为 net（已扣 fee/slippage）
- [ ] 窗口内 CLOSE 样本是否 >= 5（过少则降级结论）
- [ ] 胜率、RR、净PnL是否一致（是否出现“高胜率低RR”）
- [ ] 阻塞归因是否只统计真实阻塞（mm_block/risk_gate/exec_skip）
- [ ] 是否出现参数连续漂移（最近6次优化）
- [ ] 是否出现执行层限流过高（exec_skip:rate_limited）
- [ ] 是否出现风控过冷（mm_block:cooldown、risk_gate:cooldown异常高）

输出建议：
1) 本轮最可疑指标
2) 最可能的口径误差
3) 建议修复项（最多3条）
4) 是否允许继续自动优化（是/否）
