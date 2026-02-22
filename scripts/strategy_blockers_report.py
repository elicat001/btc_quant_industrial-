#!/usr/bin/env python3
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = Path('logs/run.log')
if not log.exists():
    print('无日志')
    raise SystemExit(0)

since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
pat_ts = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
reasons = Counter()

for line in log.read_text(errors='ignore').splitlines()[-60000:]:
    m = pat_ts.match(line)
    if not m:
        continue
    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    if ts < since:
        continue

    # 1) 策略层阻塞：仅统计“明确阻塞标记”
    if 'fused_signal=' in line:
        mm = re.search(r'mm_block\(([^)]*)\)', line)
        if mm:
            for reason in [x.strip() for x in mm.group(1).split(',') if x.strip()]:
                reasons[f'mm_block:{reason}'] += 1

        if 'low_vol<' in line:
            reasons['low_vol_block'] += 1

        # HOLD 但未出现 buy/sell via，可作为“模型未达入场条件”的弱阻塞
        if ' | HOLD' in line and ('BUY via' not in line and 'SELL via' not in line):
            reasons['model_hold'] += 1

    # 2) 风控/执行层阻塞：来自 gate_reject 与执行器跳过
    if 'gate_reject risk_check' in line:
        m2 = re.search(r'reason=([a-zA-Z_]+)', line)
        if m2:
            reasons[f'risk_gate:{m2.group(1)}'] += 1

    if "EXEC {'status': 'SKIP'" in line:
        m3 = re.search(r"'info': '([^']+)'", line)
        if m3:
            reasons[f'exec_skip:{m3.group(1)}'] += 1

print('策略阻塞归因（最近1小时，修正版）')
for k, v in reasons.most_common(12):
    print(f'- {k}: {v}')
