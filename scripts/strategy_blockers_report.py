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

for line in log.read_text(errors='ignore').splitlines()[-30000:]:
    m = pat_ts.match(line)
    if not m:
        continue
    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    if ts < since:
        continue
    if 'fused_signal=' not in line:
        continue
    for key in ['mm_block(basic_dir)','low_vol<','low_vol=True','dir_margin','p_min=','HOLD']:
        if key in line:
            reasons[key] += 1

print('策略阻塞归因（最近1小时）')
for k,v in reasons.most_common(8):
    print(f'- {k}: {v}')
