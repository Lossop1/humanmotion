#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
python - <<'PYEOF'
import json, re
from datetime import datetime
from pathlib import Path

for job in ("job_k6", "job_route"):
    p = Path("outputs") / job / "run.log"
    if not p.exists():
        continue
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]\s*(.*)", ln.strip())
        if m:
            rows.append((datetime.strptime(m.group(1), "%H:%M:%S"), m.group(2)))
    print("== %s ==" % job)
    marks = []
    for t, msg in rows:
        tag = None
        if "A02 分镜" in msg:
            tag = "规划(LLM)"
        elif "A07 seg_001" in msg:
            tag = "生成起点"
        elif "A12 seg_006" in msg or "A12 seg_005" in msg:
            tag = "候选选择"
        elif "A16 参数空间" in msg:
            tag = "拟合+装配"
        elif "A18 渲染完成" in msg:
            tag = "渲染"
        elif "A19 交付件" in msg:
            tag = "裁切+字幕"
        if tag:
            marks.append((tag, t))
    for i in range(len(marks) - 1):
        d = (marks[i + 1][1] - marks[i][1]).total_seconds()
        print("  %-12s → 下一步 %.1f 秒" % (marks[i][0], d))
    if marks:
        print("  合计跨度 %.1f 秒" % (marks[-1][1] - marks[0][1]).total_seconds())
    print()
PYEOF
