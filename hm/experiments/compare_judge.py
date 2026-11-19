"""把大模型判官的结果与人工标签比对，算一致率。"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
J = json.loads((ROOT / "outputs/_semcheck/vlm_judge.json").read_text(encoding="utf-8"))
CSV = (Path(__file__).resolve().parent.parent / "outputs/_semcheck/checklist_filled.csv")

judge = {r["file"]: r for r in J}
print("  判官结果 %d 段" % len(judge))

# 人工标签来自本地清单，这里由调用方传入一份
label_path = (Path(__file__).resolve().parent.parent / "outputs/_semcheck/human_labels.csv")
if not label_path.exists():
    print("  缺 human_labels.csv")
    sys.exit(0)
human = {}
for r in csv.DictReader(open(label_path, encoding="utf-8-sig")):
    human[r["file"]] = (r.get("人判(是/否/部分)") or "").strip()

rows = []
for f, h in human.items():
    j = judge.get(f)
    if not j:
        continue
    rows.append((f, j["zh"], h, j["verdict"], (j.get("Q1_依据") or "")[:36]))

agree = sum(1 for r in rows if r[2] == r[3])
print("  可比对 %d 段，完全一致 %d 段 = %.1f%%" % (len(rows), agree, 100 * agree / len(rows)))
print()
print("  %-24s %-10s %-4s %-4s %s" % ("file", "动作", "人判", "判官", "判官依据"))
for r in sorted(rows, key=lambda x: x[2] != x[3]):
    flag = "" if r[2] == r[3] else "   ← 不一致"
    print("  %-24s %-10s %-4s %-4s %s%s" % (r[0][:24], r[1], r[2], r[3], r[4], flag))
print()
# 混淆
from collections import Counter
c = Counter((r[2], r[3]) for r in rows)
print("  混淆（人判, 判官）：%s" % dict(c))
fp = sum(1 for r in rows if r[2] == "否" and r[3] == "是")
fn = sum(1 for r in rows if r[2] == "是" and r[3] == "否")
print("  人判否、判官说「是」：%d 段（漏放）" % fp)
print("  人判是、判官说「否」：%d 段（误杀）" % fn)
