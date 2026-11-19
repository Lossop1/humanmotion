"""核对：加速率上限后，还有多少交界低于 9 度/帧所需帧数。"""
import glob
import json
from pathlib import Path

rows = []
for p in sorted(glob.glob("/home/chuan/motion-work/outputs/*/assembled/bounds.json")):
    job = Path(p).parent.parent.name
    if not job.startswith("eval_"):
        continue
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    for j in d.get("junctions", []):
        rows.append((job, j["pose_gap_deg"], j["transition_frames"]))

print("  评测任务里的交界总数：%d" % len(rows))
if rows:
    short = [r for r in rows if r[2] < r[1] / 9.0 - 0.5]
    gaps = [r[1] for r in rows]
    print("  姿态差 中位 %.0f 度，最大 %.0f 度" % (sorted(gaps)[len(gaps) // 2], max(gaps)))
    print("  过渡帧数不足（低于 9 度/帧所需）的交界：%d/%d" % (len(short), len(rows)))
    for r in short[:8]:
        print("    %-10s 差 %3.0f 度  过渡 %2d 帧  需 %.1f" % (r[0], r[1], r[2], r[1] / 9.0))
