"""看最差的交界长什么样：列出每处交界的段、动作、差值与过渡帧数。"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path("/home/chuan/motion-work/outputs")

rows = []
for d in sorted(glob.glob(str(ROOT / "big_s*_r1" / "assembled" / "bounds.json"))):
    job = Path(d).parent.parent
    bj = json.loads(Path(d).read_text(encoding="utf-8"))
    sb = json.loads((job / "storyboard.json").read_text(encoding="utf-8"))
    act = {s["segment_id"]: (s.get("text_gen_final") or "") for s in sb["segments"]}
    for j in bj.get("junctions", []):
        a, b = j["between"]
        rows.append((job.name, act.get(a, "")[:34], act.get(b, "")[:34],
                     j["pose_gap_deg"], j["transition_frames"],
                     round(j["pose_gap_deg"] / max(j["transition_frames"], 1), 1)))

rows.sort(key=lambda r: -r[3])
print("  最差的 12 处交界：")
print("  %-12s %-34s %-34s %5s %4s %6s" % ("任务", "前段", "后段", "差", "过渡", "度/帧"))
for r in rows[:12]:
    print("  %-12s %-34s %-34s %5.0f %4d %6.1f" % (r[0][:12], r[1], r[2], r[3], r[4], r[5]))

import statistics as st
gaps = [r[3] for r in rows]
print()
print("  共 %d 处：中位 %.0f，>90 度的 %d 处，过渡帧数中位 %.0f"
      % (len(gaps), st.median(gaps), sum(1 for g in gaps if g > 90),
         st.median([r[4] for r in rows])))
