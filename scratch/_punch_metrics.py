"""出拳段的候选比较：谁真的打出去了。

量三个量（都相对身体朝向，不依赖绝对坐标）
    reach   手腕相对肩，沿身体前方的最远距离（米）
    speed   手腕沿前方的最大瞬时速度（米/秒）
    height  出手时手腕高度相对肩（米，负数表示在肩下）
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

JOB = Path("/home/chuan/motion-work/outputs/job_bank6")


def body_forward(Jf):
    acc = np.zeros(3)
    for a, t in ((7, 10), (8, 11)):
        v = np.asarray(Jf[t]) - np.asarray(Jf[a])
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            acc += v / n
    n = float(np.linalg.norm(acc))
    return acc / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def punch_metrics(J):
    fwd = body_forward(J[0])
    wrist = J[:, 21, :] - J[:, 16, :]          # 右手腕相对右肩
    along = wrist @ fwd
    speed = np.linalg.norm(np.diff(J[:, 21, :], axis=0), axis=1) * 20
    return float(along.max()), float(speed.max()), float(wrist[np.argmax(along), 1])


sb = json.loads((JOB / "storyboard.json").read_text(encoding="utf-8"))
seg = [s for s in sb["segments"] if "punch" in (s.get("text_gen_final") or "")][0]
print("  段 %s  选中 %s" % (seg["segment_id"], seg.get("selected")))
rows = []
for d in sorted(glob.glob(str(JOB / "segments" / seg["segment_id"] / "candidate_*" / "joints22.npy"))):
    J = np.load(d)
    reach, speed, h = punch_metrics(J)
    m = json.loads((Path(d).parent / "metrics.json").read_text(encoding="utf-8"))
    rows.append((Path(d).parent.name, reach, speed, h, m.get("decision")))
    print("    %-11s 前伸 %.2f 米  腕速 %.1f 米/秒  出手高度 %+.2f 米  判据 %s"
          % (Path(d).parent.name, reach, speed, h, m.get("decision")))
best = max(rows, key=lambda r: r[1])
print("  前伸最大的候选：%s（%.2f 米）" % (best[0], best[1]))
