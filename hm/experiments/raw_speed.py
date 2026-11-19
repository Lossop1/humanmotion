"""直接对当前 params 打印每个过渡窗口内的逐帧速度，与 bounds.json 对齐。"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = str(Path(__file__).resolve().parents[2])
job = sys.argv[1] if len(sys.argv) > 1 else "job_p01"
P = np.load("%s/outputs/%s/assembled/params.npy" % (ROOT, job))
b = json.load(open("%s/outputs/%s/assembled/bounds.json" % (ROOT, job), encoding="utf-8"))
T = len(P)
rots = R.from_rotvec(P[:, 0:72].reshape(T * 24, 3))
sp = np.zeros(T - 1)
for t in range(T - 1):
    rel = rots[(t + 1) * 24:(t + 2) * 24] * rots[t * 24:(t + 1) * 24].inv()
    sp[t] = np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1)).max()

print("  %s 总帧 %d，段帧数 %s" % (job, T, [s["frames"] for s in b["segments"]]))
cur = 0
segs = []
trans = []
for i, s in enumerate(b["segments"]):
    segs.append((cur, cur + s["frames"]))
    cur += s["frames"]
    if i < len(b["junctions"]):
        n = b["junctions"][i]["transition_frames"]
        trans.append((cur, cur + n))
        cur += n
print("  段区间 %s   过渡区间 %s" % (segs, trans))
for k, (a, bb) in enumerate(trans):
    win = sp[max(0, a - 1):min(len(sp), bb)]
    pre = sp[segs[k][0]:max(1, segs[k][1] - 1)]
    nxt = sp[min(len(sp) - 1, segs[k + 1][0]):max(1, segs[k + 1][1] - 2)]
    print("    交界%d 窗口 %d–%d  逐帧 %s  均值 %.2f | 前 %.2f 后 %.2f 比值 %.2f"
          % (k + 1, a, bb, " ".join("%.1f" % v for v in win),
             win.mean(), pre.mean(), nxt.mean(), win.mean() / max(pre.mean(), 1e-6)))
