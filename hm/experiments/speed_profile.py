"""把「自然」落成可测量的量：跨交界的速度剖面。

如果一个交界处，过渡段每帧的转角明显高于两侧动作段，那就是可测的突变；
如果没有差别，那「改成缓动会更自然」这个说法就没有依据。
"""
import glob
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = "/home/chuan/motion-work"

for f in sorted(glob.glob(ROOT + "/outputs/job_p0*/assembled/params.npy")):
    job = f.split("/")[-3]
    P = np.load(f)
    T = len(P)
    rots = R.from_rotvec(P[:, 0:72].reshape(T * 24, 3))
    # 相邻帧每个关节的相对转角（度），取最大值作为该帧的「动作速度」
    speed = np.zeros(T - 1)
    for t in range(T - 1):
        rel = rots[(t + 1) * 24:(t + 2) * 24] * rots[t * 24:(t + 1) * 24].inv()
        ang = np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1))
        speed[t] = ang.max()

    b = json.load(open(f.replace("params.npy", "bounds.json"), encoding="utf-8"))
    if b.get("assembly") != "param_space":
        continue
    print("  === %s ===" % job)
    cur = 0
    segs = []
    trans = []
    for i, s in enumerate(b["segments"]):
        segs.append((cur, cur + s["frames"]))
        cur += s["frames"]
        if i < len(b["junctions"]):
            n = b["junctions"][i]["transition_frames"]
            trans.append((cur, cur + n, b["junctions"][i]["pose_gap_deg"]))
            cur += n
    for k, (a, bb, gap) in enumerate(trans):
        t0, t1 = max(0, a - 1), min(len(speed), bb)
        tr = speed[t0:t1] if t1 > t0 else np.array([0.0])
        prev = speed[max(0, segs[k][0]):max(1, segs[k][1] - 1)]
        nxt = speed[min(len(speed) - 1, segs[k + 1][0]):max(1, segs[k + 1][1] - 2)]
        print("    交界 %d  差 %5.1f 度  过渡 %2d 帧 | 过渡内 %.2f 度/帧  前段 %.2f  后段 %.2f  比值 %.2f"
              % (k + 1, gap, bb - a,
                 tr.mean(), prev.mean() if len(prev) else 0,
                 nxt.mean() if len(nxt) else 0,
                 (tr.mean() / max(prev.mean(), 1e-6)) if len(prev) else 0))
    print("    全片速度：中位 %.2f  最大 %.2f 度/帧" % (np.median(speed), speed.max()))
    print()
