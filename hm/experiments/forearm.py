"""量两条前臂之间到底有没有相交（X 形），并检查相机取景比例。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import generate, to_joints


def seg_dist(p1, q1, p2, q2):
    """两条线段的最短距离。"""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c = d1 @ r
    b = d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def analyze(prompt, frames=40, tag=""):
    F = generate.generate([prompt], [frames])[0]
    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
    T = len(J)
    dmin, ang = [], []
    for t in range(T):
        dmin.append(seg_dist(J[t, 18], J[t, 20], J[t, 19], J[t, 21]))
        # 肘弯曲角：上臂与前臂夹角
        u = J[t, 18] - J[t, 16]
        v = J[t, 20] - J[t, 18]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        ang.append(np.degrees(np.arccos(np.clip(u @ v / max(nu * nv, 1e-9), -1, 1))))
    dmin = np.array(dmin)
    # 手腕高度相对肩高
    wr = (J[:, 20, 1] + J[:, 21, 1]) / 2
    sh = (J[:, 16, 1] + J[:, 17, 1]) / 2
    print("  %-34s 前臂最小间距 %5.1f 厘米  肘角 %4.0f–%4.0f 度  腕高/肩高 %.2f"
          % (tag or prompt[:32], 100 * dmin.min(), min(ang), max(ang), float(np.mean(wr / sh))))
    return dmin, np.array(ang), wr / sh


print("  动作                  （前臂间距 < 8 厘米即视为互相插入）")
analyze("a person claps their hands", 40, "鼓掌（旧）")
analyze("a person claps their hands repeatedly", 40, "鼓掌（新）")
analyze("a person crosses both arms on the chest", 40, "抱臂（对照，本来就交叉）")
analyze("a person puts both hands on the hips", 36, "叉腰（对照，不交叉）")
analyze("a person waves hello with the right hand", 40, "挥手（对照）")
