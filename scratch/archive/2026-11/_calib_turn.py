"""把「过渡突变」与「弓腰后退」变成可校准的两个量，用旧片/新片当校准对。

校准集
    旧片 outputs/eval_s8   转身段走 MDM：过渡进转身时突变；转身看起来是弓着腰后退
    新片 outputs/job_s8b   同一句、转身走 MoMask：两处问题都消失

两个量
    jerk_at_junction   交界前后逐帧最大骨向变化（度/帧）的峰值，
                       以及它相对两侧段内中位速率的倍数
    turn_lean          转身段躯干（骨盆→脊柱3）相对竖直的前倾中位/最大角
    turn_backstep      转身段根节点沿「朝向反方向」的位移（米），正数表示在后退
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm.finish_job import BONES, dirs_of, rot_between

JOBS = {
    "旧片(MDM 转身)": "/home/chuan/motion-work/outputs/eval_s8",
    "新片(MoMask 转身)": "/home/chuan/motion-work/outputs/job_s8b",
}


def frame_rate(J):
    """逐帧最大骨向变化（度/帧）。"""
    out = []
    for t in range(len(J) - 1):
        d0, _ = dirs_of(J[t])
        d1, _ = dirs_of(J[t + 1])
        out.append(max(math.degrees(rot_between(d0[k], d1[k])[1]) for k in range(len(BONES))))
    return np.array(out)


def body_forward(Jf):
    acc = np.zeros(3)
    for a, tt in ((7, 10), (8, 11)):
        v = np.asarray(Jf[tt]) - np.asarray(Jf[a])
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            acc += v / n
    n = float(np.linalg.norm(acc))
    return acc / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def analyse(jobdir, key):
    job = Path(jobdir)
    bj = json.loads((job / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    seq = np.load(job / "assembled" / "joints22.npy")
    rates = frame_rate(seq)
    bounds = bj["segments"]
    juncs = bj.get("junctions", [])

    starts = []
    t = 0
    for i, b in enumerate(bounds):
        starts.append((b["segment_id"], t, t + b["frames"]))
        t += b["frames"]
        if i < len(juncs):
            t += juncs[i]["transition_frames"]

    print("  == %s ==" % key)
    for i, j in enumerate(juncs):
        a_id, a0, a1 = starts[i]
        b_id, b0, b1 = starts[i + 1]
        lo, hi = max(0, a1 - 3), min(len(rates), b0 + 3)
        window = rates[lo:hi]
        inside = rates[a0:a1 - 1] if a1 - 1 > a0 else np.array([1.0])
        base = float(np.median(inside)) if len(inside) else 1.0
        peak = float(window.max()) if len(window) else 0.0
        print("    %s→%s 过渡 %d 帧  峰值 %.1f 度/帧  前段中位 %.1f  倍数 %.1f"
              % (a_id[-3:], b_id[-3:], j["transition_frames"], peak, base, peak / max(base, 1e-6)))

    # 转身段专项
    for sid, s0, s1 in starts:
        seg_j = seq[s0:s1]
        v = seg_j[:, 9] - seg_j[:, 0]
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        lean = np.degrees(np.arccos(np.clip(v[:, 1], -1, 1)))
        fwd = body_forward(seg_j[0])
        disp = np.diff(seg_j[:, 0, :], axis=0)
        back = -(disp @ fwd)
        if lean.max() > 6.0 or back.sum() < -0.05:
            print("    段 %s：前倾中位 %.1f 最大 %.1f 度；沿反朝向前进 %.2f 米"
                  % (sid, float(np.median(lean)), float(lean.max()), float(back.sum())))


for k, v in JOBS.items():
    analyse(v, k)
    print()
