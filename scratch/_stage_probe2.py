"""修正对比：用骨向（与位置无关）比较原始段与装配段，并允许时间偏移。"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm.finish_job import BONES, dirs_of, rot_between
from hm.job import BONE_NAMES

JOB = Path("/home/chuan/motion-work/outputs/job_bank4")


def dirs_seq(X):
    return np.array([dirs_of(X[t])[0] for t in range(len(X))])


def main():
    sb = json.loads((JOB / "storyboard.json").read_text(encoding="utf-8"))
    seg = [s for s in sb["segments"] if "punch" in (s.get("text_gen_final") or "")][0]
    n = int(str(seg["selected"]).split("_")[-1])
    J = np.load(JOB / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy")
    bj = json.loads((JOB / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    seq = np.load(JOB / "assembled" / "joints22.npy")

    t, span = 0, None
    for i, b in enumerate(bj["segments"]):
        if b["segment_id"] == seg["segment_id"]:
            span = (t, t + b["frames"])
            break
        t += b["frames"]
        if i < len(bj.get("junctions", [])):
            t += bj["junctions"][i]["transition_frames"]
    a, z = span
    S = seq[a:z]
    print("  原始 %d 帧｜装配段 %d–%d（%d 帧）" % (len(J), a, z, len(S)))

    dJ, dS = dirs_seq(J), dirs_seq(S)
    m = min(len(dJ), len(dS))

    print()
    print("  逐帧骨向差（度），不同时间偏移下：")
    for off in range(-3, 4):
        vals = []
        for t2 in range(m):
            u = t2 + off
            if 0 <= u < len(dS):
                vals.append(max(math.degrees(rot_between(dJ[t2][k], dS[u][k])[1])
                                for k in range(len(BONES))))
        if vals:
            print("    偏移 %+d：中位 %.1f 度，最大 %.1f 度" % (off, np.median(vals), max(vals)))

    print()
    print("  逐关节差（偏移 0 时，按平均角差排序前 4）：")
    per = np.zeros(len(BONES))
    for t2 in range(m):
        for k in range(len(BONES)):
            per[k] += math.degrees(rot_between(dJ[t2][k], dS[t2][k])[1])
    per /= m
    for k in np.argsort(-per)[:4]:
        print("    %-12s 平均 %.1f 度" % (BONE_NAMES[k], per[k]))

    def reach(X):
        w = X[:, 21, :] - X[:, 16, :]
        return np.linalg.norm(w, axis=1)
    rJ, rS = reach(J), reach(S)
    print()
    print("  手腕到肩距离：原始 最小 %.2f 最大 %.2f｜装配 最小 %.2f 最大 %.2f"
          % (rJ.min(), rJ.max(), rS.min(), rS.max()))
    print("  达峰帧：原始 %d｜装配 %d" % (int(np.argmax(rJ)), int(np.argmax(rS))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
