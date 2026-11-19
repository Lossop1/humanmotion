"""测裁剪的效果：两侧各裁 k 帧后，交界姿态差怎么变。

用已有的候选关节，不重新生成。对每处交界，A 段取倒数第 k+1 帧之前、B 段取第 k+1 帧之后。
"""
import glob
from pathlib import Path
import json
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from hm import finish_job as FJ

KS = [0, 2, 4, 6, 8]


def load_seg(jobdir, seg_id, cand):
    n = int(str(cand).split("_")[-1])
    return np.load(ROOT + "/outputs/%s/segments/%s/candidate_%d/joints22.npy"
                   % (jobdir, seg_id, n))


rows = []
for f in sorted(glob.glob(ROOT + "/outputs/job_p0*/assembled/bounds.json")):
    job = f.split("/")[-3]
    b = json.load(open(f, encoding="utf-8"))
    if b.get("assembly") != "param_space":
        continue
    segs = b["segments"]
    for k, j in enumerate(b["junctions"]):
        A = load_seg(job, segs[k]["segment_id"], segs[k]["candidate"])
        B = load_seg(job, segs[k + 1]["segment_id"], segs[k + 1]["candidate"])
        row = {"job": job, "j": k + 1, "gap0": j["pose_gap_deg"], "vals": {}}
        for kk in KS:
            if kk >= min(len(A), len(B)) - 2:
                row["vals"][kk] = None
                continue
            a_end = A[len(A) - 1 - kk]
            b_trim = B[kk:]
            b_al, _ = FJ.align_pair(A[:len(A) - kk], b_trim)
            row["vals"][kk] = FJ.pose_gap(a_end, b_al[0])
        rows.append(row)

print("  %-6s %-4s %8s %8s %8s %8s %8s %8s" %
      ("片", "交界", "差(0)", "裁2", "裁4", "裁6", "裁8", "降幅(裁8)"))
tot0 = tot8 = 0.0
n = 0
for r in rows:
    v = r["vals"]
    s = "  %-6s %-4d %8.1f %8s %8s %8s %8s" % (
        r["job"], r["j"], r["gap0"],
        *["%.1f" % v[k] if v.get(k) else "—" for k in (2, 4, 6, 8)])
    if v.get(8):
        drop = 100 * (1 - v[8] / r["gap0"])
        s += " %7.0f%%" % drop
        tot0 += r["gap0"]
        tot8 += v[8]
        n += 1
    print(s)
print()
if n:
    print("  可算的 %d 处：总差 %.0f 度 → %.0f 度，降幅 %.0f%%"
          % (n, tot0, tot8, 100 * (1 - tot8 / tot0)))
print()
print("  逐处看 k 的边际收益（每处选降幅最大者）：")
for r in rows:
    v = r["vals"]
    best = min((vv, kk) for kk, vv in v.items() if vv)
    print("    %s 交界%d  差 %.1f  →  %s（裁 %d 帧）"
          % (r["job"], r["j"], r["gap0"],
             "%.1f 度，降 %.0f%%" % (best[0], 100 * (1 - best[0] / r["gap0"])), best[1]))
