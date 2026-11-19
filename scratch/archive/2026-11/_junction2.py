"""交界差分解（按 bounds.json 的真实字段）。"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm.finish_job import BONES, align_pair, dirs_of, rot_between
from hm.job import BONE_NAMES

ROOT = Path("/home/chuan/motion-work/outputs")


def load_seg(jobdir, sbs, sid):
    n = int(str(sbs[sid].get("selected", "cand_01")).split("_")[-1])
    return np.load(jobdir / "segments" / sid / ("candidate_%d" % n) / "joints22.npy")


rows = []
for p in sorted(glob.glob(str(ROOT / "*" / "assembled" / "bounds.json")))[-14:]:
    jobdir = Path(p).parent.parent
    try:
        bj = json.loads(Path(p).read_text(encoding="utf-8"))
        sbs = {s["segment_id"]: s for s in json.loads(
            (jobdir / "storyboard.json").read_text(encoding="utf-8"))["segments"]}
    except Exception:
        continue
    for j in bj.get("junctions", []):
        a, b = j["between"]
        if a not in sbs or b not in sbs:
            continue
        try:
            Ja, Jb = load_seg(jobdir, sbs, a), load_seg(jobdir, sbs, b)
        except Exception:
            continue
        Jb2, _ = align_pair(Ja, Jb)
        dA, _ = dirs_of(Ja[-1])
        dB, _ = dirs_of(Jb2[0])
        angs = sorted(((math.degrees(rot_between(dA[k], dB[k])[1]), BONE_NAMES[k])
                       for k in range(len(BONES))), reverse=True)
        rows.append({"job": jobdir.name, "a": a, "b": b,
                     "gap": j.get("pose_gap_deg"), "fr": j.get("transition_frames"),
                     "rate": j.get("boundary_rate_deg_per_frame"),
                     "win": j.get("w_in_deg_per_frame"), "wout": j.get("w_out_deg_per_frame"),
                     "top": angs[:3]})

print("  交界总数 %d" % len(rows))
if rows:
    gaps = [r["gap"] for r in rows]
    fr = [r["fr"] for r in rows]
    rr = [r["rate"] for r in rows]
    print("  姿态差   中位 %.0f  最大 %.0f 度" % (np.median(gaps), max(gaps)))
    print("  过渡帧数 中位 %.0f  范围 %d-%d" % (np.median(fr), min(fr), max(fr)))
    print("  边界速度 中位 %.1f 度/帧  最大 %.1f" % (np.median(rr), max(rr)))
    print()
    print("  差最大的 10 条：")
    for r in sorted(rows, key=lambda x: -x["gap"])[:10]:
        need = r["gap"] / 9.0
        print("    %-11s %s->%s 差 %3.0f 度 过渡 %2d 帧 速率 %5.1f (入 %.1f 出 %.1f) 需 %.1f 帧  主因 %s"
              % (r["job"][:11], r["a"][-3:], r["b"][-3:], r["gap"], r["fr"], r["rate"],
                 r["win"] or 0, r["wout"] or 0, need,
                 "、".join("%s %.0f度" % (n, v) for v, n in r["top"])))
    excess = [r for r in rows if r["fr"] < r["gap"] / 9.0]
    print()
    print("  过渡帧数不足（低于 9 度/帧所需）的交界：%d/%d" % (len(excess), len(rows)))
