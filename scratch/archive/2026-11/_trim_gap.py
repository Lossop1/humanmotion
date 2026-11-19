"""截尾能否缩小交界差：对指定的两段，量不同截尾量下的姿态差。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import finish_job as FJ
from hm import signature

CASES = ["big_s03_r9", "big_s19_r9"]
TRIMS = [0, 2, 4, 6, 8, 12]


def pair_joints(job, sb, i):
    a, b = sb["segments"][i], sb["segments"][i + 1]
    if not a.get("selected") or not b.get("selected"):
        return None
    d = Path("outputs") / job
    na = int(str(a["selected"]).split("_")[-1])
    nb = int(str(b["selected"]).split("_")[-1])
    Ja = np.load(d / "segments" / a["segment_id"] / ("candidate_%d" % na) / "joints22.npy")
    Jb = np.load(d / "segments" / b["segment_id"] / ("candidate_%d" % nb) / "joints22.npy")
    return a, b, Ja, Jb


def main():
    for job in CASES:
        d = Path("outputs") / job
        sb = json.loads((d / "storyboard.json").read_text(encoding="utf-8"))
        if not (d / "assembled" / "bounds.json").exists():
            continue
        for i in range(len(sb["segments"]) - 1):
            got = pair_joints(job, sb, i)
            if not got:
                continue
            a, b, Ja, Jb = got
            b2, _ = FJ.align_pair(Ja, Jb)
            base = FJ.pose_gap(Ja[-1], b2[0])
            if base < 60:
                continue
            print("  %s %s → %s 原始差 %.0f 度（A 段 %d 帧）"
                  % (job, a["segment_id"], b["segment_id"], base, len(Ja)))
            for t in TRIMS:
                if t >= len(Ja) - 8:
                    break
                K = Ja[: len(Ja) - t]
                b3, _ = FJ.align_pair(K, Jb)
                g = FJ.pose_gap(K[-1], b3[0])
                ok, bad, _det = signature.check(a.get("text_gen_final") or "",
                                                signature.signature(K))
                print("    截 %2d 帧 → 差 %5.1f 度 ｜ 动作签名 %s"
                      % (t, g, "通过" if ok else ("仍可疑" if ok is False else "无法判")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
