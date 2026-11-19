"""实验二：候选数 k 增大时，交界差的最小值能降到多少。

装配时是动态规划在候选里挑「相邻交界差最小」的组合，
所以真正起作用的是候选集合里的最小值，不是平均值。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, to_joints
from hm.finish_job import align_pair, pose_gap

JOB = "/home/chuan/motion-work/outputs/job_route"
CASES = [
    ("seg_002", "seg_003", "a person raises the right hand in place"),
    ("seg_004", "seg_005", "a person turns around in place"),
]


def main():
    for prev_seg, next_seg, prompt in CASES:
        A = np.load("%s/segments/%s/candidate_1/joints22.npy" % (JOB, prev_seg))
        k = 8
        feats = generate.generate([prompt] * k, [40] * k, name="momask")
        gaps = []
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            J2, _ = align_pair(A, J)
            gaps.append(pose_gap(A[-1], J2[0]))
        g = np.array(gaps)
        print("  %s→%s  k=%d  交界差 %s" % (prev_seg, next_seg, k, np.round(g, 0)))
        print("     最小 %.0f 度（k=2 时通常只能拿到 80–150）｜中位 %.0f" % (g.min(), np.median(g)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
