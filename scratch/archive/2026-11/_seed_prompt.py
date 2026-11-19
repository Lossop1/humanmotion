"""实验：用提示词让后段「从前段结束姿态起步」，看交界差能不能降。

做法
    取一段真实的交界（前段末帧 A_end），对后段的动作生成两批候选：
        plain      原始提示词
        seeded     原始提示词 + 一句「从前段的姿态开始」的英文限定
    两批都解码成关节、按装配方式对齐，量与前段末帧的姿态差。
    只改提示词，不改模型、不改装配——这正是「LLM 影响生成」这条线的能力边界。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, to_joints
from hm.finish_job import BONES, align_pair, dirs_of, pose_gap, rot_between
from hm.job import BONE_NAMES

JOB = "/home/chuan/motion-work/outputs/job_route"


def last_of(seg, cand):
    return np.load("%s/segments/%s/candidate_%d/joints22.npy" % (JOB, seg, cand))


def describe_pose(J):
    """粗描述末帧姿态，用于写进提示词。只看手臂与躯干。"""
    d, _ = dirs_of(J)
    names = [n for _, n in BONE_NAMES] if False else BONE_NAMES
    def ang(idx):
        # 骨向与竖直向上 / 向下的夹角
        v = d[idx]
        up = float(np.clip(v[1], -1, 1))
        return np.degrees(np.arccos(up))
    upper = {"l": 15, "r": 16}  # 左右上臂在 BONES 里的下标需按实际顺序
    return d


def main():
    A = last_of("seg_002", int(1))          # 前段：停下
    a_end = A[-1]
    print("  前段末帧：%s 共 %d 帧" % (A.shape, len(A)))

    plain = "a person raises both hands in place"
    seeded = ("a person raises both hands in place, "
              "starting with the arms relaxed at the sides")
    feats_p = generate.generate([plain] * 4, [40] * 4, name="momask")
    feats_s = generate.generate([seeded] * 4, [40] * 4, name="momask")

    def gaps(feats):
        out = []
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            J2, _ = align_pair(A, J)
            out.append(pose_gap(a_end, J2[0]))
        return np.array(out)

    gp, gs = gaps(feats_p), gaps(feats_s)
    print("  plain  提示词：交界差 %s  中位 %.0f 度" % (np.round(gp, 0), np.median(gp)))
    print("  seeded 提示词：交界差 %s  中位 %.0f 度" % (np.round(gs, 0), np.median(gs)))
    print("  差值的差：%.0f 度（负值表示 seeded 更接近前段）" % (np.median(gs) - np.median(gp)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
