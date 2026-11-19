"""量高频扭曲：逐关节相邻帧的旋转变化率，以及绕骨轴的扭转分量。

对比解析 IK 与优化式两种拟合。
"""
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/chuan/motion-work")
from hm import fit as FM
from hm.ik_init import analytic_pose

FM._patch_paths()
m = FM._load_smpl("cuda")
parents = m.parents.detach().cpu().numpy().tolist()
with torch.no_grad():
    rest = m(betas=torch.zeros(1, 10).cuda(), global_orient=torch.zeros(1, 3).cuda(),
             body_pose=torch.zeros(1, 69).cuda(), transl=torch.zeros(1, 3).cuda(),
             return_joints=True).joints[0].cpu().numpy()

BONE = {3: 0, 6: 3, 9: 6, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14,
        18: 16, 19: 17, 20: 18, 21: 19, 1: 0, 2: 0, 4: 1, 5: 2, 7: 4, 8: 5,
        10: 7, 11: 8}


def split_swing_twist(Rm, axis):
    """把旋转分成绕 axis 的扭转角与其余部分。"""
    v = Rm @ axis
    ang = np.arccos(np.clip(float(np.dot(v, axis)), -1, 1))
    return np.degrees(ang)


def analyse(P, tag):
    T = len(P)
    rot = Rotation.from_rotvec(P[:, 0:72].reshape(T * 24, 3))
    out = {}
    for j in list(BONE):
        axis = rest[j] - rest[BONE[j]]
        axis = axis / np.linalg.norm(axis)
        Rs = [rot[t * 24 + j] for t in range(T)]
        step, tw = [], []
        for t in range(T - 1):
            d = Rs[t + 1] * Rs[t].inv()
            step.append(np.degrees(np.linalg.norm(d.as_rotvec())))
            tw.append(split_swing_twist(d.as_matrix(), axis))
        out[j] = (np.median(step), np.percentile(step, 95), np.median(tw), np.percentile(tw, 95))
    print("  %s（度/帧）" % tag)
    print("    关节   总变化中位  总变化95分位  扭转中位  扭转95分位")
    for j in sorted(out, key=lambda k: -out[k][3])[:8]:
        a, b, c, d = out[j]
        print("    %-5d %8.2f %11.2f %10.2f %10.2f" % (j, a, b, c, d))
    return out


J = np.load("/home/chuan/motion-work/outputs/job_perf3/segments/seg_003/candidate_1/joints22.npy").astype(np.float64)
_, P_opt = FM.fit(J, iters=8, batch_size=64)
_, P_ik = FM.fit(J, method="ik")
analyse(P_opt, "优化式 8 轮")
print()
analyse(P_ik, "解析 IK")
