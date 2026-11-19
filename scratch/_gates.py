"""交付前自检：在没有人的情况下，能不能自己抓出「高频扭曲」。

思路
    拟合是「关节 → 网格」的映射。若映射本身平滑，网格的高频能量应与关节同量级；
    若拟合引入了抖动，网格的高频能量会显著高于关节。这是一个不需要人看的交叉检查：
    用源信号（关节）去核对输出（网格）。

指标
    rot_step    逐关节相邻帧转角（度）：中位数与 95 分位
    hf_ratio    网格加速度能量 / 关节加速度能量（二阶差分，按幅度归一）
    accel_max   单帧最大顶点加速度（米/帧²）
"""
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def second_diff_energy(X):
    """X (T,N,3) → 二阶差分的平均模长。"""
    if len(X) < 3:
        return 0.0
    d2 = X[2:] - 2 * X[1:-1] + X[:-2]
    return float(np.mean(np.linalg.norm(d2, axis=-1)))


def rot_step(P, joints=(1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)):
    T = len(P)
    rot = Rotation.from_rotvec(P[:, 0:72].reshape(T * 24, 3))
    vals = []
    for j in joints:
        for t in range(T - 1):
            d = rot[(t + 1) * 24 + j] * rot[t * 24 + j].inv()
            vals.append(np.degrees(np.linalg.norm(d.as_rotvec())))
    vals = np.array(vals)
    return float(np.median(vals)), float(np.percentile(vals, 95))


def gates(verts, joints, params):
    e_v = second_diff_energy(verts)
    e_j = second_diff_energy(joints)
    d2 = verts[2:] - 2 * verts[1:-1] + verts[:-2] if len(verts) >= 3 else np.zeros((1, 1, 3))
    accel_max = float(np.max(np.linalg.norm(d2, axis=-1))) if len(verts) >= 3 else 0.0
    med, p95 = rot_step(params)
    return {
        "rot_med": med,
        "rot_p95": p95,
        "hf_ratio": float(e_v / max(e_j, 1e-9)),
        "accel_max": accel_max,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 用法：python -m hm.gates <params.npy> <joints22.npy>
    P = np.load(sys.argv[1])
    J = np.load(sys.argv[2])
    m = None
    try:
        from hm import fit as FM
        FM._patch_paths()
        m = FM._load_smpl("cuda")
        with torch.no_grad():
            t = torch.from_numpy(P).float().cuda()
            V = m(betas=t[:, 72:82], global_orient=t[:, 0:3], body_pose=t[:, 3:72],
                  transl=t[:, 82:85], return_verts=True).vertices.cpu().numpy()
    except Exception as e:
        print("  无法前向，只算旋转指标：%s" % e)
        V = None
    if V is None:
        print(gates(np.zeros((len(P), 2, 3)), J, P))
    else:
        r = gates(np.asarray(V, dtype=np.float64), np.asarray(J, dtype=np.float64), P)
        print("  rot 中位 %.2f 度/帧 ｜ 95 分位 %.2f ｜ hf_ratio %.2f ｜ accel_max %.4f"
              % (r["rot_med"], r["rot_p95"], r["hf_ratio"], r["accel_max"]))
