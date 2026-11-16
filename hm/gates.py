"""自检门限：用归一化比值检「拟合是否引入了抖动」。

校准证据（同一段 52 帧，举手动作）
    Kabsch 版坏解   rot中位 70.78 度/帧   hf_ratio 43.93   → 必须判不通过
    最小弧版        rot中位  0.38 度/帧   hf_ratio  1.64   → 必须判通过
    优化式          rot中位  0.24 度/帧   hf_ratio  1.52   → 必须判通过

为什么不用绝对加速度
    跑、急停这类真实快动作的顶点加速度本来就高（0.37），
    用绝对阈值会把它们误判。归一化比值 hf_ratio = 网格二阶差分 / 关节二阶差分
    与动作快慢无关，只反映「网格是否比源信号更抖」。
"""
import numpy as np
from scipy.spatial.transform import Rotation

LIMITS = {"rot_med": 5.0, "hf_ratio": 3.0}


def second_diff_energy(X):
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
    v = np.array(vals)
    return float(np.median(v)), float(np.percentile(v, 95))


def gates(verts, joints, params):
    e_v = second_diff_energy(verts)
    e_j = second_diff_energy(joints)
    med, p95 = rot_step(params)
    return {"rot_med": med, "rot_p95": p95,
            "hf_ratio": float(e_v / max(e_j, 1e-9))}


def check(verts, joints, params):
    """返回 (ok, metrics, 超限项)。"""
    r = gates(np.asarray(verts, np.float64), np.asarray(joints, np.float64), params)
    bad = [k for k, lim in LIMITS.items() if r[k] > lim]
    return (not bad), r, bad


if __name__ == "__main__":
    import sys
    P = np.load(sys.argv[1])
    J = np.asarray(np.load(sys.argv[2]), dtype=np.float64)
    from hm import fit as FM
    import torch
    FM._patch_paths()
    m = FM._load_smpl("cuda")
    with torch.no_grad():
        t = torch.from_numpy(P).float().cuda()
        V = m(betas=t[:, 72:82], global_orient=t[:, 0:3], body_pose=t[:, 3:72],
              transl=t[:, 82:85], return_verts=True).vertices.cpu().numpy()
    ok, r, bad = check(V, J, P)
    print("  rot中位 %.2f ｜ rot95 %.2f ｜ hf_ratio %.2f ｜ %s"
          % (r["rot_med"], r["rot_p95"], r["hf_ratio"],
             "通过" if ok else "不通过：" + ",".join(bad)))


def junction_jerk(seq, junctions, segments):
    """交界突变：过渡窗口内逐帧最大骨向变化的峰值，与前段段内中位的比值。

    校准：MDM 转身那一处峰值 44.2 度/帧（13.6 倍）；MoMask 同一处 5.7 度/帧（1.6 倍）。
    因此 比值 > 6 且 峰值 > 25 度/帧 记为可疑（两项都要满足，
    否则前段几乎静止时比值会被放大成误报）。
    """
    import math
    from hm.finish_job import BONES, dirs_of, rot_between

    seq = np.asarray(seq, dtype=np.float64)
    rates = []
    for i in range(len(seq) - 1):
        d0, _ = dirs_of(seq[i])
        d1, _ = dirs_of(seq[i + 1])
        rates.append(max(math.degrees(rot_between(d0[k], d1[k])[1]) for k in range(len(BONES))))
    rates = np.array(rates) if rates else np.zeros(1)

    starts, t = [], 0
    for i, b in enumerate(segments):
        starts.append((t, t + b["frames"]))
        t += b["frames"]
        if i < len(junctions):
            t += junctions[i]["transition_frames"]

    out = []
    for i, j in enumerate(junctions):
        a1 = starts[i][1]
        b0 = starts[i + 1][0]
        lo, hi = max(0, a1 - 2), min(len(rates), b0 + 2)
        peak = float(rates[lo:hi].max()) if hi > lo else 0.0
        base = float(np.median(rates[starts[i][0]:max(starts[i][0] + 1, a1 - 1)])) \
            if a1 - starts[i][0] > 1 else 1.0
        ratio = peak / max(base, 1e-6)
        out.append({"between": j.get("between"), "peak_deg_per_frame": round(peak, 1),
                    "ratio": round(ratio, 1),
                    "suspect": bool(ratio > 6.0 and peak > 25.0)})
    return out
