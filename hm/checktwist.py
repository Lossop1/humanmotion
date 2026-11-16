"""逐骨头的表面扭转检查。

骨架只给关节位置。每根骨头周围表面的「朝向」在拟合里不受约束，
可以绕骨头自身轴转任意角度而不改变任何关节位置。

做法：
    静止姿态下，对每根骨头取中点附近、朝 +Z 的那个顶点当该骨头的「正面点」。
    某帧里，把这个顶点相对骨头轴的径向方向，与「期望的正面方向」比较，
    夹角就是这根骨头表面的扭转量。
    期望正面方向 = 骨盆正面在垂直于骨头轴的平面上的投影。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hm import config, fit as FM, join_params as JP, trial as TR  # noqa: E402

BONE_NAMES = ["骨盆→左髋", "左髋→左膝", "左膝→左踝", "左踝→左脚",
              "骨盆→右髋", "右髋→右膝", "右膝→右踝", "右踝→右脚",
              "骨盆→脊柱1", "脊柱1→脊柱2", "脊柱2→脊柱3", "脊柱3→颈",
              "颈→头", "脊柱3→左锁骨", "左锁骨→左肩", "左肩→左肘",
              "左肘→左腕", "脊柱3→右锁骨", "右锁骨→右肩", "右肩→右肘",
              "右肘→右腕"]


def smpl():
    FM._patch_paths()
    return FM._load_smpl("cuda")


def rest_data(m):
    import torch
    T = torch.zeros(1, 85, dtype=torch.float32, device=next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True, return_verts=True)
    V = o.vertices.detach().cpu().numpy()[0]
    J = o.joints.detach().cpu().numpy()[0][:22]
    refs = []
    for c, p in TR.BONES:
        mid = 0.5 * (J[c] + J[p])
        d = V - mid
        # 取中点附近、且在 +Z 一侧最远的顶点
        near = np.where(np.linalg.norm(d, axis=1) < 0.09)[0]
        if len(near) == 0:
            near = np.array([int(np.argmin(np.linalg.norm(d, axis=1)))])
        refs.append(int(near[int(np.argmax(V[near][:, 2]))]))
    return V, J, refs


def rest_refs_other_side(m):
    """同一批顶点取 +Z 侧的对立面，用来在方向取反时改用另一侧。"""
    import torch
    T = torch.zeros(1, 85, dtype=torch.float32, device=next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True, return_verts=True)
    V = o.vertices.detach().cpu().numpy()[0]
    J = o.joints.detach().cpu().numpy()[0][:22]
    refs = []
    for c, p in TR.BONES:
        mid = 0.5 * (J[c] + J[p])
        d = V - mid
        near = np.where(np.linalg.norm(d, axis=1) < 0.09)[0]
        if len(near) == 0:
            near = np.array([int(np.argmin(np.linalg.norm(d, axis=1)))])
        refs.append(int(near[int(np.argmin(V[near][:, 2]))]))
    return refs


def joints_of(m, P):
    import torch
    T = torch.from_numpy(np.asarray(P, dtype=np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    return o.joints.detach().cpu().numpy()[:, :22, :]


def pelvis_fwd(P):
    from scipy.spatial.transform import Rotation as R
    return np.array([R.from_rotvec(np.asarray(P[t, 0:3], dtype=np.float64)).apply(
        np.array([0.0, 0.0, 1.0])) for t in range(len(P))])


def twist_per_bone(J, V, refs, fwd):
    """每根骨头表面扭转的角度（0 = 正面朝前，180 = 完全转到背面）。"""
    n = len(J)
    out = np.zeros((n, len(TR.BONES)))
    for t in range(n):
        f = fwd[t].copy()
        for k, (c, p) in enumerate(TR.BONES):
            axis = J[t, c] - J[t, p]
            na = np.linalg.norm(axis)
            if na < 1e-6:
                continue
            axis = axis / na
            radial = V[t][refs[k]] - J[t, p]
            radial = radial - axis * float(np.dot(radial, axis))
            nr = np.linalg.norm(radial)
            if nr < 1e-6:
                continue
            radial = radial / nr
            want = f - axis * float(np.dot(f, axis))
            nw = np.linalg.norm(want)
            if nw < 1e-6:
                continue
            want = want / nw
            out[t, k] = math.degrees(math.acos(float(np.clip(np.dot(radial, want), -1, 1))))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Q4"
    run = config.OUT_ROOT / name
    m = smpl()
    _, Jrest, refs = rest_data(m)
    refs2 = rest_refs_other_side(m)
    # 用挥手段自动判定哪一侧是正面：哪一组让脊柱的扭转更小就用哪一组
    calib = run / "A1_挥手"
    if calib.exists():
        Pc = np.load(calib / "params.npy"); Vc = np.load(calib / "verts.npy")
        Jc = joints_of(m, Pc); fc = pelvis_fwd(Pc)
        best = None
        for cand in (refs, refs2):
            Tc = twist_per_bone(Jc, Vc, cand, fc)
            score = float(np.median(Tc[:, [8, 9, 10]]))
            if best is None or score < best[0]:
                best = (score, cand)
        refs = best[1]
        print("  校准：脊柱扭转基准 %.1f 度（越小说明这一侧取对了）" % best[0])
    for d in sorted([x for x in run.iterdir() if x.is_dir() and (x / "params.npy").exists()]):
        P = np.load(d / "params.npy")
        V = np.load(d / "verts.npy")
        J = joints_of(m, P)
        T = twist_per_bone(J, V, refs, pelvis_fwd(P))
        med = np.median(T, axis=0)
        order = np.argsort(-med)
        print("  === %s  %d 帧 ===" % (d.name, len(P)))
        shown = 0
        for k in order:
            if med[k] < 25.0:
                break
            print("    %-14s 表面扭转 中位 %5.1f 度   最大 %5.1f 度" % (
                BONE_NAMES[k], med[k], T[:, k].max()))
            shown += 1
        if shown == 0:
            print("    所有骨头表面扭转都在 25 度以内")
        print()


if __name__ == "__main__":
    main()
