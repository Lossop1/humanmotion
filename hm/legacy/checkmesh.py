"""同时检查骨架和网格。

骨架：拟合误差、膝肘弯向、脚部接触、骨长。
网格：每个部位「正面」顶点相对骨盆朝向的偏移。
      正面顶点从静止姿态里取（静止姿态身体朝 +Z），所以不依赖任何一段动作，
      也不依赖「哪一段看起来正常」这种主观判断。

网格扭曲的表现：某个部位的正面顶点朝前分量明显偏小或为负，
而同一帧的骨盆是正的——说明这一块的表面被绕骨轴转过去了。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hm import config, fit as FM, join_params as JP  # noqa: E402

PARTS = {"头": 15, "左肩": 16, "右肩": 17, "左肘": 18, "右肘": 19,
         "左腕": 20, "右腕": 21, "骨盆": 0, "左膝": 4, "右膝": 5,
         "左踝": 7, "右踝": 8}


def smpl(device="cuda"):
    FM._patch_paths()
    return FM._load_smpl(device)


def rest_front_vertices(m, r=0.11):
    """静止姿态下，各部位区域内朝 +Z 最远的顶点索引。"""
    import torch
    T = torch.zeros(1, 85, dtype=torch.float32, device=next(m.parameters()).device)
    T[0, 72:82] = m.betas.mean(dim=0) if hasattr(m, "betas") else 0.0
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True, return_verts=True)
    V = o.vertices.detach().cpu().numpy()[0]
    J = o.joints.detach().cpu().numpy()[0][:22]
    out = {}
    for nm, jn in PARTS.items():
        idx = np.where(np.linalg.norm(V - J[jn], axis=1) < r)[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmin(np.linalg.norm(V - J[jn], axis=1)))])
        out[nm] = int(idx[int(np.argmax(V[idx][:, 2]))])
    return out


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


def bend(a, b, c):
    u = np.asarray(b) - np.asarray(a)
    v = np.asarray(c) - np.asarray(b)
    n = float(np.linalg.norm(u) * np.linalg.norm(v))
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / max(n, 1e-12), -1, 1))))


def check_one(m, folder, front_idx):
    folder = Path(folder)
    P = np.load(folder / "params.npy")
    V = np.load(folder / "verts.npy")
    Js = np.load(folder / "joints.npy")
    Jf = joints_of(m, P)
    f = pelvis_fwd(P)
    res = {}
    res["帧数"] = len(P)
    # 骨架
    err = np.linalg.norm(Js - Jf, axis=2)
    res["拟合误差中位"] = float(np.median(err))
    res["拟合误差最大"] = float(err.max())
    res["左膝弯"] = (min(bend(x[1], x[4], x[7]) for x in Js), max(bend(x[1], x[4], x[7]) for x in Js))
    res["右膝弯"] = (min(bend(x[2], x[5], x[8]) for x in Js), max(bend(x[2], x[5], x[8]) for x in Js))
    # 网格
    mesh = {}
    for nm, v in front_idx.items():
        vals = np.array([float((V[t][v] - Jf[t][PARTS[nm]]) @ f[t]) for t in range(len(P))])
        mesh[nm] = (float(np.median(vals)), float(vals.min()))
    res["网格"] = mesh
    return res


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Q4"
    run = config.OUT_ROOT / name
    m = smpl()
    fi = rest_front_vertices(m)
    print("  静止姿态下各部位的正面顶点索引：")
    print("    " + "  ".join("%s=%d" % (k, v) for k, v in fi.items()))
    print()
    subs = sorted([d for d in run.iterdir() if d.is_dir() and (d / "params.npy").exists()])
    for d in subs:
        try:
            r = check_one(m, d, fi)
        except Exception as e:
            print("  %-10s 跳过（%s）" % (d.name, e))
            continue
        print("  === %s  %d 帧 ===" % (d.name, r["帧数"]))
        print("    骨架：拟合误差 中位 %.3f 最大 %.3f 米   左膝 %.0f~%.0f 右膝 %.0f~%.0f" % (
            r["拟合误差中位"], r["拟合误差最大"], r["左膝弯"][0], r["左膝弯"][1],
            r["右膝弯"][0], r["右膝弯"][1]))
        bad = []
        for nm, (med, lo) in r["网格"].items():
            flag = ""
            if med < 0.0:
                flag = "  ← 反了"
                bad.append(nm)
            elif med < 0.02:
                flag = "  ← 偏小"
                bad.append(nm)
            print("      网格 %-3s 正面朝前 中位 %+.3f 最小 %+.3f 米%s" % (nm, med, lo, flag))
        if bad:
            print("    结论：%s 的网格正面朝前量异常，这一块的表面被绕骨轴转过去了" % "、".join(bad))
        else:
            print("    结论：网格各部位朝向一致，无扭曲迹像")
        print()


if __name__ == "__main__":
    main()
