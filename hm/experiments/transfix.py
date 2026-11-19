"""对照实验：举手→鼓掌这一对（156 度差），两种过渡构造。

  A 版：现在的做法，关节方向空间插值（方向 slerp，骨长锁定）→ 整条拟合
  B 版：参数空间插值（每个关节的四元数 slerp）→ 分段拟合后拼接

两版都渲出来，看手臂是否还出现「画一圈后反关节」。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hm import config, fit as FM, finish_job as FJ

FM._patch_paths()
BASE = ROOT / "outputs" / "job_long04"
OUT = ROOT / "outputs" / "_transfix"
OUT.mkdir(parents=True, exist_ok=True)
FACES = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"

SEG_A, CAND_A = "seg_002", 1
SEG_B, CAND_B = "seg_003", 2
N = 23


def load(seg, n):
    return np.load(BASE / "segments" / seg / ("candidate_%d" % n) / "joints22.npy")


def verts_of(params):
    """参数 → 顶点（与 A17 里同样的前向）。"""
    m = FM._load_smpl("cuda")
    T = torch.from_numpy(np.asarray(params, dtype=np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    return o.vertices.detach().cpu().numpy(), o.joints.detach().cpu().numpy()[:, :22, :]


def render(params, name):
    V, _ = verts_of(params)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
    np.save(d / "params.npy", np.asarray(params, dtype=np.float64))
    if FACES.exists():
        shutil.copyfile(FACES, d / "faces.npy")
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "render")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp4 = d / "render" / "out.mp4"
    print("   %-18s %s" % (name, "ok" if mp4.exists() else "FAIL"), flush=True)
    return mp4


def slerp_transition(Pa, Pb, n):
    a = np.concatenate([Pa[0:3], Pa[3:72]])
    b = np.concatenate([Pb[0:3], Pb[3:72]])
    ra = R.from_rotvec(a.reshape(-1, 3))
    rb = R.from_rotvec(b.reshape(-1, 3))
    out = np.zeros((n, 85))
    ts = np.linspace(0, 1, n + 2)[1:-1]
    for i, t in enumerate(ts):
        v = np.zeros(24 * 3)
        for j in range(24):
            qa = R.from_rotvec(a.reshape(-1, 3)[j]).as_quat()
            qb = R.from_rotvec(b.reshape(-1, 3)[j]).as_quat()
            if float(np.dot(qa, qb)) < 0.0:
                qb = -qb
            q = qa * (1.0 - t) + qb * t
            q = q / max(np.linalg.norm(q), 1e-9)
            v[j * 3:(j + 1) * 3] = R.from_quat(q).as_rotvec()
        out[i, 0:3] = v[0:3]
        out[i, 3:72] = v[3:72]
        out[i, 72:82] = 0.5 * (Pa[72:82] + Pb[72:82])
        out[i, 82:85] = Pa[82:85] * (1 - t) + Pb[82:85] * t
    return out


def main():
    JA, JB = load(SEG_A, CAND_A), load(SEG_B, CAND_B)
    JB2, dyaw = FJ.align_pair(JA, JB)
    print("  交界姿态差 %.1f 度，对齐旋转 %.1f 度" % (FJ.pose_gap(JA[-1], JB2[0]), dyaw))

    # A 版
    trA = FJ.interp_dirs(JA[-1], JB2[0], N)
    seqA = np.concatenate([JA, trA, JB2], axis=0)
    print("  拟合 A 版整条（%d 帧）..." % len(seqA), flush=True)
    _, PA_all = FM.fit(seqA)
    paramsA = np.asarray(PA_all, dtype=np.float64)

    # B 版
    print("  拟合 B 版的 A 段与 B 段 ...", flush=True)
    _, PJa = FM.fit(JA)
    _, PJb = FM.fit(JB2)
    PJa, PJb = np.asarray(PJa, dtype=np.float64), np.asarray(PJb, dtype=np.float64)
    trB = slerp_transition(PJa[-1], PJb[0], N)
    paramsB = np.concatenate([PJa, trB, PJb], axis=0)

    render(paramsA, "A_direction_space")
    render(paramsB, "B_param_space")

    # 两版在过渡区的关节弯曲符号
    for tag, params in (("A 方向空间", paramsA), ("B 参数空间", paramsB)):
        _, J = verts_of(params)
        fl = FJ.bend_flips(np.asarray(J))
        trans = {k: [x for x in v if 40 <= x <= 62] for k, v in fl.items()}
        trans = {k: v for k, v in trans.items() if v}
        print("   %s 过渡区(40–62)弯曲翻转：%s" % (tag, trans or "无"))
    print("\n  产出：%s" % OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
