"""把 263 维里的头部旋转补回 SMPL 参数，看点头/摇头能不能表达出来。

做法：
  1. 生成 → 从特征里恢复每个关节的旋转（67:193 那块）
  2. 拟合得到 SMPL 参数（85 维）
  3. 取头部（SMPL 15 号关节）的相对转动增量，叠加到参数的 body_pose[14] 上
  4. 前后各渲一段，比较头部的实际转动
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hm import config, fit as FM, generate
from hm.experiments.probe_rot2 import joint_rots

FM._patch_paths()
OUT = ROOT / "outputs" / "_headfix"
OUT.mkdir(parents=True, exist_ok=True)
FACES = ROOT / "runs" / "R01-wave-turn-walk" / "faces.npy"

HEAD_JOINT = 15
BP_IDX = HEAD_JOINT - 1          # body_pose 里对应的第 14 组
COL = 3 + BP_IDX * 3             # 参数向量里的列偏移


def angle_of(Rm):
    tr = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def head_delta_from_feat(F):
    """头部相对自身的转动增量 (T,3,3)，以首帧为基准。"""
    Rj = joint_rots(F)                     # (T,22,3,3)
    h = Rj[:, HEAD_JOINT]
    return h @ h[0].T


def apply_delta(P, delta):
    """把增量叠到拟合参数的头部旋转上。"""
    P = P.copy()
    for t in range(len(P)):
        cur = R.from_rotvec(P[t, COL:COL + 3]).as_matrix()
        P[t, COL:COL + 3] = R.from_matrix(delta[t] @ cur).as_rotvec()
    return P


def render(params, name):
    m = FM._load_smpl("cuda")
    import torch
    T = torch.from_numpy(np.asarray(params, dtype=np.float32)).to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(betas=T[:, 72:82], global_orient=T[:, 0:3], body_pose=T[:, 3:72],
              transl=T[:, 82:85], return_joints=True)
    V = o.vertices.detach().cpu().numpy()
    J = o.joints.detach().cpu().numpy()[:, :22, :]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "verts.npy", np.asarray(V, dtype=np.float64))
    np.save(d / "params.npy", np.asarray(params, dtype=np.float64))
    if FACES.exists():
        shutil.copyfile(FACES, d / "faces.npy")
    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str(ROOT / "hm" / "render_blender.py"), "--", str(d), str(d / "render")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = (d / "render" / "out.mp4").exists()
    print("    %-22s %s" % (name, "ok" if ok else "FAIL"), flush=True)
    return J


def head_motion_from_params(P):
    """参数里头部旋转相对首帧的累计角度。"""
    h = R.from_rotvec(P[:, COL:COL + 3]).as_matrix()
    return angle_of(h @ h[0].T)


def main():
    cases = [
        ("nod", "a person nods their head", 32, "mdm"),
        ("shake", "a person shakes the head", 32, "mdm"),
        ("walk", "a person walks forward", 40, "momask"),
    ]
    from hm import to_joints
    for key, prompt, frames, model in cases:
        print("  %s：%s（%s）" % (key, prompt, model), flush=True)
        F = generate.generate([prompt], [frames], name=model)[0]
        delta = head_delta_from_feat(F)
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        _, P = FM.fit(J)
        P = np.asarray(P, dtype=np.float64)

        before = head_motion_from_params(P)
        P2 = apply_delta(P, delta)
        after = head_motion_from_params(P2)

        render(P, key + "_before")
        render(P2, key + "_after")
        print("     头部累计转动：修前最大 %.1f 度，修后最大 %.1f 度"
              % (before.max(), after.max()), flush=True)
        print("     （特征里的头部转动 %.1f 度）" % angle_of(delta).max(), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
