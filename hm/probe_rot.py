"""验证：263 维里的旋转块（67:193）到底有没有头部旋转信息。

对三个动作各生成一段，取 recover_rot 的第 15 号关节（头）的旋转，
看它相对自身的转动是否随动作起伏。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mdm" / "data_loaders" / "humanml" / "scripts"))

from hm import generate
from motion_process import recover_rot

JOINTS = {12: "颈", 15: "头", 20: "左腕", 18: "左肘"}


def cont6d_to_mat(d):
    """(T,J,6) → (T,J,3,3)"""
    a1, a2 = d[..., 0:3], d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotvec_angle(R):
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def probe(prompt, frames, model="momask"):
    F = generate.generate([prompt], [frames], name=model)[0]
    import torch
    T = torch.from_numpy(np.asarray(F, dtype=np.float32))[None]
    c6 = recover_rot(T).numpy()[0]          # (T, 23, 6)
    R = cont6d_to_mat(c6)                    # (T, 23, 3, 3)
    out = {}
    for j, name in JOINTS.items():
        rj = R[:, j]
        # 相对首帧的累计转动
        rel = np.einsum("tji,tjk->tik", rj, rj[0])
        ang = rotvec_angle(rel)
        # 相邻帧的瞬时转动
        step = rotvec_angle(np.einsum("tji,tjk->tik", rj[1:], rj[:-1]))
        out[name] = (float(ang.max()), float(np.median(step)), float(step.max()))
    return out


CASES = [
    ("点头", "a person nods their head", 32, "mdm"),
    ("摇头", "a person shakes the head", 32, "mdm"),
    ("挥手", "a person waves hello with the right hand", 40, "momask"),
    ("走路", "a person walks forward", 52, "momask"),
    ("站住", "a person stands still", 40, "momask"),
]

print("  %-6s %-6s %10s %10s %10s" % ("动作", "关节", "累计转动(度)", "中位步长", "最大步长"))
for zh, prompt, frames, model in CASES:
    r = probe(prompt, frames, model)
    for jn, (mx, med, stp) in r.items():
        print("  %-6s %-6s %10.1f %10.2f %10.2f" % (zh, jn, mx, med, stp))
    print()
