"""验证 263 维里的旋转块是否带头部旋转（自己实现，避免依赖 mdm 的包结构）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import generate

JOINTS = {12: "颈", 15: "头", 18: "左肘", 20: "左腕"}


def quat_to_mat(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def cont6d_to_mat(d):
    a1, a2 = d[..., 0:3], d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def joint_rots(F):
    """(T,263) → (T,22,3,3)：根旋转 + 21 个局部旋转（cont6d）。"""
    data = np.asarray(F, dtype=np.float64)
    T = len(data)
    rot_vel = data[:, 0]
    ang = np.zeros(T)
    ang[1:] = np.cumsum(rot_vel[:-1])
    q = np.zeros((T, 4))
    q[:, 0] = np.cos(ang)
    q[:, 2] = np.sin(ang)
    root = quat_to_mat(q)                      # (T,3,3)
    block = data[:, 67:67 + 21 * 6].reshape(T, 21, 6)
    local = cont6d_to_mat(block)               # (T,21,3,3)
    return np.concatenate([root[:, None], local], axis=1)   # (T,22,3,3)


def angle_of(R):
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def probe(prompt, frames, model="momask"):
    F = generate.generate([prompt], [frames], name=model)[0]
    R = joint_rots(F)
    out = {}
    for j, name in JOINTS.items():
        rj = R[:, j]
        rel = rj @ rj[0].T
        step = angle_of(rj[1:] @ np.transpose(rj[:-1], (0, 2, 1)))
        out[name] = (float(angle_of(rel).max()), float(np.median(step)), float(step.max()))
    return out


CASES = [
    ("点头", "a person nods their head", 32, "mdm"),
    ("摇头", "a person shakes the head", 32, "mdm"),
    ("挥手", "a person waves hello with the right hand", 40, "momask"),
    ("走路", "a person walks forward", 52, "momask"),
    ("站住", "a person stands still", 40, "momask"),
]

print("  %-6s %-5s %10s %10s %10s" % ("动作", "关节", "累计(度)", "中位步长", "最大步长"))
for zh, prompt, frames, model in CASES:
    r = probe(prompt, frames, model)
    for jn, (mx, med, stp) in r.items():
        print("  %-6s %-5s %10.1f %10.2f %10.2f" % (zh, jn, mx, med, stp))
    print()
