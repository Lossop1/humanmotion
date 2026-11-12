"""解析 IK 修正版：分叉处用 Kabsch，单子骨关节用最小弧。

原因
    单子骨关节的方向约束秩为 1，Kabsch 的扭转分量完全不受约束，逐帧乱翻
    （实测旧版 70–173 度/帧，优化式只有 1.4 度）。
    改成：只有一个子骨的关节用「最小弧」旋转，它是目标方向的连续函数；
    有两个以上子骨的关节（骨盆 3 个、脊柱3 有 3 个）用 Kabsch，秩够，扭转唯一。

局限
    手腕（20/21）没有子骨，局部旋转取零（跟随前臂），手的朝向仍是猜的。
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _min_arc(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1, 1))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(a, axis)
        perp /= np.linalg.norm(perp)
        return Rotation.from_rotvec(perp * np.pi).as_matrix()
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def _kabsch(A, B):
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def analytic_pose(J, rest_joints, parents):
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    rest = np.asarray(rest_joints, dtype=np.float64)[:22]
    ch = {i: [k for k, p in enumerate(parents) if p == i and k < 22] for i in range(24)}

    pose = np.zeros((T, 72), dtype=np.float64)
    transl = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        C = {}
        for j in range(22):
            kids = ch.get(j, [])
            p = parents[j]
            Cp = C.get(p, np.eye(3)) if p >= 0 else np.eye(3)
            if not kids:
                C[j] = Cp                      # 无子骨：局部旋转取零
                continue
            if len(kids) == 1:
                a = rest[kids[0]] - rest[j]
                b = J[t, kids[0]] - J[t, j]
                R = _min_arc(a, Cp.T @ b)      # 在父框架下解最小弧
                C[j] = Cp @ R
            else:
                A, B = [], []
                for k in kids:
                    av = rest[k] - rest[j]
                    bv = J[t, k] - J[t, j]
                    if np.linalg.norm(av) < 1e-9 or np.linalg.norm(bv) < 1e-9:
                        continue
                    A.append(av / np.linalg.norm(av))
                    B.append((Cp.T @ bv) / np.linalg.norm(bv))
                if not A:
                    C[j] = Cp
                else:
                    C[j] = Cp @ _kabsch(np.array(A), np.array(B))
        for j in range(22):
            p = parents[j]
            Cp = C.get(p, np.eye(3)) if p >= 0 else np.eye(3)
            rv = Rotation.from_matrix(Cp.T @ C[j]).as_rotvec()
            if j == 0:
                pose[t, 0:3] = rv
            else:
                pose[t, 3 + (j - 1) * 3: 3 + j * 3] = rv
        transl[t] = J[t, 0] - rest[0]
    return pose, transl
