"""解析 IK：由 22 关节位置直接解出 SMPL 的 72 维姿态。

原理
    SMPL 的前向是「逐级旋转」：关节 j 的子骨方向 = C_j @ 静止姿态下的子骨方向，
    其中 C_j 是 j 及所有祖先的局部旋转之积。
    于是对每个关节 j，已知它的子骨在静止姿态下的方向 A 和在目标姿态下的方向 B，
    解一个最小二乘旋转 C_j 使 C_j A ≈ B（Kabsch），
    就能自根到叶逐级求出所有 C_j，再换算成本地旋转 R_j = C_parent⁻¹ C_j。
    有多个子骨的关节（骨盆 3 个、脊柱 2 个）用全部子骨一起解，扭转就定了。

已知局限
    手腕（20/21）在 22 关节里没有子骨，扭转无法确定，置为静止姿态。
"""
import numpy as np

BONES = None


def _children(parents, limit=22):
    ch = {i: [] for i in range(len(parents))}
    for k, p in enumerate(parents):
        if p >= 0 and k < limit:
            ch[int(p)].append(k)
    return ch


def _kabsch(A, B):
    """求旋转 R 使 R @ a ≈ b（a、b 为 (k,3) 单位向量）。"""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def _rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def analytic_pose(J, rest_joints, parents):
    """J (T,22,3) 米、y 向上 → pose (T,72) 轴角、根平移 (T,3)。"""
    from scipy.spatial.transform import Rotation

    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    rest = np.asarray(rest_joints, dtype=np.float64)[:22]
    ch = _children(parents)
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    pose = np.zeros((T, 72), dtype=np.float64)
    transl = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        C = {}
        for j in order:
            kids = ch[j]
            if not kids:
                C[j] = C.get(parents[j], np.eye(3)) if parents[j] >= 0 else np.eye(3)
                continue
            A, B = [], []
            for k in kids:
                a = rest[k] - rest[j]
                na = np.linalg.norm(a)
                if na < 1e-8:
                    continue
                b = J[t, k] - J[t, j]
                nb = np.linalg.norm(b)
                if nb < 1e-8:
                    continue
                A.append(a / na)
                B.append(b / nb)
            if not A:
                C[j] = np.eye(3)
                continue
            C[j] = _kabsch(np.array(A), np.array(B))

        # 局部旋转
        for j in order:
            p = parents[j]
            Cp = C.get(p, np.eye(3)) if p >= 0 else np.eye(3)
            Rl = Cp.T @ C[j]
            rv = _rotvec(Rl)
            if j == 0:
                pose[t, 0:3] = rv
            else:
                pose[t, 3 + (j - 1) * 3: 3 + j * 3] = rv

        # 根平移：让骨盆对上目标
        R0 = Rotation.from_rotvec(pose[t, 0:3]).as_matrix()
        pelvis = R0 @ (rest[0] - rest[0]) + rest[0]
        transl[t] = J[t, 0] - pelvis
    return pose, transl
