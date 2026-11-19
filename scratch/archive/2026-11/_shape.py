"""按目标骨长反解 SMPL 体型参数（betas）。

现在 betas 固定为均值 0，组装时报的骨长误差 3.2 厘米就来自这里。
做法
  骨长对 betas 的雅可比用有限差分求（10 次前向，几毫秒），
  再解一个最小二乘，让 21 根骨头的长度尽量贴近目标骨架。
"""
import numpy as np
import torch


def estimate_betas(model, joints, device="cuda", iters=3, damping=1e-3):
    """joints (T,22,3) → betas (10,)。按平均骨长匹配。"""
    from hm.job import BONES

    J = np.asarray(joints, dtype=np.float64)
    # 目标骨长：整段取中位数，减少个别帧噪声
    target = np.median(
        np.stack([np.linalg.norm(J[:, c] - J[:, p], axis=1) for c, p in BONES]), axis=1)

    def bone_lengths_of(betas):
        with torch.no_grad():
            out = model(betas=betas.view(1, -1), global_orient=torch.zeros(1, 3, device=device),
                        body_pose=torch.zeros(1, 69, device=device),
                        transl=torch.zeros(1, 3, device=device), return_joints=True)
        Jj = out.joints[0, :22].cpu().numpy().astype(np.float64)
        return np.array([np.linalg.norm(Jj[c] - Jj[p]) for c, p in BONES])

    b = np.zeros(10, dtype=np.float64)
    for _ in range(iters):
        base = bone_lengths_of(torch.from_numpy(b).float().to(device))
        Jac = np.zeros((len(BONES), 10))
        for k in range(10):
            eps = 1e-2
            bp = b.copy()
            bp[k] += eps
            Jac[:, k] = (bone_lengths_of(torch.from_numpy(bp).float().to(device)) - base) / eps
        r = target - base
        A = Jac.T @ Jac + damping * np.eye(10)
        b = b + np.linalg.solve(A, Jac.T @ r)
        b = np.clip(b, -3.0, 3.0)
    final = bone_lengths_of(torch.from_numpy(b).float().to(device))
    err_cm = float(np.mean(np.abs(final - target))) * 100
    return b.astype(np.float32), err_cm
