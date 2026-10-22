"""263 维特征 → 22 关节坐标。

用 HumanML3D 官方的 recover_from_ric。不重新实现。
"""
import numpy as np
import torch

from hm import config

config.setup_python()


def from_feats(F):
    """(T,263) → (T,22,3)。"""
    from utils.motion_process import recover_from_ric
    F = np.asarray(F, dtype=np.float32)
    if F.ndim != 2 or F.shape[-1] != config.FEAT_DIM:
        raise ValueError("期望 (T,263)，得到 %s" % (F.shape,))
    J = recover_from_ric(torch.from_numpy(F)[None], config.JOINTS).numpy()[0]
    return J.astype(np.float64)


def selftest():
    """自检：走路的关节应当有合理身高与位移。"""
    from hm import generate
    F = generate.generate(["a person walks forward"], [56])[0]
    J = from_feats(F)
    ok = True
    height = float(J[..., 1].max())
    pelvis = float(J[:, 0, 1].mean())
    travel = float(np.linalg.norm(J[-1, 0, [0, 2]] - J[0, 0, [0, 2]]))
    if J.shape != (56, 22, 3):
        ok = False
        print("  形状不对:", J.shape)
    if not (1.4 < height < 2.0):
        ok = False
        print("  身高不合理: %.3f" % height)
    if not (0.7 < pelvis < 1.2):
        ok = False
        print("  骨盆高度不合理: %.3f" % pelvis)
    if not (1.0 < travel < 6.0):
        ok = False
        print("  走路行程不合理: %.3f" % travel)
    print("  转关节自检: %s  身高 %.2f 骨盆 %.2f 行程 %.2f" % (
        "通过" if ok else "不通过", height, pelvis, travel))
    return ok


if __name__ == "__main__":
    selftest()
