"""22 关节 → SMPL 网格（每帧 6890 个顶点）。

用的是 mdm 仓库自带的 SMPLify3D。不重新实现优化器。

三处适配，全部在本文件里完成，依赖目录一个字节都不改：
    1. mdm 的 config 用相对路径。本文件把它们改成绝对路径。
    2. mdm 的 prior.py 直接 pickle.load 那个 GMM 文件。那个文件带 Windows
       换行，直接读会报 "the STRING opcode argument must be quoted"。
       本文件在构造模型期间替换 pickle.load，读完立刻还原。
    3. mdm 的 SMPL 模型要求 chumpy。本文件用 assets 里那份纯 numpy 的副本。
"""
import os
import pickle
import sys
import types

import numpy as np
import torch

from hm import config

SMPL_PY = config.MDM / "visualize" / "joints2smpl"
SRC = SMPL_PY / "src"

ASSETS = config.ROOT / "assets"
SMPL_MODEL = ASSETS / "SMPL_NEUTRAL.pkl"
MEAN_POSE = ASSETS / "mean_pose.npy"
MEAN_SHAPE = ASSETS / "mean_shape.npy"


def _patch_paths():
    """把 mdm 的相对路径改成绝对路径，并把三个目录加进 import 路径。

    customloss.py 用 `from visualize.joints2smpl.src import config` 导入，
    所以 mdm 仓库根目录也必须在 sys.path 里。
    """
    for p in (str(config.MDM), str(SMPL_PY), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # 注意：同一个 config.py 会被导入两次，形成两个模块对象。
    #   mdm 自己的代码用 `from visualize.joints2smpl.src import config`
    #   我如果用 `import config`，改的是另一个对象，改了也不生效。
    # 所以两个都要改。
    import config as _c1
    from visualize.joints2smpl.src import config as _c2
    for c in (_c1, _c2):
        c.SMPL_MODEL_DIR = str(ASSETS) + "/"
        c.GMM_MODEL_DIR = str(SMPL_PY / "smpl_models")
    return _c1


class _CRLFUnpickler(pickle.Unpickler):
    """读之前把 CRLF 规范成 LF。只为兼容那个被转过换行的 GMM 文件。"""

    def __init__(self, f, **kw):
        raw = f.read()
        if isinstance(raw, bytes) and b"\r\n" in raw:
            raw = raw.replace(b"\r\n", b"\n")
        import io
        super().__init__(io.BytesIO(raw), **kw)


def _load_smpl(device="cpu"):
    """加载 SMPL 模型。用 assets 里那份不需要 chumpy 的副本。"""
    import smplx
    if not SMPL_MODEL.exists():
        raise FileNotFoundError("缺 SMPL 模型: %s" % SMPL_MODEL)
    # smplx 要求 model_path/<gender>/<model>.pkl 的布局
    root = SMPL_MODEL.parent / "smpl_body"
    (root / "smpl").mkdir(parents=True, exist_ok=True)
    target = root / "smpl" / "SMPL_NEUTRAL.pkl"
    if not target.exists() or target.stat().st_size != SMPL_MODEL.stat().st_size:
        import shutil
        shutil.copy2(SMPL_MODEL, target)
    return smplx.create(str(root), model_type="smpl", gender="neutral",
                        ext="pkl", batch_size=1, use_pca=False,
                        flat_hand_mean=False).to(device)


_SMPL_CACHE = {}


def _get_smplify(device, iters, batch_size):
    """缓存 SMPL 模型与 SMPLify 实例，避免每次拟合都重新加载。"""
    key = (str(device), int(iters), int(batch_size))
    if key in _SMPL_CACHE:
        return _SMPL_CACHE[key]
    import pickle as _pk
    from smplify import SMPLify3D

    model = _load_smpl(device)
    orig_load, orig_unpickler = _pk.load, _pk.Unpickler
    _pk.Unpickler = _CRLFUnpickler
    _pk.load = lambda f, **kw: _CRLFUnpickler(f, **kw).load()
    try:
        smplify = SMPLify3D(smplxmodel=model, batch_size=batch_size, num_iters=iters,
                            use_collision=False, joints_category="AMASS",
                            device=device)
    finally:
        _pk.load, _pk.Unpickler = orig_load, orig_unpickler
    _SMPL_CACHE[key] = (model, smplify)
    return model, smplify


def fit(joints, batch_size=32, iters=8, device=None):
    """(T,22,3) → (T,6890,3)。

    joints 是米为单位、y 轴向上。
    分批优化，每批 16 帧，批内所有帧一起算。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _patch_paths()

    from smplify import SMPLify3D

    model, smplify = _get_smplify(device, iters, batch_size)

    mean_pose = np.load(MEAN_POSE) if MEAN_POSE.exists() else np.zeros(72, np.float32)
    mean_shape = np.load(MEAN_SHAPE) if MEAN_SHAPE.exists() else np.zeros(10, np.float32)

    J = np.asarray(joints, dtype=np.float32)
    T = len(J)
    conf = torch.ones(22, device=device)
    conf[[7, 8, 10, 11]] = 1.5      # 脚踝与脚，权重高一点

    out_v = np.zeros((T, 6890, 3), dtype=np.float32)
    out_p = np.zeros((T, 85), dtype=np.float32)

    pose0 = torch.from_numpy(mean_pose).float().to(device)
    betas0 = torch.from_numpy(mean_shape).float().to(device)

    done = 0
    while done < T:
        n = min(batch_size, T - done)
        chunk = J[done:done + n]
        kp = torch.from_numpy(chunk).float().to(device)
        if n < batch_size:
            kp = torch.cat([kp, kp[-1:].repeat(batch_size - n, 1, 1)], 0)
        pp = pose0[None].repeat(batch_size, 1).contiguous()
        pb = betas0[None].repeat(batch_size, 1).contiguous()
        pc = torch.zeros(batch_size, 3, device=device)

        verts, _j, pose, betas, cam_t, _jl = smplify(pp, pb, pc, kp,
                                                     conf_3d=conf, seq_ind=done)
        pose = pose.reshape(batch_size, -1)
        betas = betas.reshape(batch_size, -1)
        cam_t = cam_t.reshape(batch_size, 3)
        with torch.no_grad():
            out = model(betas=betas, global_orient=pose[:, :3],
                        body_pose=pose[:, 3:], transl=cam_t, return_verts=True)
        v = out.vertices.detach().cpu().numpy()
        out_v[done:done + n] = v[:n]
        out_p[done:done + n] = np.concatenate(
            [pose.detach().cpu().numpy(), betas.detach().cpu().numpy(),
             cam_t.detach().cpu().numpy()], axis=1)[:n]
        # 用本批最后一帧做下一批的初值，保证时间连续
        pose0 = pose[-1:].squeeze(0).detach()
        betas0 = betas[-1:].squeeze(0).detach()
        done += n
    return out_v, out_p


def selftest():
    """自检：一段走路的关节拟合后，关节误差要小。"""
    from hm import generate, to_joints
    F = generate.generate(["a person walks forward"], [48])[0]
    J = to_joints.from_feats(F)
    V, P = fit(J, batch_size=16, iters=8)
    ok = True
    if V.shape[1] != 6890:
        ok = False
        print("  顶点数不对:", V.shape)
    h = float(V[..., 1].max() - V[..., 1].min())
    if not (1.4 < h < 2.1):
        ok = False
        print("  网格身高不合理: %.2f" % h)
    if not np.isfinite(V).all():
        ok = False
        print("  顶点含 NaN/Inf")
    print("  拟合自检: %s  顶点 %s  身高 %.2f m" % (
        "通过" if ok else "不通过", V.shape, h))
    return ok


if __name__ == "__main__":
    selftest()
