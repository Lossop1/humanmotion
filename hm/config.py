"""路径与常量。所有模块只从这里读路径，不自己拼死路径。"""
import os
from pathlib import Path

# 工作根目录：本文件所在目录的上一级
ROOT = Path(__file__).resolve().parent.parent

# 依赖根目录：默认就是 ROOT；模型/数据集/外部仓库放在别处时设 HM_DEPS
DEPS = Path(os.environ.get("HM_DEPS", str(ROOT))).expanduser()
MOTIUS = DEPS / "motius"
MOMASK_TORCH = DEPS / "momask-torch"
MODELS = DEPS / "models"
DATASETS = DEPS / "datasets" / "HumanML3D"
MDM = DEPS / "mdm"


def find_blender():
    """定位 Blender 可执行文件。优先级：HM_BLENDER > PATH 中的 blender > DEPS 下的解压目录。

    找不到时返回一个不存在的路径（render.py 会据此给出明确报错），而不是抛异常，
    这样不渲染的流程（规划、单测）不受影响。
    """
    env = os.environ.get("HM_BLENDER")
    if env:
        return Path(env).expanduser()
    import shutil
    exe = "blender.exe" if os.name == "nt" else "blender"
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    for base in (DEPS, DEPS / "blender"):
        if base.is_dir():
            hits = sorted(base.glob("blender-*/" + exe))
            if hits:
                return hits[-1]
    return DEPS / "blender" / exe


BLENDER = find_blender()

# 模型
MODEL_MOMASK = MODELS / "Motius-MoMask"
MODEL_MDM = MODELS / "Motius-MDM"
MODEL_MOTIONLCM = MODELS / "Motius-MotionLCM"
MODEL_EVALUATOR = MODELS / "Motius-Evaluator-HumanML3D"

# 输出
# 统一输出根：主链写 outputs/，这里必须同源。
# 历史遗留：默认曾是 runs/，导致 job.py 写 outputs/、config.OUT_ROOT 指 runs/ 的双根目录。
OUT_ROOT = Path(os.environ.get("HM_OUT", str(ROOT / "outputs"))).expanduser()

# 动作参数
FPS = 20
JOINTS = 22
FEAT_DIM = 263

# 默认模型
DEFAULT_MODEL = "momask"

MODEL_PATHS = {
    "momask": MODEL_MOMASK,
    "mdm": MODEL_MDM,
    "motionlcm": MODEL_MOTIONLCM,
}


def ensure_out(name):
    """建输出目录并返回。"""
    p = OUT_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_python():
    """把依赖加到 import 路径。所有模块开头调用。"""
    import sys
    for p in (str(MOTIUS), str(MOMASK_TORCH)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # momask-torch 用了 np.float，新版 numpy 移除了它
    import numpy as np
    if not hasattr(np, "float"):
        np.float = float
