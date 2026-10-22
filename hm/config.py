"""路径与常量。所有模块只从这里读路径，不自己拼死路径。"""
import os
from pathlib import Path

# 工作根目录：本文件所在目录的上一级
ROOT = Path(__file__).resolve().parent.parent

# 依赖
DEPS = ROOT
MOTIUS = DEPS / "motius"
MOMASK_TORCH = DEPS / "momask-torch"
MODELS = DEPS / "models"
DATASETS = DEPS / "datasets" / "HumanML3D"
MDM = DEPS / "mdm"
BLENDER = DEPS / "blender-4.5.0-linux-x64" / "blender"

# 模型
MODEL_MOMASK = MODELS / "Motius-MoMask"
MODEL_MDM = MODELS / "Motius-MDM"
MODEL_MOTIONLCM = MODELS / "Motius-MotionLCM"
MODEL_EVALUATOR = MODELS / "Motius-Evaluator-HumanML3D"

# 输出
OUT_ROOT = ROOT / "runs"

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
