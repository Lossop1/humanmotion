"""标定数据目录的唯一解析点。

使用者要求：标定必须放在一个文件夹里，别散乱堆放。
历史上数据在 outputs/_reference；现在迁到仓库根目录的 calibration/，
这里做一次兼容：新目录存在就用新的，否则退回旧路径（老的分支/旧副本仍能跑）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "calibration"
OLD = ROOT / "outputs" / "_reference"


def ref():
    """标定数据根目录（含能力表、提示词库、判据阈值）。"""
    return NEW if NEW.exists() else OLD
