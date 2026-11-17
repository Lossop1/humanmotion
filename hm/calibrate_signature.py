"""判据量登记表：哪个量能把哪个动作与「像但不同」的动作分开。

为什么要能重跑
    之前每条规则都是手工标定一次、数字写进文档，过一阵就说不清来路。
    这里把「正对照 / 负对照 / 分离点」变成可执行脚本，表格随时可重生成。

判定
    正对照最小 > 负对照最大 → 可分，阈值取两者中点
    否则 → 不可分，不设规则（或只作提示）
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, signature, to_joints

K = 6
FRAMES = 44

CASES = [
    ("走", "a person walks forward", "disp_m",
     ["a person stands still", "a person turns around in place", "a person raises both hands"]),
    ("转身", "a person turns around in place", "yaw_deg",
     ["a person walks forward", "a person stands still", "a person raises both hands"]),
    ("举手", "a person raises both hands", "wrist_above",
     ["a person walks forward", "a person stands still", "a person bends forward at the waist"]),
    ("坐下", "a person sits down", "pelvis_end_drop",
     ["a person walks forward", "a person stands still", "a person waves hello"]),
    ("蹲下", "a person squats down", "pelvis_drop",
     ["a person walks forward", "a person stands still", "a person waves hello"]),
    ("弯腰", "a person bends forward at the waist", "lean_delta",
     ["a person walks forward", "a person stands still", "a person raises both hands"]),
    ("跳", "a person jumps up in place", "pelvis_up",
     ["a person walks forward", "a person stands still", "a person waves hello"]),
    ("踢腿", "a person kicks forward", "kick_reach",
     ["a person walks forward", "a person stands still", "a person marches in place"]),
    ("出拳", "a person punches forward", "punch_reach",
     ["a person walks forward", "a person stands still", "a person raises both hands"]),
    ("推", "a person pushes forward with both hands", "hands_forward",
     ["a person walks forward", "a person stands still", "a person raises both hands"]),
]


def dist(prompt, field):
    feats = generate.generate([prompt] * K, [FRAMES] * K, name="momask")
    return np.array([signature.signature(np.asarray(to_joints.from_feats(F), dtype=np.float64))[field]
                     for F in feats])


def main():
    print("| 动作 | 量 | 正最小 | 负最大 | 判定 | 阈值 |")
    print("|---|---|---|---|---|---|")
    for name, pos_prompt, field, negatives in CASES:
        pos = dist(pos_prompt, field)
        neg = np.concatenate([dist(p, field) for p in negatives])
        sep = pos.min() > neg.max()
        th = (0.5 * (pos.min() + neg.max())) if sep else None
        print("| %s | %s | %.3f | %.3f | %s | %s |"
              % (name, field, pos.min(), neg.max(),
                 "可分" if sep else "不可分",
                 "%.3f" % th if th is not None else "—"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
