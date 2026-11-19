"""为四类动作标定判据：出拳、踢腿、推拉、走停。

方法（与之前的校准一致）
    正对照：该动作的提示词；负对照：看起来像但不同的动作（走、站、挥手）。
    各生成 6 个片段，看指标分布在两边的分离点，用它当阈值。
    分不开就说明这个量不能用，如实报告。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, signature, to_joints

K = 6
FRAMES = 44

CASES = [
    ("出拳", "a person punches forward", "punch_reach",
     ["a person walks forward", "a person stands still", "a person waves hello with the right hand"]),
    ("踢腿", "a person kicks forward", "kick_reach",
     ["a person walks forward", "a person stands still", "a person marches in place"]),
    ("推", "a person pushes forward with both hands", "hands_forward",
     ["a person walks forward", "a person stands still", "a person raises both hands"]),
    ("走停", "a person stands still", "disp_m",
     ["a person walks forward", "a person runs forward", "a person jogs forward"]),
]


def values(prompt, field):
    feats = generate.generate([prompt] * K, [FRAMES] * K, name="momask")
    out = []
    for F in feats:
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        out.append(signature.signature(J)[field])
    return np.array(out)


def main():
    for name, pos_prompt, field, negatives in CASES:
        pos = values(pos_prompt, field)
        print("  %s（%s）：正对照 %s 中位 %.3f，最小 %.3f" %
              (name, field, pos_prompt[:34], np.median(pos), pos.min()))
        allneg = []
        for neg_prompt in negatives:
            neg = values(neg_prompt, field)
            allneg.append(neg)
            print("      负对照 %-42s 中位 %.3f，最大 %.3f"
                  % (neg_prompt[:42], np.median(neg), neg.max()))
        flat = np.concatenate(allneg)
        sep = pos.min() > flat.max()
        print("      分离：%s（正最小 %.3f vs 负最大 %.3f）"
              % ("可分" if sep else "有重叠", pos.min(), flat.max()))
        if sep:
            print("      建议阈值：%.3f" % (0.5 * (pos.min() + flat.max())))
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
