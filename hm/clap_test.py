"""鼓掌的时间序列 + 试几种提示词写法，看哪种能产生「反复击掌」。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import generate, to_joints


def series(J):
    d = np.linalg.norm(J[:, 20] - J[:, 21], axis=1)
    return d


def cycles(d):
    """腕距的局部极小次数（每次极小≈一次击掌）。"""
    n = 0
    for i in range(1, len(d) - 1):
        if d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] < np.median(d) - 0.02:
            n += 1
    return n


CASES = [
    ("现用", "a person claps their hands", 40),
    ("加频率", "a person claps their hands repeatedly", 40),
    ("数据写法", "the person claps their hands several times", 40),
    ("拍手短语", "a person is clapping their hands", 40),
    ("双手对拍", "a person brings both hands together and apart", 40),
    ("鼓掌礼仪", "a person applauds with both hands", 40),
]

print("  %-10s %-46s %8s %8s %6s" % ("写法", "提示词", "最小腕距", "最大腕距", "击掌次数"))
for tag, prompt, frames in CASES:
    J = None
    for _ in range(2):
        F = generate.generate([prompt], [frames])[0]
        J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
        d = series(J)
        c = cycles(d)
        print("  %-10s %-46s %7.1f%% %7.1f%% %6d"
              % (tag, prompt[:44], 100 * d.min(), 100 * d.max(), c), flush=True)
