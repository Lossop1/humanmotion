"""提示词措辞实验：同一动作换几种常见写法，看几何判据能不能过。

背景
    能力表里「sit」登记的是 "a person sits down and stands back up"（坐下再站起），
    直接复用会把动作变成两个，几何判据全不过。所以措辞要单独试。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, to_joints

PHRASINGS = [
    "a person sits down",
    "a person sits down on a chair",
    "a person is sitting down",
    "a person lowers themselves into a seated position",
    "a person squats down slowly",
]


def main():
    k = 4
    for prompt in PHRASINGS:
        feats = generate.generate([prompt] * k, [48] * k, name="momask")
        oks, why = 0, ""
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            ok, hard, met, reasons = criteria_v2.check(J, prompt, None, None)
            oks += 1 if ok else 0
            if not ok and not why:
                why = reasons[0][:60] if reasons else ""
        print("  %-48s 几何通过 %d/%d  %s" % (prompt, oks, k, why))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
