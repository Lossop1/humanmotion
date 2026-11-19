"""单手举这一类动作有没有可用写法：五种措辞各测 6 个候选。

背景
    「a person raises one hand up in place」在批处理里 4 个候选全过不了硬检查，
    而库里「raises both hands」通过率很高但语义不同，不能互换。
    这里只测同语义（单手、向上）的措辞。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, to_joints

PHRASINGS = [
    "a person raises one hand up in place",
    "a person raises the right hand",
    "a person raises one arm up",
    "a person lifts one hand up",
    "a person waves with the right hand",
    "a person raises the right hand up",
]
K = 6


def main():
    for text in PHRASINGS:
        feats = generate.generate([text] * K, [44] * K, name="momask")
        passed, why = 0, ""
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            ok, hard, met, reasons = criteria_v2.check(J, text, None, None)
            passed += 1 if ok else 0
            if not ok and not why and reasons:
                why = reasons[0][:44]
        print("  %-42s %d/%d  %s" % (text, passed, K, why), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
