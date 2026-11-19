"""签名规则的分辨力：规则 × 动作矩阵。

如果「举手」的规则在走路的样本上也通过，那这条规则没有分辨力，等于没写。
做法：每个动作生成 k 条，用所有规则去判，打印矩阵。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import generate, signature, to_joints

CASES = [
    ("walk", "a person walks forward", 56),
    ("turn", "a person turns around in place", 44),
    ("raise", "a person raises both hands", 40),
    ("sit", "a person sits down on a chair", 48),
    ("jump", "a person jumps up once in place", 40),
]
PROBES = ["walk forward", "turn around", "raise both hands", "sit down", "jump up"]


def main():
    k = 3
    table = {}
    for key, prompt, frames in CASES:
        feats = generate.generate([prompt] * k, [frames] * k, name="momask")
        sigs = []
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            sigs.append(signature.signature(J))
        table[key] = sigs

    print("  行=样本所属动作，列=用哪条规则判，✓ 表示该规则认为合格")
    print("  %-8s %s" % ("", "  ".join("%-6s" % p.split()[0] for p in PROBES)))
    for key, sigs in table.items():
        row = []
        for probe in PROBES:
            oks = []
            for s in sigs:
                ok, bad, det = signature.check(probe, s)
                oks.append(True if ok is None else ok)
            row.append("%d/%d" % (sum(oks), len(oks)))
        print("  %-8s %s" % (key, "  ".join("%-6s" % c for c in row)))

    print()
    print("  说明：对角线应为全过（自己的规则自己不误杀），非对角线应大量不过（有分辨力）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
