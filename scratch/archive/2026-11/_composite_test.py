"""复合提示词能不能替代「两段拼接」：直接生成一个含两个动作的片段。

动机
    交界差压不下去（多采样只换来 5–10 度），因为两侧姿态本来就差得远。
    如果一段同时包含两个动作，就没有交界了——但前提是：
      1. 几何能过（不比单动作差）
      2. 两个动作都真的做出来了（用签名分别检查）
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, signature, to_joints

CASES = [
    ("举起手 → 跳", "a person raises one hand and jumps in place", 56,
     [("wrist_above", 0.05, lambda s: s["wrist_above"]),
      ("pelvis_up", 0.08, lambda s: s["pelvis_up"])]),
    ("停下 → 举手", "a person stops walking and raises one hand", 52,
     [("wrist_above", 0.05, lambda s: s["wrist_above"])]),
    ("跳 → 转身", "a person jumps up and turns around", 52,
     [("pelvis_up", 0.08, lambda s: s["pelvis_up"]),
      ("yaw_deg", 45.0, lambda s: s["yaw_deg"])]),
    ("伸懒腰 → 弯腰", "a person stretches and bends forward", 60,
     [("lean_delta", 20.0, lambda s: s["lean_delta"])]),
]
K = 6


def main():
    for name, prompt, frames, checks in CASES:
        feats = generate.generate([prompt] * K, [frames] * K, name="momask")
        geom = 0
        met = {c[0]: 0 for c in checks}
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            ok, _h, _m, _r = criteria_v2.check(J, prompt, None, None)
            geom += 1 if ok else 0
            s = signature.signature(J)
            for field, th, get in checks:
                met[field] += 1 if get(s) >= th else 0
        parts = "  ".join("%s %d/%d" % (f, n, K) for f, n in met.items())
        print("  %-12s 几何 %d/%d ｜ %s" % (name, geom, K, parts), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
