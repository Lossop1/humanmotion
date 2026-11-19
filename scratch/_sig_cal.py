"""校准动作签名：先看它在已知「做得对」的样本上过不过，再看能不能抓到「没做出来」的。

做法
    对若干动作，用 momask 各生成 k 个候选，量签名并按规则判定。
    同时跑一遍几何判据。
    期望：几何判据接近全过，而签名能挑出明显不对的样本——
          如果签名在好样本上就频繁报警，说明阈值不对，不能拿去筛。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import criteria_v2, generate, signature, to_joints
from hm.job import bone_lengths

CASES = [
    ("walk", "a person walks forward", 56),
    ("turn", "a person turns around in place", 44),
    ("raise", "a person raises both hands", 40),
    ("sit", "a person sits down on a chair", 48),
    ("jump", "a person jumps up once in place", 40),
    ("wave", "a person waves goodbye with the right hand", 44),
]


def geom_ok(J):
    try:
        ok, hard, met, why = criteria_v2.check(J, "", None, None)
        return ok
    except Exception:
        return None


def main():
    k = 4
    total_sig_bad = total_geom_bad = total = 0
    for key, prompt, frames in CASES:
        feats = generate.generate([prompt] * k, [frames] * k, name="momask")
        rows = []
        for F in feats:
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            sig = signature.signature(J)
            ok, bad, detail = signature.check(prompt, sig)
            g = geom_ok(J)
            rows.append((ok, bad, sig, g))
            total += 1
            total_geom_bad += 0 if g else 1
            total_sig_bad += 0 if ok in (True, None) else 1
        print("  %-6s %s" % (key, prompt))
        for ok, bad, sig, g in rows:
            flag = "签名通过" if ok else ("无法判断" if ok is None else "签名不过：%s" % bad)
            print("     几何%s  签名%s  disp %.2f 手过肩 %.2f 下蹲 %.2f 转向 %.0f 膝 %.0f"
                  % ("通过" if g else "不过", flag, sig["disp_m"], sig["wrist_above"],
                     max(sig["pelvis_drop"], sig["pelvis_up"]), sig["yaw_deg"], sig["knee_range"]))
    print()
    print("  合计 %d 条：几何判据不过 %d 条，签名不过 %d 条" % (total, total_geom_bad, total_sig_bad))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
