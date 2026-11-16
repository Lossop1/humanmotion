"""对手审的 44 段算三组量，判断每个失败是否可被程序检出。

  A 头关节的位移幅度      头部动作（点头/摇头）在关节上到底能不能表达
  B 双手最小距离、手到头的距离   鼓掌贴合、摸头穿模
  C 肘/膝弯曲方向的符号翻转      反关节
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm.job import BONES

WORK = ROOT / "outputs" / "_semcheck" / "work"
CAP = json.loads((ROOT / "outputs/_reference/capability_v4.json").read_text(encoding="utf-8"))["table"]

CHAINS = (("左肘", 16, 18, 20), ("右肘", 17, 19, 21), ("左膝", 1, 4, 7), ("右膝", 2, 5, 8))


def body_fwd(Jf):
    acc = np.zeros(3)
    for a, t in ((7, 10), (8, 11)):
        v = Jf[t] - Jf[a]
        v[1] = 0.0
        n = np.linalg.norm(v)
        if n > 1e-6:
            acc += v / n
    n = np.linalg.norm(acc)
    return acc / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def bend_sign(Jf, r, m, t):
    f = body_fwd(Jf)
    side = np.cross(np.array([0.0, 1.0, 0.0]), f)
    return float(np.dot(np.cross(Jf[m] - Jf[r], Jf[t] - Jf[m]), side))


rows = []
for key, zh, prompt, frames in [(k, v["action_zh"], v["prompt"], 0) for k, v in CAP.items()]:
    pass

from hm.actions import ACTIONS_ALL
for key, zh, prompt, frames in ACTIONS_ALL:
    v = CAP.get(key)
    if not v or v["status"] not in ("support", "approx"):
        continue
    d = WORK / ("%s_%s" % (key, v["best"]))
    f = d / "joints.npy"
    if not f.exists():
        continue
    J = np.load(f)
    T = len(J)
    # A 头关节相对颈的位移范围
    head = J[:, 15] - J[:, 12]
    a_range = float(np.ptp(head, axis=0).max())
    # B 距离
    hh = float(min(np.linalg.norm(J[t, 20] - J[t, 21]) for t in range(T)))
    hd = float(min(min(np.linalg.norm(J[t, 20] - J[t, 15]), np.linalg.norm(J[t, 21] - J[t, 15]))
                   for t in range(T)))
    # C 弯曲符号翻转
    flips = {}
    for nm, r, m, t2 in CHAINS:
        s = np.array([bend_sign(J[i], r, m, t2) for i in range(T)])
        flips[nm] = int(sum(1 for i in range(1, T) if (s[i] > 0) != (s[i - 1] > 0)))
    rows.append({"key": key, "zh": zh, "model": v["best"],
                 "head_range": a_range, "hand_hand": hh, "hand_head": hd,
                 "flips": flips, "flip_total": sum(flips.values())})

print("  %-10s %-6s %8s %8s %8s %s" % ("动作", "后端", "头位移", "双手距", "手头距", "弯曲翻转"))
for r in sorted(rows, key=lambda x: -x["head_range"]):
    print("  %-10s %-6s %7.3f %7.3f %7.3f  %d"
          % (r["zh"], r["model"], r["head_range"], r["hand_hand"], r["hand_head"], r["flip_total"]))

print()
ans = [r for r in rows if r["key"] in ("nod", "shake_head")]
print("  头部动作：")
for r in ans:
    print("    %-8s 头相对颈的位移范围 %.3f 米" % (r["zh"], r["head_range"]))
others = [r["head_range"] for r in rows if r["key"] not in ("nod", "shake_head")]
print("    其余动作的头位移范围中位 %.3f 米" % float(np.median(others)))
print()
clap = [r for r in rows if r["key"] == "clap"]
if clap:
    print("  鼓掌：双手最小距离 %.3f 米（手贴合的判据在 0.1 米以内）" % clap[0]["hand_hand"])
print()
bad = [r for r in rows if r["key"] in ("hands_hips", "nod")]
for r in bad:
    print("  %-8s 弯曲翻转 %d 次，明细 %s" % (r["zh"], r["flip_total"],
                                              {k: v for k, v in r["flips"].items() if v}))
print()
allflip = sorted(rows, key=lambda x: -x["flip_total"])[:6]
print("  弯曲翻转最多的六段：")
for r in allflip:
    print("    %-10s %d 次 %s" % (r["zh"], r["flip_total"], {k: v for k, v in r["flips"].items() if v}))
