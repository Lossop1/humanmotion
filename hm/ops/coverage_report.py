"""语义覆盖的分类报告：不再只给一个百分比。

为什么要分类
    r30 的覆盖率 67%（31/46）看上去比 r20 的 78% 差，但分母里的段不是同一批：
    无法判的段集中在「捡取 / 看 / 喝水 / 直起身」这些本来就没有判据的类别上。
    只报一个百分比会把「新增了做不到判定的动作」误读成「判据退步」。
用法
    python -m tools.coverage_report --glob "big_s*_r30"
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    ("走/跑", r"walks?|walking|runs?|running|sprint|jog|steps?"),
    ("转身", r"turns? around|turning around"),
    ("坐下", r"sits? down|sitting down|takes? a seat|is seated"),
    ("蹲/跪", r"crouch|squat|kneel"),
    ("弯腰/鞠躬", r"bends? forward|bending|bowing|bend over"),
    ("跳", r"jumps?|jumping|hops?"),
    ("踢腿", r"kicks? forward|kicking"),
    ("出拳", r"punches?|punching"),
    ("停住", r"stands? still|stops? (walking|moving)|comes? to a stop"),
    ("推拉", r"pushes?|pulls?"),
    ("捡取/够", r"picks? up|reaching|reaches"),
    ("喝水/拿杯", r"drinks?|cup"),
    ("看/转头", r"looks?|turns? (his|her) head"),
    ("站立", r"stands? (up|upright)|straightens"),
]


def classify(prompt):
    text = str(prompt).lower()
    for name, pat in CATEGORIES:
        if re.search(pat, text):
            return name
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="big_s*_r30")
    a = ap.parse_args()
    from hm import signature

    rows = []
    for d in sorted(ROOT.joinpath("outputs").glob(a.glob)):
        sp = d / "storyboard.json"
        if not sp.exists():
            continue
        for seg in json.loads(sp.read_text(encoding="utf-8"))["segments"]:
            sel = seg.get("selected")
            if not sel:
                continue
            n = int(str(sel).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            prompt = seg.get("text_gen_final") or ""
            ok, _bad, _det = signature.check(prompt, signature.signature(np.load(jp)))
            rows.append((d.name, prompt, ok, signature.why_unjudgeable(prompt), classify(prompt)))

    total = len(rows)
    judgeable = [r for r in rows if r[2] is not None]
    unknown = [r for r in rows if r[2] is None]
    print("  段数 %d｜可判 %d（%.0f%%）｜无法判 %d" % (total, len(judgeable),
          100.0 * len(judgeable) / max(total, 1), len(unknown)))
    print()
    print("  == 无法判的按原因分组 ==")
    groups = {}
    for _j, _p, _o, why, _c in unknown:
        groups.setdefault(why or "原因未登记", []).append(_p[:44])
    for why, ps in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("    %-38s %d 段  例：%s" % (why[:38], len(ps), ps[0]))
    print()
    print("  == 按动作类别 ==")
    print("    %-12s %-6s %-8s %s" % ("类别", "段数", "可判", "可判率"))
    cats = {}
    for _j, _p, ok, _w, c in rows:
        cats.setdefault(c, [0, 0])
        cats[c][0] += 1
        if ok is not None:
            cats[c][1] += 1
    for c, (n, k) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
        print("    %-12s %-6d %-8d %.0f%%" % (c, n, k, 100.0 * k / max(n, 1)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
