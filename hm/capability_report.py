"""能力表：把所有对外可引用的数字从结果目录里重新算一遍，不手抄。

用法
    python -m hm.capability_report            # 用最新一轮
    python -m hm.capability_report --round 19

输出
    一张表，每行都标出来源（哪一轮、由哪个脚本算出），便于写报告时核对。
"""
import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
EVAL = OUT / "_eval_big"


def latest_round():
    files = sorted(EVAL.glob("round*.json"), key=lambda p: int(p.stem.replace("round", "")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()

    p = (EVAL / ("round%d.json" % a.round)) if a.round else latest_round()
    if p is None or not p.exists():
        print("  没有评测结果可读")
        sys.exit(2)
    rnd = int(p.stem.replace("round", ""))
    rows = json.loads(p.read_text(encoding="utf-8"))

    from hm import signature
    tag = "r%d" % rnd
    planned = selected = passed = suspect = unknown = 0
    for d in sorted(OUT.glob("big_s*_%s" % tag)):
        sb_p = d / "storyboard.json"
        if not sb_p.exists():
            continue
        for seg in json.loads(sb_p.read_text(encoding="utf-8"))["segments"]:
            planned += 1
            sel = seg.get("selected")
            if not sel:
                continue
            selected += 1
            n = int(str(sel).split("_")[-1])
            jp = d / "segments" / seg["segment_id"] / ("candidate_%d" % n) / "joints22.npy"
            if not jp.exists():
                continue
            ok, _bad, _det = signature.check(seg.get("text_gen_final") or "",
                                             signature.signature(np.load(jp)))
            if ok is None:
                unknown += 1
            elif ok:
                passed += 1
            else:
                suspect += 1

    gaps = [g for r in rows if r.get("junctions") for g in r["junctions"]]
    ratios = [x for r in rows if r.get("ratios") for x in r["ratios"]]
    times = [r["elapsed"] for r in rows if r.get("elapsed")]
    made = sum(1 for r in rows if r.get("ok"))

    print("  能力表（来源：轮%d 的 outputs/big_s*_r%d 与 round%d.json）" % (rnd, rnd, rnd))
    print()
    print("  %-28s %-22s %s" % ("项目", "数值", "来源"))
    print("  %-28s %-22s %s" % ("出片", "%d/%d" % (made, len(rows)), "round%d.json" % rnd))
    print("  %-28s %-22s %s" % ("规划段 / 出片段", "%d / %d" % (planned, selected), "storyboard.json"))
    print("  %-28s %-22s %s" % ("丢段", "%d" % (planned - selected), "storyboard.json"))
    print("  %-28s %-22s %s" % ("语义覆盖（可判/总）",
                                "%d/%d = %.0f%%" % (passed + suspect, planned,
                                                    100.0 * (passed + suspect) / max(planned, 1)),
                                "hm.signature 校准量"))
    print("  %-28s %-22s %s" % ("其中可疑", "%d" % suspect, "同上"))
    print("  %-28s %-22s %s" % ("无法判", "%d" % unknown, "G29 证明分不开的那些动作"))
    if gaps:
        print("  %-28s %-22s %s" % ("交界处数", "%d" % len(gaps), "bounds.json"))
        print("  %-28s %-22s %s" % ("姿态差（只记录）",
                                    "中位 %.0f / 最大 %.0f 度" % (st.median(gaps), max(gaps)), "bounds.json"))
    if ratios:
        print("  %-28s %-22s %s" % ("过渡速率比合格",
                                    "%d/%d（区间 %.2f–%.2f）" % (
                                        sum(1 for x in ratios if 0.7 <= x <= 1.4), len(ratios),
                                        min(ratios), max(ratios)), "G26 判据"))
    if times:
        print("  %-28s %-22s %s" % ("单句耗时（中位）", "%.1f 秒" % st.median(times),
                                    "eval_big 计时，缓存命中"))
    print()
    print("  注：段数是合并后的段；缓存的命中与否会明显改变耗时，比较耗时时须说明。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
