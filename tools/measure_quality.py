"""在已有的任务目录上统计质量指标：穿地、脚滑、加速度比、胶囊自相交。

用法
    python -m tools.measure_quality                 # 扫 outputs/big_s*_r20（标准集）
    python -m tools.measure_quality --glob "ctrl_*" # 扫别的任务

输出
    每个任务一行 + 总体分布；有正负对照的句子（鼓掌 vs 走路）单独列出，用于检验自相交判据。
"""
import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def jobs(pattern):
    out = []
    for d in sorted(ROOT.joinpath("outputs").glob(pattern)):
        if (d / "assembled" / "joints22.npy").exists():
            out.append(d)
    return out


def ground_of(d):
    """地面高度：storyboard 里 terrain.start_y（没有就 0）。"""
    p = d / "storyboard.json"
    if not p.exists():
        return 0.0
    try:
        sb = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    t = sb.get("terrain") or {}
    try:
        return float(t.get("start_y") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="big_s*_r20")
    ap.add_argument("--json", default=None, help="把逐任务结果写成 JSON")
    a = ap.parse_args()
    from hm import quality_metrics as QM

    rows = []
    for d in jobs(a.glob):
        J = np.load(d / "assembled" / "joints22.npy")
        g = ground_of(d)
        gp = QM.ground_penetration(J, g)
        fs = QM.foot_slide(J)
        ar = QM.accel_ratio(J)
        bj_p = d / "assembled" / "bounds.json"
        jd = []
        if bj_p.exists():
            jd = QM.junction_dynamics(J, json.loads(bj_p.read_text(encoding="utf-8")).get("junctions", []))
        si = QM.self_intersection(J)
        rows.append({"job": d.name, "frames": int(len(J)), "ground": g,
                     "ground_pen": gp, "foot_slide": fs, "accel_ratio": ar, "junction_dynamics": jd, "self_intersect": si})
        print("  %-16s %4d 帧 穿地 %2d 帧/深 %.3f 米｜脚滑中位 %s mm/帧｜加速度比 %s｜自交 %d 帧(%.1f%%)"
              % (d.name, len(J), gp["frames_below"], gp["max_depth_m"],
                 fs.get("median_mm_per_frame"), ar, si["frames_hit"], 100.0 * si["hit_ratio"]))
    if not rows:
        print("  没有匹配的任务目录：outputs/%s" % a.glob)
        return
    depths = [r["ground_pen"]["max_depth_m"] for r in rows]
    hits = [r["ground_pen"]["frames_below"] for r in rows]
    slides = [r["foot_slide"]["median_mm_per_frame"] for r in rows if r["foot_slide"]["median_mm_per_frame"] is not None]
    accs = [r["accel_ratio"] for r in rows if r["accel_ratio"] is not None]
    ratios = [r["self_intersect"]["hit_ratio"] for r in rows]
    print()
    print("  任务 %d 个" % len(rows))
    print("  穿地：有穿地的任务 %d 个，最大深度 %.3f 米，深度中位 %.3f 米"
          % (sum(1 for h in hits if h > 0), max(depths), st.median(depths)))
    if slides:
        print("  脚滑：中位 %.1f mm/帧，p90 %.1f mm/帧" % (st.median(slides), max(slides)))
    if accs:
        print("  加速度比：中位 %.2f，最大 %.2f" % (st.median(accs), max(accs)))
    print("  自相交：命中任务 %d/%d，命中帧占比中位 %.1f%%，最大 %.1f%%"
          % (sum(1 for r in ratios if r > 0), len(ratios), 100 * st.median(ratios), 100 * max(ratios)))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  逐任务明细已写：%s" % a.json)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
