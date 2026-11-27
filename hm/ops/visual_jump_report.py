"""画面级跳变检测：直接看渲染出来的像素，而不是关节。

为什么需要
    人说的「画面不连续」是视觉现象。实测关节侧的量（单帧转动、根/关节位移、
    相对中位数的倍数）都不能把 10 条人判跳变分出来：单帧转动 >60 只命中 4/10，
    速度倍数在跳变组与非跳变组完全重叠。
    所以换到图像空间：相邻帧的像素差，如果某一帧突然远大于本片常态，就是一次跳。

用法
    python -m tools.visual_jump_report --dir videos/blind_review/_review/clips
输出
    每条片段的 峰值/中位 像素差与倍数，以及和人工标注的对照。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 人工标注「跳变/不连续」的 10 条
HUMAN_JUMP = {"clip_05", "clip_08", "clip_09", "clip_17", "clip_18",
              "clip_22", "clip_24", "clip_26", "clip_33", "clip_36"}


def frame_diffs(path, size=240):
    cap = cv2.VideoCapture(str(path))
    prev = None
    diffs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.asarray(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT.parent / "videos" / "blind_review" / "_review" / "clips"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    if not d.exists():
        print("  找不到片段目录：%s" % d)
        return
    rows = []
    for p in sorted(d.glob("clip_*.mp4")):
        diffs = frame_diffs(p)
        if len(diffs) < 5:
            continue
        med = float(np.median(diffs))
        # 局部离群：每帧与「前后 10 帧的中位」比，去掉整体快慢的影响
        local = []
        for i in range(len(diffs)):
            lo, hi = max(0, i - 10), min(len(diffs), i + 11)
            win = np.concatenate([diffs[lo:i], diffs[i + 1:hi]])
            local.append(float(diffs[i]) / max(float(np.median(win)) if len(win) else 1e-6, 1e-6))
        rows.append({"clip": p.stem, "median": round(med, 2), "max": round(float(diffs.max()), 2),
                     "ratio": round(float(diffs.max()) / max(med, 1e-6), 2),
                     "local_ratio": round(max(local), 2),
                     "peak_frame": int(diffs.argmax()) + 1})
    if not rows:
        print("  没有可分析的片段")
        return
    print("  %-9s %-8s %-8s %-7s %-9s %s" % ("clip", "中位差", "峰值差", "倍数", "局部倍数", "人判跳变"))
    for r in rows:
        print("  %-9s %-8.2f %-8.2f %-7.2f %-9.2f %s" % (r["clip"], r["median"], r["max"], r["ratio"],
              r["local_ratio"], "是" if r["clip"] in HUMAN_JUMP else ""))
    pos = [r for r in rows if r["clip"] in HUMAN_JUMP]
    neg = [r for r in rows if r["clip"] not in HUMAN_JUMP]
    print()
    print("  跳变组 倍数：中位 %.2f｜非跳变组 中位 %.2f（最大 %.2f）"
          % (float(np.median([r["ratio"] for r in pos])), float(np.median([r["ratio"] for r in neg])),
             max(r["ratio"] for r in neg)))
    print("  %-6s %-8s %-10s %s" % ("阈值", "命中", "误报", "灵敏度/特异度"))
    for th in (1.5, 2.0, 2.5, 3.0, 4.0):
        tp = sum(1 for r in pos if r["ratio"] > th)
        fp = sum(1 for r in neg if r["ratio"] > th)
        print("  %-6.1f %-8d %-10d %.0f%% / %.0f%%"
              % (th, tp, fp, 100.0 * tp / max(len(pos), 1), 100.0 * (len(neg) - fp) / max(len(neg), 1)))
    print("  -- 局部倍数（每帧 vs 前后 10 帧中位） --")
    for th in (2.0, 3.0, 4.0, 5.0, 6.0):
        tp = sum(1 for r in pos if r["local_ratio"] > th)
        fp = sum(1 for r in neg if r["local_ratio"] > th)
        print("  %-6.1f %-8d %-10d %.0f%% / %.0f%%"
              % (th, tp, fp, 100.0 * tp / max(len(pos), 1), 100.0 * (len(neg) - fp) / max(len(neg), 1)))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
