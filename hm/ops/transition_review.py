"""过渡段盲评素材生成（固定方法，不再临时手写脚本）。

用途
    交付前必须执行的第二步：把一个任务的**每处交界完整帧**（含前后若干帧上下文）
    导出成图，交给一个没有任何上下文的子agent 独立描述，它的原话里必须能看出
    是否迈步、是否并脚拖、是否跳帧、是否浮空。

为什么固定成工具
    之前几次是临时写脚本挑帧，挑错过区间、也漏过交界末尾；盲评素材必须是完整过渡段。

用法
    python -m hm.ops.transition_review --job fresh8_1
    python -m hm.ops.transition_review --job fresh8_1 --pad 6 --height 480
    python -m hm.ops.transition_review --all    # 产物目录下所有含交界的任务

产物（写在 outputs/<job>/review/）
    trans_<k>.png      第 k 处交界的完整过渡段（每帧一格，左上角标帧号）
    trans_<k>.mp4      同一段落的原速短片（按帧解出来再编码，保持原帧序）
    manifest.json      每处交界的帧区间、帧数、成片帧数、是否完整
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def count_frames(path):
    """逐帧解码计数。Blender 的 mp4 没有索引，不能用 CAP_PROP_FRAME_COUNT。"""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def junction_ranges(bounds):
    """返回 [(起始帧, 结束帧(含), 交界帧数)]。"""
    out, cur = [], 0
    for i, seg in enumerate(bounds["segments"]):
        cur += seg["frames"]
        if i < len(bounds.get("junctions", [])):
            n = bounds["junctions"][i]["transition_frames"]
            out.append((cur, cur + n - 1, n))
            cur += n
    return out


def read_frames(path, wanted):
    """按帧号取帧（顺序解码一次取全，避免随机 seek 拿到错帧）。"""
    cap = cv2.VideoCapture(str(path))
    got, t = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if t in wanted:
            got[t] = f.copy()
        t += 1
    cap.release()
    return got


def crop_body(f, height):
    h, w = f.shape[:2]
    top, bot = int(h * 0.28), int(h * 0.99)
    left, right = int(w * 0.15), int(w * 0.85)
    c = f[top:bot, left:right]
    scale = height / float(c.shape[0])
    return cv2.resize(c, (int(c.shape[1] * scale), height))


def review_job(job_id, pad=5, height=380, fps=20.0):
    job = ROOT / "outputs" / job_id
    bp = job / "assembled" / "bounds.json"
    if not bp.exists():
        print("  %s 没有 bounds.json" % job_id)
        return None
    bounds = json.loads(bp.read_text(encoding="utf-8"))
    mp4 = job / "render" / "final_captioned.mp4"
    if not mp4.exists():
        mp4 = job / "render" / "final.mp4"
    if not mp4.exists():
        print("  %s 没有成片" % job_id)
        return None
    total = count_frames(mp4)
    jr = junction_ranges(bounds)
    if not jr:
        print("  %s 没有交界（只有一段动作）" % job_id)
        return {"job": job_id, "total_frames": total, "junctions": []}
    out = job / "review"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"job": job_id, "source": str(mp4.name), "total_frames": total,
                "pad": pad, "junctions": []}
    for k, (a, b, n) in enumerate(jr, start=1):
        lo, hi = max(0, a - pad), min(total - 1, b + pad)
        wanted = set(range(lo, hi + 1))
        frames = read_frames(mp4, wanted)
        missing = sorted(wanted - set(frames))
        imgs = []
        for t in sorted(frames):
            c = crop_body(frames[t], height)
            cv2.putText(c, "f%d" % t, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            if a <= t <= b:
                cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), (0, 128, 255), 2)
            imgs.append(c)
        png = out / ("trans_%d.png" % k)
        cv2.imwrite(str(png), np.hstack(imgs))
        # 原速短片（只含这段，帧序不变）
        h0, w0 = frames[sorted(frames)[0]].shape[:2]
        vp = out / ("trans_%d.mp4" % k)
        w = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
        for t in sorted(frames):
            w.write(frames[t])
        w.release()
        manifest["junctions"].append({
            "index": k, "junction": [a, b], "frames": n,
            "exported": [lo, hi], "exported_count": len(imgs),
            "missing_frames": missing, "complete": not missing and lo <= a and hi >= b,
            "png": png.name, "mp4": vp.name,
        })
        flag = "完整" if manifest["junctions"][-1]["complete"] else "不完整"
        print("  交界%d 帧 %d–%d（%d 帧）→ %s 导出 %d–%d 共 %d 帧 [%s]"
              % (k, a, b, n, png.name, lo, hi, len(imgs), flag))
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pad", type=int, default=5)
    ap.add_argument("--height", type=int, default=380)
    a = ap.parse_args()
    if a.all:
        base = ROOT / "outputs"
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "assembled" / "bounds.json").exists():
                review_job(d.name, pad=a.pad, height=a.height)
        return
    if not a.job:
        print("  用法：python -m hm.ops.transition_review --job <任务> ｜ --all")
        return
    review_job(a.job, pad=a.pad, height=a.height)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
