"""把已渲染的视频裁成上半身特写。

不碰相机：按每帧的人体轮廓框，取上方 55% 高度、60% 宽度的窗口，放大到 720×720。
这样手部、头部的细节一定看得清，且与相机参数无关。
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def silhouette(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(np.median(g))
    mask = g > bg + 12
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst / "out.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (720, 720))
    n = 0
    boxes = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        b = silhouette(f)
        if b is None:
            boxes.append(None)
        else:
            boxes.append(b)
        n += 1
    cap.release()

    # 用全片中位轮廓框，逐帧固定窗口，避免抖动
    valid = [b for b in boxes if b]
    if not valid:
        print("  找不到人体")
        return
    xs0 = int(np.median([b[0] for b in valid]))
    ys0 = int(np.median([b[1] for b in valid]))
    xs1 = int(np.median([b[2] for b in valid]))
    ys1 = int(np.median([b[3] for b in valid]))
    bh, bw = ys1 - ys0, xs1 - xs0
    print("  轮廓框中位：高 %d 宽 %d（画面 %dx%d）" % (bh, bw, w0, h0))

    # 上半身窗口：轮廓上方 55% 高、1.6 倍宽（留出手臂活动范围）
    ch = int(bh * 0.58)
    cw = int(max(bh * 0.58, bw * 1.9))
    cx = (xs0 + xs1) // 2
    y_top = max(0, ys0 - int(bh * 0.06))
    x0 = max(0, cx - cw // 2)
    y0 = y_top
    cw = min(cw, w0 - x0)
    ch = min(ch, h0 - y0)
    print("  裁切窗口：x %d–%d  y %d–%d" % (x0, x0 + cw, y0, y0 + ch))

    cap = cv2.VideoCapture(str(src))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        crop = f[y0:y0 + ch, x0:x0 + cw]
        if crop.size == 0:
            crop = f
        out.write(cv2.resize(crop, (720, 720), interpolation=cv2.INTER_CUBIC))
        i += 1
    cap.release()
    out.release()
    print("  写出 %s（%d 帧）" % (dst / "out.mp4", i))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
