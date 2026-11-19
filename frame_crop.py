"""按人物包围盒裁切并缩放到 720，同时重新高质量编码。

为什么这么做
    相机取景在 Blender 里靠投影算，和真实渲染差三倍多，改了几次都不可靠。
    换个稳的做法：按更高分辨率渲染，实测人物包围盒后裁到人身上，
    再缩到 720。人物拿到的真实像素变多，背景少占地方。
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent / "outputs"


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def measure_bbox(path, samples=12, thr=18):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    top, bot, left, right = h, 0, w, 0
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / samples))
        ok, f = cap.read()
        if not ok:
            continue
        bg = f[3, 3].astype(int)
        mask = np.abs(f.astype(int) - bg).sum(axis=2) > thr
        ys, xs = np.where(mask)
        if not len(ys):
            continue
        top, bot = min(top, int(ys.min())), max(bot, int(ys.max()))
        left, right = min(left, int(xs.min())), max(right, int(xs.max()))
    cap.release()
    return (left, top, right, bot), (w, h)


def crop_square(job_id, out_name="final_cropped.mp4", out_size=720, margin=1.38):
    job = ROOT / job_id
    src = job / "render" / "final.mp4"
    (l, t, r, b), (w, h) = measure_bbox(src)
    ch, cw = b - t, r - l
    side = int(max(ch, cw) * margin)
    cx, cy = (l + r) // 2, (t + b) // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    side = min(side, w - x0, h - y0)
    dst = job / "render" / out_name
    vf = "crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos" % (side, side, x0, y0, out_size, out_size)
    cmd = [_ffmpeg(), "-y", "-i", str(src), "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frac_before = ch / h
    frac_after = ch / side
    print("  %s：源 %dx%d，人物 %dx%d（占高 %.0f%%）→ 裁 %d 方形 → %d²，占高 %.0f%%"
          % (job_id, w, h, cw, ch, 100 * frac_before, side, out_size, 100 * frac_after))
    return dst


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for jid in sys.argv[1:]:
        crop_square(jid)
