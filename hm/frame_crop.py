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

# 这个模块原在仓库根；搬进 hm/ 后 parent 是包目录，仓库根要往上两级
ROOT = Path(__file__).resolve().parents[1] / "outputs"


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


def measure_bbox_motion(path, samples=24, thr=10):
    """用「帧间差异」找人：静止的棋盘地面不会变，动的人才变。

    为什么需要：地面加棋盘格后，按颜色比背景的做法会把整块地面当成主体
    （实测 bbox 变成 0,16–719,719，裁切等于没裁、人变小）。
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prev = None
    acc = np.zeros((h, w), dtype=np.float32)
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / samples))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            acc = np.maximum(acc, np.abs(g - prev))
        prev = g
    cap.release()
    mask = acc > thr
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def crop_square(job_id, out_name="final_cropped.mp4", out_size=720, margin=1.38):
    job = ROOT / job_id
    src = job / "render" / "final.mp4"
    (l, t, r, b), (w, h) = measure_bbox(src)
    # 地面有棋盘纹理时，按颜色比背景会把整块地面算成主体（实测 bbox = 整幅），
    # 之后裁切会缩到「几乎没裁」并把人缩得很小。这种情况直接用渲染原画（相机本来就按人取景）。
    if (r - l) > 0.85 * w or (b - t) > 0.85 * h:
        vf_skip = "scale=%d:%d:flags=lanczos" % (out_size, out_size)
        subprocess.run([_ffmpeg(), "-y", "-i", str(src), "-vf", vf_skip,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(job / "render" / out_name)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return job / "render" / out_name
    ch, cw = b - t, r - l
    side = int(max(ch, cw) * margin)
    # 永不放大：① side 不能超过源画面（否则裁切区域比源还大，等于放大）
    #          ② 输出尺寸不能超过 side（否则同样是把小图拉大）
    side = min(side, min(w, h))
    if side < out_size:
        out_size = max(240, side)
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
