"""给视频加分镜标识（克制版），并用 x264 高质量重编码。

之前用 cv2 的 mp4v 写，288 帧只剩 481 KB（约 27 kbps），画面糊。
现在把帧通过管道喂给 ffmpeg（libx264 / crf 16 / preset slow），
并在结束时对比源文件码率，低于源就报警——不靠肉眼判断是否糊。
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/chuan/motion-work/outputs")
CJK = ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
       "/usr/share/fonts/truetype/arphic/uming.ttc")
LATIN = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _font(size, latin=False):
    for p in (LATIN if latin else CJK):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def draw_mixed(d, xy, num, name, f_num, f_cn, fill):
    x, y = xy
    d.text((x, y), num, fill=fill, font=f_num)
    x += d.textlength(num, font=f_num) + 6
    d.text((x, y), name, fill=fill, font=f_cn)


def annotate(job_id, out_name="final_captioned.mp4", crf=16):
    job = ROOT / job_id
    sb = json.loads((job / "storyboard.json").read_text(encoding="utf-8"))
    bj = json.loads((job / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    segs, bounds, junc = sb["segments"], bj["segments"], bj.get("junctions", [])

    owner = []
    for i, b in enumerate(bounds):
        owner += [i] * b["frames"]
        if i < len(junc):
            owner += [i] * junc[i]["transition_frames"]
    labels = [(str(i + 1), (s.get("action") or "")[:8]) for i, s in enumerate(segs)]

    src = job / "render" / "final_cropped.mp4"
    if not src.exists():
        src = job / "render" / "final.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst = job / "render" / out_name
    if dst.exists():
        dst.unlink()

    cmd = [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (w, h), "-r", "%.3f" % fps, "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    fs = max(15, w // 42)
    fn, fc = _font(fs, latin=True), _font(fs)
    fsmall = _font(max(11, fs - 5), latin=True)
    total = len(owner)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok or n >= total:
            break
        i = owner[n]
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        num, name = labels[i]
        tw = d.textlength(num, font=fn) + 6 + d.textlength(name, font=fc)
        pad_x, pad_y, m = 12, 6, max(12, w // 48)
        bx0, by0 = m, h - fs - 2 * pad_y - m
        d.rounded_rectangle([bx0, by0, bx0 + tw + 2 * pad_x, by0 + fs + 2 * pad_y],
                            radius=int((fs + 2 * pad_y) / 2.6), fill=(0, 0, 0, 130))
        draw_mixed(d, (bx0 + pad_x, by0 + pad_y - 1), num, name, fn, fc, (255, 255, 255, 240))
        d.text((w - m - 46, m), "%.1fs" % (n / fps), fill=(255, 255, 255, 120), font=fsmall)
        img = Image.alpha_composite(img, layer)
        out = np.array(img.convert("RGB"))
        cv2.line(out, (0, h - 2), (int(w * n / max(total, 1)), h - 2), (70, 140, 235), 3)
        cv2.line(out, (0, h - 1), (w, h - 1), (255, 255, 255), 1)
        proc.stdin.write(out.tobytes())
        n += 1
    cap.release()
    proc.stdin.close()
    proc.wait()

    a, b = src.stat().st_size, dst.stat().st_size
    secs = max(n / fps, 1e-6)
    flag = "ok" if b >= a * 0.8 else "WARN bitrate below source"
    print("  %s -> %s (%d frames %dx%d) source %.0f kbps -> new %.0f kbps  %s"
          % (job_id, dst.name, n, w, h, a * 8 / 1000 / secs, b * 8 / 1000 / secs, flag))
    return dst


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for jid in sys.argv[1:]:
        annotate(jid)
