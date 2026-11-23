"""给一个任务生成「分镜表 + 关键帧拼接图」，让人能对着句子看视频。

输出
   分镜表    每段：序号、原文片段、动作标签、英文提示词、帧数、时间区间
   交界表    每处：两段之间、姿态差、过渡帧数
   拼接图    每段取中间一帧，标上段号与动作名
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "outputs"


def report(job_id):
    job = ROOT / job_id
    sb = json.loads((job / "storyboard.json").read_text(encoding="utf-8"))
    bj = json.loads((job / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    segs = sb["segments"]
    bounds = bj["segments"]
    junctions = bj.get("junctions", [])

    # 验证状态：✓ 通过校准量 / ? 无法判 / ✗ 可疑；没有 semantics.json 就不标
    mark_of = {}
    sem_p = job / "semantics.json"
    if sem_p.exists():
        try:
            for row in json.loads(sem_p.read_text(encoding="utf-8")):
                mark_of[row["segment"][-3:]] = {"通过": "OK", "无法判": "?", "未验证": "?", 
                                                "可疑": "X", "未选中": "-"}.get(row["status"], "")
        except Exception:
            mark_of = {}

    lines = []
    t = 0.0
    marks = []
    for i, s in enumerate(segs):
        fr = bounds[i]["frames"]
        start, end = t, t + fr
        lines.append({
            "i": i + 1, "sid": s["segment_id"], "src": s.get("source_text") or "",
            "label": s.get("action") or "", "prompt": s.get("text_gen") or "",
            "frames": fr, "start_s": round(start / 20, 2), "end_s": round(end / 20, 2),
        })
        marks.append((i + 1, s.get("action") or "", int(start + fr // 2)))
        t = end
        if i < len(junctions):
            j = junctions[i]
            jt = j["transition_frames"]
            lines.append({"junction": True, "a": i + 1, "b": i + 2,
                          "gap": j["pose_gap_deg"], "tf": jt,
                          "start_s": round(t / 20, 2), "end_s": round((t + jt) / 20, 2)})
            t += jt

    mp4 = job / "render" / "final.mp4"
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for n, label, idx in marks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, max(total - 1, 0)))
        ok, f = cap.read()
        if ok:
            frames.append((n, label, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()

    if frames:
        cols = min(4, len(frames))
        rows = (len(frames) + cols - 1) // cols
        H = 320
        sheet = Image.new("RGB", (H * cols, (H + 26) * rows), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        # 这台机器上两个字体互补：Droid 有中文没 ASCII，DejaVu 有 ASCII 没中文，分开画
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 20)
        fnt_ascii = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        for i, (n, label, img) in enumerate(frames):
            x, y = (i % cols) * H, (i // cols) * (H + 26)
            sheet.paste(Image.fromarray(img).resize((H, H)), (x, y + 26))
            sid = segs[n - 1]["segment_id"][-3:] if n - 1 < len(segs) else ""
            mark = mark_of.get(sid, "")
            head = "[%s] %d " % (mark or "-", n)
            d.text((x + 6, y + 6), head, fill=(10, 10, 10), font=fnt_ascii)
            w = d.textlength(head, font=fnt_ascii)
            d.text((x + 6 + w, y + 6), label, fill=(10, 10, 10), font=fnt)
        out = job / "分镜.png"
        sheet.save(out)
        print("  拼接图：%s（%d 段，视频 %d 帧）" % (out, len(frames), total))

    print()
    print("  分镜表  %s" % job_id)
    print("  %-4s %-6s %-6s %-26s %s" % ("段", "时间(s)", "帧数", "原文", "动作 / 提示词"))
    for r in lines:
        if r.get("junction"):
            print("  %-4s %-6s %-6d %-26s 交界：姿态差 %.0f 度，过渡 %d 帧" % (
                "—", "%.2f-%.2f" % (r["start_s"], r["end_s"]), r["tf"], "", r["gap"], r["tf"]))
        else:
            print("  %-4s %-6s %-6d %-26s %s / %s" % (
                r["i"], "%.2f-%.2f" % (r["start_s"], r["end_s"]), r["frames"],
                r["src"][:24], r["label"][:12], r["prompt"][:46]))
    print()
    print("  原句：%s" % sb.get("sentence", ""))
    return lines


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for jid in sys.argv[1:]:
        report(jid)
        print("\n" + "=" * 70 + "\n")
