"""语义判官：把渲染视频抽 6 帧拼图，交给 deepseek-flash 按固定三问判断。

三问（只问这三件，不问好坏）：
    Q1  画面里出现的是不是【动作名】这个动作
    Q2  有没有混进别的动作
    Q3  有没有明显变形（反关节、穿模、肢体自交）

判定：Q1 是 且 Q2 无 且 Q3 无 → 是；Q1 是 但 Q2/Q3 有 → 部分；Q1 否 → 否

用法：python -m hm.vlm_judge [--dir ...] [--limit N]
"""
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/chuan/motion-work")
KEY = (ROOT / ".deepseek_key").read_text().strip()
API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
VIDEO_DIR = ROOT / "outputs" / "_semcheck" / "videos"
CACHE = ROOT / "outputs" / "_semcheck" / "judge_cache"
CACHE.mkdir(parents=True, exist_ok=True)

RUBRIC = """你在检查一段由文本生成的三维人体动作视频。
视频被等间隔抽成 6 帧，按 2 行 3 列排列（左上→右下即时间顺序）。
画面是一个灰色人体模型，固定机位，没有场景。

要判断的动作名（中文）：{zh}
生成这段动作时使用的英文提示词：{prompt}

只回答下面三个问题，不要评价好不好看、自不自然：

Q1 画面里出现的是不是「{zh}」这个动作？注意：只要动作能做出来就算，不要求幅度、速度、风格。
Q2 有没有混进来别的动作？如果有，说明是什么动作。
Q3 有没有明显变形？判断标准只有三条：反关节（肘或膝朝反方向弯）、穿模（肢体插进身体）、肢体自交（两条肢体穿过彼此）。

注意两点：
- 模型没有手指关节，手掌细节无法体现，不要因为「看不出手指」而判否。
- 固定在原地的模型，位移类动作（走、跑）只能从姿态变化看出，不要因为「看不出在移动」而判否。

只输出 JSON，不要任何解释文字：
{{"Q1":"是 或 否","Q1_依据":"一句话","Q2":"无 或 有：<动作>","Q3":"无 或 有：<哪种变形>","verdict":"是 或 部分 或 否"}}"""


def strip_of(mp4, cols=3, rows=2):
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, 0
    idxs = [int((i + 0.5) / (cols * rows) * n) for i in range(cols * rows)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, n - 1))
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None, n
    h, w = frames[0].shape[:2]
    sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for k, f in enumerate(frames):
        sheet.paste(Image.fromarray(f), ((k % cols) * w, (k // cols) * h))
    d = ImageDraw.Draw(sheet)
    for k in range(len(frames)):
        d.text(((k % cols) * w + 8, (k // cols) * h + 6), "#%d" % (k + 1), fill=(200, 0, 0))
    return sheet, n


def ask(sheet, zh, prompt, max_tokens=2600):
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": RUBRIC.format(zh=zh, prompt=prompt)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            txt = (msg.get("content") or "").strip()
            if txt:
                return txt, d["usage"]
            time.sleep(2)
        except urllib.error.HTTPError as e:
            print("    HTTP %d %s" % (e.code, e.read()[:120]), flush=True)
            time.sleep(3)
        except Exception as e:
            print("    异常 %s" % e, flush=True)
            time.sleep(3)
    return "", {}


def parse(txt):
    txt = txt.replace("```json", "").replace("```", "").strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    metas = json.loads((ROOT / "outputs/_reference/capability_v4.json").read_text(encoding="utf-8"))["table"]
    from hm.actions import ACTIONS_ALL
    look = {a[0]: a for a in ACTIONS_ALL}

    files = sorted(VIDEO_DIR.glob("*.mp4"))
    if only:
        files = [f for f in files if only in f.name]
    if limit:
        files = files[:limit]

    out = []
    for f in files:
        key = f.stem.rsplit("_", 1)[0]
        a = look.get(key)
        if not a:
            continue
        zh, prompt = a[1], a[2]
        cached = CACHE / (f.stem + ".json")
        if cached.exists():
            rec = json.loads(cached.read_text(encoding="utf-8"))
            out.append(rec)
            print("  %-24s 缓存 %s" % (f.stem, rec["verdict"]), flush=True)
            continue
        sheet, n = strip_of(f)
        if sheet is None:
            print("  %-24s 抽帧失败" % f.stem, flush=True)
            continue
        sheet.save(CACHE / (f.stem + ".png"))
        txt, usage = ask(sheet, zh, prompt)
        j = parse(txt)
        rec = {"file": f.name, "key": key, "zh": zh, "prompt": prompt,
               "frames": n, "verdict": None, "raw": txt[:600]}
        if j:
            rec.update({"Q1": j.get("Q1"), "Q1_依据": j.get("Q1_依据"),
                        "Q2": j.get("Q2"), "Q3": j.get("Q3"),
                        "verdict": "是" if j.get("verdict") == "是" else
                                   ("部分" if j.get("verdict") == "部分" else "否")})
        else:
            rec["verdict"] = "解析失败"
        (CACHE / (f.stem + ".json")).write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        out.append(rec)
        print("  %-24s → %s  %s" % (f.stem, rec["verdict"], (rec.get("Q1_依据") or "")[:40]),
              flush=True)

    (ROOT / "outputs/_semcheck/vlm_judge.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  判完 %d 段，写入 vlm_judge.json" % len(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
