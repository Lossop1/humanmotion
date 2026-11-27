"""修复验证包：8 条，每条专门验证一个已修或待验证的问题。

和随机句集的区别
    随机集测泛化；这个包测「改得对不对」——每条都对应使用者标注里的一条具体问题，
    sheet.csv 里写了「这条要看什么」，判的时候有的放矢。

用法（GPU 机器）
    python -m tools.make_verify_pack
产物
    outputs/_verify/clips/clip_NN.mp4
    outputs/_verify/sheet.csv     clip / 句子 / 看什么 / verdict / note
    outputs/_verify/answer_key.json
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    ("一个人后退着离开房间", "是不是真的后退（背向行进方向），而不是普通前进"),
    ("一个人向左走", "左 = 人自己的左；看方向对不对"),
    ("一个人向右走", "右 = 人自己的右；看方向对不对"),
    ("一个人向前走，然后挥手", "两个字幕动作是否都做出来了（走、挥手）"),
    ("一个人快速跑过去，然后慢慢走回来", "跑与走回来是否都做了；字幕是否与实际一致"),
    ("一个人慢慢转身，然后跑开", "转身是否真的慢；有没有浮空转身"),
    ("一个人原地踏步", "是不是原地踏步（此前会被做成站立或往前走）"),
    ("一个人向前跳两下再后退", "跳跃时脚是否离地；后退是否真的后退"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / "outputs" / "_verify"
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for i, (sent, focus) in enumerate(CASES, 1):
        clip = "clip_%02d.mp4" % i
        job = "verify_%02d" % i
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "hm.main", sent, "--job-id", job, "--k", str(a.k)],
                           capture_output=True, text=True, cwd=str(ROOT))
        dt = time.time() - t0
        src = ROOT / "outputs" / job / "render" / "final_captioned.mp4"
        if src.exists():
            shutil.copyfile(src, clips / clip)
        sb = ROOT / "outputs" / job / "storyboard.json"
        prompts = []
        if sb.exists():
            prompts = [(s.get("text_gen_final") or "") for s in
                       json.loads(sb.read_text(encoding="utf-8")).get("segments", [])]
        rows.append({"clip": clip, "sentence": sent, "focus": focus, "verdict": "", "note": ""})
        key[clip] = {"job": job, "sentence": sent, "focus": focus, "exit": p.returncode,
                     "prompts": prompts, "seconds": round(dt, 1)}
        print("  %-10s 出口码 %d｜%s｜%s" % (clip, p.returncode, "有片" if src.exists() else "无片",
              " / ".join(p[:34] for p in prompts)))
    with (out / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "sentence", "focus", "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    (out / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print("  包在 %s" % out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
