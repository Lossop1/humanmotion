"""未解决项验证包：只打使用者明确指出的未解决问题，不重复已通过的项。

为什么单独一个包
    使用者连续几轮被判同一批 8 条里「已经对了」的那几条，明确抗议：
    「为什么没问题的你要重复？」「几轮下来根本没有什么变化」。
    所以这里只放未解决的 5 项，一条一件事，判起来快、也能直接看出有没有变化。

用法
    python -m tools.make_open_pack
产物
    outputs/_open/clips/clip_NN.mp4
    outputs/_open/sheet.csv       clip / 句子 / 这次看什么 / verdict / note
    outputs/_open/answer_key.json
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
    ("一个人向前走，然后挥手", "走→挥手的过渡是否还像瞬间切换（交界已改成最少 6 帧）"),
    ("一个人快速跑过去，然后慢慢走回来", "「走回来」是否是转身走回，而不是倒退着走"),
    ("一个人慢慢转身，然后跑开", "转身之后跑的那一段，还有没有跳一下 / 脚不沾地"),
    ("一个人原地踏步", "是否真的原地（不前移）、是否抬腿"),
    ("一个人向前跳两下再后退", "腾空时有没有莫名其妙往前平移"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / "outputs" / "_open"
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for i, (sent, focus) in enumerate(CASES, 1):
        clip = "clip_%02d.mp4" % i
        job = "open_%02d" % i
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
        rows.append({"clip": clip, "sentence": sent, "check": focus, "verdict": "", "note": ""})
        key[clip] = {"job": job, "sentence": sent, "check": focus, "exit": p.returncode,
                     "prompts": prompts, "seconds": round(dt, 1)}
        print("  %-10s 出口码 %d｜%s｜%s" % (clip, p.returncode, "有片" if src.exists() else "无片",
              " / ".join(p[:30] for p in prompts)))
    with (out / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "sentence", "check", "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    (out / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print("  包在 %s（只含未解决项，共 %d 条）" % (out, len(CASES)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
