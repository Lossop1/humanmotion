"""跑随机句集并打包成盲评包（句子可见，系统判定不可见）。

用法（在 GPU 机器上）
    python -m tools.run_random_set --n 20 --seed 20261127
产物
    outputs/_random/<seed>/clips/clip_NN.mp4
    outputs/_random/<seed>/sheet.csv       clip / sentence / verdict / note
    outputs/_random/<seed>/answer_key.json clip → job、出口码、句子
    outputs/_random/<seed>/result.json    每个任务：出口码、段数、帧数、耗时、语义状态
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools.make_random_set import build

    sents = build(a.n, a.seed)
    out = ROOT / "outputs" / "_random" / str(a.seed)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, key, result = [], {}, []
    for i, s in enumerate(sents, 1):
        clip = "clip_%02d.mp4" % i
        job = "rnd%d_%02d" % (a.seed % 10000, i)
        import time
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "hm.main", s, "--job-id", job, "--k", str(a.k)],
                           capture_output=True, text=True, cwd=str(ROOT))
        dt = time.time() - t0
        d = ROOT / "outputs" / job
        src = d / "render" / "final_captioned.mp4"
        has_clip = src.exists()
        if has_clip:
            import shutil
            shutil.copyfile(src, clips / clip)
        st = []
        sp = d / "semantics.json"
        if sp.exists():
            st = [x.get("status") for x in json.loads(sp.read_text(encoding="utf-8"))]
        segs = 0
        sbp = d / "storyboard.json"
        if sbp.exists():
            segs = len(json.loads(sbp.read_text(encoding="utf-8")).get("segments", []))
        rows.append({"clip": clip, "sentence": s, "verdict": "", "note": ""})
        key[clip] = {"job": job, "sentence": s, "exit": p.returncode}
        result.append({"clip": clip, "sentence": s, "exit": p.returncode, "segments": segs,
                       "semantics": st, "seconds": round(dt, 1), "has_clip": has_clip})
        print("  %-10s 出口码 %d｜%d 段｜%s 秒｜%s" % (clip, p.returncode, segs, round(dt, 1),
              "有片" if has_clip else "无片"))
    with (out / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "sentence", "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    (out / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in result if r["has_clip"])
    blocked = sum(1 for r in result if r["exit"] == 2)
    print()
    print("  句子 %d｜出片 %d｜被拦住 %d｜包在 %s" % (len(result), ok, blocked, out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
