"""交叉验证：十条全新句子的端到端跑批。

句子刻意覆盖不同维度：单动作、并行、数量、方向、对象、含糊、长句、速度修饰。
只记录客观量（段数、帧数、耗时、几何判据、是否中止），不预设结论。
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/chuan/motion-work")
OUT = ROOT / "outputs" / "_eval_new"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    ("S1", "一个人向前走"),
    ("S2", "一个人向左跑，然后急停"),
    ("S3", "一个人先伸个懒腰，再弯腰，最后站起来"),
    ("S4", "一个人一边走一边挥手"),
    ("S5", "一个人向前走几步，然后坐下"),
    ("S6", "一个人拿起桌上的水杯喝水"),
    ("S7", "一个人做一套动作"),
    ("S8", "一个人向前走，停下，举手，跳一下，转身，最后走开"),
    ("S9", "一个人快速跑过去，然后慢慢走回来"),
    ("S10", "一个人走向桌子，然后停下来"),
]


def run(sid, sentence):
    job = "eval_%s" % sid.lower()
    jd = ROOT / "outputs" / job
    if jd.exists():
        shutil.rmtree(jd)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "hm.run_full", sentence, "--k", "6",
                        "--job-id", job], cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    rec = {"id": sid, "sentence": sentence, "job": job, "elapsed_s": round(dt, 1),
           "ok": False, "segments": None, "frames": None, "geometry": None,
           "aborted": None}
    log = (jd / "run.log")
    if log.exists():
        txt = log.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if "A02 分镜" in line:
                rec["segments"] = line.split("A02 分镜")[-1].strip()
            if "A16 参数空间装配" in line:
                rec["geometry"] = line.split("A16 参数空间装配")[-1].strip()
        if "没有可用候选" in txt:
            rec["aborted"] = [l.strip() for l in txt.splitlines() if "没有可用候选" in l][-1]
    mp4 = jd / "render" / "final.mp4"
    if mp4.exists():
        shutil.copyfile(mp4, OUT / ("%s.mp4" % sid))
        rec["ok"] = True
    rec["tail"] = (p.stdout or "")[-300:]
    rec["err"] = (p.stderr or "")[-300:]
    return rec


def main():
    rows = []
    for sid, sent in SENTENCES:
        print("  跑 %s：%s" % (sid, sent), flush=True)
        r = run(sid, sent)
        print("     %s  耗时 %.1fs  段数 %s" % ("成功" if r["ok"] else "失败",
                                              r["elapsed_s"], r["segments"]), flush=True)
        rows.append(r)
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    ok = sum(1 for r in rows if r["ok"])
    print("\n  成功 %d/%d，结果写入 %s" % (ok, len(rows), OUT / "summary.json"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
