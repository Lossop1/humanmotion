"""扩大评测：30 句，可选跑第几轮，结果按轮次存盘。

为什么要多轮
    单轮只有 11 处交界，生成又是随机的，中位在 25–67 度之间跳。
    只有同一批句子跑两轮以上，用跨轮中位数与四分位，才能判断改动是否真的有效。

用法
    python eval_big.py --round 1
    python eval_big.py --round 2
    python eval_big.py --report
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/chuan/motion-work")
OUT = ROOT / "outputs" / "_eval_big"

SENTENCES = [
    ("S01", "一个人向前走"),
    ("S02", "一个人向左跑，然后急停"),
    ("S03", "一个人先伸个懒腰，再弯腰，最后站起来"),
    ("S04", "一个人一边走一边挥手"),
    ("S05", "一个人向前走几步，然后坐下"),
    ("S06", "一个人拿起桌上的水杯喝水"),
    ("S07", "一个人做一套动作"),
    ("S08", "一个人向前走，停下，举手，跳一下，转身，最后走开"),
    ("S09", "一个人快速跑过去，然后慢慢走回来"),
    ("S10", "一个人走向桌子，然后停下来"),
    ("S11", "一个人原地跳两下"),
    ("S12", "一个人挥右手打招呼，然后向前走"),
    ("S13", "一个人蹲下，然后站起来"),
    ("S14", "一个人向后走三步，然后转身"),
    ("S15", "一个人先踢左腿，再踢右腿"),
    ("S16", "一个人弯腰捡起地上的东西"),
    ("S17", "一个人慢慢转身，然后跑开"),
    ("S18", "一个人举起双手，然后放下"),
    ("S19", "一个人侧身走两步，然后停住"),
    ("S20", "一个人出拳，然后后退"),
    ("S21", "一个人先走，再跑，最后停下"),
    ("S22", "一个人原地转一圈"),
    ("S23", "一个人向前鞠躬，然后站直"),
    ("S24", "一个人抬起右腿向前踢"),
    ("S25", "一个人走到左边，然后向右看"),
    ("S26", "一个人跳起来转身落地"),
    ("S27", "一个人原地踏步"),
    ("S28", "一个人双手向前推，然后收回"),
    ("S29", "一个人向后转身走开"),
    ("S30", "一个人先举手再放下，最后走开"),
]


FRESH_FILE = ROOT / "eval_fresh.txt"


def sentences(setname):
    if setname == "fresh":
        lines = [ln.strip() for ln in FRESH_FILE.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        return [("F%02d" % i, t) for i, t in enumerate(lines, 1)]
    return SENTENCES


def run_round(rnd, setname="std"):
    rows = []
    for sid, sent in sentences(setname):
        job = "%s_%s_r%d" % ("big" if setname == "std" else "frs", sid.lower(), rnd)
        jd = ROOT / "outputs" / job
        if jd.exists():
            shutil.rmtree(jd)
        t0 = time.time()
        pr = subprocess.run([sys.executable, "-m", "hm.run_full", sent, "--job-id", job],
                            cwd=str(ROOT), capture_output=True, text=True)
        # 子进程输出留档，失败时才有东西可查（上一轮 16 句失败却无日志）
        (ROOT / "outputs" / ("%s.log" % job)).write_text(
            (pr.stdout or "") + "\n--- stderr ---\n" + (pr.stderr or ""), encoding="utf-8")
        dt = round(time.time() - t0, 1)
        bj_p = jd / "assembled" / "bounds.json"
        rec = {"id": sid, "round": rnd, "sentence": sent, "elapsed": dt,
               "ok": bj_p.exists(), "segments": None, "frames": None,
               "junctions": [], "dropped": []}
        if bj_p.exists():
            bj = json.loads(bj_p.read_text(encoding="utf-8"))
            rec["segments"] = len(bj["segments"])
            rec["frames"] = bj.get("total_frames")
            rec["junctions"] = [j["pose_gap_deg"] for j in bj.get("junctions", [])]
            # 记速率比：新口径看的是「过渡速率与两侧是否一致」，不是姿态差
            rec["ratios"] = []
            for j in bj.get("junctions", []):
                g, fr = j["pose_gap_deg"], max(j["transition_frames"], 1)
                br = j.get("boundary_rate_deg_per_frame") or 0
                rec["ratios"].append(round((g / fr) / max(br, 1e-6), 2))
        log = jd / "run.log"
        if log.exists():
            for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if "取舍：" in ln:
                    rec["dropped"].append(ln.split("取舍：")[1].split(" ")[0])
        rows.append(rec)
        print("    %s 轮%d %5.1fs 段%s 帧%s 交界%s %s" % (
            sid, rnd, dt, rec["segments"], rec["frames"],
            [round(g) for g in rec["junctions"]], ",".join(rec["dropped"])), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (("round%d.json" % rnd) if setname == "std" else ("fresh%d.json" % rnd))
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in rows if r["ok"])
    print("  轮%d 完成 %d/%d，结果写入 %s" % (rnd, ok, len(rows), p))


def report():
    files = sorted(OUT.glob("round*.json"))
    if not files:
        print("  还没有任何轮次")
        return
    rounds = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    import statistics as st
    per = {}
    for rows in rounds:
        for r in rows:
            per.setdefault(r["id"], []).append(r)
    print("  句    完成      帧数(中位)  交界中位(每轮)        去掉段")
    allg = []
    for sid in [s for s, _ in SENTENCES]:
        rs = per.get(sid, [])
        if not rs:
            continue
        oks = sum(1 for r in rs if r["ok"])
        frames = st.median([r["frames"] for r in rs if r["frames"]]) if any(r["frames"] for r in rs) else None
        meds = [st.median(r["junctions"]) if r["junctions"] else None for r in rs]
        dropped = sum(len(r["dropped"]) for r in rs)
        allg += [g for r in rs for g in r["junctions"]]
        print("  %-4s %d/%d %8s  %-22s %d" % (sid, oks, len(rs), frames,
              " ".join("-" if m is None else "%.0f" % m for m in meds), dropped))
    if allg:
        allg.sort()
        print()
        print("  全部交界 %d 处：中位 %.0f 度，四分位 %.0f–%.0f"
              % (len(allg), st.median(allg), allg[len(allg) // 4], allg[3 * len(allg) // 4]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--set", default="std", choices=["std", "fresh"],
                    help="std=30 句标准集，fresh=未参与调参的 10 句")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.round:
        run_round(a.round, a.set)
