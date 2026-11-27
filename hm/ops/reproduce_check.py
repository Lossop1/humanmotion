"""复现性检查：同一条句子跑两遍，看产物与指标是否一致。

为什么需要
    「可复现」是答辩会问的：同一份代码、同一句输入，两次结果一样吗？
    缓存命中的「一样」不算复现（那是同一份数据读两次），所以这里用 HM_NO_CACHE=1 跑。

用法
    python -m hm.ops.reproduce_check --sentence "一个人向前走，然后转身" --twice 2
输出
    两次的帧数、交界、选中候选、质量指标，逐项标「一致 / 不同」。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run_once(sentence, job_id):
    env_cmd = ("HM_NO_CACHE=1 python -m hm.main %s --job-id %s --k 3"
               % (json.dumps(sentence, ensure_ascii=False), job_id))
    p = subprocess.run(["bash", "-lc", env_cmd], capture_output=True, text=True, cwd=str(ROOT))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def summary(job_id):
    d = ROOT / "outputs" / job_id
    out = {"job": job_id, "exists": d.exists()}
    if not out["exists"]:
        return out
    bj_p = d / "assembled" / "bounds.json"
    if bj_p.exists():
        bj = json.loads(bj_p.read_text(encoding="utf-8"))
        out["frames"] = bj.get("total_frames") or len(np.load(d / "assembled" / "joints22.npy"))
        out["junctions"] = [j.get("pose_gap_deg") for j in bj.get("junctions", [])]
        out["selected"] = [s.get("candidate") for s in bj.get("segments", [])]
        out["facing"] = [(s.get("facing") or {}).get("applied_deg") for s in bj.get("segments", [])]
    sem = d / "semantics.json"
    if sem.exists():
        out["semantics"] = [s.get("status") for s in json.loads(sem.read_text(encoding="utf-8"))]
    try:
        from hm import quality_metrics as QM
        J = np.load(d / "assembled" / "joints22.npy")
        out["ground_pen_max_m"] = QM.ground_penetration(J)["max_depth_m"]
        out["accel_ratio"] = QM.accel_ratio(J)
        out["foot_slide_mm"] = QM.foot_slide(J)["median_mm_per_frame"]
    except Exception as e:
        out["metric_error"] = str(e)[:80]
    return out


def compare(a, b):
    keys = ("frames", "junctions", "selected", "facing", "semantics",
            "ground_pen_max_m", "accel_ratio", "foot_slide_mm")
    rows = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        rows.append((k, va, vb, "一致" if va == vb else "不同"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", required=True)
    ap.add_argument("--twice", type=int, default=2)
    a = ap.parse_args()
    jobs = []
    for i in range(1, a.twice + 1):
        jid = "repro_%02d" % i
        code, log = run_once(a.sentence, jid)
        print("  第 %d 次：出口码 %d" % (i, code))
        jobs.append(summary(jid))
    print()
    print("  %-16s %-34s %-34s %s" % ("项", "第一次", "第二次", "判定"))
    diffs = 0
    for k, va, vb, verdict in compare(jobs[0], jobs[1]):
        sa = str(va)[:32]
        sb = str(vb)[:32]
        if verdict == "不同":
            diffs += 1
        print("  %-16s %-34s %-34s %s" % (k, sa, sb, verdict))
    print()
    print("  结论：%d 项中 %d 项不同" % (len(compare(jobs[0], jobs[1])), diffs))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
