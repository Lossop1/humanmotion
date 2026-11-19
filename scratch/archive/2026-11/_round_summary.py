"""汇总一轮评测：每句的段数、失败段、几何、交界差、帧数、耗时。"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path("/home/chuan/motion-work/outputs")
SENTS = {
    "s1": "一个人向前走", "s2": "一个人向左跑，然后急停",
    "s3": "一个人先伸个懒腰，再弯腰，最后站起来", "s4": "一个人一边走一边挥手",
    "s5": "一个人向前走几步，然后坐下", "s6": "一个人拿起桌上的水杯喝水",
    "s7": "一个人做一套动作", "s8": "一个人向前走，停下，举手，跳一下，转身，最后走开",
    "s9": "一个人快速跑过去，然后慢慢走回来", "s10": "一个人走向桌子，然后停下来",
}


def summarize(job):
    jd = ROOT / job
    sb_p = jd / "storyboard.json"
    bj_p = jd / "assembled" / "bounds.json"
    if not (sb_p.exists() and bj_p.exists()):
        return None
    sb = json.loads(sb_p.read_text(encoding="utf-8"))
    bj = json.loads(bj_p.read_text(encoding="utf-8"))
    total = len(sb["segments"])
    used = len(bj["segments"])
    gaps = [j["pose_gap_deg"] for j in bj.get("junctions", [])]
    log = (jd / "run.log").read_text(encoding="utf-8", errors="replace")
    dropped = [l.split("取舍：")[1].split(" 没有")[0] for l in log.splitlines() if "取舍：" in l]
    geom = "通过" if bj.get("assembled_ok") else "不通过"
    frames = bj.get("total_frames")
    return {"job": job, "total": total, "used": used, "dropped": dropped,
            "geom": geom, "frames": frames,
            "gap_med": sorted(gaps)[len(gaps) // 2] if gaps else None,
            "gap_max": max(gaps) if gaps else None}


print("  句  段数(规划/装配)  去掉的段        几何    交界差(中位/最大)  帧数")
rows = []
for sid in sorted(SENTS, key=lambda s: int(s[1:])):
    job = "eval_%s" % sid
    r = summarize(job)
    if not r:
        print("  %-4s 无产物" % sid)
        continue
    rows.append(r)
    print("  %-4s %d/%d %-14s %-6s %-16s %s"
          % (sid, r["total"], r["used"], ",".join(r["dropped"]) or "-", r["geom"],
             ("%.0f / %.0f" % (r["gap_med"], r["gap_max"])) if r["gap_med"] is not None else "-",
             r["frames"]))

ok = sum(1 for r in rows if r["geom"] == "通过")
print()
print("  几何通过 %d/%d 句；共去掉 %d 段；交界差中位 %s"
      % (ok, len(rows), sum(len(r["dropped"]) for r in rows),
         ("%.0f 度" % sorted([r["gap_med"] for r in rows if r["gap_med"] is not None])[
             len([r for r in rows if r["gap_med"] is not None]) // 2]) if rows else "-"))
