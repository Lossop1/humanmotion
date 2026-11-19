"""能力表到底有没有预测力：拿真实跑出来的段去核对。

做法
    对每个历史任务的每一段，取它的提示词与最终结果：
        表里的说法  support / approx / unverified
        实际结果    该段的候选里有没有通过几何判据的
    做一张列联表。如果表说 support 却大量不过，这张表不能作为路由依据，
    应当降级为「参考」，并回到用户说的「直接生成看看」。
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/chuan/motion-work")
from hm import backends

ROOT = Path("/home/chuan/motion-work/outputs")


def outcome(job, seg):
    """该段：有几个候选通过几何判据。"""
    cands = seg.get("candidate_metrics") or []
    if not cands:
        return None
    return sum(1 for c in cands if c.get("decision") == "pass")


rows = []
for p in sorted(glob.glob(str(ROOT / "*" / "storyboard.json"))):
    job = Path(p).parent
    try:
        sb = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        continue
    for seg in sb.get("segments", []):
        prompt = seg.get("text_gen_final") or seg.get("text_gen") or ""
        if not prompt:
            continue
        _key, entry = backends.match_entry(backends.load_table(), prompt)
        status = "no_entry"
        rate = None
        if entry:
            model = (seg.get("route") or {}).get("model") or backends.DEFAULT
            m = (entry.get("models") or {}).get(model) or {}
            status = {"support": "support"}.get(m.get("status"), m.get("status")) or "no_entry"
            rate = m.get("pass_rate")
            if entry.get("prompt_scope") == "composite":
                status = "composite"
        n_ok = outcome(job, seg)
        if n_ok is None:
            continue
        rows.append({"job": job.name, "seg": seg["segment_id"], "prompt": prompt,
                     "status": status, "rate": rate, "ok": n_ok > 0, "n_ok": n_ok})

tally = {}
for r in rows:
    tally.setdefault(r["status"], [0, 0])
    tally[r["status"]][1] += 1
    tally[r["status"]][0] += 1 if r["ok"] else 0

print("  能力表说法 → 实际有没有通过几何的候选（%d 段）" % len(rows))
print("  %-12s %6s %6s %7s" % ("表的说法", "段数", "有通过", "比例"))
for k, (good, tot) in sorted(tally.items()):
    print("  %-12s %6d %6d %6.0f%%" % (k, tot, good, 100 * good / max(tot, 1)))

print()
print("  表说 support 但实际全不过的段：")
bad = [r for r in rows if r["status"] == "support" and not r["ok"]]
for r in bad[:10]:
    print("    %-12s %s  %r" % (r["job"], r["seg"], r["prompt"][:52]))
print("  合计 %d 段" % len(bad))
