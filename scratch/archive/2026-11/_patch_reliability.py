# -*- coding: utf-8 -*-
"""两处可靠性修复：
   1 A03 改为查能力表（capability_v4.json），硬编码禁用词只作回退
   2 某段没有通过判据的候选时降级继续，不再整条中止
"""
import ast
import pathlib
import sys

BASE = pathlib.Path(r"D:\my-projects\humanmotion\_hmfetch")

# ---------- 1. run_job.py: A03 查能力表 ----------
p = BASE / "run_job.py"
s = p.read_text(encoding="utf-8")
if "_cap_lookup" in s:
    print("  run_job 已有 _cap_lookup")
else:
    helper = '''
_CAP = {}


def _cap_lookup(prompt):
    """按提示词查能力表。返回 (status, model, 中文名) 或 None。"""
    import json
    from pathlib import Path
    if not _CAP:
        p = Path(__file__).resolve().parent.parent / "outputs" / "_reference" / "capability_v4.json"
        _CAP["doc"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    doc = _CAP["doc"]
    if not doc:
        return None
    text = str(prompt).lower()
    for key, v in doc["table"].items():
        words = [w for w in v["prompt"].lower().split()[:3] if len(w) > 2]
        if key.replace("_", " ") in text or v["action_zh"] in str(prompt) or any(w in text for w in words):
            models = v.get("models", {})
            if not models:
                return None
            best = v.get("best") or max(models, key=lambda m: models[m].get("pass_rate", 0))
            st = models[best].get("status", "unverified")
            st = {"support": "supported", "supported": "supported"}.get(st, st)
            return st, best, v.get("action_zh")
    return None


'''
    anchor = "def step_a01(job, cfg):"
    assert anchor in s
    s = s.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)

    old = '''        low = (seg.get("text_gen") or "").lower()
        if any(w in low for w in UNSUPPORTED_WORDS):
            status, reason = "unsupported", "命中禁用词表（实测生成质量不可靠）"
        else:
            status, reason = "unverified", "尚无该动作类别的能力测试数据"'''
    new = '''        prompt = seg.get("text_gen") or ""
        low = prompt.lower()
        hit = _cap_lookup(prompt)
        if hit is not None:
            status, model, zh = hit
            reason = "能力表：%s 最佳后端 %s" % (zh or "-", model)
        elif any(w in low for w in UNSUPPORTED_WORDS):
            status, reason = "unsupported", "命中禁用词表（实测生成质量不可靠）"
        else:
            status, reason = "unverified", "尚无该动作类别的能力测试数据"'''
    assert old in s, "未匹配 A03 判定段"
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    ast.parse(s)
    print("  run_job：A03 改为查能力表")

# ---------- 2. finish_job.py: 无候选时降级 ----------
q = BASE / "finish_job.py"
t = q.read_text(encoding="utf-8")
if "降级" in t:
    print("  finish_job 已有降级")
else:
    old2 = '''    segs = segment_candidates(job, sb)
    if any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        raise SystemExit("这些段没有可用候选，无法装配：%s" % bad)'''
    new2 = '''    segs = segment_candidates(job, sb)
    if any(not c for _, c in segs):
        # 降级：没有通过判据的段，改用问题项最少的候选，并显式记录
        import json as _j
        relaxed = []
        for s, c in segs:
            if c:
                relaxed.append((s, c))
                continue
            best_n, best_k = None, 10 ** 6
            for cd in s.get("candidates", []):
                n = int(cd["candidate_id"].split("_")[-1])
                mp = job.dir / "segments" / s["segment_id"] / ("candidate_%d" % n) / "metrics.json"
                try:
                    mm = _j.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                k = len(mm.get("failure_reasons") or [])
                if k < best_k:
                    best_n, best_k = n, k
            if best_n is None:
                relaxed.append((s, []))
            else:
                job.log("降级：%s 无通过判据的候选，改用 candidate_%d（%d 项问题）"
                        % (s["segment_id"], best_n, best_k))
                relaxed.append((s, [best_n]))
        segs = relaxed
    if any(not c for _, c in segs):
        bad = [s["segment_id"] for s, c in segs if not c]
        raise SystemExit("这些段没有任何候选，无法装配：%s" % bad)'''
    assert old2 in t, "未匹配无候选分支"
    t = t.replace(old2, new2, 1)
    q.write_text(t, encoding="utf-8")
    ast.parse(t)
    print("  finish_job：无候选时降级继续")
