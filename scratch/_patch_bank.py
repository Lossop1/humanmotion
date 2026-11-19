"""把提示词库接进 A06：planner 的措辞如果不如库里测过的写法稳，就用库里的。

实测依据（momask，每句 4 个样本）
    punch   "throws a punch" 0/4      "punches forward" 4/4
    kick    "does a front kick" 1/4   "kicks forward" 3/4
    sit     三种写法都 0/4 —— 这个动作本身在这个后端上不可用，库也救不了
"""
import ast
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"D:\my-projects\humanmotion\repo\hm\run_job.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")

if "prompt_bank" in s:
    print("  A06 已接提示词库")
else:
    old = '''    job.log("A06 提示词：%d/%d 段沿用能力表测过的措辞" % (reused, len(sb["segments"])))
    return sb'''
    new = '''    # 第二道：提示词库。planner 的措辞不如库里测过的写法稳时，用库里的。
    try:
        import json as _json
        from pathlib import Path as _P
        _bp = _P(__file__).resolve().parent.parent / "outputs" / "_reference" / "prompt_bank.json"
        bank = _json.loads(_bp.read_text(encoding="utf-8")) if _bp.exists() else {}
    except Exception:
        bank = {}
    from_bank = 0
    if bank:
        _stop = {"a", "the", "person", "with", "and", "in", "on", "to", "of", "up",
                 "down", "forward", "backward", "left", "right", "once", "his", "her"}
        def _toks(t):
            return {w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in _stop and len(w) > 2}
        for seg in sb["segments"]:
            cur = seg.get("text_gen_final") or ""
            cur_toks = _toks(cur)
            best_item, best_rate = None, 0.0
            for action, entry in bank.items():
                if entry.get("best_rate", 0) < 0.75:
                    continue
                keys = _toks(action.replace("_", " ")) | _toks(entry.get("best", ""))
                if not (keys & cur_toks):
                    continue
                if entry["best_rate"] > best_rate:
                    best_item, best_rate = entry["best"], entry["best_rate"]
            if best_item and best_item.strip().lower() != cur.strip().lower():
                seg["text_gen_final"] = best_item
                seg["template_version"] = "prompt_bank"
                from_bank += 1
                job.log("A06 %s 改用提示词库最优写法：%r（planner 原为 %r）"
                        % (seg["segment_id"], best_item, cur))
    job.log("A06 提示词：%d/%d 段沿用能力表测过的措辞，%d/%d 段改用提示词库最优写法"
            % (reused, len(sb["segments"]), from_bank, len(sb["segments"])))
    return sb'''
    assert old in s, "未匹配 A06 结尾"
    s = s.replace(old, new, 1)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    ast.parse(s)
    print("  A06：接上提示词库")
