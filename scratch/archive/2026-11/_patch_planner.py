# -*- coding: utf-8 -*-
"""P1 规划改造：让 planner 给出原文片段与空间字段，A02 直接采用。

改动
  1 planner SYSTEM：每段多出 source（逐字摘录原文）、direction、object、relation、speed
  2 planner.plan：解析并返回这些字段，source 必须是原句子串，否则置空
  3 run_job.step_a02：用 source 填 source_text，并填 facing/object/relation/speed
     另加覆盖率检查：各段 source 拼接占原句的比例，过低则告警
"""
import ast
import pathlib

BASE = pathlib.Path(r"D:\my-projects\humanmotion\repo\hm")

# ---------------- planner.py ----------------
p = BASE / "planner.py"
s = p.read_text(encoding="utf-8")

if "source    " in s or "## 原文片段" in s:
    print("  planner 已改")
else:
    old_fields = '''3 每段给三个字段：
   label   中文动作名，只用于日志，可以自由写
   prompt  英文提示词，直接交给生成模型
   frames  这段占多少帧（20 帧 = 1 秒）
'''
    new_fields = '''3 每段给这些字段：
   label      中文动作名，只用于日志，可以自由写
   prompt     英文提示词，直接交给生成模型
   frames     这段占多少帧（20 帧 = 1 秒）
   source     原句里对应的那几个字，**必须逐字摘录，不能改写、不能翻译**
              各段的 source 依次拼起来要覆盖原句的主要部分
   direction  这段的运动方向，取 forward / backward / left / right / in_place / circle / null
   object     动作涉及的对象名词（椅子、桌子、杯子、门……），没有就 null
   relation   与对象的关系，取 near / front / behind / left / right / toward / away / null
   speed      slow / normal / fast / null

## 原文片段（source）的规则
这是最容易做错的一条：
1 必须逐字来自原句，标点也要一致。做不到逐字，就留空字符串 ""。
2 不许出现原句里没有的字，不许改写、不许概括、不许翻译。
3 各段 source 按顺序拼接，应当接近原句全文；漏掉的部分就是你没规划到的部分。
4 如果整句只描述一个含糊的动作（例如「做一套动作」），只输出一段，source 就是整句。
'''
    assert old_fields in s
    s = s.replace(old_fields, new_fields, 1)

    old_ex = '''  "segments": [
    {"label": "挥手打招呼", "prompt": "a person waves goodbye with the right hand", "frames": 40},
    {"label": "转身", "prompt": "a person turns around in place", "frames": 40}
  ],'''
    new_ex = '''  "segments": [
    {"label": "挥手打招呼", "prompt": "a person waves goodbye with the right hand", "frames": 40,
     "source": "挥手打招呼", "direction": "forward", "object": null, "relation": null, "speed": "normal"},
    {"label": "转身", "prompt": "a person turns around in place", "frames": 40,
     "source": "转身", "direction": "in_place", "object": null, "relation": null, "speed": "normal"}
  ],'''
    assert old_ex in s
    s = s.replace(old_ex, new_ex, 1)

    old_parse = '''        frames = int(s.get("frames") or 40)
        segs.append({
            "label": str(s.get("label", "")).strip() or prompt,
            "prompt": prompt,
            "frames": max(20, min(90, frames)),
        })'''
    new_parse = '''        frames = int(s.get("frames") or 40)
        src = str(s.get("source") or "").strip()
        if src and src not in sentence:
            src = ""            # 不是原句子串就作废，不假装对上了
        segs.append({
            "label": str(s.get("label", "")).strip() or prompt,
            "prompt": prompt,
            "frames": max(20, min(90, frames)),
            "source": src,
            "direction": _enum(s.get("direction"),
                               ("forward", "backward", "left", "right", "in_place", "circle")),
            "object": (str(s.get("object")).strip() or None) if s.get("object") else None,
            "relation": _enum(s.get("relation"),
                              ("near", "front", "behind", "left", "right", "toward", "away")),
            "speed": _enum(s.get("speed"), ("slow", "normal", "fast")),
        })'''
    assert old_parse in s
    s = s.replace(old_parse, new_parse, 1)

    helper = '''

def _enum(v, allowed):
    """只接受枚举内的值，其余一律 null。"""
    t = str(v).strip().lower() if v is not None else ""
    return t if t in allowed else None

'''
    anchor = "def plan(sentence):"
    s = s.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)
    p.write_text(s, encoding="utf-8")
    ast.parse(s)
    print("  planner：已加 source/direction/object/relation/speed")

# ---------------- run_job.py ----------------
q = BASE / "run_job.py"
t = q.read_text(encoding="utf-8")
if "coverage" in t:
    print("  run_job 已改")
else:
    old = '''    for i, s in enumerate(segs, 1):
        seg = sb_schema.blank_segment("seg_%03d" % i, i, s["label"], sentence)
        seg["duration_frames"] = int(s["frames"])
        seg["text_gen"] = s["prompt"]
        seg["provenance"] = {
            "action": "planner_llm(label)",
            "source_text": "fallback:whole_sentence",
            "duration_frames": "planner_llm",
            "text_gen": "planner_llm",
        }
        warnings.append("seg_%03d 的 source_text 暂用整句（planner 未切分原文）" % i)
        out.append(seg)'''
    new = '''    covered = 0
    for i, s in enumerate(segs, 1):
        src = s.get("source") or ""
        seg = sb_schema.blank_segment("seg_%03d" % i, i, s["label"], src or sentence)
        seg["duration_frames"] = int(s["frames"])
        seg["text_gen"] = s["prompt"]
        seg["facing"] = s.get("direction")
        seg["object"] = s.get("object")
        seg["relation"] = s.get("relation")
        seg["speed"] = s.get("speed")
        seg["provenance"] = {
            "action": "planner_llm(label)",
            "source_text": "planner_llm(substring)" if src else "fallback:whole_sentence",
            "duration_frames": "planner_llm",
            "text_gen": "planner_llm",
            "facing": "planner_llm" if s.get("direction") else "null",
        }
        if src:
            covered += len(src)
        else:
            warnings.append("seg_%03d 未给出原文片段，source_text 退回整句" % i)
        out.append(seg)
    coverage = covered / max(len(sentence), 1)
    if coverage < 0.6:
        warnings.append("原文覆盖率 %.0f%%（各段 source 拼接占原句比例），可能有内容未规划"
                        % (100 * coverage))
    job.log("A02 原文覆盖率 %.0f%%（%d 段）" % (100 * coverage, len(out)))'''
    assert old in t
    t = t.replace(old, new, 1)
    q.write_text(t, encoding="utf-8")
    ast.parse(t)
    print("  run_job：A02 使用原文片段与空间字段，并统计覆盖率")
