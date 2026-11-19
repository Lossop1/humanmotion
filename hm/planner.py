"""语言处理的唯一入口：一句中文 → 段落、英文提示词、帧数、过渡、地形。

设计约束
--------
不存在任何「动作名 → 英文提示词」的映射表，也没有动作白名单。
分段、翻译、帧数、过渡时长、地形，全部由 LLM 一次给出。
代码只做数值上的物理约束，不做语言判断。
"""
import json
import os
import re
import time
import urllib.request

from hm import config
from hm.jsonio import load_json

API = "https://api.deepseek.com/chat/completions"


def _capability_block():
    """把「生成模型实测能做什么」写成给大模型看的一段说明。

    为什么放在提示里
        文学描述里的动词（撺将下来、翻身闪避）不在模型词表里，直接交给生成模型只会得到泛化动作。
        与其我在后面加一层硬映射，不如把能力清单交给大模型，让它自己选最接近的可用动作，
        并在 label 里标明「近似：…」。
    """
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parent.parent / "outputs" / "_reference" / "prompt_bank.json"
    # 读不到或损坏时返回空清单：规划仍可进行，但会少了「可用动作」约束，所以损坏时要出声
    bank = load_json(p, want=dict)
    if bank is None:
        return ""
    good, bad = [], []
    for key, e in bank.items():
        rate = e.get("best_rate", 0)
        zh = e.get("action_zh") or key
        # 推荐写法里避开复合句（含 and），否则大模型会照抄成一句话包含两个动作
        cands = [r.get("text", "") for r in e.get("phrasings", [])
                 if r.get("text") and " and " not in r["text"]]
        best = (cands[0] if cands else (e.get("best") or "")).strip()
        if rate >= 0.75:
            good.append("%s → %s" % (zh, best))
        elif rate <= 0.0:
            bad.append(zh)
    if not good:
        return ""
    lines = ["## 生成模型实测能做什么（必须照此选词）", "",
             "可用动作（中文 → 推荐英文写法，均已实测通过）："]
    for i in range(0, len(good), 3):
        lines.append("  " + "｜".join(good[i:i + 3]))
    if bad:
        lines += ["", "不可用（实测 0 通过，不要使用）：" + "、".join(bad)]
    lines += ["",
              "选词规则：",
              "1 每段 prompt 都要落到上面「可用动作」的写法上，可再加方向与快慢。",
              "2 原文动词不在可用列表里时（例如「撺将下来」「翻身闪避」），",
              "  选一个最接近的可用动作，并在 label 前面加「近似：」。",
              "3 兽类动作、物体操作、情绪状态无法表现时，仍给一段最接近的人体动作，",
              "  同样在 label 里写「近似：」并说明原文说的是什么。",
              "4 source 仍然必须逐字摘录原文，不许因为改写成近似动作就改动 source。",
              "5 prompt 必须逐字取自上面「可用动作」的英文写法，只允许再补方向与快慢；",
              "  不要在列表写法之外自行添加名词、对象或姿势（例如 dough、chair、ledge、",
              "  with effort 这类），实测这些都会让生成失败。",
              "6 原文里的对象（面团、椅子、虎）不写进 prompt；它们只出现在 source 与 label 里。"]
    return "\n".join(lines)


SYSTEM = """你是人体动作分镜器。给你一句中文描述，输出一段可以直接交给动作生成模型的英文分镜。

只输出 JSON，不要解释。

## 第一部分：段落
把句子拆成按时间排列的动作段。
1 一句话有几个先后发生的动作就拆几段，一段只描述一个连续动作。
2 如果两个动作同时发生（例如「一边走一边挥手」），合并成一段，不要拆。
3 每段给这些字段：
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

## prompt 的写法
生成模型只懂英文，而且它看不到整句，只看到你给的这一句。所以每一条都要自足：
1 一律以 "a person" 或 "the person" 开头。
2 必须交代方向：forward / backward / to the left / to the right / in place /
  in a circle。不写方向的运动类动作，模型经常原地不动或自己乱拐。
3 时态整段一致，一般现在时或现在进行时都可以。
4 一条只写一个动作，不要写 "and then"。
5 控制在 12 个词以内。
6 用常见动词，例如 walk, run, jump, turn, sit, stand, wave, bend, pick, kick,
    dance, climb, crawl, throw, punch, kneel, crouch, point, salute, shrug,
    push, pull, squat, hop, sprint, sidestep。以上只是示例，其他常见动作词同样可用；
    生僻词模型没见过，表现不可控。
7 例子：
   "a person waves goodbye with the right hand"
   "a person walks forward slowly"
   "a person turns around in place"
   "a person runs forward fast"

## frames
20 帧 = 1 秒。参考范围：
  站立、挥手、点头        30–45
  转身、弯腰、蹲下        30–45
  走、跑、爬              45–70
  坐下、躺下、起身        45–60
动作越复杂给得越长，但不超过 80。

## 第二部分：过渡
每两个相邻段落之间指定一段过渡，两个字段：
   duration_s  过渡占多少秒，0.15–1.20
   ease        取 linear / ease_in_out / ease_out 之一
               两侧都慢用 linear；一快一慢用 ease_in_out；收尾要缓用 ease_out
参考：快接快 0.2–0.3 秒；慢接慢 0.5–0.8 秒；快接慢 0.4–0.6 秒。

## 第三部分：地形
整句描述里有高度变化（上台阶、下台阶、上坡、下坡、从高处跳下）才给地面高度，
否则一律为 0。
   start_y   第一段开始时地面高度（米）。下楼从高处开始就给正数。
   profile   每段开始时的地面高度，格式 [{"at_segment": 1, "y": 0.0}, ...]
原句没提地形就不要凭空造。

## 输出格式
{
  "segments": [
    {"label": "挥手打招呼", "prompt": "a person waves goodbye with the right hand", "frames": 40,
     "source": "挥手打招呼", "direction": "forward", "object": null, "relation": null, "speed": "normal"},
    {"label": "转身", "prompt": "a person turns around in place", "frames": 40,
     "source": "转身", "direction": "in_place", "object": null, "relation": null, "speed": "normal"}
  ],
  "transitions": [
    {"junction": 1, "duration_s": 0.4, "ease": "ease_in_out"}
  ],
  "terrain": {"start_y": 0.0, "profile": [{"at_segment": 1, "y": 0.0}, {"at_segment": 2, "y": 0.0}]}
}
"""


def _key():
    """优先读环境变量 DEEPSEEK_API_KEY；密钥文件只作旧版兜底并给出警告。"""
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if k:
        return k
    for p in (config.ROOT / ".deepseek_key", config.DEPS / ".deepseek_key"):
        if p.exists():
            t = p.read_text(encoding="utf-8").strip()
            if t:
                import warnings
                warnings.warn(f"从明文文件 {p} 读取 API key，建议改用环境变量 DEEPSEEK_API_KEY",
                              stacklevel=2)
                return t
    return ""


def _build_system():
    """新版系统提示（XML 分块 + 示例 + 自检），见 hm/planner_prompt.py。"""
    from hm import planner_prompt
    return planner_prompt.build(_capability_block())


def _fill_capability(system):
    """把能力清单填进系统提示。"""
    return system.replace("{CAPABILITY}", _capability_block())


def _call(system, user, tries=3):
    key = _key()
    if not key:
        raise RuntimeError("没有 API key：请设置环境变量 DEEPSEEK_API_KEY（旧版 .deepseek_key 文件也可，但不推荐）")
    body = json.dumps({
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }).encode("utf-8")
    last = None
    for k in range(tries):
        req = urllib.request.Request(API, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
            if txt and txt.strip():
                return txt
            last = "空回复"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (k + 1))
    raise RuntimeError("LLM 调用失败：%s" % last)


def _parse(txt):
    """从回复里取出 JSON，允许 ```json 围栏。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    if m:
        return json.loads(m.group(1))
    i, j = txt.find("{"), txt.rfind("}")
    if i >= 0 and j > i:
        return json.loads(txt[i:j + 1])
    raise ValueError("回复里没有 JSON：%s" % txt[:200])


def _enum(v, allowed):
    """只接受枚举内的值，其余一律 null。"""
    t = str(v).strip().lower() if v is not None else ""
    return t if t in allowed else None


def plan(sentence):
    """一句中文 → (段落列表, 过渡列表, 地形)。

    段落每项 {label, prompt, frames, source, direction, object, relation, speed}。
    同一句话只调用一次 LLM：结果按句子缓存，改下游时不必重问，也不会有两份措辞。
    """
    # 强制走兜底（HM_PLAN_FALLBACK=1）：用来量「没有大模型时指标退化多少」
    import os as _os_fb
    if _os_fb.environ.get("HM_PLAN_FALLBACK", "0") == "1":
        from hm import planner_fallback as _fb
        _segs, _trans, _terrain = _fb.plan(sentence)
        if _segs:
            return _segs, _trans, _terrain
        raise RuntimeError("兜底规划认不出这句话：%s" % sentence)

    from hm import cache as _cache
    ck = _cache.key(("plan", sentence))
    cached = _cache.get_json("plan", ck) if _cache.enabled() else None
    if cached is not None:
        return cached["segs"], cached["trans"], cached["terrain"]
    _sys = _build_system()
    try:
        data = _parse(_call(_sys, "<text>%s</text>\n\n按要求输出 JSON。" % sentence))
    except RuntimeError as e:
        # 大模型不可用（欠费/断网/限流/没配 key）时退到确定性兜底：只用中文动作词表切句。
        # 认不出就照旧报错，不假装成功；兜底结果不写入缓存。
        from hm import planner_fallback
        _segs, _trans, _terrain = planner_fallback.plan(sentence)
        if _segs:
            print("  警告：大模型不可用（%s），改用规则兜底规划" % str(e)[:60])
            return _segs, _trans, _terrain
        raise

    segs = []
    for s in data.get("segments") or []:
        prompt = str(s.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("有一个段落没有英文提示词：%s" % s)
        frames = int(s.get("frames") or 40)
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
        })
    if not segs:
        raise ValueError("LLM 没有给出任何段落")

    trans = []
    for t in data.get("transitions") or []:
        try:
            j = int(t.get("junction"))
        except (TypeError, ValueError):
            continue
        try:
            d = float(t.get("duration_s", 0.4))
        except (TypeError, ValueError):
            d = 0.4
        ease = str(t.get("ease", "ease_in_out"))
        if ease not in ("linear", "ease_in_out", "ease_out"):
            ease = "ease_in_out"
        trans.append({"junction": j, "duration_s": max(0.15, min(1.20, d)), "ease": ease})

    terrain = data.get("terrain") or {"start_y": 0.0, "profile": []}
    if not isinstance(terrain, dict):
        # 大模型有时把地形写成 "flat" 这样的字符串；容错成「平地」
        terrain = {"start_y": 0.0, "profile": []}
    if not isinstance(terrain.get("profile"), list):
        terrain["profile"] = []
    if _cache.enabled():
        _cache.put_json("plan", ck, {"segs": segs, "trans": trans, "terrain": terrain})
    return segs, trans, terrain


TERRAIN_SYSTEM = """你在判断一段动画里地面是什么形状。

你会拿到两部分信息：
 1 用户的原句，以及从原句拆出来的动作段
 2 模型实际生成出来的每段动作的竖直剖面

第二部分给每段的：根高起点、根高终点、净变化、振荡幅度、水平路径长度，
以及一个判断。判断为「单调升降」表示这段在持续地升高或降低，像在走楼梯或坡道。
判断为「上下振荡」表示这段在原地起伏、净位移接近零，像跑跳的正常弹跳。

你要决定每段**地面高度**，让动作和世界一致。规则：

如果一段是「上下振荡」，地面高度不变。那是原地弹跳，不是地形。
如果一段是「单调升降」，有两种处理，你必须选一个并说明理由：
  A 承认它是地形。地面高度跟着变化，人从高处开始往下走。
     条件：原句里有台阶、楼梯、坡、上下楼这些词。
  B 判定它是模型幻觉。地面高度不变，把升降压平。
     条件：原句描述的是平地上的简单动作，模型却生成了大幅升降。

地面高度的变化量不能超过「水平路径长度 × 0.6」，否则渲出来是滑梯不是楼梯。
每段的这个上限都会告诉你。

只输出 JSON：
{"terrain":{"start_y":0.0,"profile":[{"at_segment":1,"y":0.0}]},
 "notes":[{"segment":1,"decision":"A|B|keep","why":"一句话理由"}]}

start_y 是第一段开始时的地面高度（米）。
profile 里 {"at_segment": k, "y": h} 的意思是「段 k 开始那一刻地面高度是 h」，
段内地面从本段的值线性走到下一段的值。所以下降要写在【发生下降的那一段】的起点上。
"""


def terrain_decision(sentence, labels, profiles):
    """第二次调用：把实测竖直剖面交给 LLM，决定地面高度。"""
    lines = []
    for p in profiles:
        kind = "单调升降（像地形）" if p.get("monotone") else "上下振荡（像弹跳）"
        lines.append(
            "段%d %s：根高 %.3f → %.3f 米，净变化 %+.3f 米，振荡 %.3f 米，判为%s；"
            "水平路径 %.2f 米，最多只能落差 %.2f 米" % (
                p["segment"], p["action"], p["start_y"], p["end_y"],
                p["net"], p["osc"], kind, p.get("horiz_m", 0.0), p.get("max_drop", 0.0)))
    user = "原句：%s\n\n动作段：%s\n\n模型生成的竖直剖面：\n%s\n\n请输出 JSON。" % (
        sentence, ", ".join(labels), "\n".join(lines))
    data = _parse(_call(TERRAIN_SYSTEM, user))
    terr = data.get("terrain") or {"start_y": 0.0, "profile": []}
    notes = data.get("notes") or []
    if not isinstance(terr.get("profile"), list):
        terr["profile"] = []
    return terr, notes


# main 的旧接口
def plan_full(sentence):
    return plan(sentence)
