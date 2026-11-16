"""后端注册表：把「用哪个生成模型」和「它擅长什么」都变成可替换的数据。

为什么要这一层
    能力知识不能写死在代码里。接一个新后端时，必须同时回答两件事：
        1 怎么调它（入口、帧数上限、能否接随机种子、能否条件生成）
        2 它在各类动作上的成功率（能力表按 backend × action 记录）
    这里只放「注册信息」，成功率一律去 outputs/_reference/capability_*.json 查。

接一个新后端的完整步骤
    1 在 BACKENDS 里加一条：入口签名保持 generate(prompts, frames, name=...)
    2 跑一次能力测试：python -m hm.capability --models 新后端名 --k 10
    3 A05 路由会自动按新表选择，无需改代码

没有登记的后端一律拒绝，不做静默回退——回退会让「它到底用了什么」无从追溯。

不管速度。同一后端在 H100 与 CPU 上差几十倍，那是硬件的事；
路由只看「这个后端能不能做这个动作」，不看它跑多快。
能力表也非必需：用户给了就按表选，没给就直接生成，用结果说话。
"""
import json
import re
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "outputs" / "_reference"
TABLE_VERSION = "capability_v4"

BACKENDS = {
    "momask": {"kind": "text2motion", "entry": "hm.generate.generate", "max_frames": 196,
               "supports_seed": False, "supports_inpainting": False, "note": "主力后端，语义正确率最高"},
    "mdm": {"kind": "text2motion", "entry": "hm.generate.generate", "max_frames": 196,
            "supports_seed": True, "supports_inpainting": True, "note": "支持 inpainting，可用于接前段末帧的条件生成"},
    "motionlcm": {"kind": "text2motion", "entry": "hm.generate.generate", "max_frames": 196,
                  "supports_seed": True, "supports_inpainting": False, "note": "少步采样，速度优先，当前未做对照"},
}

DEFAULT = "momask"

# 匹配动作词时要忽略的词：通用主语、方向、修饰
STOP = {
    "a", "an", "the", "person", "with", "and", "in", "on", "to", "of", "at", "from",
    "forward", "backward", "left", "right", "up", "down", "around", "circle",
    "slowly", "fast", "normal", "pace", "place", "both", "his", "her", "their",
    "body", "motion", "movement", "action", "set", "some", "again", "then", "while",
}


def available():
    return sorted(BACKENDS)


def spec(name):
    if name not in BACKENDS:
        raise KeyError("未注册的后端：%s（已注册：%s）。先登记再使用，不做静默回退。"
                       % (name, ", ".join(available())))
    return BACKENDS[name]


def load_table():
    p = REF / ("%s.json" % TABLE_VERSION)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _stem(w):
    """极简词干：walks/walking/walked → walk。"""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _tokens(s):
    return {_stem(w) for w in re.findall(r"[a-z]+", str(s).lower())}


def match_entry(table, prompt):
    """按动作词匹配能力表条目，要求至少一个动作词命中；通用词不计分。"""
    if not table:
        return None, None
    toks = _tokens(prompt)
    best, best_v, best_score = None, None, 0
    for key, v in table.get("table", {}).items():
        kws = [_stem(w) for w in re.findall(r"[a-z]+", v.get("prompt", "").lower())
               if w not in STOP and len(w) > 2]
        score = sum(1 for w in set(kws) if w in toks)
        if v.get("action_zh") and str(v["action_zh"]) in str(prompt):
            score += 2
        if key.replace("_", " ") in str(prompt).lower():
            score += 2
        if score > best_score:
            best, best_v, best_score = key, v, score
    return (best, best_v) if best_score > 0 else (None, None)


def resolve(prompt, prefer=None):
    """给一段提示词挑后端。返回 (backend, status, reason, detail)。"""
    table = load_table()
    key, v = match_entry(table, prompt)
    if v is None:
        return DEFAULT, "unverified", "能力表无此动作，用默认后端", {"action": None}
    models = {m: d for m, d in v.get("models", {}).items() if m in BACKENDS}
    if not models:
        return DEFAULT, "unverified", "能力表里的后端都未注册", {"action": key}
    # 复合动作（"sits down and stands back up"）只对原句成立，不能外推到单动作
    if v.get("prompt_scope") == "composite":
        return DEFAULT, "unverified", "能力表条目是复合动作，不能外推到单段", {"action": key}
    best = max(models, key=lambda m: models[m].get("pass_rate", 0))
    if prefer and prefer in models:
        best = prefer
    status = models[best].get("status", "unverified")
    rate = models[best].get("pass_rate")
    return best, status, "能力表：%s 通过率 %.2f" % (key, rate or 0.0), {
        "action": key, "pass_rate": rate,
        "candidates": {m: models[m].get("pass_rate") for m in models},
    }
