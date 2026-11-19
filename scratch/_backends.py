"""后端注册表：把「用哪个生成模型」和「它擅长什么」都变成可替换的数据。

为什么要这一层
    能力知识不能写死在代码里。接一个新后端时，必须同时回答两件事：
        1 怎么调它（入口、帧数上限、能否接随机种子、能否条件生成）
        2 它在各类动作上的成功率（能力表按 backend × action 记录）
    所以这里只放「注册信息」，具体成功率一律去 outputs/_reference/capability_*.json 查。

接一个新后端的完整步骤
    1 在 BACKENDS 里加一条：入口签名保持 generate(prompts, frames, name=...)
    2 跑一次能力测试：python -m hm.capability --models 新后端名 --k 10
    3 什么都不用改，A05 路由会自动按新表选择

没有登记的后端一律拒绝，不做静默回退——回退会让「它到底用了什么」无从追溯。
"""
import json
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "outputs" / "_reference"
TABLE_VERSION = "capability_v4"

# 每条：用途、入口、帧数上限、能否接种子、能否条件生成、备注
BACKENDS = {
    "momask": {
        "kind": "text2motion",
        "entry": "hm.generate.generate",
        "max_frames": 196,
        "supports_seed": False,
        "supports_inpainting": False,
        "note": "主力后端，语义正确率最高",
    },
    "mdm": {
        "kind": "text2motion",
        "entry": "hm.generate.generate",
        "max_frames": 196,
        "supports_seed": True,
        "supports_inpainting": True,
        "note": "支持 inpainting，可用于「接前段末帧」的条件生成",
    },
    "motionlcm": {
        "kind": "text2motion",
        "entry": "hm.generate.generate",
        "max_frames": 196,
        "supports_seed": True,
        "supports_inpainting": False,
        "note": "少步采样，速度优先，当前未做对照",
    },
}

DEFAULT = "momask"


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


def match_entry(table, prompt):
    """按提示词在能力表里找动作条目。"""
    if not table:
        return None, None
    text = str(prompt).lower()
    for key, v in table.get("table", {}).items():
        words = [w for w in v.get("prompt", "").lower().split()[:3] if len(w) > 2]
        if (key.replace("_", " ") in text or v.get("action_zh", "") in str(prompt)
                or any(w in text for w in words)):
            return key, v
    return None, None


def resolve(prompt, prefer=None):
    """给一段提示词挑后端。返回 (backend, status, reason, detail)。

    规则：只要能力表里有该动作，就按「已注册后端中的最高通过率」选；
    表里没有该动作，用默认后端并标 unverified，不假装知道。
    """
    table = load_table()
    key, v = match_entry(table, prompt)
    if v is None:
        return DEFAULT, "unverified", "能力表无此动作，用默认后端", {"action": None}
    models = {m: d for m, d in v.get("models", {}).items() if m in BACKENDS}
    if not models:
        return DEFAULT, "unverified", "能力表里的后端都未注册", {"action": key}
    best = max(models, key=lambda m: models[m].get("pass_rate", 0))
    if prefer and prefer in models:
        best = prefer
    status = models[best].get("status", "unverified")
    rate = models[best].get("pass_rate")
    return best, status, "能力表：%s 通过率 %.2f" % (key, rate or 0.0), {
        "action": key, "pass_rate": rate,
        "candidates": {m: models[m].get("pass_rate") for m in models},
    }
