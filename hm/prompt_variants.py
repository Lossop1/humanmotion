"""提示词变体：把一句措辞拆出两三种更短的写法。

依据（实测）
    长措辞容易失手：punch「throws a punch」0/4、「punches forward」4/4；
    走路加「at a normal pace」「away from the camera」这类尾巴也可能反而变差。
做法
    给同一动作生成两三个变体，一起拿去生成，再用判据挑。
    变体只做规则化删减，不让 LLM 重写——重写会改语义，风险更大。
"""
import re
from hm import paths as _paths  # 标定目录（calibration/，兼容 outputs/_reference）

# 可以安全删掉的修饰尾巴
TAILS = (
    r"\s+at\s+a\s+\w+\s+pace", r"\s+slowly", r"\s+quickly", r"\s+fast",
    r"\s+away\s+from\s+the\s+camera", r"\s+toward\s+the\s+camera",
    r"\s+in\s+front\s+of\s+the\s+camera", r"\s+with\s+the\s+(right|left)\s+(hand|arm|leg|foot|fist)",
    r"\s+a\s+few\s+steps", r"\s+several\s+times", r"\s+once", r"\s+repeatedly",
)
DIRS = ("forward", "backward", "left", "right", "up", "down")


def variants(prompt, max_variants=3):
    """返回去重后的写法列表，第一项永远是原写法。"""
    p = str(prompt).strip()
    out = [p]
    short = p
    for pat in TAILS:
        short = re.sub(pat, "", short, flags=re.I)
    short = re.sub(r"\s{2,}", " ", short).strip(" ,.")
    if short and short.lower() != p.lower():
        out.append(short)
    # 没有方向词的动作句，补一个 in place，避免模型自己乱拐
    if len(out) < max_variants:
        low = out[-1].lower()
        if not any(d in low for d in DIRS) and "in place" not in low:
            out.append(out[-1] + " in place")
    seen, uniq = set(), []
    for x in out[:max_variants]:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def bank_alternatives(prompt, limit=2, min_rate=0.5):
    """从提示词库里取同动作的其它写法（方向必须一致，通过率不低于 min_rate）。"""
    import json
    from pathlib import Path
    p = _paths.ref() / "prompt_bank.json"
    if not p.exists():
        return []
    bank = json.loads(p.read_text(encoding="utf-8"))
    def _dirs_of(x):
        # 「with the right hand」的 right 是身体侧，不是行进方向
        y = re.sub(r"(right|left)\s+(hand|leg|arm|fist|foot)", " ", str(x).lower())
        return {w for w in re.findall(r"[a-z]+", y) if w in DIRS}

    text = str(prompt).lower()
    dirs = _dirs_of(text)

    def toks(x):
        # 方向词不算动作词，否则「walks forward」会匹配到「punches forward」
        stop = {"a", "the", "person", "with", "and", "in", "on", "to", "of", "up", "down",
                "forward", "backward", "left", "right", "back", "place", "around", "once"}
        return {w for w in re.findall(r"[a-z]+", str(x).lower()) if w not in stop and len(w) > 2}

    cur = toks(prompt)
    out = []
    for action, entry in bank.items():
        for row in entry.get("phrasings", []):
            cand = row["text"]
            if cand.lower() == text:
                continue
            if _dirs_of(cand) != dirs:
                continue
            if not (toks(cand) & cur):
                continue
            rate = max((m.get("rate", 0) for m in row.get("models", {}).values()), default=0)
            if rate < min_rate:
                continue
            out.append((rate, cand))
    out.sort(reverse=True)
    return [c for _, c in out[:limit]]


def bank_rate(prompt):
    """查提示词库：返回 (动作键, 最优通过率, 最优写法) 或 None。方向必须一致。"""
    import json
    from pathlib import Path
    p = _paths.ref() / "prompt_bank.json"
    if not p.exists():
        return None
    bank = json.loads(p.read_text(encoding="utf-8"))

    def _dirs_of(x):
        y = re.sub(r"(right|left)\s+(hand|leg|arm|fist|foot)", " ", str(x).lower())
        return {w for w in re.findall(r"[a-z]+", y) if w in DIRS}

    def toks(x):
        stop = {"a", "the", "person", "with", "and", "in", "on", "to", "of", "up", "down",
                "forward", "backward", "left", "right", "back", "place", "around", "once"}
        return {w for w in re.findall(r"[a-z]+", str(x).lower()) if w not in stop and len(w) > 2}

    cur, dirs = toks(prompt), _dirs_of(prompt)
    for key, entry in bank.items():
        for row in entry.get("phrasings", []):
            cand = row["text"]
            if _dirs_of(cand) != dirs:
                continue
            if not (toks(cand) & cur):
                continue
            rate = max((m.get("rate", 0) for m in row.get("models", {}).values()), default=0)
            return key, rate, entry.get("best") or cand
    return None
