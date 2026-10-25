"""A11 语义辅助分数与人工标注位；A13/A14/A15 的失败分类与修复闭环。

关于 A11：标准的文本—动作评测器（MM-Dist 等）当前未接入运行环境。
按说明书 A11 兜底条款「若标准评测器未能接入，则以盲法人工标注和
可复现的几何判据作为主评测，明确报告这一限制」，这里做两件事：
    1. 可复现的几何判据：按动作类别检查该动作应有的几何特征是否出现；
    2. 人工标注位：留出 labels.json，由人填 pass/fail，不由程序猜。
"""
import json
import math
from pathlib import Path

import numpy as np

from hm.job import BONES

# 动作类别的几何特征表。每条：关键词 → (判据函数说明, 阈值, 取值函数)
ACTION_SIGNATURES = [
    {
        "name": "位移型",
        "keywords": ("walk", "run", "后退", "向前走", "跑", "walking", "running"),
        "metric": "root_speed_mps",
        "rule": ">= 0.30",
        "threshold": 0.30,
        "note": "走跑类应有明显根位移",
    },
    {
        "name": "原地型",
        "keywords": ("stand", "stop", "turn around in place", "wave", "clap", "挥手", "鼓掌", "站"),
        "metric": "root_speed_mps",
        "rule": "<= 0.25",
        "threshold": 0.25,
        "note": "原地类动作根位移应很小",
    },
    {
        "name": "挥手型",
        "keywords": ("wave", "挥手"),
        "metric": "wrist_vertical_range_m",
        "rule": ">= 0.05",
        "threshold": 0.05,
        "note": "挥手需要手部有明显竖向运动",
    },
    {
        "name": "转身型",
        "keywords": ("turn", "转身"),
        "metric": "heading_change_deg",
        "rule": ">= 45",
        "threshold": 45.0,
        "note": "转身应带来明显朝向变化",
    },
]


def _heading(Jf):
    a = np.asarray(Jf[13]) - np.asarray(Jf[14])
    return math.degrees(math.atan2(float(a[2]), float(-a[0])))


def geometric_metrics(J, fps=20):
    J = np.asarray(J, dtype=np.float64)
    tr = J[:, 0, [0, 2]]
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1) if len(J) > 1 else np.zeros(1)
    speed = float(step.mean() * fps) if len(step) else 0.0
    wr = np.concatenate([J[:, 20, 1], J[:, 21, 1]])
    wrist_range = float(wr.max() - wr.min()) if wr.size else 0.0
    h = [_heading(J[t]) for t in range(len(J))]
    dh = 0.0
    for i in range(1, len(h)):
        d = (h[i] - h[i - 1] + 180) % 360 - 180
        dh += d
    return {"root_speed_mps": round(speed, 3),
            "wrist_vertical_range_m": round(wrist_range, 3),
            "heading_change_deg": round(abs(dh), 1)}


def semantic_proxy(J, action_text, fps=20):
    """按动作类别检查几何签名。返回 (passed_or_None, 命中规则名, 明细)。"""
    g = geometric_metrics(J, fps)
    act = (action_text or "").lower()
    hits = []
    for sig in ACTION_SIGNATURES:
        if not any(k.lower() in act for k in sig["keywords"]):
            continue
        v = g.get(sig["metric"])
        if v is None:
            continue
        ok = v >= sig["threshold"] if sig["rule"].startswith(">") else v <= sig["threshold"]
        hits.append({"signature": sig["name"], "metric": sig["metric"],
                     "value": v, "rule": sig["rule"], "passed": bool(ok),
                     "note": sig["note"]})
    if not hits:
        return None, "no_matching_signature", {"geometric": g, "signatures": []}
    passed = all(h["passed"] for h in hits)
    return passed, "geometric_signature", {"geometric": g, "signatures": hits}


# ---------------- A13 / A14 ----------------
FAILURE_TYPES = (
    "提示词语义退化",
    "抽样质量波动",
    "模型能力不足",
    "空间目标偏差",
    "数据/坐标错误",
    "评测器误报",
)

# 说明书 A14 的失败表：一次重试只改一个变量
REPAIR_TABLE = {
    "抽样质量波动": {"variable": "candidate_count", "action": "提高候选数重采样", "allowed": True},
    "提示词语义退化": {"variable": "prompt", "action": "改提示词后重新生成", "allowed": False},
    "模型能力不足": {"variable": "model", "action": "换已验证后端或标记不支持", "allowed": False},
    "空间目标偏差": {"variable": "none", "action": "检查阶段/坐标后再用受限位移对齐", "allowed": False},
    "数据/坐标错误": {"variable": "none", "action": "停机并修正数据处理", "allowed": False},
    "评测器误报": {"variable": "none", "action": "复核样本与规则，不自动修复", "allowed": False},
}


def classify(cand_metrics, pose_reasons, semantic_result):
    """给一个全失败的段定一个主要失败类型。"""
    if any(not c["hard_checks"].get("finite", True) or
           not c["hard_checks"].get("shape", True) for c in cand_metrics):
        return "数据/坐标错误", "存在非有限值或形状错误"
    rels = [c["metrics"].get("bone_length_rel_mean_max") for c in cand_metrics
            if c["metrics"].get("bone_length_rel_mean_max") is not None]
    if rels and all(r > 0.05 for r in rels):
        return "数据/坐标错误", "所有候选的骨长均值偏差都超过容差"
    if pose_reasons:
        return "模型能力不足", "通过几何检查但姿态规则不通过：%s" % pose_reasons[0][:60]
    if semantic_result is False:
        return "抽样质量波动", "几何签名不成立，可能是采样波动或动作未完成"
    return "unknown", "无法归因"


def plan_repair(failure_type, round_no, max_rounds=2):
    """返回修复计划；不允许修的情况返回 None。"""
    if round_no >= max_rounds:
        return None
    spec = REPAIR_TABLE.get(failure_type)
    if not spec or not spec["allowed"]:
        return None
    return {"round": round_no + 1, "variable": spec["variable"],
            "action": spec["action"], "failure_type": failure_type}
