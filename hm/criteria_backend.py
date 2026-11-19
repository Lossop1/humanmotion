"""按后端读取校准阈值；没有校准时只做物理必然检查，不拿别人的数字硬套。

接口
    thresholds(backend)                     → dict 或 None
    limit(backend, field, action=None)      → 数值或 None
    contract_ok(backend)                    → (ok, 说明)

原则
    物理必然（骨长恒定、数值有限、形状）永远检查，不需要校准。
    质量类阈值只在「这个后端做过校准、且这个动作在探针集里」时使用；
    否则返回 None，调用方必须按「不判」处理，不能退回别人的阈值。
"""
from pathlib import Path

from hm import backends
from hm.jsonio import load_json

REF = Path(__file__).resolve().parent.parent / "outputs" / "_reference"


def thresholds(backend):
    p = REF / ("criteria_%s.json" % backend)
    return load_json(p, want=dict)


def limit(backend, field, action=None):
    """取阈值。action 给了就先找该动作自己的分布，再退回全局阈值。"""
    doc = thresholds(backend)
    if not doc:
        return None
    if action:
        probe = (doc.get("probes") or {}).get(action)
        if probe:
            m = (probe.get("metrics") or {}).get(field)
            if m and "p95" in m:
                return float(m["p95"]) * 1.3
    return (doc.get("thresholds") or {}).get(field)


# 后端契约：本管线只认 HumanML3D 263 维、20 fps、米、22 关节。
# 新后端必须声明并满足这些，否则拒绝——不能「先跑跑看」，
# 因为坐标系或帧率不对时，所有下游数字都是错的，而且错得看不出来。
REQUIRED_CONTRACT = {
    "output": "hml263",          # 也可以是 smpl / joints，但需要先写适配器
    "fps": 20,
    "unit": "m",
    "joints": 22,
}


def contract_ok(backend):
    reg = backends.BACKENDS.get(backend)
    if not reg:
        return False, "后端未注册"
    c = reg.get("contract") or {}
    bad = []
    for k, v in REQUIRED_CONTRACT.items():
        if c.get(k) != v:
            bad.append("%s=%r（需要 %r）" % (k, c.get(k), v))
    if bad:
        return False, "契约不满足：" + "；".join(bad)
    return True, "契约满足"
