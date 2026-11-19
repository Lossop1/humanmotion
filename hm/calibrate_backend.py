"""按后端校准判据：新后端进来时，阈值从它自己的分布里取，不沿用别人的数字。

为什么要这一步
    骨长 4 厘米、反向帧 2%、交界峰值 25 度/帧这些数，是在 MoMask/MDM 的输出上
    看出来的。换一个更好的模型，这些数可能过严（把好动作拦掉），
    换一个更差的模型，可能过松（把坏动作放进来）。
    真正与后端无关的只有物理必然：骨长恒定、数值有限、形状对。
    其余判据应当「按后端、按动作」校准，并把校准来源写进文件。

校准集
    固定探针动作：走、站、转身、挥手、跳。每个后端各采 k 个样本。
    对每个动作统计：交界峰值、段内逐帧速率、前倾、反向帧占比、骨长相对偏差。
    阈值取「已知可用样本」的 95 分位再留一点余量，而不是拍一个绝对数。

产物
    outputs/_reference/criteria_<后端>.json
    {后端, 版本, 采样数, 每个动作的指标分布, 推导出的阈值, 生成时间, 判据版本}
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hm import generate, to_joints
from hm.finish_job import BONES, dirs_of, rot_between

REF = (Path(__file__).resolve().parent.parent / "outputs/_reference")
PROBES = [
    ("walk", "a person walks forward", 56),
    ("stand", "a person stands still", 40),
    ("turn", "a person turns around in place", 44),
    ("wave", "a person waves goodbye with the right hand", 44),
    ("jump", "a person jumps up once in place", 40),
]
PELVIS, SPINE3 = 0, 9


def metrics(J):
    J = np.asarray(J, dtype=np.float64)
    rates = []
    for t in range(len(J) - 1):
        d0, _ = dirs_of(J[t])
        d1, _ = dirs_of(J[t + 1])
        rates.append(max(math.degrees(rot_between(d0[k], d1[k])[1]) for k in range(len(BONES))))
    rates = np.array(rates) if rates else np.zeros(1)
    v = J[:, SPINE3] - J[:, PELVIS]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    lean = np.degrees(np.arccos(np.clip(v[:, 1], -1, 1)))
    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = float(np.linalg.norm(fwd))
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    back = float(-(J[-1, 0] - J[0, 0]) @ fwd)
    L = np.stack([np.linalg.norm(J[:, c] - J[:, p], axis=1) for c, p in BONES])
    return {"rate_med": float(np.median(rates)), "rate_p95": float(np.percentile(rates, 95)),
            "rate_max": float(rates.max()), "lean_med": float(np.median(lean)),
            "lean_p95": float(np.percentile(lean, 95)), "backward_m": back,
            "bone_rel_spread": float(np.median(L.std(axis=1) / np.maximum(L.mean(axis=1), 1e-9)))}


def calibrate(model, k=4):
    rows = {}
    for key, prompt, frames in PROBES:
        feats = generate.generate([prompt] * k, [frames] * k, name=model)
        vals = [metrics(np.asarray(to_joints.from_feats(F), dtype=np.float64)) for F in feats]
        agg = {}
        for name in vals[0]:
            arr = np.array([v[name] for v in vals])
            agg[name] = {"median": round(float(np.median(arr)), 3),
                         "p95": round(float(np.percentile(arr, 95)), 3),
                         "max": round(float(arr.max()), 3)}
        rows[key] = {"prompt": prompt, "samples": k, "metrics": agg}
        print("    %-6s 速率中位 %.2f p95 %.2f ｜ 前倾中位 %.1f ｜ 后退 %.2f"
              % (key, agg["rate_med"]["median"], agg["rate_p95"]["p95"],
                 agg["lean_med"]["median"], agg["backward_m"]["median"]), flush=True)

    # 阈值：取全部探针里各指标的 p95，再留 30% 余量
    def thr(field):
        vals = [rows[k]["metrics"][field]["p95"] for k in rows]
        return round(float(max(vals)) * 1.3, 2)

    thresholds = {
        "rate_p95_deg_per_frame": thr("rate_p95"),
        "lean_p95_deg": thr("lean_p95"),
        "bone_rel_spread": thr("bone_rel_spread"),
        "note": "由本后端探针样本的 p95 推得，含 30% 余量；换后端需重跑",
    }
    doc = {
        "backend": model,
        "criteria_version": "criteria_v5",
        "k_per_action": k,
        "probes": rows,
        "thresholds": thresholds,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = REF / ("criteria_%s.json" % model)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  已写 %s：%s" % (out.name, thresholds))
    return doc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = sys.argv[1:] or ["momask"]
    for m in models:
        print("  == 校准后端 %s ==" % m)
        calibrate(m)
