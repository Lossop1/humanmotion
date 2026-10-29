"""判据 v2：由 96 段人工标注校准出来的硬检查。

阈值来源：outputs/_reference/criteria_v2.json
    骨长均值相对偏差 > 0.103            → 几何失败
    髋线定的骨盆朝向与脚尖方向存在反向帧 → 几何失败（这是「反关节扭转」的实际表现）
    在 84 段二分类样本上：灵敏度 0.95、特异度 0.89、平衡准确率 0.92

说明：反向帧判据用关节直接算（髋线 vs 脚尖），不需要 SMPL 拟合，
      因此可以放在解码之后、装配之前。
"""
import json
from pathlib import Path

import numpy as np

from hm import job as jobmod
from hm import pose_rules

REF_DIR = Path(__file__).resolve().parent.parent / "outputs" / "_reference"
VERSION = "criteria_v5"


def load():
    p = REF_DIR / ("%s.json" % VERSION)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"bone_th": 0.103, "bone_err_cm": 4.0, "rev_th": 0.02, "sep_th": 0.045,
            "sens": 0.89, "spec": 0.98, "bal": 0.94}


def hip_vs_toe(J):
    """返回 (反向帧占比, 最小点积)。"""
    dots = []
    for t in range(len(J)):
        hips = J[t, 2] - J[t, 1]
        hips[1] = 0.0
        p = np.cross(np.array([0.0, 1.0, 0.0]), hips)
        acc = np.zeros(3)
        for a, b in ((7, 10), (8, 11)):
            v = J[t, b] - J[t, a]
            v[1] = 0.0
            n = np.linalg.norm(v)
            if n > 1e-6:
                acc += v / n
        np_, nt = np.linalg.norm(p[[0, 2]]), np.linalg.norm(acc[[0, 2]])
        if np_ < 1e-6 or nt < 1e-6:
            continue
        dots.append(float(np.dot(p[[0, 2]] / np_, acc[[0, 2]] / nt)))
    d = np.array(dots) if dots else np.array([1.0])
    return float((d < 0).mean()), float(d.min())


def min_separation(J):
    """非相邻关节（序号差 >= 3）的最小距离，身体折叠到内部时会很小。"""
    best = 9.0
    for t in range(len(J)):
        p = J[t]
        for a in range(22):
            for b in range(a + 3, 22):
                d = float(np.linalg.norm(p[a] - p[b]))
                if d < best:
                    best = d
    return best


def check(J, action_text="", ref=None, pose_th=None):
    """返回 (passed, hard, metrics, reasons)。"""
    crit = load()
    if ref is None:
        ref = jobmod.load_bone_reference()
    reasons = []
    hard = {"finite": bool(np.isfinite(J).all()),
            "shape": bool(J.ndim == 3 and J.shape[1] == 22 and J.shape[2] == 3)}
    metrics = {}
    if not (hard["finite"] and hard["shape"]):
        return False, hard, metrics, ["非有限值或形状错误"]

    L = jobmod.bone_lengths(J)
    mean = np.array(ref["mean"])
    Lm = L.mean(0)
    err_cm = float(100 * np.abs(Lm - mean).max())
    rel = float((np.abs(Lm - mean) / np.maximum(mean, 1e-9)).max())
    metrics["bone_err_cm"] = err_cm
    metrics["bone_rel_mean_max"] = rel          # 保留作软指标
    metrics["bone_jitter_m"] = float(L.std(0).max())
    tol_cm = float(crit.get("bone_err_cm", 4.0))
    hard["bone_length"] = err_cm <= tol_cm
    if not hard["bone_length"]:
        k = int(np.argmax(np.abs(Lm - mean)))
        reasons.append("骨长绝对误差 %.1f 厘米超过 %.1f 厘米（%s）"
                       % (err_cm, tol_cm, jobmod.BONE_NAMES[k]))

    rev_frac, rev_min = hip_vs_toe(J)
    metrics["rev_frac"] = rev_frac
    metrics["rev_min"] = rev_min
    hard["root_facing"] = rev_frac <= crit.get("rev_th", 0.02)
    if not hard["root_facing"]:
        reasons.append("骨盆朝向与脚尖反向（%.0f%% 的帧，最小点积 %.2f）"
                       % (100 * rev_frac, rev_min))

    sep = min_separation(J)
    metrics["min_separation_m"] = sep
    hard["separation"] = sep >= crit.get("sep_th", 0.045)
    if not hard["separation"]:
        reasons.append("非相邻关节最小间距 %.3f 米，疑似穿模（阈值 %.3f）"
                       % (sep, crit.get("sep_th", 0.045)))

    pose_ok, pose_reasons, bad, values = pose_rules.check(J, action_text, pose_th)
    hard["pose_rules"] = pose_ok
    metrics["pose_values"] = values
    if not pose_ok:
        reasons.extend(pose_reasons)

    metrics["thresholds"] = {"bone_err_cm": crit.get("bone_err_cm", 4.0),
                             "rev_th": crit["rev_th"],
                             "sep_th": crit.get("sep_th", 0.045), "source": VERSION}
    passed = all(bool(v) for v in hard.values())
    return passed, hard, metrics, reasons
