"""A09 姿态退化检查。

四条规则，阈值全部由真实动捕统计后写文件（带版本），不在代码里写死：
    R1 膝盖朝向     膝应位于髋—踝中点之前（反关节会落到之后）
    R2 头的位置     头相对颈应朝身体正前方（拟合退化时会跑到后方）
    R3 躯干直立     骨盆—颈 与竖直方向的夹角
    R4 身体与脚尖   两者方向点积（整段姿态反向时为负）

允许的动作例外写在 EXCEPTIONS 里；已知漏检写进 rules 文件的 note 字段。
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from hm import config
from hm.job import BONES

RULE_VERSION = "pose_rules_v1"
PAD_FRACTION = 0.25   # 包络两侧各放宽的比例，见 pad()
REF_DIR = Path(__file__).resolve().parent.parent / "outputs" / "_reference"

# 这些动作在高姿态下膝会向前、躯干会前倾，不按通用阈值判
EXCEPTIONS = {
    "crouch": ("R3",), "squat": ("R3",), "sit": ("R3", "R1"),
    "kneel": ("R3", "R1"), "lie": ("R3", "R1", "R4"), "lying": ("R3", "R1", "R4"),
    "bow": ("R3",), "pick": ("R3",), "bend": ("R3",),
}


def body_forward(Jf):
    acc = np.zeros(3)
    for a, t in ((7, 10), (8, 11)):
        v = np.asarray(Jf[t]) - np.asarray(Jf[a])
        v[1] = 0.0
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            acc += v / n
    n = float(np.linalg.norm(acc))
    if n > 1e-6:
        return acc / n
    hips = np.asarray(Jf[2]) - np.asarray(Jf[1])
    hips[1] = 0.0
    f = np.cross(np.array([0.0, 1.0, 0.0]), hips)
    nf = float(np.linalg.norm(f))
    return f / nf if nf > 1e-6 else np.array([0.0, 0.0, 1.0])


def measures(J):
    """逐帧算四条规则的量，返回 (T,4) 与列名。"""
    J = np.asarray(J, dtype=np.float64)
    T = len(J)
    out = np.full((T, 4), np.nan)
    for t in range(T):
        f = J[t]
        fwd = body_forward(f)
        # R1 膝盖朝前量（左右取较小者，更容易暴露反关节）
        vals = []
        for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
            mid = 0.5 * (f[hip] + f[ankle])
            vals.append(float(np.dot(f[knee] - mid, fwd)))
        out[t, 0] = min(vals)
        # R2 头相对颈朝前
        out[t, 1] = float(np.dot(f[15] - f[12], fwd))
        # R3 躯干与竖直的夹角
        up = f[12] - f[0]
        nu = float(np.linalg.norm(up))
        out[t, 2] = math.degrees(math.acos(float(np.clip(up[1] / max(nu, 1e-9), -1, 1)))) if nu > 1e-9 else 0.0
        # R4 身体朝向与脚尖方向
        toes = np.zeros(3)
        for a, t2 in ((7, 10), (8, 11)):
            v = f[t2] - f[a]
            v[1] = 0.0
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                toes += v / nv
        nt = float(np.linalg.norm(toes))
        out[t, 3] = float(np.dot(fwd, toes / nt)) if nt > 1e-6 else 0.0
    return out, ["knee_forward_m", "head_forward_m", "torso_tilt_deg", "body_vs_toes"]


def build_thresholds(pattern="real40/*.npy"):
    """从真实动捕统计四条规则的分布，取包络作为阈值。"""
    files = sorted((Path(__file__).resolve().parent.parent).glob(pattern))
    if not files:
        raise FileNotFoundError("找不到参考动作：%s" % pattern)
    all_m = []
    for f in files:
        J = np.load(f)
        if J.ndim == 3 and J.shape[1] == 22:
            m, _ = measures(J)
            all_m.append(m)
    M = np.concatenate(all_m, axis=0)
    th = {
        "version": RULE_VERSION,
        "source": pattern,
        "files": len(files),
        "frames": int(len(M)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "R1_knee_forward_m": {"min": float(np.nanmin(M[:, 0])), "max": float(np.nanmax(M[:, 0]))},
        "R2_head_forward_m": {"min": float(np.nanmin(M[:, 1])), "max": float(np.nanmax(M[:, 1]))},
        "R3_torso_tilt_deg": {"min": float(np.nanmin(M[:, 2])), "max": float(np.nanmax(M[:, 2]))},
        "R4_body_vs_toes": {"status": "requires_smpl",
                            "note": "关节层面是自指量（恒为 1），改在 A17 用拟合根朝向与脚尖方向判定"},
        "note": "阈值为参考动捕的取值包络。已知漏检：只覆盖几何可判的四类退化，"
                "不覆盖穿模、脚滑与语义错误；蹲坐一类动作按 EXCEPTIONS 放宽 R1/R3。",
    }
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / ("%s.json" % RULE_VERSION)
    p.write_text(json.dumps(th, ensure_ascii=False, indent=1), encoding="utf-8")
    return th, p


def load_thresholds():
    p = REF_DIR / ("%s.json" % RULE_VERSION)
    if not p.exists():
        th, _ = build_thresholds()
        return th
    return json.loads(p.read_text(encoding="utf-8"))


def pad(rng, frac=None):
    """把参考包络两侧各放宽 frac，避免生成数据在边界上 3 毫米级误杀。"""
    frac = PAD_FRACTION if frac is None else frac
    lo, hi = rng["min"], rng["max"]
    span = hi - lo
    return {"min": lo - frac * span, "max": hi + frac * span,
            "raw_min": lo, "raw_max": hi, "pad": frac}


def _violated(v, rng, skip_low=True):
    """超出包络即违规；返回是否违规。"""
    lo, hi = rng["min"], rng["max"]
    return v < lo or v > hi


def check(J, action_text="", th=None):
    """返回 (passed, reasons, bad_frames, values)。"""
    th = th or load_thresholds()
    M, names = measures(J)
    act = (action_text or "").lower()
    skipped = set()
    for w, rules in EXCEPTIONS.items():
        if w in act:
            skipped.update(rules)

    # R4 不在关节层面判：body_forward 本身由脚尖方向算出，
    # dot(body_forward, toes) 恒等于 1，是自指量，永远判不出问题。
    # 真正能判「整段姿态反向」的是拟合后的根朝向，放在 A17 做。
    tol = {
        "R1": ("R1_knee_forward_m", 0, skipped),
        "R2": ("R2_head_forward_m", 1, skipped),
        "R3": ("R3_torso_tilt_deg", 2, skipped),
    }
    reasons, bad = [], {}
    for rid, (key, col, skip) in tol.items():
        if rid in skip:
            continue
        rng = pad(th[key])
        v = M[:, col]
        badmask = (v < rng["min"]) | (v > rng["max"])
        if badmask.any():
            idx = np.where(badmask)[0]
            reasons.append("%s 超出参考包络（%d 帧，例如第 %d 帧 %s=%.3f，包络 %.3f~%.3f）"
                           % (rid, len(idx), int(idx[0]) + 1, names[col], v[idx[0]],
                              rng["min"], rng["max"]))
            bad[rid] = [int(x) + 1 for x in idx[:20]]
    values = {names[c]: {"median": float(np.nanmedian(M[:, c])),
                         "min": float(np.nanmin(M[:, c])),
                         "max": float(np.nanmax(M[:, c]))} for c in range(4)}
    return (len(reasons) == 0), reasons, bad, values
