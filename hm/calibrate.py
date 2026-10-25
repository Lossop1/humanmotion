"""用人工标签校准判据。

标签来自学生对 96 段视频的逐段判断（2026-10-25），分三类：
    geom     几何变形/穿模/反关节 —— 必须拦下的错误
    sem      动作不对（例如 run 生成成下楼梯、turn 生成成跳跃转圈）—— 现有几何判据抓不到
    style    风格差异、自然度不足 —— 不算错误，不拦

脚本做两件事：
    1. 把标签写进 labels.json
    2. 对每个候选指标，比较 geom 失败组与通过组的分布，报 AUC 与单阈值最好分割

用法：python -m hm.calibrate
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/chuan/motion-work")
sys.path.insert(0, str(ROOT))
from hm import job as jobmod
from hm import pose_rules

OUT = ROOT / "outputs" / "_capsheet"
BONES = jobmod.BONES

# ---------------- 人工标签 ----------------
# 值：geom / sem / pass；style 记在 note 里，不参与二分类
LABELS = {
    "bow":          {"mdm": "pass", "momask": "pass"},
    "clap":         {"mdm": "pass", "momask": "pass"},
    "crouch":       {"mdm": "geom", "momask": "geom"},
    "jump":         {"mdm": "pass", "momask": "pass"},
    "kick":         {"mdm": "pass", "momask": "pass"},
    "lie":          {"mdm": "geom", "momask": "geom"},
    "raise_hands":  {"mdm": "mixed", "momask": "pass"},
    "run":          {"mdm": "sem", "momask": "sem"},
    "sit":          {"mdm": "geom", "momask": "geom"},
    "stand":        {"mdm": "pass", "momask": "pass"},
    "throw":        {"mdm": "pass", "momask": "pass"},
    "turn":         {"mdm": "sem", "momask": "pass"},
    "walk_back":    {"mdm": "pass", "momask": "pass"},
    "walk":         {"mdm": "pass", "momask": "pass"},
    "wave_goodbye": {"mdm": "pass", "momask": "pass"},
    "wave_hello":   {"mdm": "unnatural", "momask": "pass"},
}

# 逐采样指定的例外：raise_hands 的 mdm 只有第 3 个坏了
SAMPLE_LABELS = {
    ("raise_hands", "mdm", 1): "pass",
    ("raise_hands", "mdm", 2): "pass",
    ("raise_hands", "mdm", 3): "geom",
}

NOTES = {
    "crouch": "两模型都从膝盖开始往上扭曲约 180 度，脚尖与脸部方向相反，穿模",
    "sit": "与 crouch 同类，完全反关节，但没有扭曲",
    "lie": "两模型都穿模且严重扭曲；相机角度也不对，看不到全貌",
    "raise_hands": "mdm1 开合跳、mdm2 toss（有抬手但不是单纯抬手），mdm3 严重扭曲；momask 稳定但手不高",
    "run": "两模型都把 run 生成成下楼梯，momask 更明显；动作本身没有变形或扭曲",
    "turn": "mdm 变成跳跃转圈 360 度；momask 是正常走路转 180 度；均无变形",
    "stand": "momask 默认把手臂张成大字形；两模型都无扭曲",
    "wave_hello": "mdm 仰头、内八，没变形但不自然；momask 正常",
    "clap": "两模型手掌都不贴合；贴合过于精细，不要求解决",
    "walk": "mdm 幅度更大（风格差异），与 walk_back 一致",
    "wave_goodbye": "均为正常，mdm 幅度更大（风格差异）",
    "walk_back": "均无变形，momask 更平滑",
}


def resolve(action, model, sample):
    key = (action, model, sample)
    if key in SAMPLE_LABELS:
        return SAMPLE_LABELS[key]
    v = LABELS.get(action, {}).get(model, "unlabeled")
    return v


# ---------------- 指标 ----------------
def metrics_for(name):
    """name = action_model_n。返回指标字典。"""
    action, model, n = name.rsplit("_", 2)
    J = np.load(OUT / "joints" / ("%s.npy" % name))
    out = {"action": action, "model": model, "sample": int(n)}

    # 骨长
    L = jobmod.bone_lengths(J)
    out["bone_rel_mean"] = float(np.abs(L.mean(0) - L.mean(0).mean()).mean())
    ref = jobmod.load_bone_reference()
    if ref:
        mean = np.array(ref["mean"])
        out["bone_rel_ref"] = float((np.abs(L.mean(0) - mean) / mean).max())
    out["bone_jitter"] = float(L.std(0).max())

    # 姿态规则量
    M, _ = pose_rules.measures(J)
    out["r1_knee_min"] = float(np.nanmin(M[:, 0]))
    out["r2_head_min"] = float(np.nanmin(M[:, 1]))
    out["r2_head_max"] = float(np.nanmax(M[:, 1]))
    out["r3_tilt_max"] = float(np.nanmax(M[:, 2]))

    # 扭曲：相邻帧骨向最大变化
    d0 = np.array([J[1:, c] - J[1:, p] for c, p in BONES]).transpose(1, 0, 2)
    d0 = d0 / np.maximum(np.linalg.norm(d0, axis=2, keepdims=True), 1e-9)
    d1 = np.array([J[:-1, c] - J[:-1, p] for c, p in BONES]).transpose(1, 0, 2)
    d1 = d1 / np.maximum(np.linalg.norm(d1, axis=2, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip((d0 * d1).sum(2), -1, 1)))
    out["twist_max"] = float(ang.max())
    out["twist_p95"] = float(np.percentile(ang, 95))

    # 穿模近似：不相邻关节的最小距离（折叠到身体内部时会很小）
    near = []
    for t in range(len(J)):
        p = J[t]
        best = 9.0
        for a in range(22):
            for b in range(a + 1, 22):
                if abs(a - b) <= 2:
                    continue
                dd = float(np.linalg.norm(p[a] - p[b]))
                best = min(best, dd)
        near.append(best)
    out["min_sep_m"] = float(np.min(near))
    out["min_sep_p05"] = float(np.percentile(near, 5))

    # R4：根朝向 vs 脚尖（用拟合参数）
    pfile = OUT / "render" / name / "params.npy"
    if pfile.exists():
        from scipy.spatial.transform import Rotation as R
        P = np.load(pfile)
        dots = []
        for t in range(len(P)):
            fwd = R.from_rotvec(P[t, 0:3]).apply(np.array([0.0, 0.0, 1.0]))
            foot = np.zeros(3)
            for aa, tt in ((7, 10), (8, 11)):
                v = J[t, tt] - J[t, aa]
                v[1] = 0.0
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    foot += v / nv
            nf, ng = float(np.linalg.norm(fwd[[0, 2]])), float(np.linalg.norm(foot[[0, 2]]))
            if nf < 1e-6 or ng < 1e-6:
                continue
            dots.append(float(np.dot(fwd[[0, 2]] / nf, foot[[0, 2]] / ng)))
        dots = np.array(dots) if dots else np.array([1.0])
        out["r4_neg_frac"] = float((dots < 0).mean())
        out["r4_min"] = float(dots.min())
    return out


def auc(pos, neg):
    """正类=通过，负类=失败；AUC=0.5 表示无区分力。"""
    if not len(pos) or not len(neg):
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def main():
    rows = []
    for f in sorted((OUT / "joints").glob("*.npy")):
        name = f.stem
        try:
            m = metrics_for(name)
        except Exception as e:
            print("  跳过 %s：%s" % (name, e))
            continue
        m["label"] = resolve(m["action"], m["model"], m["sample"])
        rows.append(m)

    (OUT / "labels.json").write_text(json.dumps({
        "source": "学生对 96 段视频的逐段判断，2026-10-25",
        "levels": {"geom": "几何变形/穿模/反关节，必须拦下",
                   "sem": "动作语义错，几何判据抓不到",
                   "pass": "可用", "mixed": "同动作内不同采样不一致"},
        "labels": LABELS, "sample_labels": {str(k): v for k, v in SAMPLE_LABELS.items()},
        "notes": NOTES,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    geom = [r for r in rows if r["label"] == "geom"]
    good = [r for r in rows if r["label"] == "pass"]
    sem = [r for r in rows if r["label"] == "sem"]
    print("  样本：geom 失败 %d 段，pass %d 段，sem 失败 %d 段" % (len(geom), len(good), len(sem)))
    print()
    keys = ["bone_rel_ref", "bone_jitter", "r1_knee_min", "r2_head_min", "r2_head_max",
            "r3_tilt_max", "twist_max", "twist_p95", "min_sep_m", "min_sep_p05",
            "r4_neg_frac", "r4_min"]
    print("  %-16s %10s %10s %6s" % ("指标", "geom 中位", "pass 中位", "AUC"))
    res = []
    for k in keys:
        a = [r[k] for r in good if r.get(k) is not None]
        b = [r[k] for r in geom if r.get(k) is not None]
        if not a or not b:
            continue
        A = auc(a, b)
        res.append((k, np.median(b), np.median(a), A))
        print("  %-16s %10.4f %10.4f %6.2f" % (k, np.median(b), np.median(a), A))
    print()
    best = sorted(res, key=lambda x: -abs(x[3] - 0.5))[:3]
    print("  区分力最强的三项：" + "、".join("%s(AUC %.2f)" % (b[0], b[3]) for b in best))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
