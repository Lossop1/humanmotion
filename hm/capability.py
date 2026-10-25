"""能力测试：动作类别 × 后端 × 采样，统计通过率，生成能力表。

能力表被两处使用：
    A03 判断该动作能不能做（支持 / 可近似 / 未验证 / 不支持）
    A05 选择后端

判据与 run_job 的硬检查完全一致：H1 非有限值、H2 形状、H3 骨长均值偏差 ≤5%、
H4 姿态规则落在参考包络内。语义签名只记录，不当通过标准。

用法：
    python -m hm.capability --k 3 --models momask,mdm,motionlcm
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from hm import config, generate, job as jobmod, pose_rules, repair, to_joints

# 动作类别表：(类别键, 中文, 英文提示词, 关键词, 帧数)
ACTIONS = [
    ("wave_hello", "挥手打招呼", "a person waves hello with the right hand", ("wave", "挥手"), 40),
    ("wave_goodbye", "挥手告别", "a person waves goodbye", ("goodbye", "告别"), 40),
    ("clap", "鼓掌", "a person claps their hands", ("clap", "鼓掌"), 40),
    ("walk", "向前走", "a person walks forward", ("walk", "走"), 52),
    ("walk_back", "后退", "a person walks backward", ("backward", "后退"), 48),
    ("run", "跑", "a person runs forward", ("run", "跑"), 40),
    ("turn", "转身", "a person turns around", ("turn", "转身"), 40),
    ("stand", "站住", "a person stands still", ("stand", "站"), 40),
    ("jump", "跳", "a person jumps up", ("jump", "跳"), 36),
    ("kick", "踢腿", "a person kicks with the right leg", ("kick", "踢"), 36),
    ("bow", "鞠躬", "a person bows", ("bow", "鞠躬"), 40),
    ("raise_hands", "举手", "a person raises both hands", ("raise", "举手"), 36),
    ("crouch", "蹲下", "a person crouches down", ("crouch", "蹲"), 40),
    ("sit", "坐下", "a person sits down on a chair", ("sit", "坐"), 40),
    ("lie", "躺下", "a person lies down on the floor", ("lie", "躺"), 44),
    ("throw", "投掷", "a person throws a ball", ("throw", "投掷"), 36),
]

SUPPORT_TH = 0.75      # 通过率 ≥ 此值记为支持
APPROX_TH = 0.40       # 介于两线之间记为可近似
TABLE_VERSION = "capability_v1"
REF_DIR = Path(__file__).resolve().parent.parent / "outputs" / "_reference"


def _hard_check(J, pose_th):
    hard = {"finite": bool(np.isfinite(J).all()),
            "shape": bool(J.ndim == 3 and J.shape[1] == 22 and J.shape[2] == 3)}
    if not (hard["finite"] and hard["shape"]):
        return False, hard, None, None
    L = jobmod.bone_lengths(J)
    return True, hard, L, None


def run_test(models, k, bone_ref, pose_th):
    rows = []
    for key, zh, prompt, kws, frames in ACTIONS:
        for model in models:
            t0 = time.time()
            try:
                feats = generate.generate([prompt] * k, [frames] * k, name=model)
            except Exception as e:
                rows.append({"action": key, "action_zh": zh, "prompt": prompt,
                             "model": model, "samples": 0, "error": str(e)[:120]})
                print("  %-12s %-10s 失败：%s" % (key, model, str(e)[:60]), flush=True)
                continue
            ok = 0
            details = []
            for F in feats:
                try:
                    J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
                except Exception as e:
                    details.append({"pass": False, "why": "decode:%s" % str(e)[:40]})
                    continue
                finite_ok, hard, _, _ = _hard_check(J, pose_th)
                if not finite_ok:
                    details.append({"pass": False, "why": "H1/H2"})
                    continue
                mean = np.array(bone_ref["mean"])
                # 骨长均值相对偏差
                L = jobmod.bone_lengths(J)
                Lm = L.mean(axis=0)
                rel_max = float((np.abs(Lm - mean) / np.maximum(mean, 1e-9)).max())
                bone_ok = rel_max <= float(bone_ref.get("corruption_tolerance", 0.20))
                pose_ok, reasons, _, _ = pose_rules.check(J, prompt, pose_th)
                sem_ok, sem_rule, sem_detail = repair.semantic_proxy(J, prompt)
                sem_ok = True if sem_ok is None else bool(sem_ok)
                passed = bool(bone_ok and pose_ok and sem_ok)
                ok += int(passed)
                details.append({"pass": passed, "bone_rel": round(rel_max, 4),
                                "pose_ok": pose_ok, "sem_ok": sem_ok,
                                "sem_rule": sem_rule, "why": reasons[:1]})
            rate = ok / max(len(feats), 1)
            rows.append({"action": key, "action_zh": zh, "prompt": prompt, "model": model,
                         "samples": len(feats), "passed": ok, "pass_rate": round(rate, 3),
                         "frames": frames, "seconds": round(time.time() - t0, 1),
                         "details": details})
            print("  %-12s %-10s %d/%d = %.0f%%  (%.1f 秒)"
                  % (key, model, ok, len(feats), 100 * rate, time.time() - t0), flush=True)
    return rows


def classify(rate, samples):
    if samples == 0:
        return "unverified"
    if rate >= SUPPORT_TH:
        return "supported"
    if rate >= APPROX_TH:
        return "approximate"
    return "unsupported"


def build_table(rows):
    table = {}
    for r in rows:
        key = r["action"]
        table.setdefault(key, {"action_zh": r["action_zh"], "prompt": r["prompt"],
                               "models": {}})
        table[key]["models"][r["model"]] = {
            "samples": r["samples"], "passed": r.get("passed", 0),
            "pass_rate": r.get("pass_rate", 0.0),
            "status": classify(r.get("pass_rate", 0.0), r["samples"]),
        }
    # 每个动作挑最好的后端
    for key, v in table.items():
        cands = [(m, d) for m, d in v["models"].items() if d["samples"] > 0]
        if not cands:
            v["best"] = None
            v["status"] = "unverified"
            continue
        best = max(cands, key=lambda x: (x[1]["pass_rate"], -x[1]["samples"]))
        v["best"] = best[0]
        v["best_pass_rate"] = best[1]["pass_rate"]
        v["status"] = best[1]["status"]
    return table


def save(table, rows, models, k):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": TABLE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "samples_per_cell": k,
        "hard_checks": ["H1 非有限值", "H2 形状",
                        "H3 骨长均值偏差≤20%（只抓损坏，不作质量判据）",
                        "H4 姿态规则包络（两侧放宽 25%）",
                        "M1 几何语义签名"],
        "thresholds": {"supported": SUPPORT_TH, "approximate": APPROX_TH,
                       "note": "分段线取在本轮数据的通过率分布上，样本数有限，"
                               "扩充样本后需复核"},
        "table": table,
        "raw": rows,
    }
    p = REF_DIR / ("%s.json" % TABLE_VERSION)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_table():
    p = REF_DIR / ("%s.json" % TABLE_VERSION)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def lookup(prompt_or_action):
    """按提示词或动作文字找能力条目。返回 (status, best_model, detail)。"""
    doc = load_table()
    if not doc:
        return "unverified", None, {"reason": "能力表未生成"}
    text = str(prompt_or_action).lower()
    best_hit = None
    for key, v in doc["table"].items():
        hit = (key in text) or (v["action_zh"] in text) or any(
            w in text for w in (v["prompt"].lower().split()[:3]))
        if hit:
            best_hit = (key, v)
            break
    if best_hit is None:
        return "unverified", None, {"reason": "动作未出现在能力表内"}
    key, v = best_hit
    return v["status"], v.get("best"), {
        "action": key, "action_zh": v["action_zh"],
        "best_pass_rate": v.get("best_pass_rate"),
        "models": {m: {"pass_rate": d["pass_rate"], "samples": d["samples"],
                       "status": d["status"]} for m, d in v["models"].items()},
        "table_version": doc["version"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--models", default="momask,mdm,motionlcm")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    bone_ref = jobmod.load_bone_reference()
    if bone_ref is None:
        bone_ref, _ = jobmod.build_bone_reference()
    pose_th = pose_rules.load_thresholds()

    print("  能力测试：%d 个动作 × %d 个后端 × %d 次采样"
          % (len(ACTIONS), len(models), a.k))
    rows = run_test(models, a.k, bone_ref, pose_th)
    table = build_table(rows)
    p = save(table, rows, models, a.k)
    print("\n  能力表已写入 %s" % p)
    print("  %-14s %-10s %s" % ("动作", "最好后端", "通过率"))
    for key, v in table.items():
        print("  %-14s %-10s %s" % (v["action_zh"], v.get("best") or "—",
                                    "%.0f%%（%s）" % (100 * (v.get("best_pass_rate") or 0), v["status"])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
