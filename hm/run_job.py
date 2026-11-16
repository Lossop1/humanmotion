"""按《设计说明书_原子步骤》执行一条指令：A01–A13。

用法：
    python -m hm.run_job "一个人先挥手打招呼，然后向前走开" --k 2

只做到候选筛选与失败分类，不做装配与渲染——说明书第 6 节把 SMPL 与渲染放在
局部质量稳定之后，这里先保证「每一步可追溯」。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from hm import config, generate, to_joints
from hm import job as jobmod
from hm import planner, sb_schema

# A03 的动作可支持性：目前只有能力测试的支持清单与一个禁用清单。
# 禁用的依据是实测（蹲坐类生成成跪坐、头俯仰被解到 89 度），不是猜测。
UNSUPPORTED_WORDS = ("crouch", "squat", "sit", "sitting", "kneel", "lie", "lying")
CAPABILITY_VERSION = "capability_v1"


_CAP = {}


def _cap_lookup(prompt):
    """按提示词查能力表。返回 (status, model, 中文名) 或 None。"""
    import json
    from pathlib import Path
    if not _CAP:
        p = Path(__file__).resolve().parent.parent / "outputs" / "_reference" / "capability_v4.json"
        _CAP["doc"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    doc = _CAP["doc"]
    if not doc:
        return None
    text = str(prompt).lower()
    for key, v in doc["table"].items():
        words = [w for w in v["prompt"].lower().split()[:3] if len(w) > 2]
        if key.replace("_", " ") in text or v["action_zh"] in str(prompt) or any(w in text for w in words):
            models = v.get("models", {})
            if not models:
                return None
            best = v.get("best") or max(models, key=lambda m: models[m].get("pass_rate", 0))
            st = models[best].get("status", "unverified")
            st = {"support": "supported", "supported": "supported"}.get(st, st)
            return st, best, v.get("action_zh")
    return None


def step_a01(job, cfg):
    req = job.write_request(cfg)
    job.log("A01 指令与配置已保存  job_id=%s" % job.job_id)
    return req


def step_a02(job, sentence):
    """A02：长指令 → 结构化分镜。

    planner 目前只给 {label, prompt, frames}，并不切分原文，
    因此 source_text 暂时填整句并记一条 warning，不假装已对应到片段。
    """
    segs, trans, terrain = planner.plan(sentence)
    out = []
    warnings = []
    for i, s in enumerate(segs, 1):
        seg = sb_schema.blank_segment("seg_%03d" % i, i, s["label"], sentence)
        seg["duration_frames"] = int(s["frames"])
        seg["text_gen"] = s["prompt"]
        seg["provenance"] = {
            "action": "planner_llm(label)",
            "source_text": "fallback:whole_sentence",
            "duration_frames": "planner_llm",
            "text_gen": "planner_llm",
        }
        warnings.append("seg_%03d 的 source_text 暂用整句（planner 未切分原文）" % i)
        out.append(seg)
    sb = {
        "job_id": job.job_id,
        "sentence": sentence,
        "segments": out,
        "transitions": trans,
        "terrain": terrain,
        "schema_version": "storyboard_v1",
    }
    job.log("A02 分镜 %d 段" % len(out))
    return sb, warnings


def step_a03(job, sb):
    """A03：动作可支持性。无能力表时标 unverified，禁用词标 unsupported。"""
    rows = []
    for seg in sb["segments"]:
        prompt = seg.get("text_gen") or ""
        low = prompt.lower()
        hit = _cap_lookup(prompt)
        if hit is not None:
            status, model, zh = hit
            reason = "能力表：%s 最佳后端 %s" % (zh or "-", model)
        elif any(w in low for w in UNSUPPORTED_WORDS):
            status, reason = "unsupported", "命中禁用词表（实测生成质量不可靠）"
        else:
            status, reason = "unverified", "尚无该动作类别的能力测试数据"
        seg["support_status"] = status
        seg["support_reason"] = reason
        rows.append((seg["segment_id"], status, reason))
    job.log("A03 可支持性：" + "、".join("%s=%s" % (a, b) for a, b, _ in rows))
    return sb, rows


def step_a04(job, sb):
    """A04：Schema 校验。"""
    ok, errors, warns = sb_schema.validate(sb)
    if not ok:
        job.log("A04 校验失败：" + "; ".join(errors))
        raise SystemExit("分镜未通过 Schema 校验，停止执行")
    job.save_json("storyboard.json", sb)
    job.log("A04 校验通过（%d 段，%d 条提醒）" % (len(sb["segments"]), len(warns)))
    return sb


def step_a05(job, sb):
    """A05：模型路由。没有能力表，一律默认后端并把置信度标为低。"""
    for seg in sb["segments"]:
        seg["route"] = {
            "model": config.DEFAULT_MODEL,
            "model_version": "unknown",
            "confidence": "low",
            "reason": "尚无可按动作类别排序的能力表，采用默认后端",
            "table_version": CAPABILITY_VERSION,
        }
    job.log("A05 路由：全部默认 %s，置信度 low" % config.DEFAULT_MODEL)
    return sb


def step_a06(job, sb):
    """A06：提示词构造。目前沿用 planner 给的英文提示词，成对保存。"""
    for seg in sb["segments"]:
        seg["text_gen_final"] = seg["text_gen"]
        seg["template_version"] = "planner_prompt_v1"
    job.log("A06 提示词：沿用 planner 输出，已与 source_text 成对保存")
    return sb


def step_a07(job, sb, k):
    """A07：逐段生成 K 个候选，保存原始 263 维特征。

    MoMask 不接收随机种子（实测同一提示词两次结果不同），
    因此 seed 记 null、seed_control 记 unsupported，不假装可复现。
    """
    for seg in sb["segments"]:
        if seg.get("support_status") == "unsupported":
            seg["candidates"] = []
            job.log("A07 %s 跳过：动作不支持" % seg["segment_id"])
            continue
        prompt = seg["text_gen_final"]
        n = int(seg["duration_frames"] or 40)
        t0 = time.time()
        feats = generate.generate([prompt] * k, [n] * k)
        dt = round(time.time() - t0, 1)
        cands = []
        for i, F in enumerate(feats, 1):
            cid = "cand_%02d" % i
            d = job.cand_dir(seg["segment_id"], i)
            np.save(d / "motion263.npy", F)
            cands.append({
                "candidate_id": cid,
                "frames": int(len(F)),
                "requested_frames": n,
                "gen_seconds": dt,
                "seed": None,
                "seed_control": "unsupported",
            })
        seg["candidates"] = cands
        job.log("A07 %s 生成 %d 个候选，每批 %.1f 秒" % (seg["segment_id"], k, dt))
    return sb


def step_a08_a10(job, sb, ref):
    """A08 解码与几何检查，A09 姿态规则（未实现），A10 位移与朝向指标。"""
    for seg in sb["segments"]:
        results = []
        for c in seg.get("candidates", []):
            d = job.cand_dir(seg["segment_id"], int(c["candidate_id"].split("_")[-1]))
            F = np.load(d / "motion263.npy")
            J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
            np.save(d / "joints22.npy", J)

            hard = {"finite": bool(np.isfinite(J).all()),
                    "shape": bool(J.ndim == 3 and J.shape[1] == 22 and J.shape[2] == 3),
                    "bone_length": None}
            failures = []
            if not hard["finite"]:
                failures.append("含 NaN/Inf")
            if not hard["shape"]:
                failures.append("形状不符：%s" % (J.shape,))

            metrics = {"float": None, "mm_dist_original_text": None,
                       "foot_height_min": None, "root_path_length": None,
                       "root_step_median": None, "pose_rate_max": None,
                       "bone_length_sigma_max": None,
                       "bone_length_rel_mean_max": None,
                       "bone_length_jitter_m": None}
            reasons = {}

            if hard["finite"] and hard["shape"]:
                # 判据 v3：骨长 10.3% / 反向帧 / 穿模近似，由 96 段人工标注校准
                from hm import criteria_v2 as _crit
                ok, h2, m2, why2 = _crit.check(J, seg.get("text_gen_final", ""), ref, None)
                hard.update(h2)
                metrics.update({k: v for k, v in m2.items() if k not in metrics})
                failures.extend(why2)
                L = jobmod.bone_lengths(J)
                # 逐帧抖动：263→22 的还原不是刚性的，单帧骨长会在均值附近晃。
                # 这一项单独记为软指标，不当硬失败——硬检查针对的是系统性偏差。
                metrics["bone_length_jitter_m"] = float(L.std(axis=0).max())
                if ref is not None:
                    mean = np.array(ref["mean"])
                    std = np.array(ref["std"])
                    Lm = L.mean(axis=0)
                    rel_mean = np.abs(Lm - mean) / np.maximum(mean, 1e-9)
                    tol = float(ref.get("relative_tolerance", 0.05))
                    metrics["bone_length_rel_mean_max"] = float(rel_mean.max())
                    metrics["thresholds"] = {
                        "bone_length_rel_mean": tol,
                        "bone_length_rel_mean_source": ref.get("version", "unknown"),
                        "bone_length_jitter_m": 0.02,
                        "bone_length_jitter_source": "暂定值，待测试集校准后固定",
                    }
                    hard["bone_length_legacy"] = bool(rel_mean.max() <= tol)
                    if not hard["bone_length_legacy"]:
                        k = int(np.argmax(rel_mean))
                        failures.append("骨长均值相对偏差 %.1f%% 超出容差 %.0f%%（%s）"
                                        % (100 * rel_mean.max(), 100 * tol, jobmod.BONE_NAMES[k]))
                    if float(std.min()) > 1e-6:
                        sig = np.abs(Lm - mean) / np.maximum(std, 1e-6)
                        metrics["bone_length_sigma_max"] = float(sig.max())
                metrics["foot_height_min"] = float(min(J[:, 10, 1].min(), J[:, 11, 1].min()))
                tr = J[:, 0, [0, 2]]
                step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
                metrics["root_path_length"] = float(step.sum())
                metrics["root_step_median"] = float(np.median(step)) if len(step) else 0.0
                d0 = np.array([J[1:, c] - J[1:, p] for c, p in jobmod.BONES]).transpose(1, 0, 2)
                d0 = d0 / np.maximum(np.linalg.norm(d0, axis=2, keepdims=True), 1e-9)
                d1 = np.array([J[:-1, c] - J[:-1, p] for c, p in jobmod.BONES]).transpose(1, 0, 2)
                d1 = d1 / np.maximum(np.linalg.norm(d1, axis=2, keepdims=True), 1e-9)
                ang = np.degrees(np.arccos(np.clip((d0 * d1).sum(axis=2), -1, 1)))
                metrics["pose_rate_max"] = float(ang.max()) if ang.size else 0.0
            else:
                reasons["bone_length"] = "几何检查未通过，未计算"
                reasons["foot_height_min"] = "几何检查未通过，未计算"

            reasons["float"] = "Float 指标未接入运行环境，未计算"
            reasons["mm_dist_original_text"] = "语义评测器未接入，未计算"

            keys_v3 = ("finite", "shape", "bone_length", "root_facing", "separation")
            passed = all(hard.get(k) is True for k in keys_v3 if hard.get(k) is not None)
            m = {
                "job_id": job.job_id,
                "segment_id": seg["segment_id"],
                "candidate_id": c["candidate_id"],
                "stage": "decoded",
                "model": seg["route"]["model"],
                "model_version": seg["route"]["model_version"],
                "seed": None,
                "seed_control": "unsupported",
                "frames": int(len(J)),
                "fps": 20,
                "coordinate_system": "HumanML3D",
                "hard_checks": hard,
                "metrics": metrics,
                "metrics_not_computed_reasons": reasons,
                "decision": "pass" if passed else "reject",
                "failure_reasons": failures,
            }
            job.write_candidate_metrics(seg["segment_id"], c["candidate_id"], m)
            results.append(m)
        seg["candidate_metrics"] = [
            {k: v for k, v in r.items() if k in ("candidate_id", "decision", "hard_checks", "metrics")}
            for r in results
        ]
        job.log("A08-A10 %s 检查完成：%d 个候选，%d 个通过硬检查"
                % (seg["segment_id"], len(results), sum(1 for r in results if r["decision"] == "pass")))
    return sb


def step_a12_a13(job, sb, max_retry=2):
    """A12 淘汰硬失败并排序；A13 对全失败的段做分类。"""
    for seg in sb["segments"]:
        cands = seg.get("candidate_metrics") or []
        if not cands:
            seg["selected"] = None
            seg["segment_decision"] = seg.get("support_status", "unsupported")
            job.append_event({"event": "segment_skipped", "segment_id": seg["segment_id"],
                              "reason": seg.get("support_reason", "无候选")})
            continue
        ok = [c for c in cands if c["decision"] == "pass"]
        if not ok:
            # A13：按说明书失败表给一个主要类型
            reasons = []
            for c in cands:
                reasons.extend(c["hard_checks"] and [] or [])
            has_geom = any(not c["hard_checks"].get("shape", True) or
                           not c["hard_checks"].get("bone_length", True) for c in cands)
            ftype = "数据/坐标错误" if has_geom else "模型能力不足"
            seg["selected"] = None
            seg["segment_decision"] = "reject"
            seg["failure_type"] = ftype
            seg["failure_evidence"] = "所有候选均未通过硬检查"
            job.append_event({"event": "segment_failed", "segment_id": seg["segment_id"],
                              "failure_type": ftype, "candidates": len(cands)})
            job.log("A13 %s 全失败，类型：%s" % (seg["segment_id"], ftype))
            continue
        # 软排序：骨长偏差优先，其次单帧姿态变化，再其次路径长度
        def key(c):
            m = c["metrics"]
            return (m.get("bone_length_jitter_m") or 9e9,
                    m.get("pose_rate_max") or 9e9,
                    m.get("root_path_length") or 9e9)
        ok.sort(key=key)
        seg["selected"] = ok[0]["candidate_id"]
        seg["segment_decision"] = "pass"
        seg["ranking"] = [c["candidate_id"] for c in ok]
        job.log("A12 %s 选中 %s（排序依据：骨长偏差、单帧姿态变化、路径长度）"
                % (seg["segment_id"], seg["selected"]))
    job.save_json("storyboard.json", sb)
    return sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--k", type=int, default=2, help="每段候选数")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--no-gen", action="store_true", help="只跑到 A06，不生成")
    a = ap.parse_args()

    cfg = {
        "model": config.DEFAULT_MODEL,
        "k": a.k,
        "fps": 20,
        "coordinate_system": "HumanML3D",
        "seed_control": "unsupported (backend does not accept seed)",
        "bone_reference": None,
        "capability_table": CAPABILITY_VERSION,
        "schema": "storyboard_v1",
    }

    ref = jobmod.load_bone_reference()
    if ref is None:
        jobmod.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        ref, p = jobmod.build_bone_reference()
        print("  建立了骨长参考：%s（%d 条真实序列，%d 帧）" % (p.name, ref["files"], ref["frames"]))
    cfg["bone_reference"] = ref["version"]

    job = jobmod.Job(a.sentence, job_id=a.job_id)
    print("  任务目录：%s" % job.dir)

    step_a01(job, cfg)
    sb, warns = step_a02(job, a.sentence)
    for w in warns:
        job.log("提醒：" + w)
    sb, _ = step_a03(job, sb)
    step_a04(job, sb)
    step_a05(job, sb)
    step_a06(job, sb)
    job.save_json("storyboard.json", sb)

    if a.no_gen:
        job.log("按 --no-gen 停在 A06")
        print("  完成（未生成）")
        return

    step_a07(job, sb, a.k)
    step_a08_a10(job, sb, ref)
    step_a12_a13(job, sb)

    passed = sum(1 for s in sb["segments"] if s.get("segment_decision") == "pass")
    job.log("汇总：%d/%d 段通过" % (passed, len(sb["segments"])))
    print("\n  产出目录：%s" % job.dir)
    print("  通过 %d/%d 段" % (passed, len(sb["segments"])))
    for s in sb["segments"]:
        print("    %s  %-14s  %s  候选=%s  选中=%s"
              % (s["segment_id"], s["action"][:14], s["segment_decision"],
                 len(s.get("candidates") or []), s.get("selected")))


if __name__ == "__main__":
    main()
