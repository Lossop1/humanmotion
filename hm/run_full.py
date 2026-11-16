"""把原子表走完：A01–A18。

在 run_job 的 A01–A12 之上补 A09（姿态规则）、A11（语义辅助与人工标注位）、
A13–A15（失败分类与局部重生成），最后接 A16–A18（装配、拟合、渲染）。

用法：
    python -m hm.run_full "一个人先挥手打招呼，然后向前走开" --k 2 --job-id job_full01
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hm import config, finish_job, job as jobmod, pose_rules, repair, run_job, to_joints

MAX_ROUNDS = 2


def file_hash(p):
    h = hashlib.md5()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def step_a09(job, sb, pose_th):
    """A09 姿态退化检查，结果写进该段每个候选的 metrics。"""
    for seg in sb["segments"]:
        for c in seg.get("candidates", []):
            n = int(c["candidate_id"].split("_")[-1])
            d = job.cand_dir(seg["segment_id"], n)
            J = np.load(d / "joints22.npy")
            passed, reasons, bad, values = pose_rules.check(J, seg.get("text_gen_final", ""), pose_th)
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            m["pose_checks"] = {"passed": passed, "reasons": reasons,
                                "bad_frames": bad, "values": values,
                                "rule_version": pose_rules.RULE_VERSION}
            (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
            if not passed:
                m["hard_checks"]["pose"] = False
                m["failure_reasons"].append("姿态规则不通过")
                m["decision"] = "reject"
                (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
        job.log("A09 %s 姿态检查完成" % seg["segment_id"])


def step_a11(job, sb):
    """A11 语义辅助：几何签名 + 人工标注位。不把自动指标当通过标准。"""
    labels_path = job.dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    for seg in sb["segments"]:
        for c in seg.get("candidates", []):
            n = int(c["candidate_id"].split("_")[-1])
            d = job.cand_dir(seg["segment_id"], n)
            J = np.load(d / "joints22.npy")
            passed, rule, detail = repair.semantic_proxy(J, seg.get("text_gen_final", ""))
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            m["metrics"]["semantic_proxy"] = passed
            m["metrics"]["semantic_rule"] = rule
            m["metrics"]["semantic_detail"] = detail
            m["metrics"]["mm_dist_original_text"] = None
            m["metrics_not_computed_reasons"]["mm_dist_original_text"] = (
                "标准语义评测器未接入；改用几何签名 + 人工标注（说明书 A11 兜底条款）")
            key = "%s/%s" % (seg["segment_id"], c["candidate_id"])
            m["metrics"]["human_label"] = labels.get(key, None)
            (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        sigs = [c.get("semantic_rule") for c in
                [json.loads((job.cand_dir(seg["segment_id"], int(c["candidate_id"].split("_")[-1]))
                             / "metrics.json").read_text(encoding="utf-8"))
                 for c in seg.get("candidates", [])]]
        job.log("A11 %s 语义辅助：%s" % (seg["segment_id"], "、".join(str(s) for s in sigs) or "无候选"))

    if not labels_path.exists():
        labels_path.write_text(json.dumps({
            "_说明": "人工标注位。键为 segment_id/candidate_id，值为 pass 或 fail。"
                     "留空表示尚未人工复核，程序不代填。",
            "_待填": "示例：\"seg_001/cand_01\": \"pass\"",
        }, ensure_ascii=False, indent=1), encoding="utf-8")


def reselect_with_pose(job, sb):
    """A12 重排：硬检查含姿态规则；软指标仍按抖动与单帧姿态变化。"""
    for seg in sb["segments"]:
        rows = []
        for c in seg.get("candidates", []):
            n = int(c["candidate_id"].split("_")[-1])
            m = json.loads((job.cand_dir(seg["segment_id"], n) / "metrics.json").read_text(encoding="utf-8"))
            rows.append(m)
        ok = [m for m in rows if m["decision"] == "pass"]
        if not ok:
            seg["selected"] = None
            seg["segment_decision"] = "reject"
            continue
        ok.sort(key=lambda m: (m["metrics"].get("bone_length_jitter_m") or 9e9,
                               m["metrics"].get("pose_rate_max") or 9e9,
                               m["metrics"].get("root_path_length") or 9e9))
        seg["selected"] = ok[0]["candidate_id"]
        seg["segment_decision"] = "pass"
        seg["ranking"] = [m["candidate_id"] for m in ok]
        job.log("A12 %s 选中 %s" % (seg["segment_id"], seg["selected"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    a = ap.parse_args()

    cfg = {"model": config.DEFAULT_MODEL, "k": a.k, "fps": 20,
           "coordinate_system": "HumanML3D",
           "seed_control": "unsupported (backend does not accept seed)",
           "capability_table": run_job.CAPABILITY_VERSION,
           "pose_rules": pose_rules.RULE_VERSION,
           "schema": "storyboard_v1"}

    ref = jobmod.load_bone_reference()
    if ref is None:
        ref, _ = jobmod.build_bone_reference()
    cfg["bone_reference"] = ref["version"]
    pose_th = pose_rules.load_thresholds()

    job = jobmod.Job(a.sentence, job_id=a.job_id)
    print("  任务目录：%s" % job.dir)

    run_job.step_a01(job, cfg)
    sb, warns = run_job.step_a02(job, a.sentence)
    for w in warns:
        job.log("提醒：" + w)
    sb, _ = run_job.step_a03(job, sb)
    run_job.step_a04(job, sb)
    run_job.step_a05(job, sb)
    run_job.step_a06(job, sb)
    job.save_json("storyboard.json", sb)

    for rnd in range(a.max_rounds + 1):
        todo = [s for s in sb["segments"]
                if s.get("segment_decision") != "pass" and s.get("support_status") != "unsupported"]
        if not todo:
            break
        job.log("第 %d 轮生成与检查（%d 段）" % (rnd, len(todo)))
        hashes = {}
        for s in sb["segments"]:
            if s.get("selected"):
                n = int(s["selected"].split("_")[-1])
                p = job.cand_dir(s["segment_id"], n) / "motion263.npy"
                hashes[s["segment_id"]] = file_hash(p)

        kk = a.k if rnd == 0 else a.k * 2
        for s in todo:
            s["candidates"] = []
        run_job.step_a07(job, {"segments": todo}, kk)
        run_job.step_a08_a10(job, {"segments": todo}, ref)
        step_a09(job, {"segments": todo}, pose_th)
        step_a11(job, {"segments": todo})
        reselect_with_pose(job, {"segments": todo})
        job.save_json("storyboard.json", sb)

        # A15 校验：已通过的片段文件必须没变
        changed = []
        for sid, h in hashes.items():
            if sid in [t["segment_id"] for t in todo]:
                continue
            seg = [x for x in sb["segments"] if x["segment_id"] == sid][0]
            n = int(seg["selected"].split("_")[-1])
            if file_hash(job.cand_dir(sid, n) / "motion263.npy") != h:
                changed.append(sid)
        if changed:
            job.log("A15 警告：局部重生成改动了未指定的片段 %s" % changed)

        failed = [s for s in sb["segments"] if s.get("segment_decision") != "pass"
                  and s.get("support_status") != "unsupported"]
        if not failed:
            break
        # A13 分类 + A14 计划
        for s in failed:
            ms = [json.loads((job.cand_dir(s["segment_id"], int(c["candidate_id"].split("_")[-1]))
                              / "metrics.json").read_text(encoding="utf-8"))
                  for c in s.get("candidates", [])]
            reasons = []
            for m in ms:
                reasons.extend(m.get("pose_checks", {}).get("reasons", []))
            ftype, ev = repair.classify(ms, reasons, None)
            s["failure_type"] = ftype
            s["failure_evidence"] = ev
            plan = repair.plan_repair(ftype, rnd, a.max_rounds)
            job.append_event({"event": "repair_plan", "segment_id": s["segment_id"],
                              "failure_type": ftype, "evidence": ev,
                              "plan": plan})
            job.log("A13 %s 失败类型：%s；A14 计划：%s"
                    % (s["segment_id"], ftype, plan["action"] if plan else "不修复"))
            if plan is None:
                s["segment_decision"] = "manual_review"

    passed = sum(1 for s in sb["segments"] if s.get("segment_decision") == "pass")
    job.log("A01–A15 汇总：%d/%d 段通过" % (passed, len(sb["segments"])))
    job.save_json("storyboard.json", sb)

    if passed == 0:
        print("\n  没有通过检查的片段，停在 A15。")
        print("  任务目录：%s" % job.dir)
        return

    from hm import assemble_params as AP
    mp4, bj = AP.run(job, sb)
    print("\n  A01–A18 全部走完")
    print("  合体视频：%s" % mp4)
    print("  任务目录：%s" % job.dir)


if __name__ == "__main__":
    main()
