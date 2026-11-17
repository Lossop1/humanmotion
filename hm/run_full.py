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
    """A12：只做一次选择，交给 run_job.step_a12_a13（唯一实现）。"""
    import json as _j
    for seg in sb["segments"]:
        rows = []
        for c in seg.get("candidates", []):
            n = int(c["candidate_id"].split("_")[-1])
            mp = job.cand_dir(seg["segment_id"], n) / "metrics.json"
            try:
                rows.append(_j.loads(mp.read_text(encoding="utf-8")))
            except Exception:
                continue
        seg["candidate_metrics"] = rows
    run_job.step_a12_a13(job, sb)


def plan_only(sentence, job_id=None):
    """只跑到 A06（规划与选词），不生成。用于 --explain。"""
    cfg = {"model": config.DEFAULT_MODEL, "k": None, "fps": 20,
           "coordinate_system": "HumanML3D", "schema": "storyboard_v1"}
    job = jobmod.Job(sentence, job_id=job_id)
    run_job.step_a01(job, cfg)
    sb, warns = run_job.step_a02(job, sentence)
    for w in warns:
        job.log("提醒：" + w)
    sb, _ = run_job.step_a03(job, sb)
    run_job.step_a04(job, sb)
    run_job.step_a05(job, sb)
    run_job.step_a06(job, sb)
    job.save_json("storyboard.json", sb)
    return job, sb


def run(sentence, k=6, job_id=None, max_rounds=MAX_ROUNDS):
    """跑完整条流水线。返回 (视频路径, bounds 字典)；无段通过时返回 (None, None)。"""
    cfg = {"model": config.DEFAULT_MODEL, "k": k, "fps": 20,
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

    job = jobmod.Job(sentence, job_id=job_id)
    print("  任务目录：%s" % job.dir)

    run_job.step_a01(job, cfg)
    sb, warns = run_job.step_a02(job, sentence)
    for w in warns:
        job.log("提醒：" + w)
    sb, _ = run_job.step_a03(job, sb)
    run_job.step_a04(job, sb)
    run_job.step_a05(job, sb)
    run_job.step_a06(job, sb)
    run_job.merge_short_segments(job, sb)
    job.save_json("storyboard.json", sb)

    for rnd in range(max_rounds + 1):
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

        kk = k if rnd == 0 else k * 2
        # 重试时同时加长片段：实测同一提示词 40 帧 1/4 通过、44 帧 4/4，
        # 帧数太短会把动作压成「骨盆朝向与脚尖反向」这种错误姿态。
        if rnd > 0:
            for s in todo:
                old_n = int(s.get("duration_frames") or 40)
                s["duration_frames"] = min(old_n + 8, 80)
                job.log("重试：%s 帧数 %d → %d" % (s["segment_id"], old_n, s["duration_frames"]))
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
            plan = repair.plan_repair(ftype, rnd, max_rounds)
            job.append_event({"event": "repair_plan", "segment_id": s["segment_id"],
                              "failure_type": ftype, "evidence": ev,
                              "plan": plan})
            job.log("A13 %s 失败类型：%s；A14 计划：%s"
                    % (s["segment_id"], ftype, plan["action"] if plan else "不修复"))
            if plan is None:
                s["segment_decision"] = "manual_review"

    # 合并段若没有候选能同时做出两个动作，退回未合并的两段重新生成：
    # 合并是为了去掉交界，但不能以悄悄丢掉一个动作为代价。
    # 合并段若一个候选都没过，也算失败：退回未合并重来，别整段丢掉
    import json as _json2
    for _seg in sb["segments"]:
        if not _seg.get("merged_from") or _seg.get("merged_unverified"):
            continue
        _any = False
        for _c in _seg.get("candidates", []):
            _n = int(_c["candidate_id"].split("_")[-1])
            _mp = job.cand_dir(_seg["segment_id"], _n) / "metrics.json"
            try:
                if _json2.loads(_mp.read_text(encoding="utf-8")).get("decision") == "pass":
                    _any = True
                    break
            except Exception:
                continue
        if not _any:
            _seg["merged_unverified"] = True
            job.log("合并段 %s 无候选通过，退回未合并" % _seg["segment_id"])

    _bad = [s for s in sb["segments"] if s.get("merged_unverified")]
    if _bad and sb.get("pre_merge"):
        job.log("退回未合并：%d 个合并段没有任何候选同时做出两个动作" % len(_bad))
        _orig = {s["segment_id"]: s for s in sb["pre_merge"]}
        _new = []
        for seg in sb["segments"]:
            if seg.get("merged_unverified"):
                for pid in seg["merged_from"]:
                    p0 = _orig.get(pid)
                    if p0 is None:
                        continue
                    p0["candidates"] = []
                    p0["selected"] = None
                    p0["segment_decision"] = None
                    # 退回后要给足长度：实测「单手举」40 帧 1/4 通过、44 帧 4/4，
                    # 退回时按 48 帧起步，避免刚退回又因太短全失败
                    p0["duration_frames"] = max(int(p0.get("duration_frames") or 40), 48)
                    _new.append(p0)
            else:
                _new.append(seg)
        sb["segments"] = _new
        run_job.step_a07(job, sb, k)
        run_job.step_a08_a10(job, sb, ref)
        reselect_with_pose(job, sb)
        job.save_json("storyboard.json", sb)

    # 交界修复：只对差得大的相邻段补候选，针对性花代价
    try:
        _fixed = run_job.repair_junctions(job, sb, ref, pose_th, k_extra=max(4, k), gap_limit=90.0)
        if _fixed:
            job.log("交界修复：改选 %d 处" % _fixed)
    except Exception as _e:
        job.log("交界修复跳过：%s" % str(_e)[:80])

    passed = sum(1 for s in sb["segments"] if s.get("segment_decision") == "pass")
    job.log("A01–A15 汇总：%d/%d 段通过" % (passed, len(sb["segments"])))
    job.save_json("storyboard.json", sb)

    if passed == 0:
        print("\n  没有通过检查的片段，停在 A15。")
        print("  任务目录：%s" % job.dir)
        return None, None, str(job.dir)

    from hm import assemble_params as AP
    mp4, bj = AP.run(job, sb)
    print("\n  A01–A18 全部走完")
    print("  合体视频：%s" % mp4)
    print("  任务目录：%s" % job.dir)
    return mp4, bj, str(job.dir)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    a = ap.parse_args()
    mp4, bj = run(a.sentence, k=a.k, job_id=a.job_id, max_rounds=a.max_rounds)
    raise SystemExit(0 if mp4 else 2)


if __name__ == "__main__":
    main()
