"""按《设计说明书_原子步骤》执行一条指令：A01–A13。

用法：
    python -m hm.run_job "一个人先挥手打招呼，然后向前走开" --k 2

只做到候选筛选与失败分类，不做装配与渲染——说明书第 6 节把 SMPL 与渲染放在
局部质量稳定之后，这里先保证「每一步可追溯」。
"""
import argparse
import re
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
    """按提示词查能力表。复用 backends 的匹配与 prompt_scope 规则，避免两套实现。"""
    from hm import backends
    table = backends.load_table()
    _key, entry = backends.match_entry(table, prompt)
    if not entry:
        return None
    score = backends.BACKENDS
    models = {m: d for m, d in (entry.get("models") or {}).items() if m in score}
    if not models:
        return None
    if entry.get("prompt_scope") == "composite":
        return None
    best = entry.get("best") or max(models, key=lambda m: models[m].get("pass_rate", 0))
    st = models[best].get("status", "unverified")
    st = {"support": "supported"}.get(st, st)
    return st, best, entry.get("action_zh")


_FILLERS = ("一个人", "然后", "接着", "最后", "再", "先", "一边", "并且", "着", "了", "的")


def _norm_text(t):
    """去掉虚词与标点，只留下承载动作的字，用于算原文覆盖率。"""
    import re as _re
    for w in _FILLERS:
        t = t.replace(w, "")
    return _re.sub(r"[，,。.、；;！!？?\s]", "", t)


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
    covered = 0
    for i, s in enumerate(segs, 1):
        src = s.get("source") or ""
        seg = sb_schema.blank_segment("seg_%03d" % i, i, s["label"], src or sentence)
        seg["duration_frames"] = int(s["frames"])
        seg["text_gen"] = s["prompt"]
        seg["facing"] = s.get("direction")
        seg["object"] = s.get("object")
        seg["relation"] = s.get("relation")
        seg["speed"] = s.get("speed")
        seg["provenance"] = {
            "action": "planner_llm(label)",
            "source_text": "planner_llm(substring)" if src else "fallback:whole_sentence",
            "duration_frames": "planner_llm",
            "text_gen": "planner_llm",
            "facing": "planner_llm" if s.get("direction") else "null",
        }
        if src:
            covered += len(src)
        else:
            warnings.append("seg_%03d 未给出原文片段，source_text 退回整句" % i)
        out.append(seg)
    coverage = len(_norm_text("".join(x.get("source") or "" for x in segs))) / max(
        len(_norm_text(sentence)), 1)
    if coverage < 0.6:
        warnings.append("原文覆盖率 %.0f%%（各段 source 拼接占原句比例），可能有内容未规划"
                        % (100 * coverage))
    job.log("A02 原文覆盖率 %.0f%%（%d 段）" % (100 * coverage, len(out)))
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
        elif (_bank := __import__("hm.prompt_variants", fromlist=["bank_rate"]).bank_rate(prompt)) \
                and _bank[1] == 0:
            status = "unsupported"
            reason = "提示词库实测 0/4（%s），生成不出来" % _bank[0]
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
    """A05：按能力表逐段选后端。未注册的后端直接拒绝，不静默回退。"""
    from hm import backends
    picked = []
    for seg in sb["segments"]:
        prompt = seg.get("text_gen") or ""
        model, status, reason, detail = backends.resolve(prompt)
        spec = backends.BACKENDS.get(model, {})
        seg["route"] = {
            "model": model,
            "model_version": "unknown",
            "confidence": "high" if status == "support" or status == "supported" else "low",
            "reason": reason,
            "support_status": status,
            "pass_rate": detail.get("pass_rate"),
            "max_frames": spec.get("max_frames"),
            "table_version": CAPABILITY_VERSION,
        }
        picked.append("%s→%s(%s)" % (seg["segment_id"], model, status))
        # 能力表里帧数上限更短时，截短，不交给后端去报错
        cap = spec.get("max_frames")
        if cap and seg.get("duration_frames") and int(seg["duration_frames"]) > int(cap):
            job.log("A05 %s 帧数 %d 超过 %s 上限 %d，截到上限"
                    % (seg["segment_id"], int(seg["duration_frames"]), model, int(cap)))
            seg["duration_frames"] = int(cap)
    job.log("A05 路由（按能力表逐段选）：" + "、".join(picked))
    return sb


def step_a06(job, sb):
    """A06：提示词构造。

    规则：如果能力表里有这个动作，并且它在这个后端上是被验证过的，
    就用能力表里那句**被测过的措辞**，而不是 planner 现场编的措辞。
    措辞差异对生成质量的影响，比我们想象的大的多（坐下就踩过）。
    """
    from hm import backends as _bk
    table = _bk.load_table()
    reused = 0
    for seg in sb["segments"]:
        prompt = seg["text_gen"]
        model = (seg.get("route") or {}).get("model")
        _key, entry = _bk.match_entry(table, prompt)
        final, version = prompt, "planner_prompt_v1"
        if entry:
            m = (entry.get("models") or {}).get(model) or {}
            status = {"support": "supported"}.get(m.get("status"), m.get("status"))
            tested = entry.get("prompt")
            # 能力表里可能登记的是复合动作（"sits down and stands back up"），
            # 直接复用会把一段变成两个动作，所以带连接词的一律不复用
            compound = ((" and " in (" %s " % tested.lower()))
                        if tested else False) or entry.get("prompt_scope") == "composite"
            # 只在「测过的措辞是 planner 措辞的子集」时复用，
            # 否则会把「stands up」换成「stands still」这种语义不同的写法
            def _tok(x):
                _stop = {"a", "the", "person", "with", "and", "in", "on", "to", "of", "up", "down",
                         "forward", "backward", "left", "right", "place", "from", "at", "both"}
                return {w for w in re.findall(r"[a-z]+", str(x).lower())
                        if w not in _stop and len(w) > 2}
            _subset = bool(tested) and _tok(tested) <= _tok(prompt)
            if (tested and not compound and _subset and status == "supported"
                    and tested.lower() != prompt.lower()):
                final, version = tested, "capability_tested_prompt"
                reused += 1
                job.log("A06 %s 采用能力表测过的提示词：%r（planner 原为 %r）"
                        % (seg["segment_id"], tested, prompt))
        seg["text_gen_final"] = final
        seg["template_version"] = version
    # 第二道：提示词库。planner 的措辞不如库里测过的写法稳时，用库里的。
    try:
        import json as _json
        from pathlib import Path as _P
        _bp = _P(__file__).resolve().parent.parent / "outputs" / "_reference" / "prompt_bank.json"
        bank = _json.loads(_bp.read_text(encoding="utf-8")) if _bp.exists() else {}
    except Exception:
        bank = {}
    from_bank = 0
    import os as _os_bank
    if bank and _os_bank.environ.get("HM_BANK_SUBSTITUTE", "0") == "1":
        _stop = {"a", "the", "person", "with", "and", "in", "on", "to", "of", "up",
                 "down", "forward", "backward", "left", "right", "once", "his", "her"}
        _DIRS = {"forward", "backward", "left", "right", "up", "down", "back"}

        def _toks(t):
            return {w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in _stop and len(w) > 2}

        def _dirs_of(t):
            # 「with the right hand」里的 right 指身体侧，不是行进方向，先剔除
            x = re.sub(r"(right|left)\s+(hand|leg|arm|fist|foot)", " ", str(t).lower())
            return {w for w in re.findall(r"[a-z]+", x) if w in _DIRS}
        for seg in sb["segments"]:
            cur = seg.get("text_gen_final") or ""
            cur_toks = _toks(cur)
            best_item, best_rate = None, 0.0
            for action, entry in bank.items():
                if entry.get("best_rate", 0) < 0.75:
                    continue
                keys = _toks(action.replace("_", " ")) | _toks(entry.get("best", ""))
                if not (keys & cur_toks):
                    continue
                # 方向必须一致，否则「向前走」会被换成本库里「向后走」的写法
                if _dirs_of(entry.get("best", "")) != _dirs_of(cur):
                    continue
                if entry["best_rate"] > best_rate:
                    best_item, best_rate = entry["best"], entry["best_rate"]
            # 只在 planner 的写法「没测过」时才替换；它自己测过且不差就用它，
            # 否则会把本来对的措辞也改掉（实测替换 18 段后交界中位反而升到 50 度）
            _cur_tested = None
            for _a, _e in bank.items():
                for _r in _e.get("phrasings", []):
                    if _r["text"].strip().lower() == cur.strip().lower():
                        _cur_tested = max((m.get("rate", 0) for m in _r["models"].values()),
                                          default=0)
                        break
                if _cur_tested is not None:
                    break
            if _cur_tested is not None and _cur_tested >= 0.5:
                best_item = None
            if best_item and best_item.strip().lower() != cur.strip().lower():
                seg["text_gen_final"] = best_item
                seg["template_version"] = "prompt_bank"
                from_bank += 1
                job.log("A06 %s 改用提示词库最优写法：%r（planner 原为 %r）"
                        % (seg["segment_id"], best_item, cur))
    job.log("A06 提示词：%d/%d 段沿用能力表测过的措辞，%d/%d 段改用提示词库最优写法"
            % (reused, len(sb["segments"]), from_bank, len(sb["segments"])))
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
        model = (seg.get("route") or {}).get("model") or config.DEFAULT_MODEL
        t0 = time.time()
        from hm import cache as _cache
        ck = _cache.key(("gen", model, prompt, n, k))
        cached = _cache.get_json("gen", ck) if _cache.enabled() else None
        if cached is not None:
            feats = [np.load(_cache.get_file("gen", ck, "cand%d.npy" % i))
                     for i in range(len(cached["shapes"]))]
            hit = True
        else:
            feats = generate.generate([prompt] * k, [n] * k, name=model)
            hit = False
            if _cache.enabled():
                import tempfile
                tmpd = Path(tempfile.mkdtemp())
                for i, F in enumerate(feats):
                    np.save(tmpd / ("c%d.npy" % i), np.asarray(F))
                    _cache.put_file("gen", ck, "cand%d.npy" % i, tmpd / ("c%d.npy" % i))
                _cache.put_json("gen", ck, {"shapes": [list(np.shape(np.asarray(F))) for F in feats],
                                            "model": model, "prompt": prompt,
                                            "frames": n, "k": k})
        dt = round(time.time() - t0, 1)
        # 同一动作再取库里另外两种写法，一起交给判据挑（措辞差异实测影响很大）
        from hm import prompt_variants as _pv
        # 库里其它写法默认不追加：实测三次自动替换都把语义改错了
        # （crawl→push、stands up→stands still、stand up→pick），
        # 措辞机制先只保留「实测表 + 死路识别」，需要时用 HM_BANK_VARIANTS=1 打开。
        import os as _os_var
        _vars = _pv.bank_alternatives(prompt) if _os_var.environ.get("HM_BANK_VARIANTS", "0") == "1" else []
        for _pv_text in _vars:
            try:
                _more = generate.generate([_pv_text] * max(2, k // 2),
                                          [n] * max(2, k // 2), name=model)
            except Exception as _e:
                job.log("A07 %s 变体生成失败：%s" % (seg["segment_id"], str(_e)[:60]))
                continue
            feats = list(feats) + list(_more)
            job.log("A07 %s 追加变体写法：%r（+%d 个候选）"
                    % (seg["segment_id"], _pv_text, len(_more)))
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
        job.log("A07 %s 生成 %d 个候选，每批 %.1f 秒%s" % (seg["segment_id"], len(feats), dt,
                                                   "（缓存）" if hit else ""))
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


def _goal_bonus(job, seg, cand):
    """动作偏好分：越接近目标动作分越低。只用批内相对量，不用绝对阈值。"""
    import re as _re
    from hm import signature as _sig
    prompt = (seg.get('text_gen_final') or '').lower()
    n = int(cand['candidate_id'].split('_')[-1])
    J = np.load(job.dir / 'segments' / seg['segment_id'] / ('candidate_%d' % n) / 'joints22.npy')
    s = _sig.signature(J)
    score = 0.0
    if _re.search(r'punch|kick|throw', prompt):
        score -= 3.0 * s['punch_reach']
    if _re.search(r'jump|hop', prompt):
        score -= 2.0 * s['pelvis_up']
    # 站距偏离 0.35 米越多越差（0.35 取自站立与并步的常见值）
    score += 2.0 * abs(s['stance_width'] - 0.35)
    return score


STOP_MIN_FRAMES = 64


def ensure_stop_frames(job, sb):
    """含「停住/站定」的段要给足帧数。

    实测同一提示词 «stops walking and stands still»：44 帧末段速度 1.53、56 帧 1.68（都没停），
    72 帧才 0.06（6/6 通过）。帧数不够时模型会一直走，而不是先走再停。
    """
    import re as _re
    for seg in sb["segments"]:
        prompt = seg.get("text_gen_final") or ""
        if _re.search(r"stops?|stands? still|com(e|es) to a stop|halts?", prompt.lower()):
            old_n = int(seg.get("duration_frames") or 40)
            if old_n < STOP_MIN_FRAMES:
                seg["duration_frames"] = STOP_MIN_FRAMES
                job.log("%s 含「停住」，帧数 %d → %d" % (seg["segment_id"], old_n, STOP_MIN_FRAMES))
    return sb


def merge_short_segments(job, sb, limit_frames=56):
    """把两个相邻短段合成一段，用复合提示词生成（HM_MERGE_SHORT=1 时启用）。

    为什么要合并
        交界压不下去是因为两侧姿态差得远；合并后就没有交界。
        实测「停下→举手」「跳→转身」两个复合提示词各 6/6 几何通过、两动作都做出来。
    代价
        合并段无法再逐动作做签名检查，只能整体看几何；出问题更难定位。
    """
    import os as _os
    if _os.environ.get("HM_MERGE_SHORT", "1") == "0":
        return sb
    out = []
    originals = list(sb.get("pre_merge") or [])
    i = 0
    segs = sb["segments"]
    while i < len(segs):
        a = segs[i]
        if i + 1 < len(segs):
            b = segs[i + 1]
            fa = int(a.get("duration_frames") or 40)
            fb = int(b.get("duration_frames") or 40)
            # 合并条件按「总长」判：单段 56 帧的上限会把补过帧的「停住」段排除在外，
            # 而停住段恰恰需要更长的片段才做得出来。
            # 已经是复合描述的提示词（含 and）不再合并：文学文本里两段各含动作，
            # 合起来变成一句长句，模型只能给出泛化动作（实测武松遇虎那段）
            _compound = (" and " in (a.get("text_gen_final") or "").lower()
                         or " and " in (b.get("text_gen_final") or "").lower())
            if not _compound and (fa <= limit_frames and fb <= limit_frames
                                  or (fa + fb) <= 120):
                pa = (a.get("text_gen_final") or "").strip()
                pb = (b.get("text_gen_final") or "").strip()
                body_b = pb[len("a person "):] if pb.startswith("a person ") else pb
                merged = dict(a)
                merged["text_gen"] = "%s and %s" % (pa, body_b)
                merged["text_gen_final"] = merged["text_gen"]
                merged["duration_frames"] = fa + fb
                merged["source_text"] = (a.get("source_text") or "") + (b.get("source_text") or "")
                merged["merged_from"] = [a["segment_id"], b["segment_id"]]
                import copy as _copy
                originals.extend([_copy.deepcopy(a), _copy.deepcopy(b)])
                # 合并后要重新判定可支持性：单动作「蹲下」做不出来，
                # 但「蹲下再站起」整体是能力表 10/10 的条目，不能沿用旧结论
                merged["support_status"] = "unverified"
                merged["support_reason"] = "合并段，需重新判定"
                if not merged.get("route"):
                    merged["route"] = a.get("route") or {}
                merged["selected"] = None
                merged["segment_decision"] = None
                merged["candidates"] = []
                out.append(merged)
                job.log("合并短段：%s + %s → %d 帧，提示词 %r"
                        % (a["segment_id"], b["segment_id"], fa + fb, merged["text_gen"]))
                i += 2
                continue
        out.append(a)
        i += 1
    sb["segments"] = out
    sb["pre_merge"] = originals
    return sb


def _merged_parts_ok(job, seg, n):
    """合并段的两个动作是否都做出来了。返回 (ok, 失败项)。

    合并本来是为了去掉交界，但实测 24 个合并段里有 4 个（17%）后一个动作没做出来；
    合并段整体几何能过，判据看不出这个，只能按拆开的两个提示词分别查。
    """
    from hm import signature as _sig
    jp = job.cand_dir(seg["segment_id"], n) / "joints22.npy"
    if not jp.exists():
        return None, []
    sig = _sig.signature(np.load(jp))
    fails = []
    for part in (seg.get("text_gen_final") or "").split(" and "):
        part = part.strip()
        body = part if part.lower().startswith("a person") else "a person " + part
        ok, bad, _det = _sig.check(body, sig)
        if ok is False:
            fails.append((part[:28], [b[0] for b in bad]))
    return (len(fails) == 0), fails


def repair_junctions(job, sb, ref, pose_th, k_extra=6, gap_limit=90.0):
    """交界修复：差得大的相邻段，重新生成后一段，挑一个与前一端更贴合的候选。

    为什么放在选择之后
        逐段的判据只看这一段好不好，看不到它与前一段能不能接上；
        等 DP 选完再对最差的几处补候选，是针对性的，代价只花在坏的交界上。
    不覆盖原候选
        新候选编号从 91 开始，原有文件不动，失败也不会让原来的选择消失。
    """
    from hm import finish_job as FJ
    fixed = 0
    for i in range(len(sb["segments"]) - 1):
        A, B = sb["segments"][i], sb["segments"][i + 1]
        if not A.get("selected") or not B.get("selected"):
            continue
        if B.get("support_status") == "unsupported":
            continue
        na = int(str(A["selected"]).split("_")[-1])
        Ja = np.load(job.cand_dir(A["segment_id"], na) / "joints22.npy")

        def gap_to(n):
            Jb = np.load(job.cand_dir(B["segment_id"], n) / "joints22.npy")
            b2, _ = FJ.align_pair(Ja, Jb)
            return FJ.pose_gap(Ja[-1], b2[0])

        cur = int(str(B["selected"]).split("_")[-1])
        cur_gap = gap_to(cur)
        job.log("A12 交界候选：%s→%s 原始姿态差 %.0f 度"
                % (A["segment_id"], B["segment_id"], cur_gap))
        # 截尾：末段常在收势，截掉几帧能让过渡更早开始。
        # 只在「交界变小」且「本段动作签名仍成立」时采纳，实测 s03 101→86 度、
        # s19 反而变大，所以必须逐处实测而不是一刀切。
        if cur_gap > 60:
            from hm import signature as _sig
            _best_k, _best_gap = 0, cur_gap
            for _k in (2, 4, 6, 8, 12):
                if _k >= len(Ja) - 8:
                    break
                _K = Ja[: len(Ja) - _k]
                _ok, _bad, _d = _sig.check(A.get("text_gen_final") or "", _sig.signature(_K))
                if _ok is False:
                    continue
                _b2, _ = FJ.align_pair(_K, np.load(job.cand_dir(B["segment_id"],
                                                               int(str(B["selected"]).split("_")[-1])) / "joints22.npy"))
                _g = FJ.pose_gap(_K[-1], _b2[0])
                if _g < _best_gap - 8.0:
                    _best_k, _best_gap = _k, _g
            if _best_k:
                A["trim_tail"] = _best_k
                job.log("A12 截尾：%s 截 %d 帧，交界 %.0f → %.0f 度"
                        % (A["segment_id"], _best_k, cur_gap, _best_gap))
                cur_gap = _best_gap
        if cur_gap <= gap_limit:
            continue
        prompt = B.get("text_gen_final")
        n_frames = int(B.get("duration_frames") or 40)
        model = (B.get("route") or {}).get("model") or config.DEFAULT_MODEL
        try:
            from hm import generate as _gen
            feats = _gen.generate([prompt] * k_extra, [n_frames] * k_extra, name=model)
        except Exception as e:
            job.log("A12 交界修复跳过 %s：%s" % (B["segment_id"], str(e)[:60]))
            continue
        cands = list(B.get("candidates") or [])
        for j, F in enumerate(feats, 91):
            d = job.cand_dir(B["segment_id"], j)
            np.save(d / "motion263.npy", np.asarray(F))
            cands.append({"candidate_id": "cand_%02d" % j, "frames": int(len(F)),
                          "requested_frames": n_frames, "seed": None,
                          "seed_control": "unsupported", "extra_for_junction": True})
        B["candidates"] = cands
        step_a08_a10(job, {"segments": [B]}, ref)
        def _passed(c):
            if c.get("decision"):
                return c["decision"] == "pass"
            n = int(c["candidate_id"].split("_")[-1])
            mp = job.cand_dir(B["segment_id"], n) / "metrics.json"
            try:
                return json.loads(mp.read_text(encoding="utf-8")).get("decision") == "pass"
            except Exception:
                return False

        best, best_gap = None, cur_gap
        for c in B.get("candidates", []):
            if not _passed(c):
                continue
            n = int(c["candidate_id"].split("_")[-1])
            pj = job.cand_dir(B["segment_id"], n) / "joints22.npy"
            if not pj.exists():
                continue
            g = gap_to(n)
            if g < best_gap - 5.0:
                best, best_gap = n, g
        _gaps = []
        for c in B.get("candidates", []):
            if not _passed(c):
                continue
            n = int(c["candidate_id"].split("_")[-1])
            pj = job.cand_dir(B["segment_id"], n) / "joints22.npy"
            if pj.exists():
                _gaps.append((gap_to(n), n))
        if _gaps:
            _gaps.sort()
            job.log("A12 交界尝试：%s→%s 当前 %.0f 度，新增候选最好 %.0f 度（%d 个可用）"
                    % (A["segment_id"], B["segment_id"], cur_gap, _gaps[0][0], len(_gaps)))
        if best is not None:
            B["selected"] = "cand_%02d" % best
            fixed += 1
            job.log("A12 交界修复：%s→%s 姿态差 %.0f → %.0f 度（候选 cand_%02d）"
                    % (A["segment_id"], B["segment_id"], cur_gap, best_gap, best))
    return fixed


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
        # 动作签名：几何通过的候选里，优先挑「确实做出来这个动作」的
        try:
            from hm import signature as _sig
            _prompt = seg.get("text_gen_final") or seg.get("text_gen") or ""

            def _sig_pass(c):
                n = int(c["candidate_id"].split("_")[-1])
                J = np.load(job.dir / "segments" / seg["segment_id"] /
                            ("candidate_%d" % n) / "joints22.npy")
                verdict, bad, det = _sig.check(_prompt, _sig.signature(J))
                return verdict

            _sig_of = {c['candidate_id']: _sig_pass(c) for c in ok}
            _n_good = sum(1 for v in _sig_of.values() if v is True)
            if _n_good < len(ok):
                job.log('A12 %s 动作签名（只作排序参考）：%d/%d 个候选看起来做出了该动作'
                        % (seg['segment_id'], _n_good, len(ok)))
        except Exception as _e:
            job.log("A12 %s 动作签名跳过：%s" % (seg["segment_id"], str(_e)[:80]))
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
        # 合并段：先挑出「两个动作都做出来」的候选；存在就只在这些里选。
        # 实测 24 个合并段中 4 个（17%）后一个动作没做出来，整体几何看不出来。
        if seg.get("merged_from") and ok:
            _keep = []
            for _c in ok:
                _n = int(_c["candidate_id"].split("_")[-1])
                _good, _f = _merged_parts_ok(job, seg, _n)
                if _good:
                    _keep.append(_c)
            if not _keep:
                seg["merged_unverified"] = True
            if _keep and len(_keep) < len(ok):
                seg["allowed"] = [_c["candidate_id"] for _c in _keep]
                job.log("A12 %s 合并段：%d/%d 个候选两个动作都做出来了"
                        % (seg["segment_id"], len(_keep), len(ok)))
                ok = _keep

        def key(c):
            m = c["metrics"]
            _ok = _sig_of.get(c["candidate_id"]) if "_sig_of" in dir() else None
            return (0 if _ok is True else 1,
            _goal_bonus(job, seg, c),
                    m.get("bone_length_jitter_m") or 9e9,
                    m.get("pose_rate_max") or 9e9,
                    m.get("root_path_length") or 9e9)
        ok.sort(key=key)
        seg["selected"] = ok[0]["candidate_id"]
        seg["segment_decision"] = "pass"
        seg["ranking"] = [c["candidate_id"] for c in ok]
        job.log("A12 %s 选中 %s（排序依据：动作签名、骨长偏差、单帧姿态变化、路径长度）"
                % (seg["segment_id"], seg["selected"]))
    # 收尾：在所有通过判据的候选里做一次动态规划，选交界最优组合。
    # 逐段排序只管「这一段好不好」，交界只管「两段接得顺不顺」，两者都要。
    if all(seg.get("segment_decision") == "pass" for seg in sb["segments"]):
        try:
            from hm import finish_job as _fj
            _chosen = _fj.choose_combination(job, sb)
            job.log("A12 选交界最优组合：%s"
                    % "、".join("%s→%s" % (sg["segment_id"][-3:], "cand_%02d" % n)
                               for sg, n in _chosen))
            for sg, n in _chosen:
                sg["selected"] = "cand_%02d" % n
        except Exception as _e:
            job.log("A12 交界优化跳过：%s" % str(_e)[:80])
    job.save_json("storyboard.json", sb)
    return sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--k", type=int, default=6, help="每段候选数")
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
