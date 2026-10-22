"""端到端入口。一句话 → 一个视频。

用法：
    python -m hm.main "一个人先往前走，然后转身，接着跑起来，最后挥手告别"
    python -m hm.main "..." --out myrun --no-fit
"""
import argparse
import json
import time

import numpy as np

from hm import config


def run(sentence, out_name=None, do_fit=True, do_render=True, model=None,
        iters=20, verbose=True, inpaint=True):
    t_all = time.time()
    marks = []

    def mark(name, t0):
        dt = time.time() - t0
        marks.append((name, round(dt, 1)))
        if verbose:
            print("    [%s] %.1f s" % (name, dt), flush=True)

    import os as _os
    cut_mode = _os.environ.get("HM_CUT", "min")
    curve_mode = _os.environ.get("HM_CURVE", "hermite")
    blend_mode = _os.environ.get("HM_BLEND", "linear")
    from hm import planner, storyboard, generate, to_joints

    t0 = time.time()
    segs, llm_trans, llm_terrain = planner.plan_full(sentence)
    if verbose:
        print("    分段: %s" % [(s["label"], s["frames"]) for s in segs], flush=True)
    mark("1 规划", t0)

    t0 = time.time()
    plan = storyboard.build([dict(s) for s in segs])
    plan["terrain"] = llm_terrain
    if verbose:
        for s in plan["segments"]:
                print("      段%d %-10s %3d帧  %s" % (
                    s["id"], s["label"], s["frames"], s["prompt"]), flush=True)
    mark("2 分镜", t0)

    out = config.OUT_ROOT / (out_name or time.strftime("run_%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(
        json.dumps({"sentence": sentence, "plan": plan}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    t0 = time.time()
    prompts = [s["prompt"] for s in plan["segments"]]
    lengths = [s["frames"] for s in plan["segments"]]
    feats = generate.generate(prompts, lengths, name=model)
    for s, F in zip(plan["segments"], feats):
        np.save(out / ("feats_%d.npy" % s["id"]), F)
    mark("3 生成", t0)

    # 交界修补：把相邻两段的特征接起来，只重修交界区，由模型生成过渡
    feats_used = feats
    mark("3.5 过渡准备", time.time())

    t0 = time.time()
    joints = [to_joints.from_feats(F) for F in feats_used]
    mark("4 转关节", t0)

    # 装配：在【拟合参数空间】里接，不在关节位置空间里接。
    # 关节位置空间逐骨插值，两端方向差大时肢体扫出大弧，看起来是乱甩
    # （实测左右肘单帧甩到 152 度）。参数是轴角表示，插值只沿确定的旋转轴走。
    t0 = time.time()
    from hm import join as joinmod
    from hm import join_params as JP
    from hm import transition_params as TP
    from hm import fit as fitmod

    acts = [s["label"] for s in plan["segments"]]
    # 逐段拟合
    seg_params = []
    for i, J in enumerate(joints):
        if verbose:
            print("      拟合段%d %s（%d 帧）" % (i, acts[i], len(J)), flush=True)
        _v, P = fitmod.fit(J, batch_size=16, iters=iters)
        seg_params.append(P)
        # 缓存每段拟合参数，之后的装配实验直接复用，不再重新拟合
        np.save(out / ("params_seg_%d.npy" % (i + 1)), P)
    mark("5 分段拟合", t0)

    # 逐接缝定时长、找切点
    t0 = time.time()
    jp = {}
    llm_tr = llm_trans or []
    for k, x in enumerate(llm_tr):
        x["junction"] = k + 1
    by_index = {int(x.get("junction", -1)): x for x in llm_tr if isinstance(x, dict)}
    for k in range(len(acts) - 1):
        base = TP.default_params(acts[k], acts[k + 1])
        if k in by_index:
            base.update({kk: vv for kk, vv in by_index[k].items() if kk != "junction"})
        jp[k] = TP.clamp_params(base)

    # 切点用关节算（参数里看不出姿态相似度）
    skip_list = [0]
    fade_list = [0]
    for k in range(len(joints) - 1):
        if cut_mode == "none":
            # 不截断：A 全部保留，B 从第 0 帧接。
            # 姿态差用 A 的真末帧和 B 的真首帧来量，交给过渡去吸收。
            Jk, Jk1 = joints[k], joints[k + 1]
            ci = len(Jk)
            cj = 0
            m = joinmod.max_bone_angle(Jk[-1], Jk1[0])[0]
            dh = abs(float(Jk[-1][0][1]) - Jk1[0][0][1]) if hasattr(Jk1[0], "__len__") else 0.0
            if verbose:
                print("      接缝%d 不截断，真末帧→真首帧 姿态差 %.1f 度" % (k + 1, m), flush=True)
        elif cut_mode == "active":
            ci, cj, m, dh = joinmod.find_cut_active(joints[k], joints[k + 1])
        else:
            ci, cj, m, dh = joinmod.find_cut(joints[k], joints[k + 1])
        Jk, Jk1 = joints[k], joints[k + 1]
        # 两端真实速率：上一段末一步、下一段首一步的位姿变化率
        rate_in = joinmod.max_bone_angle(Jk[ci - 2], Jk[ci - 1])[0] if ci >= 2 else 0.0
        rate_out = (joinmod.max_bone_angle(Jk1[cj], Jk1[cj + 1])[0]
                    if cj + 1 < len(Jk1) else rate_in)
        prm = jp[k]
        span = m
        # 帧数区间由两端动作算出：下界 span/m（不比两侧快），
        # 上界 3·span/m（三次 Hermite 端点导数不超过 3 才单调）。
        # LLM 的语义时长只用来在区间内选，超出就被夹回来。
        nf, lo, hi = TP.pick_frames(span, rate_in, rate_out,
                                    want_s=float(prm.get("duration_s", 0.0)) or None,
                                    fps=config.FPS)
        prm.update({"cut_at": int(ci), "skip": int(cj), "span_deg": round(span, 1),
                    "rate_in": round(rate_in, 2), "rate_out": round(rate_out, 2),
                    "n_lo": int(lo), "n_hi": int(hi), "frames_final": int(nf)})
        skip_list.append(cj)
        fade_list.append(int(nf))
        if verbose:
            print("      接缝%d 最差骨角 %.1f 度  速率 %.1f→%.1f  区间[%d,%d] → 取 %d 帧  cut@%d skip@%d" % (
                k + 1, span, rate_in, rate_out, lo, hi, fade_list[-1], ci, cj), flush=True)
    # 上一段要截断到 cut
    for k in range(len(joints) - 1):
        ci = jp[k]["cut_at"]
        seg_params[k] = seg_params[k][:ci]
        joints[k] = joints[k][:ci]
    # 每段的平均角速度、每个接缝的姿态差，供曲线端点斜率使用
    rates = {}
    for k, J in enumerate(joints):
        if len(J) >= 2:
            v = [joinmod.max_bone_angle(J[i], J[i + 1])[0] for i in range(len(J) - 1)]
            rates[k] = float(np.median(v))
    for k in range(len(joints) - 1):
        rates[("gap", k)] = float(jp[k]["span_deg"])
    merged, lay = JP.merge(seg_params, joints, skip_list, fade_list, rates=dict(rates, blend=blend_mode), curve=curve_mode)
    np.save(out / "params.npy", merged)
    np.save(out / "joints.npy", np.concatenate([j for j in joints], axis=0))
    (out / "assemble.json").write_text(
        json.dumps({"layout": lay, "junctions": {str(k): v for k, v in jp.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    info = lay
    mark("6 参数空间连接", t0)

    t0 = time.time()
    verts = JP.verts_from_params(merged)
    # 地形：地面高度由 LLM 按整句语义给出，逐帧目标从「一律归零」换成这条曲线。
    # 全平地时曲线恒为 0；「先下楼再挥手」时曲线从 +h 降到 0，人一开始就在楼上。
    # 之前的 ground.apply 强制全片平均触地归零，会把模型输出的下楼压平。
    # 第二次 LLM 调用：把模型实际生成的竖直剖面交给它，决定地面形状。
    # 这一步不能省。模型每段独立生成、看不到完整句子，它给出的竖直运动
    # 可能是弹跳（应压平），也可能是地形（必须让地面跟着变）。
    # 两者处理相反，由 LLM 看证据决定，不由代码替它选。
    from hm import vertical, terrain as terrainmod
    profs = vertical.all_profiles(joints, acts)
    terr, notes = planner.terrain_decision(sentence, acts, profs)
    if verbose:
        for p in profs:
            print("      %s 净%+.2f 振荡%.2f 单调=%s" % (
                p["action"], p["net"], p["osc"], p["monotone"]), flush=True)
        for nn in notes:
            print("      地形判断 段%s: %s（%s）" % (
                nn.get("segment"), nn.get("decision"), nn.get("why", "")), flush=True)
        print("      地形 start_y=%.2f  %s" % (
            terr.get("start_y", 0.0),
            [(x["at_segment"], x["y"]) for x in terr.get("profile", [])]), flush=True)
    # 第二次调用失败时保留第一次给出的地形，不要用空结果覆盖。
    # 第一次调用已经按原句语义给出过地形（例如「先下楼」→ start_y=3.0），
    # 第二次调用只用来在动作与句子冲突时修正，它返回空不该把信息丢掉。
    # 先决定用哪一份地形，再夹制。顺序反了的话，兜底值不会被夹制。
    # 优先用第二次调用（它看过实测剖面），没有就用第一次调用（按原句语义给的）。
    use = terr if terr.get("profile") else (plan.get("terrain") or {"start_y": 0.0, "profile": []})
    if not terr.get("profile") and use.get("profile") and verbose:
        print("      第二次调用没给地形，沿用第一次的: start_y=%.2f" % use.get("start_y", 0.0), flush=True)
    # 按每段的水平路径长度夹住能声明的地形落差：
    # 真实楼梯坡度约 0.5-0.6，落差超过 0.6 × 水平路径就渲不出楼梯，只会像滑梯。
    use = terrainmod.clamp_by_horiz(use, profs)
    if verbose:
        print("      地形（夹制后）start_y=%.2f  %s" % (
            use.get("start_y", 0.0),
            [(x["at_segment"], x["y"]) for x in use.get("profile", [])]), flush=True)
    terr = use
    plan["terrain"] = terr
    tgt = terrainmod.profile_to_frames(lay, terr, len(merged))
    verts, off = terrainmod.apply(verts, tgt, window=21)
    merged[:, 83] += off
    if verbose:
        print("      落地修正 %.3f ~ %.3f 米" % (float(off.min()), float(off.max())), flush=True)
    np.save(out / "verts.npy", verts)
    # 把逐帧地面高度存下来，渲染脚本要用它生成台阶
    np.save(out / "terrain.npy", tgt)
    from hm import render as rendermod
    rendermod.save_faces(out)
    seq = np.load(out / "joints.npy")
    mark("7 出网格", t0)
    result = {
        "sentence": sentence,
        "out": str(out),
        "frames": int(len(seq)),
        "seconds_of_video": round(len(seq) / config.FPS, 2),
        "travel_m": round(float(np.linalg.norm(seq[-1, 0, [0, 2]] - seq[0, 0, [0, 2]])), 3),
        "segments": info,
        "marks": marks,
    }

    if not do_fit:
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        if verbose:
            print("  跳过拟合与渲染。关节已存: %s" % (out / "joints.npy"))
        return result

    if not do_render:
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        return result

    t0 = time.time()
    mp4 = rendermod.render(out)
    result["video"] = mp4
    mark("7 渲染", t0)

    result["total_seconds"] = round(time.time() - t_all, 1)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    if verbose:
        print("\n  完成  %d 帧 %.1f 秒视频  行程 %.2f m" % (
            result["frames"], result["seconds_of_video"], result["travel_m"]))
        print("  视频  %s" % mp4)
        print("  用时  %.1f 分钟" % (result["total_seconds"] / 60))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    run(a.sentence, a.out, not a.no_fit, not a.no_render, a.model, a.iters,
        inpaint=not a.no_inpaint)


if __name__ == "__main__":
    main()