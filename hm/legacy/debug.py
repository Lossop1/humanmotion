"""手动调参与重渲。

先跑一次 hm.main 把每段拟合结果缓存下来，之后只调两个旋钮：
    --fades      过渡帧数（时长）
    --rate-scale 端点速率比例（速度）。1.0 严格等于邻段，越小越慢

用法
    python -m hm.tune v2 --fades 12,15 --rate-scale 1.0 --tag t1
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hm import config, join as joinmod, join_params as JP, to_joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fades", default="")
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--curve", default="hermite", choices=["hermite", "linear", "smooth"])
    ap.add_argument("--blend", default=__import__("os").environ.get("HM_BLEND", "linear"),
                    choices=["linear", "polar"])
    ap.add_argument("--cut", default=__import__("os").environ.get("HM_CUT", "min"),
                    choices=["min", "active", "none"])
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_absolute():
        run = config.OUT_ROOT / run.name if (config.OUT_ROOT / run.name).exists() else run
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))["plan"]
    segs = plan["segments"]

    # 拟合结果缓存
    from hm import fit as FM
    need = [i for i in range(len(segs)) if not (run / ("params_seg_%d.npy" % (i + 1))).exists()]
    if need:
        FM._patch_paths()
    params, joints = [], []
    for i, s in enumerate(segs):
        F = np.load(run / ("feats_%d.npy" % s["id"]))
        joints.append(to_joints.from_feats(F))
        p = run / ("params_seg_%d.npy" % (i + 1))
        if p.exists():
            P = np.load(p)
        else:
            _v, P = FM.fit(joints[i], batch_size=16, iters=a.iters)
            np.save(p, P)
            print("      拟合段%d %d 帧（已缓存）" % (i + 1, len(P)), flush=True)
        params.append(np.asarray(P, dtype=np.float64))

    # 切点
    skip_list, fade_list = [0], [0]
    for k in range(len(joints) - 1):
        if a.cut == "none":
            Jk, Jk1 = joints[k], joints[k + 1]
            ci, cj = len(Jk), 0
            m = joinmod.max_bone_angle(Jk[-1], Jk1[0])[0]
            _dh = 0.0
        elif a.cut == "active":
            ci, cj, m, _dh = joinmod.find_cut_active(joints[k], joints[k + 1])
        else:
            ci, cj, m, _dh = joinmod.find_cut(joints[k], joints[k + 1])
        skip_list.append(int(cj))
        from hm import transition_params as TP
        Jk, Jk1 = joints[k], joints[k + 1]
        rate_in = joinmod.max_bone_angle(Jk[ci - 2], Jk[ci - 1])[0] if ci >= 2 else 0.0
        rate_out = (joinmod.max_bone_angle(Jk1[cj], Jk1[cj + 1])[0]
                    if cj + 1 < len(Jk1) else rate_in)
        want = None
        if "junctions" in plan and str(k) in plan["junctions"]:
            want = float(plan["junctions"][str(k)].get("duration_s", 0.0)) or None
        nf, lo, hi = TP.pick_frames(m, rate_in, rate_out, want_s=want, fps=config.FPS)
        print("      接缝%d 最差骨角 %.1f 速率 %.1f→%.1f 区间[%d,%d] → %d 帧 cut@%d skip@%d"
              % (k + 1, m, rate_in, rate_out, lo, hi, nf, ci, cj), flush=True)
        fade_list.append(int(nf))
    for k in range(len(joints) - 1):
        if a.cut == "none":
            ci = len(joints[k])
        elif a.cut == "active":
            ci = joinmod.find_cut_active(joints[k], joints[k + 1])[0]
        else:
            ci = joinmod.find_cut(joints[k], joints[k + 1])[0]
        params[k] = params[k][:ci]
        joints[k] = joints[k][:ci]

    # 帧数按「两端姿态差 ÷ 边界处的角速度」算，不再用外部给的时长。
    # 实测接缝 2 的两端只差 33 度，而来向速度 16.6 度/帧：两帧就能走完。
    # 之前给 26 帧，平均 1.2 度/帧，比两侧慢 13 倍，画面上几乎不动再突然一跳。
    # 先做朝向对齐，再量差距。对齐把差距从 136.7 度降到 33.2 度，
    # 帧数必须按对齐后的差距算，否则会快四倍。
    for _k in range(len(params) - 1):
        _tgt = JP.pelvis_yaw(params[_k][-1])
        _J2, _P2, _d = JP.align_yaw(joints[_k + 1], params[_k + 1], _tgt)
        joints[_k + 1] = _J2
        params[_k + 1] = _P2

    for _k in range(len(params) - 1):
        _A, _B = params[_k], params[_k + 1]
        _gap = JP.param_angle(_A[skip_list[_k] - 1], _B[skip_list[_k + 1]])
        _ri = JP.param_rate(_A, len(_A) - 1, 3, tail=True)
        _ro = JP.param_rate(_B, skip_list[_k + 1], 3, tail=False)
        _n = int(round(_gap / max(_ri, _ro, 1e-6)))
        print("      接缝%d 差距 %.1f 度  边界速率 %.1f/%.1f 度/帧  → %d 帧（原 %d）"
              % (_k + 1, _gap, _ri, _ro, max(2, min(60, _n)), fade_list[_k + 1]), flush=True)
        fade_list[_k + 1] = max(2, min(60, _n))

    if a.fades:
        want = [int(x) for x in a.fades.split(",")]
        fade_list = [0] + want[:len(params) - 1]

    # 必须把每段的角速度和每处姿态差传进去，否则曲线端点斜率都取 1.0，
    # 三次曲线退化成直线：过渡全程恒定慢速，进段时突然跳一下。
    # 端点斜率用「边界处最后/最初几帧」的速率，不用整段中位数
    def _bnd(J, tail=True, w=3):
        if len(J) < 2:
            return 0.0
        idx = list(range(max(0, len(J) - 1 - w), len(J) - 1)) if tail \
            else list(range(0, min(w, len(J) - 1)))
        vals = [joinmod.max_bone_angle(J[_t], J[_t + 1])[0] for _t in idx]
        return float(np.mean(vals)) if vals else 0.0
    rates = {}
    for _k in range(len(joints) - 1):
        rates[("in", _k)] = _bnd(joints[_k], tail=True)
        rates[("out", _k + 1)] = _bnd(joints[_k + 1], tail=False)
    rates["blend"] = a.blend
    merged, lay = JP.merge(params, joints, skip_list, fade_list,
                           curve=a.curve, rates=rates)
    total = sum(int(r.get("frames", 0)) + int(r.get("fade_frames", 0)) for r in lay)
    print("  过渡帧数 %s   总帧数 %d  layout合计 %d  %s" % (
        fade_list, len(merged), total, "一致" if total == len(merged) else "不一致"), flush=True)

    out = run if a.tag is None else (run.parent / ("%s_%s" % (run.name, a.tag)))
    out.mkdir(parents=True, exist_ok=True)
    verts = JP.verts_from_params(merged)
    # 接地：逐帧最低顶点对齐到地面 0。不画地形，但人不能悬空或沉下去。
    from hm import terrain as terrainmod
    verts, off = terrainmod.apply(verts, np.zeros(len(merged)), window=21)
    merged[:, 83] += off
    print("  接地修正 %.3f ~ %.3f 米" % (float(off.min()), float(off.max())), flush=True)
    np.save(out / "params.npy", merged)
    np.save(out / "verts.npy", verts)
    import shutil
    for nm in ("faces.npy",):
        if (run / nm).exists() and not (out / nm).exists():
            shutil.copy(run / nm, out / nm)
    (out / "assemble.json").write_text(
        json.dumps({"layout": lay, "fades": fade_list, "curve": a.curve, "cut": a.cut},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("  写出 %s" % out, flush=True)

    if not a.no_render:
        cmd = [str(config.BLENDER), "--background", "--python",
               str(Path(__file__).with_name("render_blender.py")), "--",
               str(out), str(out / "render")]
        subprocess.run(cmd, check=False)
        print("  渲染 %s" % (out / "render" / "out.mp4"), flush=True)


if __name__ == "__main__":
    main()