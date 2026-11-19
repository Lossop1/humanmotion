"""用回退那一版（e2e38 时代）的装配方法跑一遍，产出可比较的视频。

旧方法的行为：
    速度      单帧差分
    混合      线性矢量
    摆放      纯平移，不做朝向对齐
    曲线      hermite_u，斜率 = 该段中位角速度 / 过渡平均角速度
    插值      24 个关节逐个球面插值
    帧数      用外部给的（默认取 plan 里的时长）

只读已有的分段参数，不重新生成动作。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hm import config, to_joints
from hm.legacy import join_params as JP
from hm.legacy.join import max_bone_angle


def median_rate(J):
    if len(J) < 2:
        return 0.0
    vals = [max_bone_angle(J[t], J[t + 1])[0] for t in range(len(J) - 1)]
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fades", default=None, help="例如 21,26；缺省取 plan 里的时长")
    ap.add_argument("--tag", default="legacy")
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_absolute():
        run = config.OUT_ROOT / run.name
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    segs = plan["plan"]["segments"]
    n = len(segs)

    joints = [to_joints.from_feats(np.load(run / ("feats_%d.npy" % s["id"]))) for s in segs]
    params = [np.load(run / ("params_seg_%d.npy" % (i + 1))) for i in range(n)]

    if a.fades:
        fades = [0] + [int(x) for x in a.fades.split(",")][: n - 1]
    else:
        fades = [0]
        for s in segs[1:]:
            fades.append(int(round(float(s.get("duration_s", 1.0)) * 20)))
    cuts = [0] * n

    rates = {}
    for i, J in enumerate(joints):
        rates[i] = median_rate(J)
    for k in range(n - 1):
        rates[("gap", k)] = float(max_bone_angle(joints[k][-1], joints[k + 1][0])[0])

    print("  旧方法：帧数 %s   切点 %s" % (fades, cuts), flush=True)
    for k in range(n - 1):
        print("    接缝%d  段内中位速率 %.1f / %.1f   两端差距 %.1f 度"
              % (k + 1, rates[k], rates[k + 1], rates[("gap", k)]), flush=True)

    merged, lay = JP.merge_legacy(params, joints, cuts, fades, rates=rates)
    total = sum(int(r.get("frames", 0)) + int(r.get("fade_frames", 0)) for r in lay)
    print("  总帧数 %d  layout合计 %d  %s"
          % (len(merged), total, "一致" if total == len(merged) else "不一致"), flush=True)

    out = run.parent / ("%s_%s" % (run.name, a.tag))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "params.npy", merged)
    (out / "assemble.json").write_text(
        json.dumps({"layout": lay, "mode": "legacy"}, ensure_ascii=False, indent=1), encoding="utf-8")

    verts = JP.verts_from_params(merged)
    from hm.legacy import terrain as terrainmod
    verts, _ = terrainmod.apply(verts, np.zeros(len(merged)), window=21)
    np.save(out / "verts.npy", verts)
    faces = np.load(run / "faces.npy")
    np.save(out / "faces.npy", faces)

    subprocess.run([str(config.BLENDER), "--background", "--python",
                    str((Path(__file__).resolve().parents[1] / "render_blender.py")), "--",
                    str(out), str(out / "render")], check=False)
    print("  视频 %s" % (out / "render" / "out.mp4"), flush=True)


if __name__ == "__main__":
    main()
