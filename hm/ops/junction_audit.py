"""交界审计（junc_v1）：独立度量交界是否「两脚并拢被拖」。

为什么单独一个工具
    被修的量和用来证明修好的量如果是同一个，得到的永远是 0（前面已经栽过一次）。
    这里只用关节位置算，不调用 pin_* 之类的修正函数，且带 5 段合成正负对照自检。

判据（阈值都有出处）
    DSD  两脚同时贴地且都在动的帧数（连续 ≥2 帧才算）    目标 0
    p90  每只脚在交界内逐帧水平位移的 90 分位（毫米/帧）  目标 ≤15
    seam 交界两端的骨盆加速度（米/秒²）                  目标 ≤5

用法
    python -m hm.ops.junction_audit --job j9            # 审计一个任务
    python -m hm.ops.junction_audit --selftest          # 5 段合成对照
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLE = (10, 11)          # 脚掌
FPS = 20.0
CONTACT_H = 0.05         # 接地高度（真人站姿 0–6 毫米/帧、失败样本 24–35 毫米）
SLIDE_MOVING = 10.0      # 毫米/帧，超过算「在动」
P90_LIMIT = 15.0
SEAM_LIMIT = 5.0         # 米/秒²


def ranges(bj):
    """返回 (段区间列表, 交界区间列表)，帧区间左闭右开。"""
    segs, juncs, cur = [], [], 0
    for i, s in enumerate(bj["segments"]):
        a, b = cur, cur + s["frames"]
        segs.append((a, b))
        cur = b
        if i < len(bj.get("junctions", [])):
            n = bj["junctions"][i]["transition_frames"]
            juncs.append((cur, cur + n))
            cur += n
    return segs, juncs


def _contact(sole, gy):
    h = sole[..., 1] - gy
    return h < CONTACT_H


def junction_stats(J, a, b):
    """一个交界的 DSD / 分脚 p90 / 缝加速度。"""
    sole = J[:, list(SOLE), :]
    gy = float(sole[..., 1].min())
    seg = sole[a:b]
    v = np.linalg.norm(np.diff(seg, axis=0), axis=2) * 1000.0     # (n-1, 2) 毫米/帧
    contact = _contact(seg, gy)
    dsd = 0
    for t in range(len(v)):
        # 两脚都贴地、且两只脚都在动 —— 只有一脚动是正常迈步，不是平移
        if contact[t].all() and contact[t + 1].all() and bool((v[t] >= SLIDE_MOVING).all()):
            dsd += 1
    # 只看「该脚接地」的帧：迈步时摆动脚本就要快速移动，不能算滑移
    p90 = []
    for k in (0, 1):
        sel = contact[:-1, k] & contact[1:, k]
        vals = v[sel, k] if sel.any() else np.array([])
        p90.append(float(np.percentile(vals, 90)) if len(vals) else 0.0)

    def seam_acc(t0):
        if t0 <= 0 or t0 + 1 >= len(J):
            return 0.0
        p0, p1, p2 = J[t0 - 1, 0], J[t0, 0], J[t0 + 1, 0]
        return float(np.linalg.norm(p2 - 2 * p1 + p0) * FPS * FPS)

    # 摆动相：交界内是否有一只脚「抬起—前移—落地」。三份盲评都说没有，
    # 所以这里把它变成可测量：抬升高度、空中帧数、水平换位距离。
    h = seg[..., 1] - gy
    swing_lift = 0.0
    swing_travel = 0.0
    swing_air = 0
    for k in (0, 1):
        lift = float(h[:, k].max() - h[:, k].min())
        air = int((h[:, k] > 0.03).sum())
        travel = float(np.linalg.norm(seg[-1, k, [0, 2]] - seg[0, k, [0, 2]]))
        if lift > swing_lift:
            swing_lift, swing_air, swing_travel = lift, air, travel
    return {"frames": b - a, "dsd": dsd, "p90": p90,
            "seam_in": seam_acc(a), "seam_out": seam_acc(b),
            "swing_lift": swing_lift, "swing_air": swing_air, "swing_travel": swing_travel}


def judge(st):
    bad = []
    if st["dsd"] > 0:
        bad.append("DSD")
    if max(st["p90"]) > P90_LIMIT:
        bad.append("p90")
    if max(st["seam_in"], st["seam_out"]) > SEAM_LIMIT:
        bad.append("seam")
    # 摆动相默认只作为观察量：站立/制动类交界本来就没有摆动相，
    # 不能一律当失败（5 段合成里 plant/brake 就是这种）。
    # 需要强制时用 HM_SWING_GATE=1。
    if __import__("os").environ.get("HM_SWING_GATE") == "1":
        if not (st["swing_lift"] >= 0.04 and st["swing_air"] >= 2 and st["swing_travel"] >= 0.10):
            bad.append("swing")
    return bad


def scan_accel(J, bj, topn=5):
    """全片根加速度扫描：交界之外也要查（实测段内出现过 221 米/秒²，而交界只有 1.6）。"""
    p = J[:, 0, :]
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) * FPS * FPS
    segs, juncs = ranges(bj)
    def where(t):
        for i, (a, b) in enumerate(segs):
            if a <= t < b:
                return "段%d" % (i + 1)
        for i, (a, b) in enumerate(juncs):
            if a <= t < b:
                return "交界%d" % (i + 1)
        return "?"
    order = np.argsort(acc)[::-1][:topn]
    out = []
    for k in order:
        t = int(k) + 1
        out.append((t, float(acc[k]), where(t)))
    return out


def shape_metrics(J):
    """形态量：平滑不等于自然。三份盲评指出的问题在这里量化。

    M1 摆动脚次序：离地最高点必须出现在区间中点附近，且之后要落地
    M2 上半身：肩-髋相对扭转幅度、腕部上下摆动幅度
    M3 竖直漂移：整段脚掌最低点漂移
    """
    sole = J[:, list(SOLE), :]
    h = sole[..., 1]
    T = len(J)
    out = {}
    for k in (0, 1):
        air = np.where(h[:, k] - h[:, k].min() > 0.03)[0]
        if len(air) == 0:
            out["M1_foot%d" % k] = {"air": 0, "peak_ratio": None, "landed": False}
            continue
        peak = int(air[int(np.argmax(h[air, k]))])
        out["M1_foot%d" % k] = {
            "air": int(len(air)),
            "peak_ratio": round(peak / float(max(T - 1, 1)), 2),
            "landed": bool(h[min(T - 1, air[-1] + 2), k] - h[:, k].min() < 0.03),
        }
    # 肩-髋相对扭转（用肩线 16/17 与髋线 1/2 在水平面的夹角）
    def yaw(a, b):
        v = J[:, b, :] - J[:, a, :]
        return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
    sh = yaw(16, 17)
    hp = yaw(1, 2)
    twist = np.degrees(np.unwrap(np.radians(sh - hp)))
    out["M2_twist_deg"] = round(float(twist.max() - twist.min()), 1)
    # 肩髋反向：肩与髋相对扭转的皮尔逊相关为负 → 真的在反相摆（走路特征）
    try:
        rel = np.degrees(np.unwrap(np.radians(sh - hp)))
        out["M2_counter_corr"] = round(float(np.corrcoef(sh, hp)[0, 1]), 2)
        out["M2_rel_range_deg"] = round(float(rel.max() - rel.min()), 1)
    except Exception:
        out["M2_counter_corr"] = None
    # 手臂摆动：腕（20/21）竖直方向行程
    try:
        wrists = J[:, [20, 21], 1]
        out["M2_wrist_swing_m"] = round(float(np.max(wrists.max(axis=0) - wrists.min(axis=0))), 3)
    except Exception:
        out["M2_wrist_swing_m"] = None
    # M3 改成「根/头高度漂移」（脚掌最低点恒定，测不出下沉）
    out["M3_root_drift_m"] = round(float(J[-1, 0, 1] - J[0, 0, 1]), 3)
    try:
        out["M3_head_drift_m"] = round(float(J[-1, 17, 1] - J[0, 17, 1]), 3)
    except Exception:
        pass
    return out


def gate_job(jid):
    """六条判据一次过：三条平滑（DSD / 滑移 / 缝）+ 三条形态（M1 / M2 / M3）。"""
    d = ROOT / "outputs" / jid
    bj = json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    J = np.load(d / "assembled" / "joints22.npy")
    _segs, juncs = ranges(bj)
    ALL = []
    for i, (a, b) in enumerate(juncs):
        st = junction_stats(J, a, b)
        sub = J[max(0, a - 2):min(len(J), b + 2)]
        sm = shape_metrics(sub)
        fails = []
        if st["dsd"] != 0:
            fails.append("DSD")
        if max(st["p90"]) > P90_LIMIT:
            fails.append("滑移")
        if max(st["seam_in"], st["seam_out"]) > SEAM_LIMIT:
            fails.append("缝")
        m1 = [v for k, v in sm.items() if k.startswith("M1_")]
        if not any(v["air"] >= 4 and v["landed"] and v["peak_ratio"] is not None
                   and 0.30 <= v["peak_ratio"] <= 0.70 for v in m1):
            fails.append("M1摆动次序")
        if (sm.get("M2_rel_range_deg") or 0) < 8.0 or (sm.get("M2_wrist_swing_m") or 0) < 0.10:
            fails.append("M2上半身")
        # M3：走↔站之间的骨盆高度本来就有 5–8 厘米变化，不能用固定 2 厘米；
        # 默认按 9 厘米判，原地类动作可用 HM_M3_LIMIT=0.02 收紧。
        _m3lim = float(__import__("os").environ.get("HM_M3_LIMIT", "0.09"))
        if abs(sm.get("M3_root_drift_m") or 0) > _m3lim:
            fails.append("M3竖直漂移")
        ALL.append(fails)
        print("  交界%d %d–%d 帧%d  DSD=%d p90=%.0f/%.0f seam=%.1f/%.1f  "
              "M1=%s M2=%.1f°/%.3f米 M3=%+.3f米  → %s" % (
                  i + 1, a, b - 1, b - a, st["dsd"], st["p90"][0], st["p90"][1],
                  st["seam_in"], st["seam_out"],
                  ",".join("air%d/%.2f/%s" % (v["air"], v["peak_ratio"] or -1, v["landed"]) for v in m1),
                  sm.get("M2_rel_range_deg") or 0, sm.get("M2_wrist_swing_m") or 0,
                  sm.get("M3_root_drift_m") or 0,
                  "PASS" if not fails else "FAIL " + ",".join(fails)))
    ok = bool(ALL) and all(not f for f in ALL)
    print("  总判：%s" % ("PASS（六条全过）" if ok else "FAIL"))
    return ok


def audit_job(jid):
    d = ROOT / "outputs" / jid
    bj = json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8"))
    J = np.load(d / "assembled" / "joints22.npy")
    _segs, juncs = ranges(bj)
    total = sum(x["frames"] for x in bj["segments"]) + sum(
        j["transition_frames"] for j in bj.get("junctions", []))
    lines = []
    if total != len(J):
        lines.append("FAIL J0 帧数核对 %d vs %d" % (total, len(J)))
    for i, (a, b) in enumerate(juncs):
        st = junction_stats(J, a, b)
        bad = judge(st)
        lines.append("junc%d %d-%d 帧%d DSD=%d p90=%.0f/%.0f seam=%.1f/%.1f "
                     "swing=%.0fmm/%d帧/%.0fmm %s" % (
            i + 1, a, b - 1, st["frames"], st["dsd"], st["p90"][0], st["p90"][1],
            st["seam_in"], st["seam_out"], st["swing_lift"] * 1000, st["swing_air"],
            st["swing_travel"] * 1000, ("FAIL " + ",".join(bad)) if bad else "PASS"))
    return lines


def _synth(kind, n=20, v0=1.0):
    """合成用例：直接拼三段轨迹（前 / 交界 / 后），不再用分支推断。"""
    step = v0 / FPS
    m = n
    pre_root = step * np.arange(n)
    base = pre_root[-1]
    if kind == "plant":
        j_root = np.full(m, base); j_l = np.full(m, base + 0.10); j_r = np.full(m, base - 0.10); j_ry = np.zeros(m)
    elif kind == "slip_one":
        j_root = base + step * np.arange(m)
        j_l = np.full(m, base + 0.10); j_r = base - 0.10 + 0.04 * np.arange(m); j_ry = np.zeros(m)
    elif kind == "drag_both":
        j_root = base + 0.06 * np.arange(m)
        j_l = base + 0.10 + 0.06 * np.arange(m); j_r = base - 0.10 + 0.06 * np.arange(m); j_ry = np.zeros(m)
    elif kind == "step":
        j_root = base + step * np.arange(m)
        j_l = np.full(m, base + 0.10)
        j_r = base - 0.10 + 0.10 * (np.arange(m) / max(m - 1, 1))
        j_ry = 0.08 * np.sin(np.pi * (np.arange(m) + 0.5) / m)
    else:   # brake：脚站住、根速线性降到 0
        j_root = base + step * np.cumsum(np.linspace(1, 0, m))
        j_l = np.full(m, base + 0.10); j_r = np.full(m, base - 0.10); j_ry = np.zeros(m)
    post_root = j_root[-1] + step * np.arange(1, n + 1)
    root = np.concatenate([pre_root, j_root, post_root])
    lz = np.concatenate([pre_root + 0.10, j_l, post_root + 0.10])
    rz = np.concatenate([pre_root - 0.10, j_r, post_root - 0.10])
    ry = np.concatenate([np.zeros(n), j_ry, np.zeros(n)])
    T = len(root)
    J = np.zeros((T, 22, 3))
    J[:, 0] = [0, 1.0, 0]
    for t in range(T):
        J[t, :, 2] = root[t]
        J[t, 10] = [0.10, 0.0, lz[t]]
        J[t, 11] = [-0.10, ry[t], rz[t]]
    return J, (n, 2 * n)


def selftest():
    expect = {"plant": False, "slip_one": True, "drag_both": True, "step": False, "brake": False}
    ok = True
    for kind, should_fail in expect.items():
        J, (a, b) = _synth(kind)
        st = junction_stats(J, a, b)
        bad = judge(st)
        passed = not bad
        good = (not passed) == should_fail
        ok = ok and good
        print("  %-10s DSD=%d p90=%.0f/%.0f seam=%.1f → %s（应%s）%s" % (
            kind, st["dsd"], st["p90"][0], st["p90"][1], max(st["seam_in"], st["seam_out"]),
            "PASS" if passed else "FAIL " + ",".join(bad),
            "判失败" if should_fail else "判通过", "" if good else "  ← 不符预期"))
    print("  自检：%s" % ("通过（5/5）" if ok else "未通过"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan", action="store_true", help="全片根加速度扫描（不限交界）")
    ap.add_argument("--shape", action="store_true", help="形态量（M1 摆动次序 / M2 上半身 / M3 漂移）")
    ap.add_argument("--gate", action="store_true", help="六条判据一次过（三条平滑 + 三条形态）")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.job:
        print("  用法：python -m hm.ops.junction_audit --job <任务> ｜ --selftest")
        return
    if a.gate:
        gate_job(a.job)
        return
    if a.shape:
        d = ROOT / "outputs" / a.job
        bj = json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8"))
        J = np.load(d / "assembled" / "joints22.npy")
        segs, juncs = ranges(bj)
        wins = juncs if juncs else segs
        for i, (aa, bb) in enumerate(wins):
            tag = ("交界%d" % (i + 1)) if juncs else ("段%d" % (i + 1))
            print("  == %s 帧 %d–%d（窗口内计算，避免把站起来算成下沉）" % (tag, aa, bb - 1))
            sub = J[max(0, aa - 2):min(len(J), bb + 2)]
            for k, v in shape_metrics(sub).items():
                print("    %-16s %s" % (k, v))
        return
    if a.scan:
        d = ROOT / "outputs" / a.job
        bj = json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8"))
        J = np.load(d / "assembled" / "joints22.npy")
        print("  全片根加速度 Top5（%s）" % a.job)
        for t, v, w in scan_accel(J, bj):
            print("    f%-4d %8.1f 米/秒²  %s" % (t, v, w))
        return
    for line in audit_job(a.job):
        print("  " + line)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
