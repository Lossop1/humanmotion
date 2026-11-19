"""两项检查：转身走样（写进签名）+ 交界突变（写进 gates 并在装配时调用）。"""
import ast
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- 1) 签名：前倾、反向前进 ----------
p = r"D:\my-projects\humanmotion\repo\hm\signature.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "backward_m" in s:
    print("  签名已有前倾/后退")
else:
    s = s.replace('''    yaws = _unwrap([_yaw(f) for f in J])''',
'''    yaws = _unwrap([_yaw(f) for f in J])
    # 躯干前倾：骨盆→脊柱3 与竖直方向的夹角
    tv = J[:, 9] - J[:, 0]
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lean = np.degrees(np.arccos(np.clip(tv[:, 1], -1, 1)))
    # 沿「朝向反方向」的净位移，正数表示在后退
    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = float(np.linalg.norm(fwd))
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    back = float(-(J[-1, 0] - J[0, 0]) @ fwd)''')
    s = s.replace('''        "knee_range": float(max(knee) - min(knee)),''',
'''        "knee_range": float(max(knee) - min(knee)),
        "lean_median": float(np.median(lean)),
        "lean_max": float(lean.max()),
        "backward_m": back,''')
    s = s.replace('''    ("crouch|squat|bend|bending|kneel|kneeling", "pelvis_drop", 0.18, "应降低重心"),''',
'''    ("crouch|squat|bend|bending|kneel|kneeling", "pelvis_drop", 0.18, "应降低重心"),
    # 校准对：MDM 转身前倾中位 8.9 度、后退 0.97 米；MoMask 3.6 度、后退 -0.05 米
    ("turns around|turn around", "lean_median", 6.0, "转身应站直，不该弓着"),
    ("turns around|turn around", "backward_m", 0.3, "转身不该边退边走"),''')
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    ast.parse(s)
    print("  signature：加入前倾与后退两个量及转身规则")

# ---------- 2) gates：交界突变 ----------
q = r"D:\my-projects\humanmotion\repo\hm\gates.py"
t = open(q, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "junction_jerk" in t:
    print("  gates 已有交界突变")
else:
    t += '''

def junction_jerk(seq, junctions, segments):
    """交界突变：过渡窗口内逐帧最大骨向变化的峰值，与前段段内中位的比值。

    校准：MDM 转身那一处峰值 44.2 度/帧（13.6 倍）；MoMask 同一处 5.7 度/帧（1.6 倍）。
    因此 比值 > 6 或 峰值 > 25 度/帧 记为可疑，只记录不拦。
    """
    import math
    from hm.finish_job import BONES, dirs_of, rot_between

    seq = np.asarray(seq, dtype=np.float64)
    rates = []
    for i in range(len(seq) - 1):
        d0, _ = dirs_of(seq[i])
        d1, _ = dirs_of(seq[i + 1])
        rates.append(max(math.degrees(rot_between(d0[k], d1[k])[1]) for k in range(len(BONES))))
    rates = np.array(rates) if rates else np.zeros(1)

    starts, t = [], 0
    for i, b in enumerate(segments):
        starts.append((t, t + b["frames"]))
        t += b["frames"]
        if i < len(junctions):
            t += junctions[i]["transition_frames"]

    out = []
    for i, j in enumerate(junctions):
        a1 = starts[i][1]
        b0 = starts[i + 1][0]
        lo, hi = max(0, a1 - 2), min(len(rates), b0 + 2)
        peak = float(rates[lo:hi].max()) if hi > lo else 0.0
        base = float(np.median(rates[starts[i][0]:max(starts[i][0] + 1, a1 - 1)])) \\
            if a1 - starts[i][0] > 1 else 1.0
        ratio = peak / max(base, 1e-6)
        out.append({"between": j.get("between"), "peak_deg_per_frame": round(peak, 1),
                    "ratio": round(ratio, 1),
                    "suspect": bool(ratio > 6.0 or peak > 25.0)})
    return out
'''
    open(q, "w", encoding="utf-8", newline="\n").write(t)
    ast.parse(t)
    print("  gates：新增 junction_jerk()")

# ---------- 3) 装配时调用并写进 bounds ----------
r = r"D:\my-projects\humanmotion\repo\hm\assemble_params.py"
u = open(r, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "junction_jerk" in u:
    print("  装配已调用交界检查")
else:
    anchor = '''    job.log("A16 自检：'''
    assert anchor in u, "未匹配自检日志"
    ins = '''    from hm import gates as GT
    _jj = GT.junction_jerk(Jn, junctions, bounds)
    for _x in _jj:
        if _x["suspect"]:
            job.log("A16 交界可疑：%s 峰值 %.1f 度/帧（前段中位的 %.1f 倍）"
                    % ("→".join(_x["between"] or []), _x["peak_deg_per_frame"], _x["ratio"]))
'''
    u = u.replace(anchor, ins + "    " + anchor.strip(), 1)
    open(r, "w", encoding="utf-8", newline="\n").write(u)
    ast.parse(u)
    print("  装配：交界可疑会打日志")
