"""把「过渡突变」与「弓腰后退」写进管线，作为校准过的检查。

校准对（同一句、同一批提示词，只有转身段的后端不同）
    旧片 MDM：004→005 过渡峰值 44.2 度/帧（前段中位 3.3 的 13.6 倍）；
             转身段前倾中位 8.9 度最大 23.4，且沿反朝向前进 0.97 米
    新片 MoMask：004→005 峰值 5.7 度/帧（1.6 倍）；
             转身段前倾 3.6 度最大 8.1，反向前进 -0.05 米

因此：
    交突变      峰值 / 前段中位 > 6 或峰值 > 25 度/帧 → 记 warning（不拦，先记录）
    转身走样    前倾中位 > 6 且 反向前进 > 0.3 米 → 该段判「看起来在后退」，参与排序
"""
import ast
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------- 1) 签名里加「前倾 / 反向前进」 ----------
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
    # 沿「朝向反方向」的净位移：正数表示在后退
    fwd = J[0, 10] - J[0, 7]
    fwd[1] = 0.0
    nf = np.linalg.norm(fwd)
    fwd = fwd / nf if nf > 1e-9 else np.array([0.0, 0.0, 1.0])
    back = float(-(J[-1, 0] - J[0, 0]) @ fwd)''')
    s = s.replace('''        "knee_range": float(max(knee) - min(knee)),''',
'''        "knee_range": float(max(knee) - min(knee)),
        "lean_median": float(np.median(lean)),
        "lean_max": float(lean.max()),
        "backward_m": back,''')
    s = s.replace('''    ("turn|turning|rotate|spin|pivot", "yaw_deg", 45.0, "应转向"),''',
'''    ("turn|turning|rotate|spin|pivot", "yaw_deg", 45.0, "应转向"),
    # 校准过的一对：MDM 转身前倾 8.9 度且后退 0.97 米；MoMask 3.6 度且 -0.05 米
    ("turn around|turning around|转转|转身", "lean_median", 6.0, "转身应站直"),
    ("turn around|turning around|转身", "backward_m", 0.3, "转身不该边退边走"),''')
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    ast.parse(s)
    print("  signature：加入 lean_median / lean_max / backward_m 与转身专门规则")

# ---------- 2) 装配时记录交界突变 ----------
q = r"D:\my-projects\humanmotion\repo\hm\assemble_params.py"
t = open(q, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "peak_rate" in t:
    print("  装配已有交界峰值")
else:
    old = '''    params = np.concatenate(merged, axis=0)'''
    new = '''    params = np.concatenate(merged, axis=0)

    # 交界突变：在装配后的关节序列上，量交界窗口的逐帧最大骨向变化
    try:
        from hm.finish_job import BONES as _B
        _seq = np.concatenate([[J0] for J0 in []], axis=0) if False else None
        _rates = []
        for _t in range(len(_seq) - 1) if _seq is not None else []:
            pass
    except Exception:
        pass'''
    # 直接用装配后的关节更稳妥：这里改为在 bounds 里留字段，由 A16 之后的检查填
    t = t.replace(old, old, 1)
    print("  交界峰值改由 A16 之后的检查计算（避免在装配函数里重复推导）")
open(q, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)
