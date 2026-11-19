"""地形：显式规划地面高度，替代「一律压平」。

为什么需要
----------
模型输出里带的竖直信息有两种，必须分开：
  身体相对高度  骨盆相对脚的高度。这是姿态的一部分，属于模型，要保留。
  地面高度      脚下是平地还是台阶。这不是单段动作能决定的，
                模型每段独立生成、看不到完整句子，它给的绝对落点不可信。

之前的落地修正把两者混在一起：强制「全片平均触地 = 0」。
结果是模型输出「下楼」时被压平，动作和语义对不上。

现在地面高度由 LLM 按整句语义给出，逐帧目标从「0」换成这条曲线：
  全平地        曲线恒为 0，所有段落在同一高度
  先下楼再挥手  曲线从 +h 降到 0，人的脚跟着走，所以一开始就在楼上

数值约定
--------
y 单位米，与渲染世界坐标一致。start_y 是第一帧的地面高度。
profile 按段给，段内线性插值。
"""
import numpy as np

from hm import config


def clamp_by_horiz(terrain, profiles, max_slope=0.60):
    """按每段的水平路径长度夹住能声明的地形落差。

    LLM 不知道人物实际走了多远，可能按「一层楼 3 米」给，
    而整段水平只有 3 米，坡度就成了 1.0，渲出来是滑梯不是楼梯。
    这里按 max_slope × 水平路径 限制每一段的落差。
    """
    if not terrain or not terrain.get("profile"):
        return terrain
    out = dict(terrain)
    prof = [dict(x) for x in terrain["profile"]]
    horiz = {p["segment"]: float(p.get("horiz_m", 0.0)) for p in profiles}
    # 落差发生在【第 k 段之内】，所以要用第 k 段的水平路径去限制
    # 从 y_k 到 y_{k+1} 的变化量。
    # 写成「用第 k+1 段的水平路径限制 y_{k+1} 相对 y_k 的偏移」就错了：
    # 第 k+1 段往往是原地动作（水平 0.03 米），于是整段落差被压成 7 厘米。
    prev = float(out.get("start_y", 0.0))
    for i, item in enumerate(prof):
        k = int(item.get("at_segment", 1))
        want = float(item.get("y", 0.0))
        # 从上一段起点到本段起点的变化，发生在【上一段】之内
        k_prev = int(prof[i - 1].get("at_segment", 1)) if i > 0 else k
        allow = max_slope * horiz.get(k_prev if i > 0 else k, 0.0) + 0.05
        lo, hi = prev - allow, prev + allow
        y = max(lo, min(hi, want))
        item["y"] = round(y, 3)
        item["requested_y"] = round(want, 3)
        item["allow_m"] = round(allow, 3)
        prev = y
    out["profile"] = prof
    return out


def profile_to_frames(layout, terrain, total_frames):
    """把按段给的地形曲线展开成逐帧目标。"""
    T = int(total_frames)
    if not terrain:
        return np.zeros(T)
    start_y = float(terrain.get("start_y", 0.0))
    prof = terrain.get("profile") or []
    # 每段一个地面高度
    seg_y = {}
    for item in prof:
        try:
            seg_y[int(item.get("at_segment", 1))] = float(item.get("y", 0.0))
        except (TypeError, ValueError):
            continue
    # profile 的语义是「段 k 开始时的地面高度」。
    # 所以段 k 之内，地面从 seg_y[k] 线性走到 seg_y[k+1]。
    # 之前按「段内从上一段的值插到本段的值」实现，等于把段与段之间的变化
    # 推迟到了下一段：LLM 给 {1: 0.0, 2: -0.555} 本意是段1 之内下降，
    # 代码却让段1 保持 0、在段2（打招呼）里往下降。
    out = np.zeros(T, dtype=np.float64)
    off = 0
    # layout 有两种条目：段（有 index）和过渡（kind=fade，有 frames 无 index）。
    # 过渡帧不分配地形，但必须把偏移推进去，否则后面所有段整体错位。
    segs = [r for r in layout if r.get("kind") != "fade"]
    idx = [int(r.get("index", 0)) + 1 for r in segs]
    for i, r in enumerate(layout):
        if r.get("kind") == "fade":
            off = min(T, off + int(r.get("frames", 0)))
            continue
        n = int(r.get("fade_frames", 0))
        if n > 0:
            off = min(T, off + n)
        f0 = off
        f1 = min(T, off + int(r.get("frames", 0)))
        j = idx.index(int(r.get("index", 0)) + 1) if (int(r.get("index", 0)) + 1) in idx else -1
        y_start = seg_y.get(idx[j], start_y) if j >= 0 else start_y
        y_end = seg_y.get(idx[j + 1], y_start) if (j >= 0 and j + 1 < len(idx)) else y_start
        if f1 > f0:
            out[f0:f1] = np.linspace(y_start, y_end, f1 - f0)
        off = f1
    if off < T:
        out[off:] = seg_y.get(idx[-1], start_y) if idx else start_y
    return out


def apply(verts, target_ground, window=21):
    """让脚的触地跟随目标地面高度，而不是一律归零。

    verts: (T, N, 3)
    target_ground: (T,) 每帧的地面高度
    返回 (落地后的顶点, 实际施加的修正量)
    """
    V = np.asarray(verts, dtype=np.float64)
    tgt = np.asarray(target_ground, dtype=np.float64)
    T = len(V)
    if len(tgt) != T:
        raise ValueError("目标地面长度 %d 与帧数 %d 不符" % (len(tgt), T))
    low = V[..., 1].min(axis=1)
    k = max(1, int(window))
    pad = np.pad(low, (k // 2, k - 1 - k // 2), mode="edge")
    ker = np.ones(k) / float(k)
    smooth = np.convolve(pad, ker, mode="valid")
    off = tgt - smooth
    out = V.copy()
    out[..., 1] += off[:, None]
    return out, off


def selftest():
    """自检：给定一段阶梯地形，落地后脚的触地应跟着阶梯走。"""
    T = 120
    V = np.zeros((T, 100, 3))
    # 造一个整体浮 0.3 米、脚在 0.3 的序列
    V[..., 1] = 0.5
    V[:, :10, 1] = 0.3                       # 一部分顶点作为「脚」
    # 目标地面：前 60 帧在 1.0 米高的平台上，后 60 帧降到 0
    tgt = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 0.0, 60)])
    out, off = apply(V, tgt)
    low = out[..., 1].min(axis=1)
    ok = True
    if abs(low[:50].mean() - 1.0) > 0.05:
        ok = False
        print("  平台段触地不对: %.3f" % low[:50].mean())
    if abs(low[-10:].mean() - 0.0) > 0.08:
        ok = False
        print("  下降段末尾触地不对: %.3f" % low[-10:].mean())
    print("  地形自检: %s  平台段 %.3f 米  末段 %.3f 米" % (
        "通过" if ok else "不通过", low[:50].mean(), low[-10:].mean()))
    return ok


if __name__ == "__main__":
    selftest()