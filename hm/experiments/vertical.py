"""竖直剖面：给 LLM 判断「动作像下楼还是像弹跳」的证据。

为什么要这组数
--------------
模型每段独立生成，看不到完整句子，它给出的竖直运动可能有两种含义：
  弹跳      上下振荡，净位移接近零。这是原地动作的正常起伏，应该压平。
  下楼/上坡  单调升降，净位移显著大于振荡。这是地形，必须让地面跟着变，
            否则动作和世界对不上。
两者处理方式完全相反，所以必须把它测出来交给 LLM，不能我在代码里替它决定。
"""
import numpy as np

from hm import config

# 真实楼梯的最大坡度（垂直/水平）。用于限制能够声明的地形落差。
MAX_STAIR_SLOPE = 0.60


def segment_profile(J):
    """一段关节的竖直剖面。

    返回 dict：
        start_y   根高起点
        end_y     根高终点
        net       净变化（米，正为上升）
        osc       去掉线性趋势后的最大起伏
        monotone  净变化是否显著大于起伏（判为地形而非弹跳）
    """
    J = np.asarray(J, dtype=np.float64)
    y = J[:, 0, 1]
    n = len(y)
    if n < 2:
        return {"start_y": float(y[0]), "end_y": float(y[0]), "net": 0.0,
                "osc": 0.0, "monotone": False}
    t = np.linspace(0.0, 1.0, n)
    trend = np.polyval(np.polyfit(t, y, 1), t)
    osc = float(np.abs(y - trend).max())
    net = float(y[-1] - y[0])
    # 水平路径长度：地形落差必须和它匹配。
    # 真实楼梯坡度约 0.5-0.6（垂直/水平）。如果声明落差远大于水平路径，
    # 渲出来是滑梯不是楼梯。之前 LLM 不知道走了多远，按「一层楼 3 米」给，
    # 而这段动作水平只有 3.07 米，坡度成了 0.98。
    h = J[:, 0, :][:, [0, 2]]
    horiz = float(np.linalg.norm(np.diff(h, axis=0), axis=1).sum())
    return {
        "start_y": round(float(y[0]), 3),
        "end_y": round(float(y[-1]), 3),
        "net": round(net, 3),
        "osc": round(osc, 3),
        "horiz_m": round(horiz, 2),
        "max_drop": round(horiz * MAX_STAIR_SLOPE, 2),
        "monotone": bool(abs(net) > 2.0 * osc and abs(net) > 0.10),
    }


def all_profiles(joints, actions):
    """每段的剖面，附带动作名。"""
    out = []
    for i, (J, a) in enumerate(zip(joints, actions)):
        p = segment_profile(J)
        p["segment"] = i + 1
        p["action"] = a
        out.append(p)
    return out


def describe(profiles):
    """把剖面写成一句话，直接塞进提示词。"""
    lines = []
    for p in profiles:
        kind = "单调升降（像地形）" if p["monotone"] else "上下振荡（像弹跳）"
        lines.append("段%d %s：根高 %.3f → %.3f 米，净变化 %+.3f 米，振荡 %.3f 米，判为%s；水平路径 %.2f 米，按真实楼梯坡度最多只能落差 %.2f 米" % (
            p["segment"], p["action"], p["start_y"], p["end_y"],
            p["net"], p["osc"], kind, p.get("horiz_m", 0.0), p.get("max_drop", 0.0)))
    return "\n".join(lines)


def selftest():
    """自检：单调下降应判为地形，纯振荡应判为弹跳。"""
    T = 60
    J1 = np.zeros((T, 22, 3))
    J1[:, 0, 1] = np.linspace(1.30, 1.00, T)          # 单调下降 0.3 米
    J2 = np.zeros((T, 22, 3))
    J2[:, 0, 1] = 0.95 + 0.15 * np.sin(np.linspace(0, 6.28, T))   # 纯振荡
    p1 = segment_profile(J1)
    p2 = segment_profile(J2)
    ok = True
    if not p1["monotone"]:
        ok = False
        print("  单调下降没判成地形:", p1)
    if p2["monotone"]:
        ok = False
        print("  纯振荡被误判成地形:", p2)
    print("  下降: 净 %+.3f 振荡 %.3f 单调=%s" % (p1["net"], p1["osc"], p1["monotone"]))
    print("  振荡: 净 %+.3f 振荡 %.3f 单调=%s" % (p2["net"], p2["osc"], p2["monotone"]))
    print("  剖面自检: %s" % ("通过" if ok else "不通过"))
    return ok


if __name__ == "__main__":
    selftest()