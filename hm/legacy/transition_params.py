import math
"""过渡参数：由 LLM 按段落语义设定，代码只负责执行。

为什么不再用「前后动作的角速度」定帧数
--------------------------------------
上一版把过渡帧数绑到前后段的实际角速度上，动作快就少插帧。
实测得到 3/7/3 帧，也就是 0.15/0.35/0.15 秒完成 30-50 度的全身姿态切换，
折合 200 度/秒，画面明显太快。
肢体摆动的快慢和「整体姿态切换该用多久」不是一回事，不能直接套用。

参数集
------
duration_s   这段过渡占多少秒。帧数 = duration_s × 20，这是唯一有时长意义的量。
ease         曲线形状。linear 匀速；ease_in_out 两端慢中间快；ease_out 起步快收尾慢。
scope        参与过渡的部位。full 全身；upper 只上半身；lower 只下半身。
anticipate   是否允许先反向蓄势。true 时曲线会先轻微反向再转到目标。

默认值按两侧动作的快慢给，LLM 可以逐接缝覆盖。
"""
from hm import config

# 动作快慢分档。决定默认过渡时长。
FAST_ACTIONS = {"run", "jog", "jump", "kick", "dance", "climb"}
SLOW_ACTIONS = {"stand", "wave", "sit", "squat", "bend", "pick", "place",
                "reach", "shake", "turn"}

DURATION_RANGE = (0.15, 1.20)
DEFAULT_DURATION = {
    ("fast", "fast"): 0.30,
    ("fast", "slow"): 0.45,
    ("slow", "fast"): 0.45,
    ("slow", "slow"): 0.60,
    ("normal", "normal"): 0.50,
}

EASE_CHOICES = ("linear", "ease_in_out", "ease_out")
SCOPE_CHOICES = ("full", "upper", "lower")


def speed_class(action):
    if action in FAST_ACTIONS:
        return "fast"
    if action in SLOW_ACTIONS:
        return "slow"
    return "normal"


def default_params(a, b):
    """给一个接缝的默认参数。"""
    sa, sb = speed_class(a), speed_class(b)
    dur = DEFAULT_DURATION.get((sa, sb), DEFAULT_DURATION[("normal", "normal")])
    ease = "linear" if (sa == "fast" and sb == "fast") else "ease_in_out"
    return {
        "duration_s": dur,
        "ease": ease,
        "scope": "full",
        "anticipate": False,
    }


def clamp_params(p):
    """把参数夹到合法范围。LLM 给的值可能越界。"""
    out = dict(p)
    lo, hi = DURATION_RANGE
    try:
        d = float(out.get("duration_s", DEFAULT_DURATION[("normal", "normal")]))
    except (TypeError, ValueError):
        d = DEFAULT_DURATION[("normal", "normal")]
    out["duration_s"] = round(max(lo, min(hi, d)), 3)
    if out.get("ease") not in EASE_CHOICES:
        out["ease"] = "ease_in_out"
    if out.get("scope") not in SCOPE_CHOICES:
        out["scope"] = "full"
    out["anticipate"] = bool(out.get("anticipate", False))
    return out


def frames_for(duration_s):
    """秒 → 帧数。"""
    p = clamp_params({"duration_s": duration_s})
    return max(1, int(round(p["duration_s"] * config.FPS)))


def ease_params(n, ease):
    """生成 n 个中间参数 u，按给定的曲线形状。"""
    import numpy as np
    t = np.linspace(0.0, 1.0, n + 2)[1:-1] if n > 0 else np.array([])
    if ease == "linear":
        return t
    if ease == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    return t * t * (3.0 - 2.0 * t)


def anticipate_params(n, ease, depth=0.12):
    """先轻微反向再转到目标，用于需要蓄势的动作。"""
    import numpy as np
    u = ease_params(n, ease)
    if n <= 0:
        return u
    k = np.linspace(0.0, 1.0, n)
    return u - depth * np.sin(np.pi * k)


def selftest():
    ok = True
    for a, b in [("run", "run"), ("walk", "run"), ("stand", "wave"), ("turn", "run")]:
        p = default_params(a, b)
        n = frames_for(p["duration_s"])
        print("  %-6s -> %-6s  默认 %.2f 秒 = %d 帧  ease=%s" % (
            a, b, p["duration_s"], n, p["ease"]))
        if not (1 <= n <= 30):
            ok = False
            print("    帧数越界")
    # 越界输入必须被夹住
    bad = clamp_params({"duration_s": 5.0, "ease": "xxx", "scope": "yyy"})
    if bad["duration_s"] > DURATION_RANGE[1] or bad["ease"] not in EASE_CHOICES:
        ok = False
        print("  越界参数没被夹住:", bad)
    print("  过渡参数自检: %s" % ("通过" if ok else "不通过"))
    return ok


if __name__ == "__main__":
    selftest()

# ---------- 帧数区间 ----------
#
# 过渡的速率不应快于两侧动作自身的速率。设
#     span  两端姿态角距离（度）
#     a,b   上段末帧、下段首帧的位姿变化率（度/帧）
#     m     max(a,b)
# 下界 N ≥ span/m：平均速率不超过两侧较快的那个。
# 上界 N ≤ 3·span/m：三次 Hermite 的端点导数 u'(0)=a·N/span 不超过 3 才单调。
# 两个界都由两端动作算出，没有人为常数。

def frames_interval(span_deg, rate_in, rate_out):
    m = max(float(rate_in), float(rate_out), 1e-3)
    lo = max(2, int(math.ceil(float(span_deg) / m)))
    hi = max(lo, int(math.ceil(3.0 * float(span_deg) / m)))
    return lo, hi


def pick_frames(span_deg, rate_in, rate_out, want_s=None, fps=20):
    """在可行区间内取帧数。want_s 是 LLM 给的语义时长，超出区间就夹回来。"""
    lo, hi = frames_interval(span_deg, rate_in, rate_out)
    if want_s is None:
        n = int(round(0.5 * (lo + hi)))
    else:
        n = int(round(float(want_s) * float(fps)))
    return max(lo, min(hi, n)), lo, hi
