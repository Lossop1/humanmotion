"""过渡的加速度连续性：速度一致不等于起停不突兀。

量什么
    逐帧最大关节角速度 w[t]（度/帧）；
    交界处加速度 = w 的差分。
    报告三件事：
      入弯加速度  过渡第 1 帧与之前一帧的速度差
      出弯加速度  过渡结束后与过渡最后一帧的速度差
      内部峰值    过渡内部加速度绝对值最大者
    三者与「两侧运动自身的加速度中位」相比：比值越接近 1 越自然。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import finish_job as FJ


def speed_series(J):
    """逐帧最大骨向变化速度（度/帧）。"""
    out = [0.0]
    for t in range(1, len(J)):
        dA, _ = FJ.dirs_of(J[t - 1])
        dB, _ = FJ.dirs_of(J[t])
        out.append(max(FJ.rot_between(dA[k], dB[k])[1] for k in range(len(dA))))
    return np.array(out)


def main():
    rows = []
    for p in sorted(glob.glob("/home/chuan/motion-work/outputs/big_s*_r12/assembled/bounds.json")):
        d = Path(p).parent
        bj = json.loads(Path(p).read_text(encoding="utf-8"))
        J = np.load(d / "joints22.npy")
        w = speed_series(J)
        acc = np.abs(np.diff(w))
        t = 0
        for i, b in enumerate(bj["segments"]):
            t += b["frames"]
            if i >= len(bj.get("junctions", [])):
                break
            j = bj["junctions"][i]
            fr = j["transition_frames"]
            start, end = t, t + fr
            if end >= len(w) - 1:
                break
            # 两侧各取 12 帧的加速度中位作为参照
            left = acc[max(0, start - 12):max(1, start - 1)]
            right = acc[end + 1:min(len(acc), end + 13)]
            ref = np.median(np.concatenate([left, right])) if len(left) + len(right) else 0.0
            ref = max(ref, 0.05)
            acc_in = abs(w[start] - w[start - 1]) if start >= 1 else 0.0
            acc_out = abs(w[end + 1] - w[end]) if end + 1 < len(w) else 0.0
            peak = float(acc[start:end].max()) if end > start else 0.0
            rows.append((d.parent.name, j["pose_gap_deg"], fr, acc_in, acc_out, peak,
                         acc_in / ref, acc_out / ref, peak / ref))
            t += fr

    print("  交界处加速度 / 两侧加速度中位（越接近 1 越平顺）")
    print("  任务         差  帧  入弯   出弯   峰比   入比   出比")
    for r in rows:
        print("  %-11s %3.0f %3d %5.1f %5.1f %5.2f %5.2f %5.2f"
              % (r[0][:11], r[1], r[2], r[3], r[4], r[8], r[6], r[7]))
    if rows:
        ins = np.array([r[6] for r in rows])
        outs = np.array([r[7] for r in rows])
        peaks = np.array([r[8] for r in rows])
        print()
        print("  入弯比：中位 %.2f 最大 %.2f｜出弯比：中位 %.2f 最大 %.2f｜峰比：中位 %.2f 最大 %.2f"
              % (np.median(ins), ins.max(), np.median(outs), outs.max(),
                 np.median(peaks), peaks.max()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
