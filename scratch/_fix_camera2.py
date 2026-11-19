"""按行号重排：把相机校正块挪到关键帧块之前。

上一版靠手写缩进匹配失败，这里改为按行号切片，行内容只用于定位。
"""
import ast
import subprocess
from pathlib import Path

P = Path("/home/chuan/motion-work/hm/render_blender.py")
subprocess.run(["git", "-C", "/home/chuan/motion-work", "checkout", "--", "hm/render_blender.py"],
               check=True)
lines = P.read_text(encoding="utf-8").split("\n")

i_kf = next(i for i, l in enumerate(lines) if l.strip().startswith("for i in range(T):"))
i_corr = next(i for i, l in enumerate(lines) if "投影把人物高度校正" in l)
i_after = next(i for i, l in enumerate(lines) if l.strip().startswith("scn.frame_start"))

assert i_kf < i_corr < i_after, (i_kf, i_corr, i_after)

kf = lines[i_kf:i_corr]
corr = lines[i_corr:i_after]
rest = lines[i_after:]

# 关键帧里把瞄准点改成带 aim_dz 的版本（与校正时一致）
kf = [l.replace("cz - height * 0.10", "cz + aim_dz") for l in kf]
# 校正块放宽目标与收敛阈值
corr = [l.replace("range(14)", "range(24)")
         .replace('0.80 if PIVOT == "upper" else 0.78', '0.72 if PIVOT == "upper" else 0.68')
         .replace("< 0.01", "< 0.005") for l in corr]

out = lines[:i_kf] + corr + kf + rest
s2 = "\n".join(out)
s2 = s2.replace("scn.eevee.taa_render_samples = 8", "scn.eevee.taa_render_samples = 16")
ast.parse(s2)
P.write_text(s2, encoding="utf-8")
print("  行号 %d / %d / %d：已把校正块移到关键帧之前" % (i_kf, i_corr, i_after))
