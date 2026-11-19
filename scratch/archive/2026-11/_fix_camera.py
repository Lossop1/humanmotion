"""把「相机校正」块移到「关键帧」块之前，避免手写缩进出错。

在原文件里按块首行定位，交换两段，交给 ast 校验后再写回。
"""
import ast
import subprocess
from pathlib import Path

P = Path("/home/chuan/motion-work/hm/render_blender.py")
subprocess.run(["git", "-C", "/home/chuan/motion-work", "checkout", "--", "hm/render_blender.py"],
               check=True)
s = P.read_text(encoding="utf-8")

kf_start = s.index("  for i in range(T):\n      cx, cy, cz = float(root[i, 0])")
kf_end = s.index("  # 用投影把人物高度校正到画面高度的目标比例。")
corr_start = kf_end
corr_end = s.index("  scn.frame_start = 1")

kf = s[kf_start:kf_end]
corr = s[corr_start:corr_end]

kf_new = kf.replace('aim = Vector((cx, cy, cz - height * 0.10))',
                    'aim = Vector((cx, cy, cz + aim_dz))')

# 校正块的迭代次数与目标比例调整
corr_new = (corr
            .replace("for _it in range(14):", "for _it in range(24):")
            .replace("0.80 if PIVOT == \"upper\" else 0.78", "0.72 if PIVOT == \"upper\" else 0.68")
            .replace("abs(frac - TARGET_FRAC) < 0.01", "abs(frac - TARGET_FRAC) < 0.005"))

s2 = s[:kf_start] + corr_new + kf_new + s[corr_end:]
# 采样从 8 提到 16（更细的边缘）
s2 = s2.replace("      scn.eevee.taa_render_samples = 8", "      scn.eevee.taa_render_samples = 16")
ast.parse(s2)
P.write_text(s2, encoding="utf-8")
print("  已交换相机校正与关键帧顺序，采样 8→16")
