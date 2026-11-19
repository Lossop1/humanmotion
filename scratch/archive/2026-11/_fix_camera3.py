"""把相机校正换成「渲染探针帧 → 按真实像素量」的闭环。

原因
    bpy_extras 的 world_to_camera_view 与真实渲染不一致（差三倍多，早先踩过），
    靠它算出来的距离与偏移会让人物偏出画面。改为渲染一张低分辨率探针帧，
    直接量人物包围盒，按实测比例调距离、按实测中心调 shift_y，再打关键帧。
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
assert i_kf < i_corr < i_after

kf = lines[i_kf:i_corr]
kf = [l.replace("cz - height * 0.10", "cz + aim_dz") for l in kf]
rest = lines[i_after:]

probe = '''  # --- 相机闭环：渲染一张低分辨率探针帧，按真实像素量出人物占比与中心 ---
  TARGET_FRAC = TARGET_ARG if TARGET_ARG else (0.72 if PIVOT == "upper" else 0.68)
  probe_f = int(max(0, min(T - 1, int(T * 0.35))))
  _rsx, _rsy = scn.render.resolution_x, scn.render.resolution_y
  scn.render.resolution_percentage = 25
  scn.frame_set(probe_f + 1)
  _tmp = "/tmp/_cam_probe.png"
  scn.render.filepath = _tmp
  scn.render.image_settings.file_format = "PNG"
  import os as _os
  for _pass in range(2):
      cx, cy, cz = float(root[probe_f, 0]), float(root[probe_f, 1]), float(root[probe_f, 2])
      eye = Vector((cx + math.sin(az) * dist, cy - math.cos(az) * dist, cz + height * 0.30))
      aim = Vector((cx, cy, cz + aim_dz))
      cam.location = eye
      cam.rotation_euler = (aim - eye).to_track_quat("-Z", "Y").to_euler()
      bpy.context.view_layer.update()
      bpy.ops.render.render(write_still=True)
      if not _os.path.exists(_tmp):
          print("  探针帧没产出，跳过校准")
          break
      img = bpy.data.images.load(_tmp, check_existing=False)
      wpx, hpx = img.size
      buf = np.empty(wpx * hpx * 4, dtype=np.float32)
      img.pixels.foreach_get(buf)
      a = buf.reshape(hpx, wpx, 4)[::-1, :, :3]     # Blender 像素自下而上
      bpy.data.images.remove(img)
      bg = a[2, 2]
      mask = (np.abs(a - bg).sum(axis=2) > 0.06)
      ys, xs = np.where(mask)
      if len(ys) < 10:
          print("  探针帧里找不到人物，跳过校准")
          break
      top, bot = ys.min(), ys.max()
      frac = (bot - top) / hpx
      center = (top + bot) / 2.0 / hpx             # 0 在上，1 在下
      print("  探针：人物占高 %.0f%%（目标 %.0f%%），中心 y=%.2f" %
            (100 * frac, 100 * TARGET_FRAC, center))
      if _pass == 0:
          dist = dist * (frac / TARGET_FRAC)
          cam.data.shift_y += (center - 0.5) * (1.0 / max(frac, 1e-3)) * 0.5
      else:
          break
  try:
      _os.remove(_tmp)
  except OSError:
      pass
  scn.render.resolution_percentage = 100
'''

out = lines[:i_kf] + probe.split("\n") + kf + rest
s2 = "\n".join(out)
s2 = s2.replace("scn.eevee.taa_render_samples = 8", "scn.eevee.taa_render_samples = 16")
ast.parse(s2)
P.write_text(s2, encoding="utf-8")
print("  已改为渲染探针帧校准（行号 %d / %d / %d）" % (i_kf, i_corr, i_after))
