"""Blender 渲染。回到最简单的形式：一块地面、白色人体、跟随相机。

不画台阶，不画地形参照物。地形只用来决定把人放在什么高度——
模型生成的动作本身有升降，地面要跟着它，但场景不渲染地形。
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
SRC, DST = Path(args[0]), Path(args[1])
MODE = args[2] if len(args) > 2 else "video"
QUALITY = args[7] if len(args) > 7 else "high"
TARGET_ARG = float(args[4]) if len(args) > 4 else None
PIVOT = args[5] if len(args) > 5 else "full"
DIST_SCALE = float(args[6]) if len(args) > 6 else 1.0
FRAMES_ARG = args[3] if len(args) > 3 else None
DST.mkdir(parents=True, exist_ok=True)

verts = np.load(SRC / "verts.npy")
faces = np.load(SRC / "faces.npy")
T = len(verts)
V = np.stack([verts[..., 0], -verts[..., 2], verts[..., 1]], axis=-1)
print("  顶点 %s  帧数 %d" % (verts.shape, T))

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
mesh = bpy.data.meshes.new("body")
mesh.from_pydata([tuple(v) for v in V[0]], [], [tuple(int(i) for i in f) for f in faces])
mesh.update()
obj = bpy.data.objects.new("body", mesh)
scn.collection.objects.link(obj)


def swap(scene):
    i = scene.frame_current - 1
    if 0 <= i < T:
        mesh.vertices.foreach_set("co", V[i].ravel())
        mesh.update()


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(swap)

bm = bpy.data.materials.new("body"); bm.use_nodes = True
bs = bm.node_tree.nodes["Principled BSDF"]
bs.inputs["Base Color"].default_value = (0.82, 0.76, 0.70, 1)
bs.inputs["Roughness"].default_value = 0.55
obj.data.materials.append(bm)

gmat = bpy.data.materials.new("ground"); gmat.use_nodes = True
gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.33, 0.35, 0.40, 1)
gmat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9

gy = float(V[..., 2].min())
root = V[:, 0, :]
bpy.ops.mesh.primitive_plane_add(size=60, location=(float(root[:, 0].mean()), float(root[:, 1].mean()), gy - 0.005))
bpy.context.active_object.data.materials.append(gmat)

key = bpy.data.lights.new("key", type="SUN"); key.energy = 3.0; key.angle = math.radians(9)
ko = bpy.data.objects.new("key", key)
ko.rotation_euler = (math.radians(54), math.radians(4), math.radians(38))
scn.collection.objects.link(ko)
fill = bpy.data.lights.new("fill", type="SUN"); fill.energy = 1.2
fo = bpy.data.objects.new("fill", fill)
fo.rotation_euler = (math.radians(68), 0, math.radians(215))
scn.collection.objects.link(fo)

world = bpy.data.worlds.new("w"); scn.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.56, 0.58, 0.63, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.85

scn.render.resolution_x = 720
scn.render.resolution_y = 720
scn.render.fps = 20
cd = bpy.data.cameras.new("cam"); cd.lens = 45
cam = bpy.data.objects.new("cam", cd); scn.collection.objects.link(cam); scn.camera = cam

per_h = V[..., 2].max(axis=1) - V[..., 2].min(axis=1)
height = float(np.median(per_h))
# upper：只看上半身（腰以上）；full：全身
if PIVOT == "upper":
    body_h = height * 0.52
    aim_dz = height * 0.26
else:
    body_h = height
    aim_dz = -height * 0.10
v_fov = 2 * math.atan(24.0 / (2 * cd.lens))
dist = (height / 0.62) / (2 * math.tan(v_fov / 2))
az = math.radians(38)
offx, offy = math.sin(az) * dist, -math.cos(az) * dist
# offx/offy 会在后面的校正里随 dist 更新
dist = dist * DIST_SCALE
for i in range(T):
    cx, cy, cz = float(root[i, 0]), float(root[i, 1]), float(root[i, 2])
    offx, offy = math.sin(az) * dist, -math.cos(az) * dist
    eye = Vector((cx + offx, cy + offy, cz + height * 0.30))
    aim = Vector((cx, cy, cz - height * 0.10))
    cam.location = eye
    cam.rotation_euler = (aim - eye).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_euler", frame=i + 1)
# 用投影把人物高度校正到画面高度的目标比例。
# 手算视场会错：Blender 的 sensor fit 与画幅比例有关，算出来的距离差三倍多。
from bpy_extras.object_utils import world_to_camera_view
TARGET_FRAC = TARGET_ARG if TARGET_ARG else (0.80 if PIVOT == "upper" else 0.78)
probe = int(T * 0.35)
cx, cy, cz = (float(root[probe, 0]), float(root[probe, 1]), float(root[probe, 2]))
frac = 0.0
for _it in range(14):
    eye = Vector((cx + math.sin(az) * dist, cy - math.cos(az) * dist, cz + height * 0.30))
    aim = Vector((cx, cy, cz + aim_dz))
    cam.location = eye
    cam.rotation_euler = (aim - eye).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    ph = world_to_camera_view(scn, cam, Vector((cx, cy, cz + aim_dz + body_h * 0.5)))
    pl = world_to_camera_view(scn, cam, Vector((cx, cy, cz + aim_dz - body_h * 0.5)))
    frac = abs(ph.y - pl.y)
    if frac < 1e-6 or abs(frac - TARGET_FRAC) < 0.01:
        break
    dist = dist * (frac / TARGET_FRAC)
print("  身高 %.2f m  取景 %s  相机距离 %.2f m  缩放 %.2f  投影占比 %.0f%%（目标 %.0f%%）"
      % (height, PIVOT, dist, DIST_SCALE, 100 * frac, 100 * TARGET_FRAC))
print("  身高 %.2f m  相机距离 %.2f m  人物占画面高度 %.0f%%" % (height, dist, 100 * TARGET_FRAC))

scn.frame_start = 1
scn.frame_end = T
try:
    scn.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scn.render.engine = "BLENDER_EEVEE"

# 画质档：fast 用于预览与批量检查，high 用于出片
if QUALITY == "fast":
    # 实测：分辨率对耗时影响最小（720→540 只省 14%），采样与阴影才是大头
    # 所以 fast 档保留 720p，只降采样、关阴影
    pass
    try:
        scn.eevee.taa_render_samples = 8
    except AttributeError:
        pass
    try:
        scn.eevee.use_shadows = False
    except AttributeError:
        pass
    try:
        scn.eevee.use_raytracing = False
    except AttributeError:
        pass
print("  画质档 %s  分辨率 %dx%d" % (QUALITY, scn.render.resolution_x, scn.render.resolution_y))

# 环境变量可单项覆盖，便于做消融
import os as _os
_res = _os.environ.get("HM_RENDER_RES")
if _res:
    scn.render.resolution_x = int(_res)
    scn.render.resolution_y = int(_res)
_smp = _os.environ.get("HM_RENDER_SAMPLES")
if _smp:
    try:
        scn.eevee.taa_render_samples = int(_smp)
    except AttributeError:
        pass
if _os.environ.get("HM_RENDER_SHADOW", "1") == "0":
    try:
        scn.eevee.use_shadows = False
    except AttributeError:
        pass
print("  生效分辨率 %dx%d" % (scn.render.resolution_x, scn.render.resolution_y))

if MODE == "range":
    # 只渲染指定帧区间，用来单独看过渡段
    a, b = [int(x) for x in (FRAMES_ARG or "1,1").split(",")]
    scn.frame_start = max(1, a)
    scn.frame_end = min(T, b)
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.filepath = str(DST / ("range_%d_%d.mp4" % (scn.frame_start, scn.frame_end)))
    print("  渲染帧 %d-%d -> %s" % (scn.frame_start, scn.frame_end, scn.render.filepath))
    bpy.ops.render.render(animation=True)
elif MODE != "still":
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.filepath = str(DST / "out.mp4")
    bpy.ops.render.render(animation=True)
else:
    scn.render.image_settings.file_format = "PNG"
    idx = ([int(x) for x in FRAMES_ARG.split(",")] if FRAMES_ARG else
           [1, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T])
    for i in idx:
        i = max(1, min(i, T))
        scn.frame_set(i)
        scn.render.filepath = str(DST / ("f%04d.png" % i))
        bpy.ops.render.render(write_still=True)
print("  完成")