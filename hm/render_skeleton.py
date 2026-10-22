"""骨架渲染：把 22 个关节画成圆柱连成的骨架。

为什么要这个：真实动捕数据只有 22 个关节坐标，没有网格。
要把它和生成的动画并排比较，必须用两边共同有的东西——骨架。
同一个相机、同一种线条，差异就只剩动作本身。
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
SRC = Path(args[0])
DST = Path(args[1])
MODE = args[2] if len(args) > 2 else "video"
FRAMES_ARG = args[3] if len(args) > 3 else None
DST.mkdir(parents=True, exist_ok=True)

J = np.load(SRC / "joints.npy")
T = len(J)
print("  骨架 %s  帧数 %d" % (J.shape, T))

# y 上 -> z 上
V = np.stack([J[..., 0], -J[..., 2], J[..., 1]], axis=-1)

PARENT = {1: 0, 4: 1, 7: 4, 10: 7, 2: 0, 5: 2, 8: 5, 11: 8,
          3: 0, 6: 3, 9: 6, 12: 9, 15: 12,
          13: 9, 16: 13, 18: 16, 20: 18, 14: 9, 17: 14, 19: 17, 21: 19}
BONES = sorted(PARENT.items())

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene

mat = bpy.data.materials.new("bone"); mat.use_nodes = True
b = mat.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.90, 0.55, 0.30, 1)
b.inputs["Roughness"].default_value = 0.4

# 每个关节一个球，每根骨头一个圆柱（用缩放和旋转摆位）
objs = {}
for j in range(22):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(0, 0, 0))
    o = bpy.context.active_object
    o.data.materials.append(mat)
    objs[("j", j)] = o
for c, p in BONES:
    bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=1.0, location=(0, 0, 0))
    o = bpy.context.active_object
    o.data.materials.append(mat)
    objs[("b", c)] = o


def place(i):
    P = V[i]
    for j in range(22):
        objs[("j", j)].location = (float(P[j, 0]), float(P[j, 1]), float(P[j, 2]))
    for c, p in BONES:
        a = P[p]
        bb = P[c]
        d = bb - a
        L = float(np.linalg.norm(d))
        cy = objs[("b", c)]
        if L < 1e-6:
            cy.scale = (0, 0, 0)
            continue
        cy.scale = (1.0, 1.0, L)
        cy.location = tuple((a + bb) / 2.0)
        # 把圆柱的 Z 轴转到 d 方向
        z = np.array([0.0, 0.0, 1.0])
        dn = d / L
        axis = np.cross(z, dn)
        s = float(np.linalg.norm(axis))
        if s < 1e-8:
            cy.rotation_mode = "QUATERNION"
            cy.rotation_quaternion = (1, 0, 0, 0) if dn[2] > 0 else (0, 1, 0, 0)
        else:
            axis = axis / s
            ang = math.acos(float(np.clip(np.dot(z, dn), -1, 1)))
            cy.rotation_mode = "QUATERNION"
            cy.rotation_quaternion = (math.cos(ang / 2), axis[0] * math.sin(ang / 2),
                                      axis[1] * math.sin(ang / 2), axis[2] * math.sin(ang / 2))
    # 地面
    pass


def upd(scene):
    i = scene.frame_current - 1
    if 0 <= i < T:
        place(i)


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(upd)
place(0)

# 地面
gy = float(V[..., 2].min()) - 0.02
bpy.ops.mesh.primitive_plane_add(size=30, location=(float(V[:, 0, 0].mean()), float(V[:, 0, 1].mean()), gy))
gm = bpy.data.materials.new("g"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.30, 0.32, 0.36, 1)
bpy.context.active_object.data.materials.append(gm)

ld = bpy.data.lights.new("k", type="SUN"); ld.energy = 3.0
lo = bpy.data.objects.new("k", ld); lo.rotation_euler = (math.radians(55), 0, math.radians(35))
scn.collection.objects.link(lo)
w = bpy.data.worlds.new("w"); scn.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.62, 0.66, 1)

cd = bpy.data.cameras.new("c"); cd.lens = 50
cam = bpy.data.objects.new("c", cd); scn.collection.objects.link(cam); scn.camera = cam
root_all = V[:, 0, :]
center = root_all.mean(axis=0)
hspan = float(V[..., 2].max() - V[..., 2].min())
# 相机距离锁成固定值，两份数据才能在同一比例下并排比较。
# 之前按各自动作的跨度算，两边比例不同，看出的差别不可信。
FIXED_DIST = 4.2
try:
    import os
    FIXED_DIST = float(os.environ.get("SK_DIST", "4.2"))
except Exception:
    pass
dist = FIXED_DIST
cam.location = (center[0] + dist * 0.6, center[1] - dist * 0.8, gy + hspan * 0.55)
cam.rotation_euler = (center + np.array([0, 0, hspan * 0.3]) - np.array(cam.location)
                      ).to_track_quat("-Z", "Y").to_euler() if hasattr(np.array(cam.location), "to_track_quat") else cam.rotation_euler
from mathutils import Vector
cam.rotation_euler = (Vector(center + np.array([0, 0, hspan * 0.3])) -
                      Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()

scn.frame_start = 1
scn.frame_end = T
scn.render.resolution_x = 640
scn.render.resolution_y = 640
scn.render.fps = 20
try:
    scn.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scn.render.engine = "BLENDER_EEVEE"

if MODE == "still":
    import numpy as np2
    scn.render.image_settings.file_format = "PNG"
    idx = ([int(x) for x in FRAMES_ARG.split(",")] if FRAMES_ARG else
           list(range(1, T + 1, max(1, T // 12))))
    for i in idx:
        i = max(1, min(i, T))
        scn.frame_set(i)
        scn.render.filepath = str(DST / ("f%04d.png" % i))
        bpy.ops.render.render(write_still=True)
else:
    scn.render.image_settings.file_format = "FFMPEG"
    scn.render.ffmpeg.format = "MPEG4"
    scn.render.ffmpeg.codec = "H264"
    scn.render.filepath = str(DST / "out.mp4")
    bpy.ops.render.render(animation=True)
print("  完成")