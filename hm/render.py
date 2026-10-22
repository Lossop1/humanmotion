"""网格 → 视频。起 Blender 子进程跑。"""
import subprocess

import numpy as np

from hm import config

SCRIPT = config.ROOT / "hm" / "render_blender.py"


def render(out_dir, name="video"):
    """out_dir 下要有 verts.npy 和 faces.npy。返回 mp4 路径。"""
    out_dir = config.ensure_out(str(out_dir)) if not hasattr(out_dir, "exists") else out_dir
    v = out_dir / "verts.npy"
    f = out_dir / "faces.npy"
    if not v.exists() or not f.exists():
        raise FileNotFoundError("缺 verts.npy 或 faces.npy：%s" % out_dir)
    if not config.BLENDER.exists():
        raise FileNotFoundError("找不到 Blender: %s" % config.BLENDER)
    dst = out_dir / "render"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [str(config.BLENDER), "--background", "--python", str(SCRIPT),
           "--", str(out_dir), str(dst), "video"]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    mp4 = dst / "out.mp4"
    if not mp4.exists():
        tail = (p.stdout or "")[-800:] + "\n" + (p.stderr or "")[-800:]
        raise RuntimeError("渲染失败:\n" + tail)
    return str(mp4)


def save_faces(out_dir):
    """从 SMPL 模型取出面片索引，存到 out_dir。只需做一次。"""
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path = __import__("pathlib").Path
    p = src(str(config.ROOT)) / "assets" / "SMPL_NEUTRAL.pkl"
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    faces = np.asarray(d["f"], dtype=np.int32)
    np.save(out_dir / "faces.npy", faces)
    return faces


def selftest():
    """自检：造一个原地旋转的方块，渲成视频，确认文件非空。"""
    out = config.ensure_out("selftest_render")
    T = 20
    # 用 SMPL 的顶点做输入，但只让骨盆上下动，检查流程通不通
    import pickle
    from pathlib import Path
    p = Path(str(config.ROOT)) / "assets" / "SMPL_NEUTRAL.pkl"
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    V0 = np.asarray(d["v_template"], dtype=np.float64)
    faces = np.asarray(d["f"], dtype=np.int32)
    V = np.repeat(V0[None], T, axis=0)
    V[..., 1] += np.linspace(0, 0.2, T)[:, None]
    np.save(out / "verts.npy", V.astype(np.float32))
    np.save(out / "faces.npy", faces)
    mp4 = render(out)
    import os
    size = os.path.getsize(mp4)
    ok = size > 10000
    print("  渲染自检: %s  %s  %.1f KB" % ("通过" if ok else "不通过", mp4, size / 1024))
    return ok


if __name__ == "__main__":
    selftest()
