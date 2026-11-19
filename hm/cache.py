"""内容寻址缓存：同一份输入只算一次。

为什么缓存属于「我们的方法」而不是硬件
    改一句里的一段话、调一次判据，重跑整条句子时，
    没变的部分（同样的提示词、同样的关节、同样的参数）不该重算。
    这跟 GPU 多快无关，是调度问题。

三处缓存
    生成  key = (后端, 提示词, 帧数, 候选数)      → feats_<i>.npy
    拟合  key = (关节字节, 方法, 迭代, 精修轮数)   → params.npy
    渲染  key = (参数字节, 画质, 分辨率)           → final.mp4

失效方式
    每个 key 前缀带 CACHE_VERSION，改算法时手动加一；
    环境变量 HM_NO_CACHE=1 完全绕过缓存，便于做干净的对照实验。
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

from hm.jsonio import load_json

CACHE_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent / "outputs" / "_cache"


def enabled():
    return os.environ.get("HM_NO_CACHE", "0") != "1"


def _key(*parts):
    h = hashlib.sha1()
    h.update(("v%d|" % CACHE_VERSION).encode())
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def key(parts):
    return _key(*parts)


def path(kind, k, suffix):
    d = ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / ("%s.%s" % (k, suffix))


def get_json(kind, k):
    # 缓存损坏当作没命中，重算即可，所以不必吵；但程序错误不吞
    return load_json(path(kind, k, "json"), warn_on_corrupt=False)


def put_json(kind, k, obj):
    path(kind, k, "json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def get_file(kind, k, suffix):
    p = path(kind, k, suffix)
    return p if p.exists() else None


def put_file(kind, k, suffix, src):
    dst = path(kind, k, suffix)
    shutil.copyfile(src, dst)
    return dst


def stats():
    if not ROOT.exists():
        return {}
    out = {}
    for d in sorted(ROOT.iterdir()):
        if d.is_dir():
            out[d.name] = len(list(d.glob("*")))
    return out
