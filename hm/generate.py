"""文本 → 263 维特征。封装 motius 的 pipeline。"""
import numpy as np

from hm import config

config.setup_python()

_PIPES = {}


def load(name=None, device=None):
    """加载并缓存一个模型。name 取 momask / mdm / motionlcm。"""
    name = name or config.DEFAULT_MODEL
    if name in _PIPES:
        return _PIPES[name]
    path = config.MODEL_PATHS.get(name)
    if path is None:
        raise ValueError("未知模型: %s" % name)
    if not path.exists():
        raise FileNotFoundError("模型目录不存在: %s" % path)
    device = device or "cuda"
    from motius.pipelines.auto import Pipeline
    pipe = Pipeline.from_pretrained(str(path), bundle_kwargs={"device": device})
    _PIPES[name] = pipe
    return pipe


def generate(prompts, lengths, name=None):
    """一批提示词 → 一批特征。返回 [(T,263), ...]。"""
    pipe = load(name)
    out = pipe.infer_t2m(list(prompts), [int(n) for n in lengths])
    res = []
    for a in out:
        F = np.asarray(a, dtype=np.float32)
        if F.ndim == 3:
            F = F[0]
        if F.shape[-1] != config.FEAT_DIM:
            raise ValueError("特征维度不是 %d: %s" % (config.FEAT_DIM, F.shape))
        res.append(F)
    return res


def selftest():
    """自检：生成一段走路的特征。"""
    ok = True
    feats = generate(["a person walks forward"], [56])
    F = feats[0]
    if F.shape != (56, 263):
        ok = False
        print("  形状不对:", F.shape)
    if not np.isfinite(F).all():
        ok = False
        print("  特征含 NaN/Inf")
    print("  生成自检: %s  形状 %s  范围 %.2f ~ %.2f" % (
        "通过" if ok else "不通过", F.shape, F.min(), F.max()))
    return ok


if __name__ == "__main__":
    selftest()
