"""任务骨架：job 目录、阶段命名、候选元数据、统一记录格式。

对应《设计说明书_原子步骤》第 2 节与第 4 节：
    - 一条请求一个 job_id，所有文件在该任务目录内
    - 产物标注所处阶段（raw / decoded / retimed / assembled / smpl_fit / rendered）
    - 每个候选的 metrics.json 按固定字段写；没算的指标写 null 并注明原因，不填 0
    - 版本信息取不到时写 unknown，不留空
"""
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from hm import config

# 处理阶段。任何指标都必须标注它是在哪个阶段量的。
STAGES = ("raw", "decoded", "retimed", "assembled", "smpl_fit", "rendered")

# 决策取值，限定这五种。
DECISIONS = ("pass", "reject", "retry", "manual_review", "unsupported")

# 21 根骨头：(子关节, 父关节)
BONES = [(1, 0), (4, 1), (7, 4), (10, 7),
         (2, 0), (5, 2), (8, 5), (11, 8),
         (3, 0), (6, 3), (9, 6), (12, 9),
         (15, 12), (13, 9), (16, 13), (18, 16), (20, 18),
         (14, 9), (17, 14), (19, 17), (21, 19)]

BONE_NAMES = ["骨盆-左髋", "左髋-左膝", "左膝-左踝", "左踝-左脚",
              "骨盆-右髋", "右髋-右膝", "右膝-右踝", "右踝-右脚",
              "骨盆-脊柱1", "脊柱1-脊柱2", "脊柱2-脊柱3", "脊柱3-颈",
              "颈-头", "脊柱3-左锁骨", "左锁骨-左肩", "左肩-左肘",
              "左肘-左腕", "脊柱3-右锁骨", "右锁骨-右肩", "右肩-右肘",
              "右肘-右腕"]

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def bone_lengths(J):
    """(T,22,3) → (T,21) 每帧每根骨头的长度。"""
    J = np.asarray(J, dtype=np.float64)
    return np.stack([np.linalg.norm(J[:, c] - J[:, p], axis=1) for c, p in BONES], axis=1)


def build_bone_reference(pattern="real40/*.npy", out_dir=None):
    """用真实动捕统计骨长的容许区间，写成一个带版本号的文件。

    依据说明书 A08：「容许区间由训练/验证数据统计确定并记录版本，
    不能临时为单个坏候选放宽」。
    """
    files = sorted(config.ROOT.glob(pattern)) if hasattr(config, "ROOT") else []
    if not files:
        root = Path(__file__).resolve().parent.parent
        files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError("找不到参考动作：%s" % pattern)
    all_L = []
    for f in files:
        J = np.load(f)
        if J.ndim == 3 and J.shape[1] == 22:
            all_L.append(bone_lengths(J))
    L = np.concatenate(all_L, axis=0)
    ref = {
        "version": "real40_v1",
        "source": pattern,
        "files": len(files),
        "frames": int(len(L)),
        "mean": [float(x) for x in L.mean(axis=0)],
        "std": [float(x) for x in L.std(axis=0)],
        "bone_names": BONE_NAMES,
        "tolerance_sigma": 3.0,
        "relative_tolerance": 0.05,
        "corruption_tolerance": 0.20,
        "relative_tolerance_note": "5% 是质量偏好（软指标）；20% 才是硬失败阈值。第一轮能力测试 96 个样本的分布：P50=6.3%、P90=16.5%、最大 50.5%，5% 会误杀 60% 的正常样本，15% 又让蹲坐躺全部通过，因此硬阈值只用来抓损坏",
        "note": "40 条真实动捕的骨长标准差约等于 0（同一套骨架重定向），sigma 判据退化，因此改用相对偏差 5% 作为容许带",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir = Path(out_dir) if out_dir else OUTPUT_ROOT / "_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / ("bone_length_%s.json" % ref["version"])
    p.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    return ref, p


def load_bone_reference(path=None):
    if path is None:
        cand = sorted((OUTPUT_ROOT / "_reference").glob("bone_length_*.json"))
        if not cand:
            return None
        path = cand[-1]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_rev(root):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):   # 没装 git / 超时：版本号只是记录用
        return "unknown"


class Job:
    """一个任务的全部目录与记录。"""

    def __init__(self, sentence, job_id=None, note=""):
        if not str(sentence).strip():
            raise ValueError("指令为空")
        self.sentence = str(sentence)
        self.job_id = job_id or time.strftime("job_%Y%m%d_%H%M%S")
        self.dir = OUTPUT_ROOT / self.job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.note = note
        self._log_lines = []

    # ---------- 记录 ----------
    def log(self, msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        self._log_lines.append(line)
        with open(self.dir / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print("    " + msg, flush=True)

    def write_request(self, config_snapshot):
        """A01：原样保存指令与运行配置快照。"""
        req = {
            "job_id": self.job_id,
            "sentence_original": self.sentence,
            "sentence_normalized": " ".join(self.sentence.split()),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_rev": _git_rev(Path(__file__).resolve().parent.parent),
            "config": config_snapshot,
            "note": self.note,
        }
        (self.dir / "request.json").write_text(
            json.dumps(req, ensure_ascii=False, indent=1), encoding="utf-8")
        return req

    def save_json(self, rel, obj):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def load_json(self, rel):
        return json.loads((self.dir / rel).read_text(encoding="utf-8"))

    # ---------- 路径 ----------
    def seg_dir(self, seg_id):
        d = self.dir / "segments" / seg_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cand_dir(self, seg_id, n):
        d = self.seg_dir(seg_id) / ("candidate_%d" % n)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def assembled_dir(self):
        d = self.dir / "assembled"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def smpl_dir(self):
        d = self.dir / "smpl"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def render_dir(self):
        d = self.dir / "render"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------- 记录格式 ----------
    def write_candidate_metrics(self, seg_id, cand_id, m):
        """按说明书第 4 节的字段写 metrics.json；缺字段直接报错，避免写出半张表。"""
        need = ["job_id", "segment_id", "candidate_id", "stage", "model",
                "model_version", "seed", "frames", "fps", "coordinate_system",
                "hard_checks", "metrics", "decision", "failure_reasons"]
        miss = [k for k in need if k not in m]
        if miss:
            raise ValueError("metrics 缺字段：%s" % miss)
        if m["stage"] not in STAGES:
            raise ValueError("未知阶段：%s" % m["stage"])
        if m["decision"] not in DECISIONS:
            raise ValueError("未知决策：%s" % m["decision"])
        n = int(str(cand_id).split("_")[-1])
        p = self.cand_dir(seg_id, n) / "metrics.json"
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def append_event(self, event):
        """修复与失败事件：追加写入，不覆盖旧记录。"""
        p = self.dir / "events.jsonl"
        rec = dict(event)
        rec.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        rec.setdefault("job_id", self.job_id)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self):
        return {
            "job_id": self.job_id,
            "dir": str(self.dir),
            "elapsed_s": round(time.time() - self.t0, 1),
        }
