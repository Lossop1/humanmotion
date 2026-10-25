"""能力测试的可视化与人工标注材料。

阶段一（快）：16 动作 × 2 后端 × 3 采样 → 关节序列 + 每动作一张联络图
阶段二（慢）：逐段 SMPL 拟合与渲染 → 短片，供人工逐段判定

产物：
    outputs/_capsheet/joints/<action>_<model>_<n>.npy
    outputs/_capsheet/sheet_<action>.png        联络图（2 行后端 × 3 列采样，每格 3 个关键帧）
    outputs/_capsheet/index.json                每段的指标，供事后校准
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chuan/motion-work")
from hm import config, generate, job as jobmod, pose_rules, repair, to_joints
from hm.capability import ACTIONS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/chuan/motion-work")
OUT = ROOT / "outputs" / "_capsheet"
(OUT / "joints").mkdir(parents=True, exist_ok=True)

BONES = jobmod.BONES
MODELS = ("momask", "mdm")
K = 3


def draw(ax, J, color="#1f4e79"):
    ax.clear()
    for c, p in BONES:
        ax.plot([J[p, 0], J[c, 0]], [J[p, 1], J[c, 1]], "-", lw=2.2, color=color, solid_capstyle="round")
    ax.plot(J[:, 0], J[:, 1], "o", ms=2.0, color=color)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(J[:, 0].min() - 0.35, J[:, 0].max() + 0.35)
    ax.set_ylim(J[:, 1].min() - 0.15, J[:, 1].max() + 0.15)


def sheet(action_key, zh, store):
    fig, axes = plt.subplots(len(MODELS) * 2, K * 3,
                             figsize=(K * 3 * 1.55, len(MODELS) * 2 * 1.75))
    for mi, model in enumerate(MODELS):
        for ki in range(K):
            J = store.get((model, ki))
            for fi, frac in enumerate((0.15, 0.5, 0.85)):
                for vi, (title, row) in enumerate((("正面", 0), ("侧面", 1))):
                    ax = axes[mi * 2 + row, ki * 3 + fi]
                    if J is None:
                        ax.axis("off")
                        continue
                    t = int(frac * (len(J) - 1))
                    Jf = J[t] if row == 0 else J[t][:, [2, 1, 0]]
                    draw(ax, Jf, "#1f4e79" if model == "momask" else "#a0522d")
                    if row == 0:
                        ax.set_title("%s #%d  %d%%" % (model, ki + 1, int(frac * 100)), fontsize=7)
            axes[mi * 2, ki * 3].text(-0.08, 0.5, "", transform=axes[mi * 2, ki * 3].transAxes)
    fig.suptitle("%s  （%s；每行一后端，每列一采样；上排正面，下排侧面）"
                 % (zh, action_key), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / ("sheet_%s.png" % action_key)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main():
    bone_ref = jobmod.load_bone_reference()
    pose_th = pose_rules.load_thresholds()
    index = []
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for key, zh, prompt, kws, frames in ACTIONS:
        if only and key != only:
            continue
        store = {}
        for model in MODELS:
            feats = generate.generate([prompt] * K, [frames] * K, name=model)
            for i, F in enumerate(feats, 1):
                J = np.asarray(to_joints.from_feats(F), dtype=np.float64)
                np.save(OUT / "joints" / ("%s_%s_%d.npy" % (key, model, i)), J)
                store[(model, i)] = J
                v = jobmod.bone_lengths(J)
                rel = float((np.abs(v.mean(axis=0) - np.array(bone_ref["mean"]))
                             / np.array(bone_ref["mean"])).max())
                pose_ok, reasons, _, _ = pose_rules.check(J, prompt, pose_th)
                sem_ok, sem_rule, sem_detail = repair.semantic_proxy(J, prompt)
                index.append({
                    "action": key, "action_zh": zh, "model": model, "sample": i,
                    "file": str((OUT / "joints" / ("%s_%s_%d.npy" % (key, model, i)))
                                .relative_to(ROOT)),
                    "bone_rel_max": round(rel, 4),
                    "pose_ok": bool(pose_ok), "pose_reasons": reasons[:2],
                    "semantic_ok": sem_ok, "semantic_rule": sem_rule,
                    "semantic_detail": sem_detail,
                    "label": None,
                })
            print("  %-12s %-8s 关节已存" % (key, model), flush=True)
        p = sheet(key, zh, store)
        print("  联络图 %s" % p.name, flush=True)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  index.json 已写，共 %d 段" % len(index))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
