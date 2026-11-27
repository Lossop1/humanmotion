"""随机句集：按种子生成一批句子，用来做"没见过的输入"检验。

为什么不用固定集
    使用者要求：检验不能重复用同一批句子，应该随机生成，才算真检验。
    所以这里按模板随机组合，把种子记下来（同一个种子可以复现同一批句子），
    但不同种子得到不同句子。

生成的三类（每类都要有）
    A 可做的组合：动作在能力表内、方向明确（应当出片且做对）
    B 死路动作：爬行/蹲下/躺下/跳舞（应当被拦住、出口码 2）
    C 长句/复合：两三个动作连写（考验拆分与编排）

用法
    python -m tools.make_random_set --n 20 --seed 20261127 > outputs/_random/set_seed.txt
"""
import argparse
import random
import sys

WALK = ["向前走", "向后走", "向左走", "向右走", "慢慢向前走", "快速向前跑", "向前冲刺几步"]
SECOND = ["挥手", "举起双手", "出拳", "踢腿", "跳一下", "停下", "鞠躬", "转身"]
ADVERB = ["慢慢", "快速", "缓缓", "突然"]
DEAD = ["爬行", "蹲下", "躺下", "跳舞", "跪下"]
PLACES = ["离开房间", "走到桌子边", "走向门口", "走到窗边"]


def build(n, seed):
    rng = random.Random(seed)
    out = []
    # A 类：可做组合（约 60%）
    while len(out) < int(n * 0.6):
        k = rng.choice([1, 1, 2])
        parts = [rng.choice(WALK)]
        if k == 2:
            parts.append(rng.choice(ADVERB) + rng.choice(SECOND))
        if rng.random() < 0.25:
            parts.append(rng.choice(WALK))
        s = "一个人" + "，然后".join(parts)
        if s not in out:
            out.append(s)
    # B 类：死路动作（约 20%）
    while len(out) < int(n * 0.8):
        s = "一个人" + rng.choice(ADVERB) + rng.choice(DEAD)
        if s not in out:
            out.append(s)
    # C 类：长句/复合（约 20%）
    while len(out) < n:
        parts = [rng.choice(WALK), rng.choice(ADVERB) + rng.choice(SECOND), rng.choice(WALK)]
        s = "一个人先" + "，再".join(parts[:2]) + "，最后" + parts[2]
        if s not in out:
            out.append(s)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261127)
    a = ap.parse_args()
    for i, s in enumerate(build(a.n, a.seed), 1):
        print(s)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
