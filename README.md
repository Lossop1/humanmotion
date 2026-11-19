# humanmotion

把一句中文长句变成一段人体动作视频：

```
中文长句 → 分镜（LLM 规划） → 逐段生成动作（MoMask / MDM） → 语义与几何校验
        → 段间装配 → 关节到 SMPL 网格 → Blender 渲染 → final.mp4 + 分镜图
```

项目的重点不是"能出片"，而是**每一步都有可校准的判据**：出的片附带 `semantics.json`，
说明每一段是被验证通过、可疑，还是无法判断；评测数字由代码从存储结果重算，不手抄。

设计依据、各节点的判据、实测数字和被否证的方案，都记录在 [`目标树.md`](目标树.md)
（唯一维护文档）。**先读它，再改代码。**

## 目录

```
hm/                 主包：规划、生成、装配、拟合、渲染、判据、评测（入口 main → run_full → run_job）
hm/legacy/          run_full 之前的旧拼接/检查链路，只留作对照；核心代码不许依赖它
hm/experiments/     一次性探针、诊断与小实验，结论已记在 目标树.md；需要模型与数据才能运行
tests/              单元测试（纯 CPU，不需要模型与 GPU）；test_layout 守住包结构与导入
outputs/_reference/ 能力表、提示词库、判据校准数据（入库，评测依赖它们）
eval_big.py         30 句 × 两轮的评测；eval_fresh.txt 是未参与调参的新句集
scratch/            一次性实验脚本的归档，不属于产品代码，不保证能运行
目标树.md            目标、判据、状态、否证记录
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                               # 只做规划/单测：够用
pip install -e ".[gpu,render]"                        # 完整流水线：另需下面的外部依赖
```

### 外部依赖（不在仓库里）

| 依赖 | 默认位置 | 说明 |
|---|---|---|
| `momask-torch`、`motius`、`mdm` | `<DEPS>/` | 动作生成后端 |
| 模型权重 `models/Motius-*` | `<DEPS>/models/` | MoMask / MDM / MotionLCM / Evaluator |
| HumanML3D 数据集 | `<DEPS>/datasets/HumanML3D` | 仅校准与评测需要 |
| Blender 4.5 | 自动探测 | 渲染需要 |

`<DEPS>` 默认是仓库根目录，可用 `HM_DEPS` 改。以上目录都已在 `.gitignore` 中。
Blender 的探测顺序：`HM_BLENDER` → `PATH` 中的 `blender` → `<DEPS>/blender-*/`。

### LLM 规划器的密钥

分镜规划调用 DeepSeek。请用环境变量提供密钥：

```bash
export DEEPSEEK_API_KEY=...        # Windows PowerShell: $env:DEEPSEEK_API_KEY="..."
```

旧版的明文文件 `.deepseek_key` 仍能读取，但会发出警告，不建议使用。
大模型不可用（欠费、断网、没配密钥）时，规划自动退到规则兜底，只认词表里的中文动作。

## 使用

```bash
python -m hm.main "一个人先向前走，然后转身，最后挥手"      # 出片
python -m hm.main --explain "一句话"                         # 只规划，说明能保证到什么程度
python -m hm.main --batch sentences.txt                      # 批量，每行一句
python -m hm.main --check                                    # 冒烟：三句各跑一遍
```

出口码：`0` 出片且几何通过；`1` 出片但有问题（有段被丢弃或几何不过）；`2` 没出片。

每个任务写到 `runs/<job-id>/`：`final.mp4`、`final_cropped.mp4`、`final_captioned.mp4`、
`分镜.png`、`semantics.json`。

## 常用环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HM_DEPS` | 仓库根 | 模型、数据集、外部仓库所在目录 |
| `HM_BLENDER` | 自动探测 | Blender 可执行文件 |
| `HM_OUT` | `<repo>/runs` | 输出根目录 |
| `HM_NO_CACHE` | `0` | `1` 绕过内容寻址缓存（做对照实验时用） |
| `HM_PLAN_FALLBACK` | `0` | `1` 强制用规则兜底规划（量化没有大模型时的退化） |
| `HM_DEGRADE` | `drop` | 有段失败时的策略：`drop` / `clip` / `abort` |
| `HM_RENDER_QUALITY` | `fast` | 渲染质量；`HM_RENDER_RES`、`HM_RENDER_SAMPLES` 可单独覆盖 |
| `DEEPSEEK_API_KEY` | 无 | 规划器密钥 |

## 开发

```bash
pytest                       # 单元测试，秒级
ruff check .                 # 目前只拦语法错误与未定义名，规则会逐步收紧
python -m hm.selfcheck       # 提交前自检：守门 + 最近一轮评测摘要
```

约定：

- 新指标先在已知对照上校准（负对照判空、正对照判中），不能分离就记入"否证"，不做门禁。
- 改动影响评测数字时，同时更新 `目标树.md` 里对应节点；数字从 `hm.capability_report` 取，不手写。
- 评测只报"逐句两轮平均"并附四分位，单轮数字不可用。
- 新句子进 `eval_fresh.txt` 的前提是它没被用来调过参。
- 实验脚本放 `scratch/`，要沉淀成产品能力再搬进 `hm/` 并补测试。
