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

## 目录与入口（唯一规范）

```
python -m hm.cli --help        唯一用户入口：出片、守门、评测、判定包、度量、报告、判定统计
python -m hm.main "一句话"      等价 hm.cli run，保留旧习惯
python -m hm.guard             守门八项自检
```

```
hm/                 主包：规划 / 生成 / 装配 / 拟合 / 渲染 / 判据 / CLI（hm.cli）
hm/ops/             评测、判定包、度量、报告等操作的实现（不再以独立脚本形式放在仓库根）
calibration/        标定数据唯一目录：能力表、提示词库、判据阈值（实际路径 outputs/_reference，
                    代码里所有引用都指向它；不要在这些数据之外新建标定文件）
tests/              单元测试（纯 CPU）
outputs/<job-id>/   每个任务的产物
docs/               只留两份文档：证据度量.md（数字与命令）、设计说明书.md
docs/_archive/      历史文档，仅供查阅，不再维护
目标树.md            目标、判据、状态、否证记录（改代码前先读）
```

仓库根不再有零散脚本（hm/ops/ 已并入 hm/ops/）。需要新功能请加到 `hm.cli` 的子命令里。

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
python -m hm.cli run "一个人先向前走，然后转身，最后挥手"     # 出片
python -m hm.cli run "一句话" --k 3                          # 指定候选数
python -m hm.cli guard                                      # 守门八项
python -m hm.cli eval --round 36                            # 30 句标准集评测
python -m hm.cli pack open                                  # 生成「只含未解决项」的判定包
python -m hm.cli pack verify                                # 8 条全项判定包
python -m hm.cli pack random --n 20 --seed 20261127         # 随机句集
python -m hm.cli score --verdicts 判定.json                  # 人工判定统计
```

出口码：`0` 出片且几何通过、没有被判可疑或近似代替的段；`1` 出片但有可疑段、近似代替的段或几何不过；`2` 没出片（做不了，或所有段都没通过检查）。

每个任务写到 `outputs/<job-id>/`：

```
render/final.mp4             原始渲染（1080 方形）
render/final_cropped.mp4     裁到人物
render/final_captioned.mp4   字幕 + 验证角标（看这个）
分镜.png                     每段一格，标题前带状态：OK 通过 / ? 未验证 / X 可疑
semantics.json               逐段语义状态与原因
storyboard.json              分镜与选择记录（prompt / source / 候选 / 选中项）
run.log                      逐步日志
```

## 常用环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HM_NO_CACHE` | 0 | 关内容缓存；改词表/提示词后必须设 1，否则同一句会拿到旧规划 |
| `HM_SEED` | 无 | 生成种子；复现同一批候选时用 |
| `HM_FORCE_BACKEND` | 无 | 强制后端（momask / mdm） |
| `HM_FIT_METHOD` | opt | 拟合方式（opt 或 ik） |
| `HM_TRANS_CAP` | 内置 | 交界过渡帧上限 |
| `HM_SMOOTH_FILTER` | 0 | 候选平滑过滤；开启会触发重生成，很慢 |
| `HM_SPEED_WARP` | 0 | 是否用帧数伸缩表达快慢；默认关（关掉才不会像快放） |
| `HM_RENDER_RES` | 1080 | 渲染分辨率 |
| `HM_RENDER_SAMPLES` | 内置 | 渲染采样数 |
| `HM_DEBUG_FACE` | 0 | 打印朝向校正的调试值 |
| `HM_DEPS` | 仓库根 | 模型、数据集所在目录 |
| `HM_BLENDER` | 自动探测 | Blender 可执行文件 |
| `HM_OUT` | `outputs` | 输出根目录 |
| `DEEPSEEK_API_KEY` | 无 | 规划用大模型的密钥；缺失时退规则兜底 |


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
