# 项目结构

一句话：中文描述 → 人体动作视频。规划交给大模型，生成、拟合、装配、渲染都在本机 GPU 上。

## 目录

```
hm/                 主包，41 个模块，只放「会被别的代码 import 或作为命令入口」的东西
  main.py             唯一入口：一句中文 → 视频 + 分镜（--explain / --compare / --check / --batch）
  run_full.py         流程编排（A01–A19）
  run_job.py          各步骤实现（规划、路由、生成、检查、选择、交界修复、合并）
  planner*.py         分镜规划（prompt 负责提示词，fallback 负责大模型不可用时的兜底）
  generate.py         动作生成后端调用
  to_joints.py        263 维特征 → 22 关节
  fit.py ik_init.py shape.py   关节 → SMPL 参数（解析 IK + 体型反解）
  assemble_params.py  参数空间装配（过渡、截尾、交界）
  render_blender.py   渲染（Blender 后台）
  signature.py        动作签名 + 校准过的判据（G29 登记表）
  criteria_v2.py criteria_backend.py gates.py   几何/后端判据与自检门限
  cache.py            内容寻址缓存（HM_NO_CACHE=1 关闭）
  guard.py selfcheck.py capability_report.py    守门、自检、能力表
  prompt_bank.py prompt_bank_build.py prompt_variants.py prompt_normalize.py  措辞与规范化
  experiments/        35 个一次性探查/校准脚本，不被主链引用，允许跑不通
  legacy/             18 个被替代的旧实现，保留备查
tests/              153 个测试，其中 test_layout.py 守包结构（导入必须解析、核心不许反向依赖 legacy）
scratch/            临时脚本，按月归档到 scratch/archive/YYYY-MM/，不进导入链
eval_big.py         评测：30 句标准集 / 10 句新句集，按轮次存结果
eval_fresh.txt      新句集（未参与调参）
eval_new.py         早期十句评测（保留）
annotate_video.py frame_crop.py job_report.py frame_kit.py   后处理：裁切、字幕、分镜、交付件
目标树.md            唯一维护文档：目标、判据、实测、边界
docs/               设计说明书、PPT 等材料
```

## 常用命令

```
python -m hm.main "一句话"                 一条命令出片，出口码 0/1/2
python -m hm.main "一句话" --explain        只规划，说明这句话能保证到什么程度
python -m hm.main "一句话" --compare        预测与实际并排
python -m hm.main --check                  三句冒烟
python -m hm.main --batch 文件.txt          批量
python -m hm.guard                         守门七项（含新句集可疑数必须为 0）
python -m hm.selfcheck                     守门 + 最新一轮评测摘要
python -m hm.capability_report             能力表（每行标来源，可重算）
python -m pytest tests -q                  单元测试
python eval_big.py --round N [--set fresh]  跑评测
```

## 约定

1 一个功能只有一处实现：选择、入口、查表都曾出现两套实现互相覆盖，代价是记录与成片不一致。
2 改完必须跑一条完整句子并看日志，确认走的是改过的那条路径。
3 判据只用「先校准过」的量；校不开就不设规则，把「无法判」显式写出来。
4 数字不手抄：报告里的数一律由 `hm.capability_report` 从结果目录重算。
5 远端（GPU 机器）是运行环境，版本以本仓库为准；同步方式见「部署」。

## 部署

```
远端  chuan@100.124.24.52:~/motion-work   模型、Blender、环境变量 env.sh 都在那里
取回  scp <远端>:~/motion-work/outputs/<任务>/render/final_captioned.mp4 .
产物  每个任务目录：render/final*.mp4、分镜.png、semantics.json、storyboard.json、run.log
```
