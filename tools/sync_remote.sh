#!/usr/bin/env bash
# 把本仓库的当前提交同步到 GPU 机器，让「跑的代码」与「仓库代码」是同一份。
#
# 用法（在本地仓库根目录执行）：
#   bash tools/sync_remote.sh [远端地址]
# 默认远端：chuan@100.124.24.52:~/motion-work
#
# 做法：git archive 出当前 HEAD 的干净快照 → scp → 远端删掉旧 hm/ 再解包 → 记 MIRROR.txt → 跑守门
# 不做的事：不动远端的数据目录（outputs/models/datasets/motius/blender）与 env.sh 里的环境变量
set -euo pipefail

REMOTE="${1:-chuan@100.124.24.52:~/motion-work}"
SHA="$(git rev-parse --short HEAD)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 打包 HEAD=$SHA =="
git archive --format=tar -o "$TMP/repo.tar" HEAD

echo "== 上传 =="
scp -q "$TMP/repo.tar" "${REMOTE%%:*}:~/repo.tar"

echo "== 远端替换 hm/ 并解包 =="
ssh "${REMOTE%%:*}" "cd ~/motion-work && \
  mv hm /tmp/hm.old.\$(date +%s) && \
  tar -xf ~/repo.tar && rm -f ~/repo.tar && \
  echo 'MIRROR=$SHA  synced_at='\"\$(date -Iseconds)\" > MIRROR.txt && \
  echo \"  hm 顶层 \$(ls hm/*.py | wc -l)｜experiments \$(ls hm/experiments/*.py | wc -l)｜legacy \$(ls hm/legacy/*.py | wc -l)\""

echo "== 远端守门 =="
ssh "${REMOTE%%:*}" "cd ~/motion-work && source env.sh >/dev/null 2>&1 && python -m hm.guard 2>&1 | tail -n 4"