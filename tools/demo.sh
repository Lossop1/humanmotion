#!/usr/bin/env bash
# 一键演示：跑一句 → 打印四件产物 → 打印该段语义状态与交界结论 → 退出码与失败原因
#
# 用法（在装了环境的机器上跑；本项目的环境在 GPU 机器）
#   bash tools/demo.sh                                  # 用默认句子
#   bash tools/demo.sh "一个人先向前走，然后转身，最后挥手告别"
#   bash tools/demo.sh "句子" --k 4 --job-id demo1
#   bash tools/demo.sh --check                          # 只做冒烟（三句）
#
# 从别的机器调用：
#   ssh chuan@100.124.24.52 "cd ~/motion-work && bash tools/demo.sh '一句话'"
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "找不到仓库根目录"; exit 2; }

SENTENCE="${1:-}"
if [ "$SENTENCE" = "--check" ]; then
  shift || true
  python -m hm.main --check "$@"
  exit $?
fi
if [ -z "$SENTENCE" ]; then
  SENTENCE="一个人先向前走，然后转身，最后挥手告别"
fi
shift || true

JOB="demo_$(date +%H%M%S)"
K=4
while [ $# -gt 0 ]; do
  case "$1" in
    --k) K="$2"; shift 2;;
    --job-id) JOB="$2"; shift 2;;
    *) echo "未知参数：$1"; exit 2;;
  esac
done

echo "=============================================================="
echo " 演示：$SENTENCE"
echo " 任务：$JOB（每段候选 k=$K）"
echo "=============================================================="

LOG="$(mktemp)"
rm -rf "outputs/$JOB"
python -m hm.main "$SENTENCE" --job-id "$JOB" --k "$K" >"$LOG" 2>&1
RC=$?

# 过程日志只留关键行，其余不刷屏；出错时把尾部原样给出来
grep -E "规划|分镜|可支持性|强制后端|提示词：|合并短段|段/帧|几何|去掉的段|交界|语义验证|近似代替|截尾|耗时" "$LOG" | sed 's/^/   /'

if [ "$RC" != "0" ] && [ "$RC" != "1" ] && [ "$RC" != "2" ]; then
  echo
  echo " 运行异常（退出码 $RC），日志尾部："
  tail -n 20 "$LOG" | sed 's/^/   /'
  rm -f "$LOG"
  exit "$RC"
fi

DIR="outputs/$JOB"
echo
echo " 产物 ------------------------------------------------------"
for f in "$DIR/render/final.mp4" "$DIR/render/final_cropped.mp4" "$DIR/render/final_captioned.mp4" "$DIR/分镜.png"; do
  if [ -f "$f" ]; then
    printf "   %-42s %8.1f KB\n" "${f#outputs/$JOB/}" "$(awk "BEGIN{print $(stat -c%s "$f")/1024}")"
  else
    printf "   %-42s %s\n" "${f#outputs/$JOB/}" "缺失"
  fi
done

if [ -f "$DIR/semantics.json" ] && [ -f "$DIR/assembled/bounds.json" ]; then
  python - "$DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
sem = json.loads((d / "semantics.json").read_text(encoding="utf-8"))
bj = json.loads((d / "assembled" / "bounds.json").read_text(encoding="utf-8"))

print()
print(" 语义状态 --------------------------------------------------")
marks = {"通过": "OK", "未验证": "??", "无法判": "??", "可疑": "XX", "未选中": "--"}
for row in sem:
    print("   seg_%s  %-4s %s" % (row["segment"], marks.get(row["status"], "?"),
                                  row.get("detail") or ""))
n_ok = sum(1 for r in sem if r["status"] == "通过")
n_un = sum(1 for r in sem if r["status"] in ("未验证", "无法判"))
n_bad = sum(1 for r in sem if r["status"] == "可疑")
print("   小结：通过 %d / 未验证 %d / 可疑 %d（共 %d 段）" % (n_ok, n_un, n_bad, len(sem)))

print()
print(" 交界结论 --------------------------------------------------")
js = bj.get("junctions", [])
if not js:
    print("   无交界（单段，或全部合并成一段）")
for j in js:
    gap = j["pose_gap_deg"]
    fr = max(j["transition_frames"], 1)
    rate = gap / fr
    br = j.get("boundary_rate_deg_per_frame") or 0
    ratio = rate / br if br else 0
    verdict = "合格" if 0.7 <= ratio <= 1.4 else "不合格"
    print("   %s→%s  姿态差 %.0f 度、过渡 %d 帧 → %.1f 度/帧；两侧 %.1f → 速率比 %.2f（%s）"
          % (j["between"][0], j["between"][1], gap, fr, rate, br, ratio, verdict))
print("   骨长相对波动 %.2e（物理不变量，应接近 0）"
      % bj.get("metrics", {}).get("bone_length_rel_spread_max", 0))
PYEOF
fi

echo
case "$RC" in
  0) echo " 判定：正常（退出码 0）——出片、几何通过、没有被判可疑或近似代替的段" ;;
  1) echo " 判定：出片但有问题（退出码 1）——往上找「未验证 / 近似代替 / 可疑 / 去掉的段」" ;;
  2) echo " 判定：没出片（退出码 2）——上面已给出「这句话做不了」或「所有段都没通过检查」的原因" ;;
esac
echo " 任务目录：$DIR"
echo " 提示：看片用 render/final_captioned.mp4；看每段状态用 分镜.png 或 semantics.json"
rm -f "$LOG"
exit "$RC"
