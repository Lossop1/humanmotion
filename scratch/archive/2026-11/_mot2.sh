#!/bin/bash
source /home/chuan/motion-work/env.sh >/dev/null
cd /home/chuan/motion-work
echo "  motius/pipelines 下的文件："
ls motius/motius/pipelines/ 2>/dev/null | head -n 20
echo
echo "  infer_t2m 定义位置："
grep -rn "def infer_t2m" motius/ 2>/dev/null | head -n 6
echo
echo "  inpaint / prefix 相关："
grep -rln "inpaint" motius/ 2>/dev/null | head -n 8
grep -rn "def infer_inpaint\|inpainting" motius/motius/pipelines/*.py 2>/dev/null | head -n 10
