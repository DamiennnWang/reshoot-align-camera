#!/bin/zsh
# 渲染全部三维镜头（横、竖各一套）。已渲染的帧会跳过，可中断后续跑。
cd "${0:A:h}"
B=~/Applications/Blender.app/Contents/MacOS/Blender
for o in h v; do for s in tunnel end scan poses phones; do
  echo "== $s $o $(date +%T)"
  $B -b -P scene.py -- $s $o 2>&1 | grep -E "Traceback|Error: [^S]" -A5
done; done
echo "== done $(date +%T)"
