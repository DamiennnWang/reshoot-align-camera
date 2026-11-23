#!/bin/sh
# 容器启动脚本：第一次在持久卷里建 Python 虚拟环境（之后重启直接用），然后启动网站服务。
set -e
APP=/data/app
VENV=/data/venv
REQ_HASH=$(md5sum "$APP/requirements.txt" | cut -c1-32)
if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$VENV/.req" 2>/dev/null)" != "$REQ_HASH" ]; then
  echo "安装依赖…"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir -i "${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" -r "$APP/requirements.txt"
  echo "$REQ_HASH" > "$VENV/.req"
fi
export SITE_DIR=$APP/site TOOLKIT_DIR=$APP/server DATA_DIR=/data/work PORT=${PORT:-8080}
exec "$VENV/bin/python" -u "$APP/server/server.py"
