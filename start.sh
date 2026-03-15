#!/bin/bash
# 语料评测平台 - 启动脚本

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/app.pid"
LOG_FILE="$APP_DIR/app.log"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"

cd "$APP_DIR"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[INFO] 服务已在运行 (PID: $OLD_PID)，如需重启请先执行 ./stop.sh"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# 创建必要目录
mkdir -p "$APP_DIR/uploads" "$APP_DIR/data"

# 检查 Python 环境
if [ -d "$APP_DIR/venv" ]; then
    source "$APP_DIR/venv/bin/activate"
fi

# 检查依赖
if ! python3 -c "import flask" 2>/dev/null; then
    echo "[INFO] 安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动方式：优先 gunicorn，回退 flask 内置服务器
if python3 -c "import gunicorn" 2>/dev/null; then
    echo "[INFO] 使用 gunicorn 启动 ($WORKERS workers)..."
    nohup gunicorn \
        --bind "$HOST:$PORT" \
        --workers "$WORKERS" \
        --timeout 300 \
        --access-logfile "$LOG_FILE" \
        --error-logfile "$LOG_FILE" \
        --pid "$PID_FILE" \
        app:app \
        >> "$LOG_FILE" 2>&1 &
    sleep 1
    echo "[OK] 服务已启动: http://$HOST:$PORT (PID: $(cat $PID_FILE))"
else
    echo "[INFO] 未安装 gunicorn，使用 Flask 内置服务器启动 (仅建议开发环境)..."
    nohup python3 app.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    echo "[OK] 服务已启动: http://$HOST:$PORT (PID: $(cat $PID_FILE))"
fi

echo "[INFO] 日志文件: $LOG_FILE"
