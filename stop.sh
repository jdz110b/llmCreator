#!/bin/bash
# 语料评测平台 - 停止脚本

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/app.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[WARN] PID 文件不存在，服务可能未运行"
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] 正在停止服务 (PID: $PID)..."
    kill "$PID"
    # 等待进程退出
    for i in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    # 如果还没退出，强制终止
    if kill -0 "$PID" 2>/dev/null; then
        echo "[WARN] 进程未响应，强制终止..."
        kill -9 "$PID"
    fi
    echo "[OK] 服务已停止"
else
    echo "[INFO] 进程 $PID 已不存在"
fi

rm -f "$PID_FILE"
