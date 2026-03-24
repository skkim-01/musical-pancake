#!/bin/bash

# PaddleOCR 4개 인스턴스 시작 스크립트
# 포트: 58581, 58582, 58583, 58584 (모두 GPU 7)

PADDLEOCR_DIR="/home/genai09/changhwi.hong/paddleocr"
PORTS=(58581 58582 58583 58584)
GPU_ID=6

cd "$PADDLEOCR_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv directory not found in $PADDLEOCR_DIR"
    exit 1
fi

echo "========================================"
echo "Starting ${#PORTS[@]} PaddleOCR instances on GPU $GPU_ID"
echo "========================================"

for PORT in "${PORTS[@]}"; do
    PID_FILE="server_${PORT}.pid"
    LOG_FILE="paddlex_${PORT}.log"
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "[PORT $PORT] Already running (PID: $PID)"
            continue
        else
            rm "$PID_FILE"
        fi
    fi
    
    echo "[PORT $PORT] Starting..."
    nohup paddlex --serve --pipeline PaddleOCR-VL.yaml --device gpu:$GPU_ID --port $PORT >> "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "[PORT $PORT] Started with PID: $! (log: $LOG_FILE)"
    
    # 인스턴스 간 시작 간격 (메모리 할당 충돌 방지)
    sleep 5
done

echo "========================================"
echo "All instances started!"
echo "========================================"
echo ""
echo "Check status: $0 --status"
echo "View logs: tail -f $PADDLEOCR_DIR/paddlex_PORT.log"
