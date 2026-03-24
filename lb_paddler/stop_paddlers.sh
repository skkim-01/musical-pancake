#!/bin/bash

# PaddleOCR 모든 인스턴스 중지 스크립트

PADDLEOCR_DIR="/home/genai09/changhwi.hong/paddleocr"
PORTS=(58581 58582 58583 58584)

cd "$PADDLEOCR_DIR"

echo "========================================"
echo "Stopping all PaddleOCR instances"
echo "========================================"

for PORT in "${PORTS[@]}"; do
    PID_FILE="server_${PORT}.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "[PORT $PORT] Not running (no PID file)"
        continue
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "[PORT $PORT] Stopping PID $PID..."
        kill $PID
        sleep 2
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "[PORT $PORT] Force killing..."
            kill -9 $PID
        fi
        
        rm "$PID_FILE"
        echo "[PORT $PORT] Stopped"
    else
        echo "[PORT $PORT] Process not found, removing stale PID file"
        rm "$PID_FILE"
    fi
done

echo "========================================"
echo "All instances stopped!"
echo "========================================"
