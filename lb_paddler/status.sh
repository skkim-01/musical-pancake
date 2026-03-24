#!/bin/bash

# PaddleOCR 인스턴스 상태 확인

PADDLEOCR_DIR="/home/genai09/changhwi.hong/paddleocr"
PORTS=(58581 58582 58583 58584)

cd "$PADDLEOCR_DIR"

echo "========================================"
echo "PaddleOCR Instance Status"
echo "========================================"

for PORT in "${PORTS[@]}"; do
    PID_FILE="server_${PORT}.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "[PORT $PORT] ❌ Not running"
        continue
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        # Get memory usage
        MEM=$(ps -p $PID -o rss= 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        echo "[PORT $PORT] ✅ Running (PID: $PID, Memory: $MEM)"
    else
        echo "[PORT $PORT] ⚠️  Dead (PID file exists but process not found)"
    fi
done

echo "========================================"

# Check load balancer
LB_DIR="/home/genai09/skkim-01/study/lb_paddler"
if [ -f "$LB_DIR/lb.pid" ]; then
    LB_PID=$(cat "$LB_DIR/lb.pid")
    if ps -p $LB_PID > /dev/null 2>&1; then
        echo "[LB :18005] ✅ Running (PID: $LB_PID)"
    else
        echo "[LB :18005] ⚠️  Dead"
    fi
else
    echo "[LB :18005] ❌ Not running"
fi

echo "========================================"
