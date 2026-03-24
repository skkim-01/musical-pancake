#!/bin/bash

# 통합 모니터링 대시보드
# LB 상태 + 패들러 로그 통합

PADDLEOCR_DIR="/home/genai09/changhwi.hong/paddleocr"
LB_URL="http://localhost:18005/lb/stats"
PORTS=(58581 58582 58583 58584)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    clear
    echo -e "${BLUE}========================================"
    echo "  PaddleOCR Load Balancer Dashboard"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "========================================${NC}"
}

print_lb_stats() {
    echo ""
    echo -e "${YELLOW}📊 Load Balancer Stats${NC}"
    echo "----------------------------------------"
    
    STATS=$(curl -s "$LB_URL" 2>/dev/null)
    
    if [ -z "$STATS" ]; then
        echo -e "${RED}  ❌ LB not responding${NC}"
        return
    fi
    
    # Parse JSON with python
    echo "$STATS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Total Requests:   {data['total_requests']}\")
print(f\"  Success:          {data['success_requests']}\")
print(f\"  Failed:           {data['failed_requests']}\")
print(f\"  Queued:           {data['queued_requests']}\")
print(f\"  Queue Size:       {data['queue_size']}/{data['queue_capacity']}\")
print()
print('  Backends:')
for b in data['backends']:
    url = b['url'].split(':')[-1]
    if b.get('circuit_open', False):
        status = f\"🚫 BLOCKED ({b.get('circuit_open_secs', 0)}s)\"
    elif b['healthy']:
        status = '✅'
    else:
        status = f\"❌ (fail:{b.get('consecutive_failures', 0)})\"
    print(f\"    [{url}] {status} Active: {b['active_conns']}\")
"
}

print_recent_logs() {
    echo ""
    echo -e "${YELLOW}📜 Recent Logs (last 3 per instance)${NC}"
    echo "----------------------------------------"
    
    for PORT in "${PORTS[@]}"; do
        LOG_FILE="$PADDLEOCR_DIR/paddlex_${PORT}.log"
        if [ -f "$LOG_FILE" ]; then
            echo -e "${GREEN}[PORT $PORT]${NC}"
            tail -3 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
        fi
    done
}

print_gpu_status() {
    echo ""
    echo -e "${YELLOW}🎮 GPU 6 Status${NC}"
    echo "----------------------------------------"
    nvidia-smi -i 6 --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "  Memory: %s / %s | GPU: %s | Temp: %s\n", $1, $2, $3, $4}'
}

# Main loop
if [ "$1" == "--watch" ] || [ "$1" == "-w" ]; then
    while true; do
        print_header
        print_lb_stats
        print_gpu_status
        print_recent_logs
        echo ""
        echo -e "${BLUE}Press Ctrl+C to exit. Refreshing in 3s...${NC}"
        sleep 3
    done
else
    print_header
    print_lb_stats
    print_gpu_status
    print_recent_logs
    echo ""
    echo -e "${BLUE}Use '$0 --watch' for live monitoring${NC}"
fi
