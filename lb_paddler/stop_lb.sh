#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ ! -f lb.pid ]; then
    echo "Load balancer is not running (no PID file)"
    exit 1
fi

PID=$(cat lb.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping load balancer (PID: $PID)..."
    kill $PID
    sleep 2
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process didn't stop, force killing..."
        kill -9 $PID
    fi
    
    rm lb.pid
    echo "Stopped."
else
    echo "Process not found (PID: $PID), removing stale PID file..."
    rm lb.pid
fi
