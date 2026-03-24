#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if already running
if [ -f lb.pid ]; then
    PID=$(cat lb.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Load balancer is already running (PID: $PID)"
        exit 1
    else
        echo "Removing stale PID file..."
        rm lb.pid
    fi
fi

# Build if binary doesn't exist
if [ ! -f lb_paddler ]; then
    echo "Binary not found, building..."
    ./build.sh
fi

# Start the load balancer
echo "Starting load balancer..."
nohup ./lb_paddler -config config.json >> lb.log 2>&1 &

# Save PID
echo $! > lb.pid
echo "Load balancer started with PID: $!"
echo "Logs: tail -f $DIR/lb.log"
