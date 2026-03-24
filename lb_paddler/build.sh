#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Build the load balancer
echo "Building load balancer..."
go build -o lb_paddler main.go

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binary: $DIR/lb_paddler"
else
    echo "Build failed!"
    exit 1
fi
