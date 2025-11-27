#!/bin/bash

# Script to start FastAPI server and ngrok tunnel
# Usage: ./start_server.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
FASTAPI_PORT=8000
NGROK_DOMAIN="normal-similarly-colt.ngrok-free.app"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$FASTAPI_PID" ]; then
        echo "Stopping FastAPI server (PID: $FASTAPI_PID)"
        kill $FASTAPI_PID 2>/dev/null || true
    fi
    if [ ! -z "$NGROK_PID" ]; then
        echo "Stopping ngrok (PID: $NGROK_PID)"
        kill $NGROK_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}Error: ngrok is not installed${NC}"
    echo "Please install ngrok: https://ngrok.com/download"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

# Check if uvicorn is installed
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}Error: uvicorn is not installed${NC}"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Change to script directory
cd "$SCRIPT_DIR"

echo -e "${GREEN}Starting FastAPI server on port $FASTAPI_PORT...${NC}"
# Start FastAPI server in background
python3 -m uvicorn main:app --host 0.0.0.0 --port $FASTAPI_PORT &
FASTAPI_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo -e "${RED}Error: FastAPI server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}FastAPI server started (PID: $FASTAPI_PID)${NC}"
echo -e "${GREEN}Starting ngrok tunnel with domain: $NGROK_DOMAIN...${NC}"

# Start ngrok tunnel
ngrok http $FASTAPI_PORT --domain=$NGROK_DOMAIN &
NGROK_PID=$!

# Wait a moment for ngrok to start
sleep 2

# Check if ngrok is running
if ! kill -0 $NGROK_PID 2>/dev/null; then
    echo -e "${RED}Error: ngrok failed to start${NC}"
    kill $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}ngrok tunnel started (PID: $NGROK_PID)${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FastAPI server: http://localhost:$FASTAPI_PORT${NC}"
echo -e "${GREEN}Public URL: https://$NGROK_DOMAIN${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Wait for processes
wait

