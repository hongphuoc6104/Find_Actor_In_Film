#!/bin/bash
# ===========================================
# FACE CLUSTERING - START ALL SERVICES
# ===========================================
# Usage: bash start.sh
# ===========================================

set -e

echo "==========================================="
echo "🚀 STARTING FACE CLUSTERING SYSTEM"
echo "==========================================="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: bash setup.sh"
    exit 1
fi

# Set Python path
export PYTHONPATH=$PYTHONPATH:.

# Kill any existing processes on ports
echo "🔄 Cleaning up ports..."
fuser -k 8000/tcp > /dev/null 2>&1 || true
fuser -k 5173/tcp > /dev/null 2>&1 || true

# Create required directories
mkdir -p Data/video temp_uploads

# Start Backend in background
echo "🖥️  Starting Backend API (port 8000)..."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start Frontend in background
echo "🌐 Starting Frontend (port 5173)..."
cd frontend-client
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

echo ""
echo "==========================================="
echo "✅ ALL SERVICES STARTED!"
echo "==========================================="
echo ""
echo "📺 Frontend:  http://localhost:5173"
echo "🔌 Backend:   http://localhost:8000"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to kill both processes
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    fuser -k 8000/tcp > /dev/null 2>&1 || true
    fuser -k 5173/tcp > /dev/null 2>&1 || true
    echo "✅ All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
wait
