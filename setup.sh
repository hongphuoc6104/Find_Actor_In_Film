#!/bin/bash
# ===========================================
# FACE CLUSTERING - ONE-CLICK SETUP
# ===========================================
# Usage: bash setup.sh
# ===========================================

set -e

echo "==========================================="
echo "🚀 FACE CLUSTERING - SETUP SCRIPT"
echo "==========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}❌ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Check Node.js
echo -e "\n${YELLOW}[2/5] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    echo "   → https://nodejs.org/en/download/"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✅ Node.js $NODE_VERSION${NC}"

# Create virtual environment
echo -e "\n${YELLOW}[3/5] Setting up Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✅ Created .venv${NC}"
else
    echo -e "${GREEN}✅ .venv already exists${NC}"
fi

# Activate and install Python dependencies
echo -e "\n${YELLOW}[4/5] Installing Python dependencies...${NC}"
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✅ Python dependencies installed${NC}"

# Install frontend dependencies
echo -e "\n${YELLOW}[5/5] Installing frontend dependencies...${NC}"
cd frontend-client
npm install --silent
cd ..
echo -e "${GREEN}✅ Frontend dependencies installed${NC}"

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p Data/video Data/frames Data/face_crops Data/embeddings
mkdir -p warehouse/parquet warehouse/cluster_previews warehouse/evaluation
echo -e "${GREEN}✅ Directories created${NC}"

# Done
echo ""
echo "==========================================="
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Add your video to: Data/video/YOUR_MOVIE.mp4"
echo "  2. Run: bash start.sh"
echo "  3. Open: http://localhost:5173"
echo ""
