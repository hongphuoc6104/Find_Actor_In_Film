#!/bin/bash
# Script chạy server siêu nhẹ
export PYTHONPATH=$PYTHONPATH:.
# Dọn dẹp port
fuser -k 8000/tcp > /dev/null 2>&1 || true

# Tạo thư mục video nếu chưa có
mkdir -p Data/video

# Khởi chạy
echo "🚀 Server đang chạy tại: http://localhost:8000"
echo "👉 Hãy mở trình duyệt để test Demo."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload