#!/bin/bash
# cleanup_safe.sh - Script xóa files không cần thiết một cách an toàn

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CLEANUP SCRIPT - XÓA FILES KHÔNG LIÊN QUAN LOGIC         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Kiểm tra xem đang ở đúng thư mục
if [ ! -f "configs/config.yaml" ]; then
    echo "❌ ERROR: Vui lòng chạy script từ thư mục gốc của project!"
    exit 1
fi

echo "📁 Working directory: $(pwd)"
echo ""

# Hỏi người dùng có muốn backup không
read -p "🔄 Bạn có muốn backup trước khi xóa? (y/n): " backup_choice

if [ "$backup_choice" = "y" ] || [ "$backup_choice" = "Y" ]; then
    BACKUP_DIR=~/backup_myproject_$(date +%Y%m%d_%H%M%S)
    echo "📦 Tạo backup tại: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Backup poster files
    [ -d "poster_images" ] && cp -r poster_images "$BACKUP_DIR/" 2>/dev/null || true
    [ -f "Pts.png" ] && cp Pts.png "$BACKUP_DIR/" 2>/dev/null || true
    [ -f "poster_nghien_cuu.png" ] && cp poster_nghien_cuu.png "$BACKUP_DIR/" 2>/dev/null || true
    [ -f "Screenshot from 2025-12-15 14-01-15.png" ] && cp "Screenshot from 2025-12-15 14-01-15.png" "$BACKUP_DIR/" 2>/dev/null || true
    
    echo "✅ Backup hoàn tất!"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "BƯỚC 1: XÓA POSTER & DEMO FILES"
echo "═══════════════════════════════════════════════════════════════"

files_to_delete=(
    "poster_images"
    "Pts.png"
    "poster_nghien_cuu.png"
    "Screenshot from 2025-12-15 14-01-15.png"
    "POSTER_IMAGES_INDEX.md"
    "poster_files_summary.txt"
)

for item in "${files_to_delete[@]}"; do
    if [ -e "$item" ]; then
        echo "🗑️  Xóa: $item"
        rm -rf "$item"
    else
        echo "⏭️  Bỏ qua (không tồn tại): $item"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "BƯỚC 2: XÓA CACHE & GENERATED FILES"
echo "═══════════════════════════════════════════════════════════════"

echo "🗑️  Xóa __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "🗑️  Xóa *.pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "🗑️  Xóa .pytest_cache..."
[ -d ".pytest_cache" ] && rm -rf .pytest_cache

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "BƯỚC 3: XÓA IDE SETTINGS (tuỳ chọn)"
echo "═══════════════════════════════════════════════════════════════"

read -p "⚙️  Xóa .idea/ và .vscode/? (y/n): " ide_choice
if [ "$ide_choice" = "y" ] || [ "$ide_choice" = "Y" ]; then
    [ -d ".idea" ] && rm -rf .idea && echo "✅ Xóa .idea/"
    [ -d ".vscode" ] && rm -rf .vscode && echo "✅ Xóa .vscode/"
else
    echo "⏭️  Giữ lại IDE settings"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "BƯỚC 4: XÓA VIRTUAL ENV CŨ (tuỳ chọn)"
echo "═══════════════════════════════════════════════════════════════"

if [ -d ".venv1" ]; then
    read -p "⚠️  Tìm thấy .venv1/. Xóa không? (CẢNH BÁO: ~500MB) (y/n): " venv_choice
    if [ "$venv_choice" = "y" ] || [ "$venv_choice" = "Y" ]; then
        rm -rf .venv1
        echo "✅ Đã xóa .venv1/"
    else
        echo "⏭️  Giữ lại .venv1/"
    fi
else
    echo "⏭️  Không tìm thấy .venv1/"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "BƯỚC 5: DỌN DẸP TEMP_UPLOADS (tuỳ chọn)"
echo "═══════════════════════════════════════════════════════════════"

if [ -d "temp_uploads" ]; then
    file_count=$(find temp_uploads -type f | wc -l)
    if [ "$file_count" -gt 0 ]; then
        read -p "📁 Tìm thấy $file_count files trong temp_uploads/. Xóa không? (y/n): " temp_choice
        if [ "$temp_choice" = "y" ] || [ "$temp_choice" = "Y" ]; then
            rm -rf temp_uploads/*
            echo "✅ Đã xóa nội dung temp_uploads/"
        else
            echo "⏭️  Giữ lại temp_uploads/"
        fi
    else
        echo "⏭️  temp_uploads/ đã trống"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    HOÀN TẤT CLEANUP!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Tính toán dung lượng tiết kiệm (ước tính)
echo "📊 Ước tính đã tiết kiệm:"
echo "   • Poster files: ~10-15 MB"
echo "   • Cache files: ~1-2 MB"
if [ "$ide_choice" = "y" ] || [ "$ide_choice" = "Y" ]; then
    echo "   • IDE settings: ~500 KB"
fi
if [ "$venv_choice" = "y" ] || [ "$venv_choice" = "Y" ]; then
    echo "   • Virtual env: ~500 MB"
fi

echo ""
echo "✅ Các file CORE LOGIC đã được giữ nguyên:"
echo "   • tasks/ - Pipeline modules"
echo "   • services/ - Recognition services"
echo "   • utils/ - Utility functions"
echo "   • warehouse/ - FAISS index & embeddings"
echo "   • Data/ - Processed data"
echo "   • configs/ - Configuration"
echo ""

if [ "$backup_choice" = "y" ] || [ "$backup_choice" = "Y" ]; then
    echo "💾 Backup đã lưu tại: $BACKUP_DIR"
fi

echo "🎉 Cleanup hoàn tất!"
