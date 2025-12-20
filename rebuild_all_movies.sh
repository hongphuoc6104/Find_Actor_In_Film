#!/bin/bash
# Script to rebuild ALL movies from scratch (skip auto-labeling)

set -e

echo "=========================================="
echo "🎬 REBUILDING ALL 11 MOVIES"
echo "=========================================="
echo ""

# All movies to process
MOVIES=(
    "CHUYENXOMTUI"
    "DENAMHON"
    "EMCHUA18"
    "GAIGIALAMCHIEU"
    "HEMCUT"
    "KEANDANH"
    "Nang2"
    "NGUOIVOCUOICUNG"
    "NHAGIATIEN"
    "SIEULAYGAPSIEULUA"
    "TAMCAM"
)

TOTAL=${#MOVIES[@]}
CURRENT=0

for movie in "${MOVIES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Processing: $movie"
    echo "=========================================="
    
    python -m flows.pipeline --movie "$movie" --skip-ingestion
    
    if [ $? -eq 0 ]; then
        echo "✅ $movie completed successfully"
    else
        echo "❌ $movie failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "⚠️  SKIPPING AUTO-LABELING (no labeled_faces)"
echo "=========================================="
echo "User will add labels manually later"

echo ""
echo "=========================================="
echo "🔍 REBUILDING SEARCH INDEX"
echo "=========================================="
python -c "from utils.indexer import clear_index_cache, build_character_index; clear_index_cache(); build_character_index(force_rebuild=True)" || echo "Index rebuild completed with warnings (expected without labels)"

echo ""
echo "=========================================="
echo "✅ ALL 11 MOVIES CLUSTERING COMPLETED!"
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo "   1. Add labeled face images to warehouse/labeled_faces/"
echo "   2. Run: python -c 'from tasks.assign_labels_task import assign_labels_task; assign_labels_task()'"
echo "   3. Run: python -c 'from utils.indexer import clear_index_cache, build_character_index; clear_index_cache(); build_character_index(force_rebuild=True)'"
