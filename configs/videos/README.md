# Per-Video Configuration - Quick Start Guide

## What Changed

The system now supports **video-specific configuration files** to prevent parameter conflicts.

## Directory Structure

```
configs/
├── config.yaml              # Base config (defaults for all videos)
└── videos/                  # Video-specific overrides
    ├── CHUYENXOMTUI.yaml
    ├── EMCHUA18.yaml
    └── [your_video].yaml
```

## How It Works

1. **Run pipeline normally**:
   ```bash
   python flows/pipeline.py --movie CHUYENXOMTUI
   ```

2. **Pipeline automatically**:
   - Loads `configs/config.yaml` (base)
   - Checks if `configs/videos/CHUYENXOMTUI.yaml` exists
   - If exists → merges overrides
   - If not → uses base config

## Creating Video-Specific Config

### Option 1: Manual Creation

Create `configs/videos/VIDEO_NAME.yaml`:

```yaml
# Comments about this video
clustering:
  distance_threshold:
    default: 0.88

merge:
  within_movie_threshold: 0.50
```

### Option 2: Save Current Config

After tuning parameters for a video, save them:

```bash
python utils/save_video_config.py --movie VIDEO_NAME \
  --distance-threshold 0.88 \
  --merge-threshold 0.50 \
  --description "Tuned for bright video"
```

## Benefits

✅ **No more conflicts**: Each video has isolated parameters  
✅ **Reproducible**: Re-running same video → same results  
✅ **Optional**: Videos without specific config use base config  
✅ **Easy management**: YAML files are transparent and editable

## Migration

For existing videos with good results:

1. Note the parameters that work well
2. Create `configs/videos/{video}.yaml` with those params
3. Next run will use saved params automatically

## Examples

See `configs/videos/CHUYENXOMTUI.yaml` and `EMCHUA18.yaml` for reference.
