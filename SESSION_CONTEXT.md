# Session Context - Clustering Optimization

**Date:** 2025-12-16  
**Objective:** Optimize 3-stage merge pipeline for different video profiles

---

## 📊 Config Overview

### Baseline Parameters (Default cho tất cả video)

```yaml
# quality_filters (dòng 57-65)
quality_filters:
  min_det_score: 0.45
  min_blur_clarity: 40.0
  min_face_size: 50
  landmark_quality_filter:
    min_score_hard_cutoff: 0.55  # Chỉ nhận faces >= 70% visibility
    min_score_for_core: 0.70

# clustering (dòng 38-44)
clustering:
  distance_threshold:
    default: 0.85

# merge (dòng 46-47)
merge:
  within_movie_threshold: 0.60

# post_merge (dòng 49-52)
post_merge:
  distance_threshold: 0.60

# filter_clusters (dòng 54-55)
filter_clusters:
  min_size: 5

# min_size_rules (dòng 101-107)
auto_tuning.min_size_rules:
  duration_base:
    Short: 5
    Medium: 7
    Long: 10
```

---

## 🎬 Auto-Tuning Profiles

### 1. Blurry (EMCHUA18)

**Điều kiện match:** `clarity: "Blurry"`

```yaml
# Dòng 111-123 trong config.yaml
- conditions:
    clarity: "Blurry"
  overrides:
    quality_filters:
      min_blur_clarity: 25.0
      min_det_score: 0.50
    clustering:
      distance_threshold:
        default: 0.90
    merge:
      within_movie_threshold: 0.45
    post_merge:
      distance_threshold: 0.60
```

**Kết quả:** ~15-20 clusters

---

### 2. Dark + Blurry (DENAMHON)

**Điều kiện match:** `lighting: "Dark", clarity: "Blurry"`

> [!IMPORTANT]
> Rule này được áp dụng SAU rule Blurry (đè lên các giá trị trùng)

```yaml
# Dòng 126-140 trong config.yaml
- conditions:
    lighting: "Dark"
    clarity: "Blurry"
  overrides:
    quality_filters:
      min_det_score: 0.30
      min_blur_clarity: 20.0
      min_face_size: 40
    clustering:
      distance_threshold:
        default: 0.90
    merge:
      within_movie_threshold: 0.45
    post_merge:
      distance_threshold: 0.70  # Cao hơn Blurry để gom nhiều hơn
```

**Kết quả:** ~11-15 clusters

---

## 📋 Parameter Cheat Sheet

| Tham số | Tăng → | Giảm → |
|---------|--------|--------|
| `min_score_hard_cutoff` | Ít faces, sạch hơn | Nhiều faces |
| `clustering.distance_threshold` | Cụm lớn, có thể lẫn | Cụm nhỏ, sạch |
| `merge.within_movie_threshold` | Ít merge, tách lẻ | Nhiều merge |
| `post_merge.distance_threshold` | Nhiều gom | Ít gom |

---

## 🔧 Trade-offs Đã Tìm Thấy (DENAMHON)

| Config | Kết quả |
|--------|---------|
| clustering=0.70, merge=0.48, pm=0.65 | Nhân vật A biến mất |
| clustering=0.75, merge=0.45, pm=0.70 | A tách 2-3 cụm, sạch |
| clustering=0.80, merge=0.45, pm=0.70 | Merge nhầm 2 người |
| **clustering=0.90, merge=0.45, pm=0.70** | **Đang sử dụng** |

---

## 📝 Notes

1. **Rule Order:** Rules trong `auto_tuning.rules` được áp dụng theo thứ tự, rule sau đè lên rule trước
2. **Baseline ảnh hưởng embedding:** `min_score_hard_cutoff` là baseline, thay đổi sẽ ảnh hưởng TẤT CẢ video
3. **Video khó (Dark+Blurry):** Chấp nhận một số nhân vật bị tách do embedding không nhất quán ở góc nghiêng
