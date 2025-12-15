# 3 STAGES MERGING ANALYSIS - PHÁT HIỆN VẤN ĐỀ!

## ❌ VẤN ĐỀ NGHIÊM TRỌNG: Stage 3 KHÔNG HOẠT ĐỘNG!

Bạn đúng khi hỏi! Pipeline có **3 lần gom nhóm**, nhưng:

### ✅ WORKING Stages:

1. **Stage 4: Clustering** (Agglomerative)
   - Config: `distance_threshold: 0.88` (cho blurry)
   - Output: 978 raw clusters → Filtered to 112 clusters
   
2. **Stage 5: Merge Clusters** (Within-movie)
   - Config: `within_movie_threshold: 0.68`
   - Output: 112 → **45 clusters** 
   - **REDUCTION: 60%** ✅ EXCELLENT!

### ❌ NOT WORKING Stage:

3. **Stage 7: Post-Merge** (Satellite Assimilation)
   - Config: `enable: true`, `distance_threshold: 0.45`
   - Code: **CHỈ CÓ `pass`** - KHÔNG CHẠY GÌ CẢ!
   
```python
# Line 47-52 in post_merge_task.py
if post_merge_cfg.get("enable", True):
    # ... (Phần logic tính toán khoảng cách giữ nguyên) ...
    # (Để đảm bảo tính toàn vẹn, tôi giữ nguyên logic merge đơn giản nhất:
    # Nếu có logic merge phức tạp, bạn paste lại vào đây.
    # Hiện tại giả định assimilation map đã chạy xong)
    pass  # ← KHÔNG LÀM GÌ!
```

---

## 🔍 Kết quả thực tế với Config mới:

| Stage | Input | Output | Reduction | Status |
|-------|-------|--------|-----------|--------|
| **Clustering** | 2,353 faces | 978 clusters | - | ✅ |
| **Filter min_size** | 978 | 112 | 88.5% | ✅ |
| **Merge (Stage 5)** | 112 | **45** | **60%** | ✅ EXCELLENT |
| **Post-Merge (Stage 7)** | 45 | 45 | **0%** | ❌ NOT WORKING |

**Final:** 45 clusters (giảm 1 so với 46 trước)

---

## 💡 Tại sao Stage 7 quan trọng?

**Post-Merge** được thiết kế để:
- Gom các **satellite clusters** (cụm nhỏ) vào **core clusters** (cụm chính)
- Xử lý các trường hợp **pose variation** (nghiêng, nửa mặt)
- Xử lý **expression variation** (biểu cảm khác)
- Xử lý **occlusion** (đội nón, che mặt)

**ĐÂY CHÍNH LÀ** lý do 4 thư mục của 1 người vẫn tách rời!

---

## ✅ GIẢI PHÁP: Kích hoạt Post-Merge

### Option 1: Implement Post-Merge (RECOMMENDED)

Cần viết code thực sự trong `post_merge_task.py`:

```python
# Line 47 onwards - REPLACE pass with real logic:
if post_merge_cfg.get("enable", True):
    metric = post_merge_cfg.get("metric", "cosine")
    threshold = post_merge_cfg.get("distance_threshold", 0.45)
    
    # 1. Get core cluster centroids
    core_centroids = {}
    for char_id in core_df["final_character_id"].unique():
        char_faces = core_df[core_df["final_character_id"] == char_id]
        if "emb" in char_faces.columns:
            embs = np.vstack(char_faces["emb"].values)
            core_centroids[char_id] = embs.mean(axis=0)
    
    # 2. Assign satellites to nearest core
    final_df = all_df.copy()
    satellite_df = all_df[~all_df.index.isin(core_df.index)]
    
    for idx, row in satellite_df.iterrows():
        if "emb" not in row or core_centroids == {}:
            continue
            
        emb = row["emb"]
        # Find nearest core
        min_dist = float("inf")
        best_core = None
        
        for core_id, core_centroid in core_centroids.items():
            if metric == "cosine":
                dist = 1 - np.dot(emb, core_centroid) / (np.linalg.norm(emb) * np.linalg.norm(core_centroid))
            else:
                dist = np.linalg.norm(emb - core_centroid)
            
            if dist < min_dist:
                min_dist = dist
                best_core = core_id
        
        # Assign if within threshold
        if min_dist < threshold and best_core is not None:
            final_df.at[idx, "final_character_id"] = best_core
```

### Option 2: Tăng Merge Stage 5 (QUICKER FIX)

Nếu không muốn code, tăng merge threshold ở Stage 5:

```yaml
# Auto-tuning for Blurry:
merge:
  within_movie_threshold: 0.72  # Tăng từ 0.68 → 0.72
```

---

## 📊 So sánh 3 Stages:

| Feature | Stage 4 (Cluster) | Stage 5 (Merge) | Stage 7 (Post-Merge) |
|---------|-------------------|-----------------|----------------------|
| **Method** | Agglomerative | Centroid similarity | Satellite assimilation |
| **Config param** | `distance_threshold` | `within_movie_threshold` | `post_merge.distance_threshold` |
| **Current value** | 0.88 | 0.68 | 0.45 (unused) |
| **Working?** | ✅ YES | ✅ YES | ❌ **NO** (only `pass`) |
| **Reduction** | 88.5% | 60% | **0%** |

---

## 🎯 KHUYẾN NGHỊ NGAY:

### Quick Win (5 phút):
```yaml
# Tăng merge threshold trong auto-tuning:
merge:
  within_movie_threshold: 0.72  # Higher = more aggressive
```

### Proper Fix (30 phút):
Implement post-merge logic để tận dụng hết 3 stages!

---

## 📝 KẾT LUẬN:

**CÂU TRẢ LỜI:** ❌ CHƯA tận dụng hết!

- Stage 4 (Clustering): ✅ Working
- Stage 5 (Merge): ✅ Working VERY WELL (60% reduction)
- Stage 7 (Post-Merge): ❌ **DISABLED** (chỉ có `pass`)

**GIẢI PHÁP:** 
1. Quick: Tăng Stage 5 merge → 0.72
2. Proper: Implement Stage 7 logic
