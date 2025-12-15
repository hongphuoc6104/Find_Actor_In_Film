# Config Changes: Gom Pose & Expression Variations

## 🎯 Vấn đề

Bạn phát hiện: **1 người bị tách 4 thư mục:**

1. ✅ Mặt đẹp thẳng
2. ❌ Nghiêng >45° / nửa mặt → Tách riêng
3. ❌ Mờ/tối/blur → Tách riêng  
4. ❌ Đội nón + biểu cảm lạ → Tách riêng

**Nguyên nhân:** Embedding khác nhau theo pose/expression/occlusion

---

## ✅ Giải pháp: TĂNG MẠNH merge

### Điều chỉnh Auto-Tuning (Blurry):

```yaml
# TRƯỚC (không đủ):
clustering:
  distance_threshold:
    default: 0.95        # Vẫn cao
merge:
  within_movie_threshold: 0.62  # Không đủ mạnh

# SAU (mạnh hơn):
clustering:
  distance_threshold:
    default: 0.88        # ↓ Giảm 7.4% → Gom chặt hơn
merge:
  within_movie_threshold: 0.68  # ↑ Tăng 9.7% → Merge mạnh hơn
```

---

## 📊 Dự kiến kết quả:

| Metric | Trước | Sau (dự kiến) |
|--------|-------|---------------|
| Raw clusters | 978 | ~1100-1200 (threshold thấp hơn) |
| After merge | 46 | ~30-35 (merge mạnh hơn) |
| Merge reduction | 58.9% | ~70-75% |
| **4 thư mục → 1** | ❌ | ✅ |

---

## 🧪 Test ngay:

```bash
# Xóa preview cũ
rm -rf warehouse/cluster_previews/EMCHUA18/

# Chạy lại (skip embedding để nhanh)
python -m flows.pipeline --movie EMCHUA18 --skip-ingestion --skip-embedding

# Kiểm tra
ls warehouse/cluster_previews/EMCHUA18/ | wc -l  # Nên ~30-35

# Verify visual
eog warehouse/cluster_previews/EMCHUA18/4_merged_*/000.jpg
```

---

## ⚖️ Trade-offs

### ✅ Pros:
- Gom đúng người (4 thư mục → 1)
- Ít phân mảnh hơn
- Dễ review hơn

### ⚠️ Cons (có thể):
- **Risk cao hơn**: Merge 0.68 có thể ghép nhầm 2 người giống nhau
- Cần kiểm tra KỸ sau khi test

---

## 🔍 Sau khi test:

### Nếu tốt (4 thư mục → 1, không ghép nhầm):
✅ **APPLY cho tất cả video**

### Nếu vẫn tách (4 thư mục vẫn còn):
→ Tăng merge thêm: `0.68 → 0.72`
→ Giảm threshold thêm: `0.88 → 0.85`

### Nếu ghép nhầm 2 người:
→ Giảm merge: `0.68 → 0.65`
→ Tăng threshold: `0.88 → 0.90`

---

## 📝 Parameters Summary

```yaml
# For BLURRY videos (EMCHUA18):
Auto-tuning:
  distance_threshold: 0.88  # Gom chặt (từ 0.95)
  merge_threshold: 0.68     # Merge mạnh (từ 0.62)
  
# For NORMAL videos:
Baseline (unchanged):
  distance_threshold: 1.15
  merge_threshold: 0.55
```

**Lý do:** Blurry video có nhiều noise → cần merge mạnh hơn để gom pose/expression variations
