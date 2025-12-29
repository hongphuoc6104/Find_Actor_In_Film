# 📊 Hướng Dẫn Tham Số Chi Tiết

Tài liệu này giải thích tất cả các tham số có thể điều chỉnh trong hệ thống Face Clustering.

---

## 📋 Mục Lục

1. [Stage 1-2: Trích Xuất & Nhận Diện](#stage-1-2-trích-xuất--nhận-diện)
2. [Stage 4-7: Phân Cụm & Gộp](#stage-4-7-phân-cụm--gộp)
3. [Tracklet Settings](#tracklet-settings)
4. [Mối Quan Hệ Giữa Các Tham Số](#mối-quan-hệ-giữa-các-tham-số)
5. [Presets Theo Độ Dài Video](#presets-theo-độ-dài-video)

---

## Stage 1-2: Trích Xuất & Nhận Diện

> ⚠️ **Lưu ý:** Thay đổi các tham số này cần chạy "Xử lý từ đầu" vì ảnh hưởng đến việc trích xuất.

### 1. `min_det_score` (Độ tin cậy nhận diện)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 0.45 |
| **Range** | 0.2 - 0.9 |
| **Vị trí config** | `quality_filters.min_det_score` |

**Ý nghĩa:** Ngưỡng confidence score từ detector. Chỉ nhận faces có score ≥ giá trị này.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.2-0.35)** | Phát hiện nhiều mặt hơn, bao gồm mặt mờ/xa/nghiêng. Có thể nhận nhầm vật thể là mặt. |
| **↑ Cao (0.6-0.9)** | Chỉ giữ mặt rõ ràng, bỏ qua mặt mờ. Có thể bỏ sót nhân vật xa camera. |

**🔗 Liên quan:**
- Nếu giảm `min_det_score` → nên tăng `min_blur_clarity` để lọc ảnh mờ
- Nếu video chất lượng thấp → giảm xuống 0.35-0.40

---

### 2. `min_face_size` (Kích thước mặt tối thiểu)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 50 px |
| **Range** | 20 - 120 px |
| **Vị trí config** | `quality_filters.min_face_size` |

**Ý nghĩa:** Kích thước tối thiểu (pixels) của bounding box khuôn mặt.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (20-35)** | Giữ mặt nhỏ/xa camera. Chất lượng embedding kém hơn. |
| **↑ Cao (80-120)** | Chỉ giữ mặt lớn/gần camera. Bỏ sót người ở xa. |

**🔗 Liên quan:**
- Nếu video có nhiều cảnh xa (phim hành động, đám đông) → giảm xuống 35-40
- Nếu video chủ yếu cận cảnh (phỏng vấn) → tăng lên 70-100

---

### 3. `min_blur_clarity` (Độ rõ nét tối thiểu)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 40.0 |
| **Range** | 15 - 80 |
| **Vị trí config** | `quality_filters.min_blur_clarity` |

**Ý nghĩa:** Độ sắc nét của ảnh mặt (Laplacian variance). Cao = rõ nét.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (15-25)** | Chấp nhận ảnh mờ/blur. Có thể gom nhầm cluster. |
| **↑ Cao (60-80)** | Chỉ giữ ảnh sắc nét. Bỏ nhiều frames chuyển động. |

**🔗 Liên quan:**
- Video chuyển động nhanh (action, MV) → giảm xuống 25-35
- Video tĩnh (phỏng vấn, podcast) → tăng lên 50-60

---

### 4. `landmark_hard_cutoff` (Ngưỡng landmark cứng)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 0.55 |
| **Range** | 0.3 - 0.8 |
| **Vị trí config** | `quality_filters.landmark_quality_filter.min_score_hard_cutoff` |

**Ý nghĩa:** Ngưỡng visibility score của landmarks (mắt, mũi, miệng). Score thấp = mặt nghiêng nhiều.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.3-0.45)** | Chấp nhận mặt nghiêng 45-60°. Embedding có thể không chính xác. |
| **↑ Cao (0.65-0.8)** | Chỉ nhận mặt gần như thẳng (≤15°). Bỏ sót nhiều cảnh. |

**🔗 Liên quan:**
- Nếu nhân vật hay quay đầu → giảm xuống 0.45-0.50
- Nếu cần chính xác cao → tăng lên 0.65-0.70

---

### 5. `landmark_core` (Ngưỡng landmark core)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 0.70 |
| **Range** | 0.5 - 0.9 |
| **Vị trí config** | `quality_filters.landmark_quality_filter.min_score_for_core` |

**Ý nghĩa:** Ngưỡng để chọn ảnh **đại diện** chất lượng cao cho mỗi cluster.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.5-0.6)** | Ảnh đại diện có thể nghiêng. Không ảnh hưởng clustering. |
| **↑ Cao (0.75-0.9)** | Ảnh đại diện rất đẹp nhưng có thể không có đủ. |

> 💡 Tham số này **không ảnh hưởng clustering**, chỉ ảnh hưởng ảnh preview.

---

## Stage 4-7: Phân Cụm & Gộp

> 🔄 Có thể chạy "Gom nhóm lại" mà không cần xử lý từ đầu.

### 6. `distance_threshold` (Ngưỡng clustering)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 1.15 |
| **Range** | 0.4 - 1.5 |
| **Vị trí config** | `clustering.distance_threshold.default` |

**Ý nghĩa:** Khoảng cách tối đa giữa 2 embeddings để được gom vào cùng cluster.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.4-0.7)** | Chặt chẽ, ít gom nhầm. Có thể tạo nhiều cluster cho cùng người. |
| **↑ Cao (1.0-1.5)** | Gom nhiều hơn. Có thể gom nhầm người giống nhau. |

**🔗 Liên quan:**
- Nếu tạo quá nhiều cluster cùng người → tăng lên 1.2-1.3
- Nếu gom nhầm 2 người khác nhau → giảm xuống 0.8-1.0
- Video ngắn (<10 phút) → giảm xuống 0.6-0.8

---

### 7. `merge_threshold` (Ngưỡng merge cụm)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 0.55 |
| **Range** | 0.35 - 0.75 |
| **Vị trí config** | `merge.within_movie_threshold` |

**Ý nghĩa:** Độ tương đồng cosine **tối thiểu** giữa 2 cluster centroids để gộp lại.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.35-0.45)** | Dễ merge, gộp nhiều clusters. Có thể gộp nhầm. |
| **↑ Cao (0.6-0.75)** | Khó merge, chỉ gộp clusters rất giống. |

**🔗 Liên quan:**
- Luôn nên có `merge_threshold` > `distance_threshold` (sau khi convert sang similarity)
- Nếu vẫn còn nhiều cluster cùng người sau clustering → giảm merge_threshold

---

### 8. `min_track_size` (Số frame tối thiểu/track)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 3 |
| **Range** | 1 - 10 |
| **Vị trí config** | `filter.min_track_size` |

**Ý nghĩa:** Số lần xuất hiện **liên tục** tối thiểu của 1 khuôn mặt.

```
Ví dụ với min_track_size = 3:
Track A: [F1][F2][F3] → 3 faces → GIỮ ✅
Track B: [F1][F2]     → 2 faces → LOẠI ❌
```

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **= 1** | Giữ mọi detection, kể cả xuất hiện 1 frame. |
| **= 3** | Lọc detection lẻ tẻ (thường là false positives). |
| **≥ 5** | Chỉ giữ người xuất hiện liên tục. Bỏ sót nhiều. |

**🔗 Liên quan:**
- Video ngắn/cắt nhanh → giảm xuống 1-2
- Video dài/cảnh tĩnh → giữ 3 hoặc tăng lên 4-5

---

### 9. `min_cluster_size` (Số ảnh tối thiểu/cụm)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 15 |
| **Range** | 1 - 50 |
| **Vị trí config** | `filter.min_cluster_size` |

**Ý nghĩa:** Số faces **tổng cộng** tối thiểu để cluster được giữ lại.

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **= 2-5** | Giữ cả nhân vật phụ (xuất hiện ít). Phù hợp video ngắn. |
| **= 10-20** | Chỉ giữ nhân vật chính. Loại bỏ người lướt qua. |
| **≥ 25** | Rất nghiêm ngặt. Chỉ phù hợp video >1 giờ. |

**🔗 Liên quan:**
- Video ngắn (<10 phút) → 2-5
- Video trung bình (10-40 phút) → 10-15
- Video dài (>40 phút) → 15-25

---

### 10. `post_merge_threshold` (Ngưỡng post-merge)

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 0.60 |
| **Range** | 0.40 - 0.80 |
| **Vị trí config** | `post_merge.distance_threshold` |

**Ý nghĩa:** Ngưỡng để hấp thụ clusters nhỏ (satellite) vào clusters lớn (core).

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **↓ Thấp (0.4-0.5)** | Dễ hấp thụ, gộp nhiều clusters nhỏ. |
| **↑ Cao (0.7-0.8)** | Khó hấp thụ, giữ clusters nhỏ riêng biệt. |

---

## Tracklet Settings

### 11. `tracklet.max_age`

| Thuộc tính | Giá trị |
|------------|---------|
| **Mặc định** | 3 |
| **Range** | 1 - 10 |
| **Vị trí config** | `tracklet.max_age` |

**Ý nghĩa:** Số frames chờ đợi khi mất mặt trước khi đóng track.

```
Ví dụ: Người A bị che mặt 2 frames rồi xuất hiện lại

max_age = 1: Tạo Track mới (2 tracks riêng biệt)
max_age = 3: Nối tiếp Track cũ (1 track dài)
```

| Giá trị | Ảnh hưởng |
|---------|-----------|
| **= 1-2** | Tạo nhiều tracks ngắn. Phù hợp cảnh đông người. |
| **= 5-10** | Nối tracks bị gián đoạn. Phù hợp cảnh ít người. |

---

## Mối Quan Hệ Giữa Các Tham Số

### Nhóm 1: Quality Filters (Stage 1-2)

```
min_det_score ↓  →  Cần tăng min_blur_clarity để bù lại
min_face_size ↓  →  Cần giảm landmark_hard_cutoff (mặt nhỏ thường nghiêng)
```

### Nhóm 2: Clustering (Stage 4-7)

```
distance_threshold ↑  →  Có thể giảm merge_threshold
min_track_size ↓      →  Nên tăng min_cluster_size để bù lại noise
```

### Video Ngắn (<10 phút)

| Vấn đề | Giải pháp |
|--------|-----------|
| Ít faces | Giảm `min_cluster_size` xuống 2-5 |
| Tracks ngắn | Giảm `min_track_size` xuống 1 |
| Cảnh cắt nhanh | Tăng `max_age` lên 5 |

### Video Dài (>40 phút)

| Vấn đề | Giải pháp |
|--------|-----------|
| Quá nhiều clusters | Tăng `distance_threshold` lên 1.2+ |
| Nhân vật phụ nhiều | Tăng `min_cluster_size` lên 20-25 |
| Noise nhiều | Tăng `min_track_size` lên 4-5 |

---

## Presets Theo Độ Dài Video

| Preset | Độ dài | Các thay đổi chính |
|--------|--------|-------------------|
| **MV/Clip** | <10 phút | `min_cluster_size: 2`, `distance_threshold: 0.6`, `min_track_size: 1` |
| **Phim ngắn** | 10-40 phút | `min_cluster_size: 10`, `distance_threshold: 0.8` |
| **Phim dài** | >40 phút | `min_cluster_size: 20`, `distance_threshold: 1.15` |

---

## 🔍 Troubleshooting

| Vấn đề | Nguyên nhân có thể | Giải pháp |
|--------|-------------------|-----------|
| Không có cluster nào | `min_cluster_size` quá cao | Giảm xuống 2-5 |
| Cùng người nhưng 2 clusters | `distance_threshold` quá thấp | Tăng lên hoặc giảm `merge_threshold` |
| 2 người khác bị gộp chung | `distance_threshold` quá cao | Giảm xuống |
| Bỏ sót nhiều người | `min_det_score` quá cao | Giảm xuống 0.35-0.40 |
| Ảnh mờ trong clusters | `min_blur_clarity` quá thấp | Tăng lên 50-60 |

---

*Tài liệu được tạo tự động - Cập nhật: 2025*
