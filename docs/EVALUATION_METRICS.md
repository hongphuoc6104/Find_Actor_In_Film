# Đánh giá Chất lượng Face Clustering

## Tổng quan

Tài liệu này mô tả các phương pháp đánh giá chất lượng của hệ thống Face Clustering khi có **ground truth labels** (nhãn chuẩn). Các metrics này giúp đo lường độ chính xác của thuật toán clustering so với dữ liệu đã được gán nhãn sẵn.

## Điều kiện áp dụng

- ✅ Có ground truth labels (tên diễn viên/nhân vật)
- ✅ Mỗi ảnh đã được gán nhãn đúng
- ✅ Cấu trúc thư mục: `test_dataset/[actor_name]/[images]`

---

## 1. Purity (Độ tinh khiết)

### Định nghĩa
Đo lường xem mỗi cluster có chứa chủ yếu các ảnh của **cùng một người** không.

### Công thức
```
Purity = (1/N) × Σᵢ max_j |cluster_i ∩ class_j|
```

Trong đó:
- `N`: Tổng số ảnh
- `cluster_i`: Cluster thứ i
- `class_j`: Class (người) thứ j
- `|cluster_i ∩ class_j|`: Số ảnh thuộc cả cluster i và class j

### Ý nghĩa
- **Giá trị**: 0.0 → 1.0
- **1.0 (Hoàn hảo)**: Mỗi cluster chỉ chứa ảnh của 1 người
- **0.0 (Tệ nhất)**: Các ảnh phân bố ngẫu nhiên

### Ưu điểm
- ✅ Dễ hiểu, trực quan
- ✅ Dễ giải thích trong báo cáo

### Nhược điểm
- ⚠️ Có thể bị "hack": Tạo nhiều cluster nhỏ → Purity cao nhưng vô nghĩa
- ⚠️ Không xử phạt việc tách một người thành nhiều cluster

### Khi nào sử dụng
- Báo cáo cho người không chuyên
- So sánh nhanh giữa các cấu hình

---

## 2. Normalized Mutual Information (NMI)

### Định nghĩa
Đo lường **lượng thông tin chung** giữa clustering kết quả và ground truth, được chuẩn hóa về đoạn [0,1].

### Công thức
```
NMI(C, T) = 2 × I(C;T) / [H(C) + H(T)]
```

Trong đó:
- `I(C;T)`: Mutual Information giữa clustering C và true labels T
- `H(C)`, `H(T)`: Entropy của C và T

### Chi tiết tính toán
```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
```

### Ý nghĩa
- **Giá trị**: 0.0 → 1.0
- **1.0**: Clustering hoàn toàn giống ground truth
- **0.0**: Clustering hoàn toàn ngẫu nhiên
- **~0.5**: Có correlation nhưng không hoàn hảo

### Ưu điểm
- ✅ Chuẩn trong các bài báo học thuật
- ✅ Không bị ảnh hưởng bởi số lượng cluster
- ✅ Symmetric (đối xứng)
- ✅ Xử phạt cả over-clustering và under-clustering

### Nhược điểm
- ⚠️ Khó giải thích cho người không chuyên

### Khi nào sử dụng
- **Báo cáo khoa học/luận văn**
- So sánh với các research papers khác
- Đánh giá tổng thể chất lượng clustering

---

## 3. Adjusted Rand Index (ARI)

### Định nghĩa
So sánh tất cả các **cặp ảnh** để xem chúng có được gom đúng không, có điều chỉnh cho random clustering.

### Công thức
```
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
```

Trong đó:
- `RI`: Rand Index (số cặp đúng / tổng số cặp)
- Adjusted để loại bỏ yếu tố ngẫu nhiên

### Chi tiết tính toán
```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(true_labels, predicted_clusters)
```

### Ý nghĩa
- **Giá trị**: -1.0 → 1.0
- **1.0**: Clustering hoàn hảo
- **0.0**: Clustering ngẫu nhiên
- **<0**: Tệ hơn random

### Ưu điểm
- ✅ Xử lý tốt các trường hợp edge-case
- ✅ Điều chỉnh cho random chance
- ✅ Được sử dụng rộng rãi trong research

### Nhược điểm
- ⚠️ Có thể cho giá trị âm (khó giải thích)
- ⚠️ Sensitive với imbalanced clusters

### Khi nào sử dụng
- Nghiên cứu chuyên sâu
- So sánh với baseline ngẫu nhiên

---

## 4. Precision, Recall, F1-Score

### Định nghĩa
Áp dụng metrics phân loại truyền thống vào bài toán clustering theo 2 cách:

#### 4.1. Cluster-Level Metrics

**Precision của một cluster**:
```
Precision_i = max_j |cluster_i ∩ class_j| / |cluster_i|
```
→ Trong cluster i, bao nhiêu % là người chiếm đa số?

**Recall của một class**:
```
Recall_j = max_i |cluster_i ∩ class_j| / |class_j|
```
→ Của người j, bao nhiêu % được gom vào cluster đúng nhất?

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 4.2. BCubed Metrics (Khuyến nghị cho clustering)

Đánh giá từng ảnh riêng lẻ:

**BCubed Precision**:
```
Precision_i = (Số ảnh cùng người trong cluster của i) / (Kích thước cluster của i)
```

**BCubed Recall**:
```
Recall_i = (Số ảnh cùng người trong cluster của i) / (Tổng số ảnh của người đó)
```

**BCubed F1**:
```
F1 = Trung bình điều hòa của BCubed Precision và Recall
```

### Ý nghĩa
- **Precision cao**: Clusters tinh khiết (ít nhiễu)
- **Recall cao**: Ít bị tách rời
- **F1 cao**: Cân bằng giữa hai yếu tố

### Ưu điểm
- ✅ Dễ hiểu (quen thuộc với ML basics)
- ✅ Phản ánh cả lỗi "gom sai" và "tách sai"
- ✅ BCubed phù hợp với bài toán clustering

### Nhược điểm
- ⚠️ Cần định nghĩa rõ "positive/negative" trong clustering
- ⚠️ Cluster-level metrics có thể thiên lệch

### Khi nào sử dụng
- Khi cần phân tích chi tiết lỗi
- Khi muốn balance giữa "gom đủ" và "gom đúng"

---

## So sánh các Metrics

| Metric | Giá trị | Dễ hiểu | Robust | Khuyến nghị |
|--------|---------|---------|--------|-------------|
| **Purity** | 0→1 | ⭐⭐⭐ | ⭐ | Poster/Demo |
| **NMI** | 0→1 | ⭐⭐ | ⭐⭐⭐ | **Research/Luận văn** |
| **ARI** | -1→1 | ⭐ | ⭐⭐⭐ | Research chuyên sâu |
| **F1** | 0→1 | ⭐⭐⭐ | ⭐⭐ | Phân tích lỗi |

---

## Combo đề xuất cho các mục đích

### 🎯 Để Poster/Báo cáo Đồ án
```
1. Purity (dễ giải thích)
2. NMI (chuẩn học thuật)
3. F1-Score (cân bằng precision/recall)
```

### 🎓 Để Luận văn/Paper
```
1. NMI (bắt buộc)
2. ARI (khuyến nghị)
3. BCubed F1 (nếu cần chi tiết)
```

### ⚡ Để Debug/Tuning nhanh
```
1. Purity (nhanh, trực quan)
2. F1-Score (phát hiện lỗi)
```

---

## Cấu trúc Test Dataset

### Thư mục chuẩn
```
test_dataset/
├── Tom_Cruise/
│   ├── tom_cruise_001.jpg
│   ├── tom_cruise_002.jpg
│   └── tom_cruise_003.jpg
├── Brad_Pitt/
│   ├── brad_pitt_001.jpg
│   └── brad_pitt_002.jpg
├── Leonardo_DiCaprio/
│   └── ...
└── metadata.json
```

### metadata.json (optional)
```json
{
  "dataset_name": "Hollywood Actors Test Set",
  "num_classes": 10,
  "num_images": 150,
  "min_images_per_class": 10,
  "max_images_per_class": 20,
  "classes": ["Tom_Cruise", "Brad_Pitt", ...],
  "description": "Bộ test gồm 10 diễn viên Hollywood nổi tiếng"
}
```

---

## Yêu cầu về Test Dataset

### Số lượng khuyến nghị
- **Số người (classes)**: 5-20 người
- **Số ảnh mỗi người**: 10-50 ảnh
- **Tổng ảnh**: 100-500 ảnh

### Chất lượng ảnh
- ✅ Đa dạng góc chụp
- ✅ Nhiều biểu cảm khác nhau
- ✅ Các điều kiện ánh sáng khác nhau
- ✅ Độ phân giải đủ cao (>=224x224)

### Lưu ý
- ⚠️ Tránh ảnh quá dễ (chân dung chuẩn)
- ⚠️ Nên có ảnh trong phim (nhiều người, góc nghiêng)
- ⚠️ Cân bằng số ảnh giữa các người

---

## Interpretation Guide (Hướng dẫn giải thích kết quả)

### Kết quả tốt
```
Purity: > 0.85
NMI: > 0.75
ARI: > 0.70
F1: > 0.80
```
→ Hệ thống hoạt động tốt, sẵn sàng production

### Kết quả trung bình
```
Purity: 0.70-0.85
NMI: 0.60-0.75
ARI: 0.50-0.70
F1: 0.65-0.80
```
→ Cần tinh chỉnh tham số

### Kết quả kém
```
Purity: < 0.70
NMI: < 0.60
ARI: < 0.50
F1: < 0.65
```
→ Cần xem xét lại thuật toán hoặc feature extraction

### Phân tích lỗi
- **Purity cao, Recall thấp**: Tạo quá nhiều cluster nhỏ (over-segmentation)
- **Precision thấp**: Gom nhầm nhiều người khác nhau
- **Recall thấp**: Tách một người thành nhiều cluster
- **NMI thấp mà Purity cao**: Over-clustering

---

## Tài liệu tham khảo

1. **Purity**: Manning et al., "Introduction to Information Retrieval" (2008)
2. **NMI**: Strehl & Ghosh, "Cluster ensembles" (2003)
3. **ARI**: Hubert & Arabie, "Comparing partitions" (1985)
4. **BCubed**: Bagga & Baldwin, "Entity-based cross-document coreferencing" (1998)

## Cập nhật

- **2025-12-16**: Tạo tài liệu ban đầu với 4 metrics chính
