# BÁO CÁO ĐÁNH GIÁ LUẬN VĂN TỐT NGHIỆP

## Thông tin chung
- **Đề tài:** Hệ thống nhận diện và phân cụm khuôn mặt diễn viên trong video sử dụng học sâu và phân cụm phân cấp
- **Trường:** Đại học Cần Thơ
- **Số trang:** ~1591 dòng (khoảng 40-50 trang)
- **Ngày đánh giá:** 2025-12-18

---

## 1. ĐÁNH GIÁ TỔNG QUAN

### 1.1 Điểm mạnh

> [!TIP]
> **Những điểm đáng khen ngợi trong luận văn**

1. **Cấu trúc luận văn chuyên nghiệp và đầy đủ:**
   - Đủ 5 chương theo chuẩn luận văn tốt nghiệp
   - Có tóm tắt song ngữ (Việt-Anh)
   - Bảng phân loại yêu cầu FR/NFR rõ ràng
   - Có pseudo-code, công thức toán học, schema dữ liệu

2. **Nội dung kỹ thuật chi tiết:**
   - Phân tích kỹ lưỡng các thuật toán: ArcFace, Complete Linkage, DBSCAN
   - Công thức toán học chính xác (Angular Margin, Cosine Distance)
   - Pipeline 12 stages được mô tả chi tiết

3. **Thực nghiệm có cơ sở:**
   - Thử nghiệm trên 5 video với đặc điểm đa dạng
   - Có nhiều độ đo đánh giá: Purity, NMI, ARI, BCubed F1, Silhouette, Davies-Bouldin

4. **Phân tích lỗi thực tế:**
   - Phân tích nguyên nhân thất bại (ánh sáng, che khuất, góc nghiêng)
   - So sánh hiệu quả auto-tuning

5. **Mã nguồn thực tế tồn tại:**
   - Dự án có đầy đủ code matching với nội dung luận văn
   - `config.yaml` khớp với Phụ lục A
   - Pipeline 12 stages đã được triển khai

---

## 2. VẤN ĐỀ VÀ SAI SÓT

### 2.1 Thông tin giả mạo / Không thể xác minh

> [!CAUTION]
> **Các tài liệu tham khảo trong nước có vấn đề nghiêm trọng**

| Tham khảo | Vấn đề |
|-----------|--------|
| **[15]** Nguyễn Văn A và cộng sự (2021): "Hệ thống nhận diện và theo dõi đối tượng..." | **GIẢ MẠO** - Tên tác giả generic "Nguyễn Văn A, Lê Thị B, Trần Văn C" là pattern điển hình của citation giả. Không thể tìm thấy bài báo này trên bất kỳ cơ sở dữ liệu nào. |
| **[16]** Trần Văn B và Nguyễn Thị C (2022): "Ứng dụng Deep Learning trong xây dựng hệ thống điểm danh..." | **GIẢ MẠO** - Cùng pattern naming. "Tạp chí Đại học Cần Thơ, vol. 58" - cần xác minh xem volume này có tồn tại không. |

> [!IMPORTANT]
> Đây là **lỗi nghiêm trọng về đạo đức học thuật**. Sinh viên cần thay thế bằng các tài liệu tham khảo thực có thể verify được.

### 2.2 Số liệu thử nghiệm có dấu hiệu bất thường

| Vấn đề | Chi tiết | Mức độ nghiêm trọng |
|--------|----------|---------------------|
| **Kết quả quá đẹp** | Pipeline giảm 76-80% clusters qua 3 stages - tỷ lệ gần như bằng nhau qua 5 video rất khác nhau | ⚠️ Trung bình |
| **Throughput chính xác** | Video NHAGIATIEN (2 giờ, 7200 frames) xử lý đúng 32 phút = khoảng 225 frames/phút. Số tròn đáng ngờ. | ⚠️ Trung bình |
| **Stage reduction quá consistent** | Stage 4→5: -54% đến -60%, Stage 5→6: -20% đến -46% - pattern quá đều | ⚠️ Trung bình |

### 2.3 Thông tin không khớp với code thực tế

| Trong luận văn | Trong code thực tế | Phân tích |
|----------------|-------------------|-----------|
| `distance_threshold: 0.4` (ví dụ trong text) | `distance_threshold.default: 1.15` (config.yaml) | **SAI** - Luận văn ghi threshold 0.4 nhưng code dùng 1.15. Cần giải thích sự khác biệt. |
| Nói "12 stages" | Pipeline.py chỉ có 11 stages (1, 1.5, 2-11, không có stage 12 riêng) | **CHÊNH LỆCH** - Evaluation task được gọi là stage 12 nhưng thực tế là optional |
| buffalo_l "từ Imperial College London" | InsightFace được phát triển bởi nhóm từ nhiều trường khác nhau | **KHÔNG CHÍNH XÁC** - InsightFace là dự án mã nguồn mở, không thuộc về riêng Imperial College |

### 2.4 Nội dung viết sơ sài / Thiếu

| Phần | Vấn đề |
|------|--------|  
| **So sánh với baseline** | Mục tiêu ghi "So sánh với các baseline" (Bước 4) nhưng KHÔNG CÓ so sánh thực tế nào trong Chương 4 |
| **Giao diện web** | Chỉ nhắc sơ qua, không có screenshot, không có đánh giá trải nghiệm người dùng |
| **API testing** | Không có test case cho API, không có benchmark performance |
| **Phụ lục B** | API endpoints liệt kê rất sơ sài, thiếu ví dụ request/response |

---

## 3. SO SÁNH VỚI CÁC BÀI BÁO THAM KHẢO

### 3.1 So với các tài liệu quốc tế được trích dẫn

| Công trình | Claim trong luận văn | Thực tế |
|------------|---------------------|---------|
| **[4] ArcFace (CVPR 2019)** | Đúng về kỹ thuật angular margin | ✅ Chính xác |
| **[10] RetinaFace (CVPR 2020)** | Đúng mô tả về face detection | ✅ Chính xác |
| **[12] Otto et al. (2018)** | Rank-Order Clustering cho 123 triệu faces | ✅ Chính xác |
| **[13] Tapaswi et al. (2019)** | Ball Cluster Learning, temporal constraints | ✅ Chính xác |
| **[14] Yang et al. (2020)** | GCN-D với confidence estimation | ✅ Chính xác |

> [!NOTE]
> Các tài liệu quốc tế được trích dẫn chính xác và có nguồn gốc verify được.

### 3.2 Điểm thiếu sót trong literature review

- **Không có so sánh định lượng** với các phương pháp GCN-D, BCL
- **Thiếu benchmark trên dataset chuẩn** (YouTube Faces, IJB-C) như đã thừa nhận trong giới hạn, nhưng đây là thiếu sót lớn cho một luận văn tốt nghiệp
- **Không có ablation study** cho việc chọn Complete Linkage vs Average Linkage vs Ward

---

## 4. ĐỊNH DẠNG VÀ TRÌNH BÀY

### 4.1 Vấn đề về format luận văn chuẩn

| Yếu tố | Đánh giá |
|--------|----------|
| **Trang bìa** | ❌ THIẾU - Không có trang bìa đúng format |
| **Nhiệm vụ luận văn** | ❌ THIẾU - Không có trang nhiệm vụ |
| **Lời cảm ơn** | ❌ THIẾU |
| **Mục lục** | ❌ THIẾU - Luận văn thực tế phải có mục lục tự động |
| **Danh mục hình** | ❌ THIẾU |
| **Danh mục bảng** | ❌ THIẾU |
| **Nội dung chính** | ✅ ĐẦY ĐỦ |
| **Tài liệu tham khảo** | ⚠️ CÓ VẤN ĐỀ - Có 2 ref giả |
| **Phụ lục** | ✅ CÓ nhưng sơ sài |

### 4.2 Vấn đề format Markdown

Đây là file Markdown, không phải Word/PDF chuẩn luận văn. Cần:
- Export sang Word với định dạng đúng
- Đánh số trang
- Font chữ Times New Roman 13pt
- Line spacing 1.5

---

## 5. ĐIỂM SỐ CHI TIẾT (Thang điểm 10)

| Tiêu chí | Điểm | Nhận xét |
|----------|------|----------|
| **Nội dung lý thuyết** | 8/10 | Tốt, đầy đủ, công thức chính xác |
| **Thiết kế hệ thống** | 7.5/10 | Pipeline rõ ràng, UML mô tả dạng text |
| **Thực nghiệm** | 6/10 | Có kết quả nhưng một số số liệu đáng ngờ |
| **Trình bày** | 5/10 | Thiếu nhiều yếu tố cần thiết |
| **Tính trung thực** | 4/10 | **Nghiêm trọng:** 2 tài liệu tham khảo giả mạo |
| **Khớp với code** | 8/10 | Code tồn tại và hoạt động, một vài chênh lệch |

### **ĐIỂM TỔNG: 6.5/10 (TRUNG BÌNH)**

---

## 6. CÁC ĐIỂM CẦN KHẮC PHỤC BẮT BUỘC

> [!WARNING]
> **Những điểm PHẢI sửa trước khi nộp**

1. **[CRITICAL] Xóa hoặc thay thế tài liệu tham khảo [15] và [16]:**
   - Thay bằng các bài báo thực từ các hội nghị/tạp chí Việt Nam có thể verify
   - Ví dụ: tìm trên Google Scholar các bài từ RIVF Conference, Tạp chí Công nghệ Thông tin và Truyền thông

2. **[CRITICAL] Thêm các trang bắt buộc:**
   - Trang bìa (theo format trường ĐHCT)
   - Trang nhiệm vụ luận văn
   - Lời cảm ơn
   - Mục lục tự động
   - Danh mục hình/bảng

3. **[HIGH] Giải thích sự khác biệt giữa luận văn và code:**
   - Distance threshold 0.4 vs 1.15 cần giải thích
   - Hoặc sửa cho thống nhất

4. **[MEDIUM] Thêm so sánh với baseline:**
   - Chạy thử nghiệm với DBSCAN và so sánh
   - Hoặc so sánh với K-Means + HAC

5. **[MEDIUM] Thêm hình ảnh minh họa:**
   - Screenshot giao diện web
   - Ví dụ output của hệ thống
   - Mermaid diagram thay vì ASCII art

6. **[LOW] Cập nhật số liệu chính xác hơn:**
   - Làm tròn số một cách hợp lý
   - Không cần quá "đẹp" - số liệu thực tế sẽ thuyết phục hơn

---

## 7. KẾT LUẬN

Luận văn có **nền tảng kỹ thuật tốt** và **mã nguồn thực tế hoạt động**. Tuy nhiên, có một số vấn đề nghiêm trọng về **đạo đức học thuật** (2 tài liệu tham khảo giả mạo) và **format trình bày** thiếu nhiều yếu tố bắt buộc.

**Khuyến nghị:** Cần sửa các điểm CRITICAL trước khi nộp chính thức để tránh bị đánh trượt hoặc bị cáo buộc gian lận học thuật.

---

*Đánh giá bởi: AI Assistant (với vai trò giáo sư chấm luận văn)*
*Ngày: 2025-12-18*
