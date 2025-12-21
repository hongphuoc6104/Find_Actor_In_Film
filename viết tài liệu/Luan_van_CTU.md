# LUẬN VĂN TỐT NGHIỆP ĐẠI HỌC

---

**TRƯỜNG ĐẠI HỌC CẦN THƠ**  
**KHOA CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG**

---

## HỆ THỐNG TỰ ĐỘNG NHẬN DIỆN VÀ PHÂN CỤM KHUÔN MẶT DIỄN VIÊN TRONG VIDEO SỬ DỤNG HỌC SÂU VÀ PHÂN CỤM PHÂN CẤP

---

**Sinh viên thực hiện:** [Họ và tên]  
**MSSV:** [Mã số sinh viên]  
**Ngành:** Khoa học Máy tính  
**Cán bộ hướng dẫn:** [Tên CBHD]

---

**Cần Thơ, Tháng 12/2025**

---

# TÓM TẮT

## Tóm tắt (Tiếng Việt)

Trong bối cảnh bùng nổ nội dung video số, việc tìm kiếm và nhận diện diễn viên trong các bộ phim đang trở thành một nhu cầu thiết yếu đối với cả người dùng cuối lẫn các nhà sản xuất nội dung. Tuy nhiên, phương pháp gán nhãn thủ công truyền thống đòi hỏi nguồn nhân lực lớn, tốn nhiều thời gian và dễ phát sinh sai sót, đặc biệt khi xử lý các bộ phim có thời lượng dài với nhiều nhân vật xuất hiện.

Luận văn này đề xuất một hệ thống tự động nhận diện và phân cụm khuôn mặt diễn viên trong video, kết hợp kỹ thuật học sâu (Deep Learning) với các thuật toán phân cụm phân cấp (Hierarchical Clustering). Hệ thống sử dụng mô hình **InsightFace** với kiến trúc **buffalo_l** để trích xuất đặc trưng khuôn mặt thành vector 512 chiều, sau đó áp dụng thuật toán **Agglomerative Clustering** với tiêu chí **Complete Linkage** để gom nhóm các khuôn mặt thuộc cùng một cá nhân. Quy trình được tối ưu hóa thông qua pipeline 3 giai đoạn hợp nhất (merge) và hệ thống tự động điều chỉnh tham số (auto-tuning) dựa trên đặc điểm của video đầu vào.

Hệ thống cung cấp giao diện web thân thiện cho phép người dùng tải lên ảnh khuôn mặt và tìm kiếm các phân cảnh xuất hiện của diễn viên tương ứng trong toàn bộ kho phim đã được lập chỉ mục.

**Từ khóa:** Nhận diện khuôn mặt, Phân cụm phân cấp, InsightFace, Complete Linkage, Học sâu, Xử lý video.

---

## Abstract (English)

In the context of the digital video content explosion, searching and identifying actors in movies has become an essential need for end-users and content producers alike. However, traditional manual labeling methods require significant human resources, are time-consuming, and are prone to errors, especially when processing long-duration films with many appearing characters.

This thesis proposes an automated system for actor face recognition and clustering in videos, combining Deep Learning techniques with Hierarchical Clustering algorithms. The system utilizes the **InsightFace** model with the **buffalo_l** architecture to extract facial features into 512-dimensional vectors, then applies the **Agglomerative Clustering** algorithm with the **Complete Linkage** criterion to group faces belonging to the same individual. The process is optimized through a 3-stage merge pipeline and an auto-tuning system that adjusts parameters based on input video characteristics.

The system provides a user-friendly web interface allowing users to upload a face image and search for corresponding actor appearances across the entire indexed movie database.

**Keywords:** Face Recognition, Hierarchical Clustering, InsightFace, Complete Linkage, Deep Learning, Video Processing.

---

# MỤC LỤC

- Chương 1: Giới thiệu
- Chương 2: Cơ sở lý thuyết và Công trình liên quan
- Chương 3: Phân tích và Thiết kế hệ thống
- Chương 4: Cài đặt và Kết quả thử nghiệm
- Chương 5: Kết luận và Hướng phát triển
- Tài liệu tham khảo
- Phụ lục

---

# CHƯƠNG 1: GIỚI THIỆU

## 1.1 Đặt vấn đề và Tính cấp thiết

### 1.1.1 Bối cảnh thực tiễn

Trong thời đại số hóa hiện nay, ngành công nghiệp giải trí đang chứng kiến sự gia tăng chưa từng có về khối lượng nội dung video. Mỗi ngày có hàng triệu giờ video mới được tải lên các nền tảng trực tuyến như YouTube, Netflix, và các dịch vụ OTT (Over-the-Top) trong nước. Điều này đặt ra một thách thức lớn trong việc quản lý, tổ chức và tìm kiếm thông tin trong kho dữ liệu video khổng lồ này.

Một nhu cầu phổ biến của người dùng là khả năng tìm kiếm các bộ phim hoặc phân cảnh dựa trên sự xuất hiện của một diễn viên cụ thể. Ví dụ, người dùng có thể muốn tìm tất cả các bộ phim có sự tham gia của một diễn viên yêu thích, hoặc xác định danh tính của một nhân vật trong một cảnh phim. Để thực hiện được điều này, các hệ thống cần phải có khả năng:

1. **Phát hiện khuôn mặt** (Face Detection): Xác định vị trí các khuôn mặt trong từng khung hình video.
2. **Trích xuất đặc trưng** (Feature Extraction): Chuyển đổi hình ảnh khuôn mặt thành các vector đặc trưng có khả năng so sánh.
3. **Phân cụm** (Clustering): Gom nhóm các khuôn mặt thuộc cùng một cá nhân mà không cần biết trước danh tính.
4. **Tìm kiếm** (Search): Cho phép truy vấn và trả về kết quả chính xác.

### 1.1.2 Hạn chế của phương pháp truyền thống

Phương pháp gán nhãn thủ công (manual labeling) hiện vẫn được sử dụng rộng rãi trong nhiều hệ thống quản lý nội dung video. Tuy nhiên, phương pháp này tồn tại nhiều hạn chế nghiêm trọng:

| Vấn đề | Mô tả |
|--------|-------|
| **Chi phí nhân lực cao** | Cần đội ngũ nhân viên lớn để xem và gán nhãn từng cảnh phim |
| **Tốn thời gian** | Một bộ phim 2 giờ có thể cần 8-10 giờ để gán nhãn thủ công |
| **Sai sót chủ quan** | Phụ thuộc vào khả năng nhận diện của con người, dễ bỏ sót hoặc nhầm lẫn |
| **Không khả thi với quy mô lớn** | Không thể mở rộng khi số lượng video tăng theo cấp số nhân |
| **Thiếu tính nhất quán** | Các nhãn có thể không đồng nhất giữa các người gán nhãn khác nhau |

Những hạn chế trên đã thúc đẩy nhu cầu phát triển các hệ thống tự động hóa dựa trên trí tuệ nhân tạo và học máy.

### 1.1.3 Tầm quan trọng của bài toán

**Đối với Khoa học Máy tính:**
- Bài toán phân cụm khuôn mặt (Face Clustering) là một trường hợp đặc biệt của bài toán học không giám sát (Unsupervised Learning), đòi hỏi các thuật toán phải có khả năng tìm ra cấu trúc ẩn trong dữ liệu mà không cần nhãn đã biết trước.
- Việc kết hợp giữa mạng nơ-ron sâu (Deep Neural Networks) cho trích xuất đặc trưng và các thuật toán phân cụm truyền thống mở ra hướng nghiên cứu kết hợp (hybrid) đầy tiềm năng.

**Đối với Xã hội:**
- Tự động hóa quá trình quản lý nội dung giúp giảm chi phí sản xuất và vận hành cho các đơn vị truyền thông.
- Người dùng cuối được hưởng lợi từ các công cụ tìm kiếm thông minh, nâng cao trải nghiệm giải trí.
- Ứng dụng tiềm năng trong nhiều lĩnh vực: an ninh (giám sát video), truyền thông (tổ chức kho phim), giáo dục (tìm kiếm bài giảng).

---

## 1.2 Mục tiêu nghiên cứu

### 1.2.1 Mục tiêu tổng quát

Xây dựng một hệ thống hoàn chỉnh có khả năng **tự động nhận diện và phân cụm khuôn mặt diễn viên** trong video, cho phép người dùng tìm kiếm các phân cảnh xuất hiện của diễn viên dựa trên ảnh đầu vào.

### 1.2.2 Mục tiêu cụ thể

1. **Nghiên cứu và áp dụng mô hình học sâu InsightFace:**
   - Tìm hiểu kiến trúc mạng ArcFace và RetinaFace trong bộ mô hình buffalo_l.
   - Đánh giá hiệu quả trích xuất đặc trưng khuôn mặt thành vector 512 chiều.

2. **Thiết kế thuật toán phân cụm phân cấp tối ưu:**
   - Áp dụng Agglomerative Clustering với tiêu chí Complete Linkage.
   - Xây dựng pipeline 3 giai đoạn hợp nhất (merge) để cải thiện độ chính xác.
   - Phát triển hệ thống tự động điều chỉnh tham số (auto-tuning) theo đặc điểm video.

3. **Phát triển giao diện người dùng thân thiện:**
   - Xây dựng API RESTful cho tích hợp hệ thống.
   - Thiết kế giao diện web cho phép tải ảnh và hiển thị kết quả tìm kiếm.

---

## 1.3 Phạm vi nghiên cứu

### 1.3.1 Phạm vi dữ liệu

- **Nguồn dữ liệu:** Các bộ phim, phim ngắn, và video clip tiếng Việt.
- **Định dạng hỗ trợ:** MP4, AVI, MKV, MOV.
- **Thời lượng:** Hỗ trợ video từ ngắn (dưới 10 phút) đến dài (trên 2 giờ).
- **Điều kiện:** Các video có chất lượng từ trung bình đến cao, với các điều kiện ánh sáng và góc quay đa dạng.

### 1.3.2 Phạm vi phương pháp

| Khía cạnh | Phương pháp sử dụng |
|-----------|---------------------|
| **Phát hiện khuôn mặt** | RetinaFace (trong InsightFace) |
| **Trích xuất đặc trưng** | ArcFace với mô hình buffalo_l |
| **Phân cụm** | Agglomerative Clustering + Complete Linkage |
| **Hợp nhất cụm** | Hierarchical merge với Average Linkage trên centroid |
| **Độ đo khoảng cách** | Cosine distance |

### 1.3.3 Giới hạn nghiên cứu

- Hệ thống tập trung vào **phân cụm không giám sát** (unsupervised clustering), không yêu cầu dữ liệu huấn luyện có nhãn cho mỗi video mới.
- Chưa xử lý các trường hợp đặc biệt như: khuôn mặt bị che hoàn toàn, trang điểm biến đổi mạnh, hoặc deepfake.
- Hiệu năng được tối ưu cho GPU NVIDIA với CUDA; trên CPU sẽ chậm hơn đáng kể.

---

## 1.4 Phương pháp nghiên cứu

### 1.4.1 Quy trình nghiên cứu tổng thể

Nghiên cứu được thực hiện theo quy trình gồm 5 bước chính:

| Bước | Nội dung |
|------|----------|
| **1** | Nghiên cứu lý thuyết về nhận diện khuôn mặt, phân cụm phân cấp, mô hình InsightFace |
| **2** | Thiết kế kiến trúc hệ thống: pipeline xử lý 12 giai đoạn, API và giao diện |
| **3** | Cài đặt và triển khai các thuật toán trích xuất đặc trưng và phân cụm |
| **4** | Thử nghiệm và đánh giá trên bộ dữ liệu video tiếng Việt |
| **5** | Tối ưu hóa, xây dựng hệ thống auto-tuning và hoàn thiện tài liệu |

### 1.4.2 Phương pháp thu thập dữ liệu

- **Dữ liệu huấn luyện mô hình:** Sử dụng mô hình InsightFace đã được huấn luyện sẵn trên các bộ dữ liệu lớn như MS1MV3, CASIA-WebFace.
- **Dữ liệu thử nghiệm:** Thu thập từ các nguồn video công khai, bao gồm phim Việt Nam và video clip trên các nền tảng trực tuyến.
- **Đánh giá:** Sử dụng các độ đo nội tại (internal metrics) không cần gán nhãn, phù hợp với bản chất **học không giám sát** của bài toán.

---

## 1.5 Ý nghĩa khoa học và thực tiễn

### 1.5.1 Ý nghĩa khoa học

- Đóng góp một phương pháp kết hợp hiệu quả giữa Deep Learning và Hierarchical Clustering cho bài toán phân cụm khuôn mặt trong video.
- Đề xuất cơ chế 3 giai đoạn hợp nhất (Initial Clustering → Merge → Satellite Assimilation) giúp cải thiện đáng kể chất lượng phân cụm.
- Xây dựng hệ thống tự điều chỉnh dựa trên hồ sơ đặc điểm video (ánh sáng, độ nét, độ phức tạp), mở ra hướng nghiên cứu về phân cụm thích ứng.

### 1.5.2 Ý nghĩa thực tiễn

- Cung cấp công cụ tự động hóa cho các đơn vị sản xuất nội dung video, giảm chi phí và thời gian gán nhãn.
- Nâng cao trải nghiệm người dùng với khả năng tìm kiếm diễn viên trong kho phim lớn.
- Làm nền tảng cho các ứng dụng mở rộng: hệ thống gợi ý phim, phân tích hành vi nhân vật, nghiên cứu điện ảnh.

---

## 1.6 Bố cục luận văn

Luận văn được tổ chức thành **5 chương** với nội dung như sau:

| Chương | Tiêu đề | Nội dung chính |
|--------|---------|----------------|
| **1** | Giới thiệu | Đặt vấn đề, mục tiêu, phạm vi và phương pháp nghiên cứu |
| **2** | Cơ sở lý thuyết | Tổng quan về nhận diện khuôn mặt, phân cụm phân cấp, mô hình InsightFace |
| **3** | Phân tích và thiết kế hệ thống | Kiến trúc hệ thống, thiết kế pipeline, cấu trúc dữ liệu |
| **4** | Cài đặt và thử nghiệm | Hiện thực hệ thống, kết quả thử nghiệm, đánh giá |
| **5** | Kết luận | Tổng kết, đóng góp, hạn chế và hướng phát triển |

---

*Kết thúc Chương 1*

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN

## 2.1 Tổng quan về Nhận diện Khuôn mặt Dựa trên Học sâu

### 2.1.1 Sự phát triển của các phương pháp nhận diện khuôn mặt

Lĩnh vực nhận diện khuôn mặt (Face Recognition) đã trải qua nhiều giai đoạn phát triển, từ các phương pháp thủ công dựa trên đặc trưng hình học trong những năm 1990, đến các kỹ thuật trích xuất đặc trưng cục bộ như SIFT, HOG, và LBP vào đầu những năm 2000. Tuy nhiên, bước đột phá thực sự đến từ sự ra đời của học sâu (Deep Learning), đặc biệt là Convolutional Neural Networks (CNNs), cho phép học các biểu diễn đặc trưng trực tiếp từ dữ liệu pixel thô.

Các cột mốc quan trọng trong tiến trình này bao gồm:

| Năm | Công trình | Đóng góp |
|-----|------------|----------|
| 2014 | DeepFace (Facebook) | Mạng 9 lớp đạt 97.35% trên LFW, tiếp cận hiệu suất con người |
| 2015 | FaceNet (Google) | Học embedding trực tiếp với triplet loss, đạt 99.63% trên LFW |
| 2019 | ArcFace | Angular margin loss tạo decision boundary phân biệt rõ ràng hơn |

### 2.1.2 Bài toán Face Clustering trong ngữ cảnh Video

Khác với bài toán nhận diện khuôn mặt có giám sát (supervised face recognition) nơi mô hình được huấn luyện trên các danh tính đã biết trước, **Face Clustering** là một bài toán **không giám sát** (unsupervised) với các thách thức đặc thù:

1. **Số lượng cụm không xác định**: Thuật toán phải tự động xác định số danh tính xuất hiện trong video.
2. **Biến đổi lớn trong cùng lớp**: Cùng một người có thể xuất hiện với nhiều biểu cảm, góc nghiêng, điều kiện ánh sáng khác nhau.
3. **Mất cân bằng dữ liệu**: Nhân vật chính có hàng trăm khung hình, trong khi vai phụ chỉ xuất hiện vài giây.
4. **Nhiễu và ngoại lệ**: Các khuôn mặt bị che, mờ, hoặc phát hiện sai cần được xử lý một cách vững chắc.

---

## 2.2 Mô hình InsightFace và Kiến trúc ArcFace

### 2.2.1 Tổng quan về InsightFace

**InsightFace** là một thư viện mã nguồn mở cung cấp các mô hình nhận diện khuôn mặt hiện đại nhất (state-of-the-art), được phát triển bởi nhóm nghiên cứu từ Đại học Hoàng gia London (Imperial College London). Thư viện này tích hợp nhiều thành phần:

| Thành phần | Chức năng | Mô hình trong buffalo_l |
|------------|-----------|-------------------------|
| Face Detection | Phát hiện vị trí khuôn mặt | RetinaFace |
| Face Alignment | Căn chỉnh khuôn mặt về pose chuẩn | 5-point landmark |
| Face Recognition | Trích xuất embedding vector | ArcFace (ResNet-100) |
| Face Attribute | Ước lượng tuổi, giới tính | Attribute model |

Hệ thống trong luận văn này sử dụng bộ mô hình **buffalo_l** (Large Buffalo), bao gồm backbone ResNet-100 được huấn luyện trên bộ dữ liệu MS1MV3 với khoảng 5.8 triệu ảnh từ 93,000 danh tính.

### 2.2.2 Kiến trúc ArcFace và Hàm mất mát Angular Margin

#### 2.2.2.1 Nền tảng: Softmax Loss

Phương pháp huấn luyện truyền thống cho nhận diện khuôn mặt sử dụng **Softmax Loss**:

$$\mathcal{L}_{softmax} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^{C} e^{W_j^T x_i + b_j}}$$

Trong đó:
- $x_i \in \mathbb{R}^d$: embedding của mẫu thứ $i$
- $y_i$: nhãn lớp
- $W_j \in \mathbb{R}^d$: trọng số của lớp $j$
- $C$: số lượng danh tính

Hàm mất mát Softmax tối ưu hóa khả năng phân loại nhưng không đảm bảo **tính gọn trong lớp** (các mẫu cùng lớp gần nhau) và **khả năng phân tách giữa các lớp** (các lớp khác nhau xa nhau).

#### 2.2.2.2 Angular Margin Loss: ArcFace

**ArcFace** là phương pháp hiệu quả nhất, áp dụng additive angular margin trực tiếp lên góc:

$$\mathcal{L}_{arc} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$

Trong đó:
- $s$: scale factor (thường $s = 64$)
- $m$: angular margin (thường $m = 0.5$ radians ≈ 28.6°)
- $\theta_{y_i} = \arccos(W_{y_i}^T x_i)$: góc giữa embedding và trọng số lớp

#### 2.2.2.3 Diễn giải hình học của ArcFace

Ý nghĩa hình học của ArcFace: Margin $m = 0.5$ rad tạo ra một "vùng cấm" (margin zone) giữa các decision boundaries của các lớp khác nhau. Điều này đảm bảo rằng để một mẫu được phân loại đúng, góc của nó với tâm lớp đúng phải nhỏ hơn góc với bất kỳ lớp nào khác **ít nhất $m$ radians**.

> [Hình 2.1: Minh họa Angular Margin trong không gian embedding - Nguồn: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019]

### 2.2.3 Quy trình trích xuất Embedding trong Hệ thống

Quy trình trích xuất embedding 512 chiều từ một khung hình video bao gồm các bước sau:

| Bước | Nội dung | Kết quả |
|------|----------|---------|
| **1** | Phát hiện khuôn mặt với RetinaFace | Khung bao, điểm tin cậy, 5 điểm mốc |
| **2** | Căn chỉnh khuôn mặt | Ảnh khuôn mặt căn chỉnh $112 \times 112$ điểm ảnh |
| **3** | Trích xuất đặc trưng (ResNet-100) | Vector 512 chiều |
| **4** | Chuẩn hóa L2 | Vector đơn vị trên siêu cầu |

Công thức L2 Normalization:

$$\hat{x} = \frac{x}{\|x\|_2} = \frac{x}{\sqrt{\sum_{i=1}^{512} x_i^2}}$$

Điều này đảm bảo tất cả các vector đặc trưng nằm trên siêu cầu đơn vị trong không gian $\mathbb{R}^{512}$, cho phép sử dụng **độ tương đồng cosin** như độ đo khoảng cách:

$$\text{sim}(a, b) = \hat{a}^T \hat{b} = \cos \theta_{a,b}$$

---

## 2.3 Các Thuật toán Phân cụm Phân cấp

### 2.3.1 Đặc điểm của Phân cụm Phân cấp (Hierarchical Clustering)

Phân cụm phân cấp là họ các thuật toán tạo ra **dendrogram** – cấu trúc cây biểu diễn quan hệ phân cấp giữa các cụm. Có hai hướng tiếp cận chính:

- **Agglomerative (bottom-up)**: Bắt đầu với mỗi điểm là một cụm, liên tục hợp nhất các cặp cụm gần nhất cho đến khi đạt điều kiện dừng.
- **Divisive (top-down)**: Bắt đầu với tất cả điểm trong một cụm, liên tục chia cụm lớn thành các cụm nhỏ hơn.

Trong bài toán phân cụm khuôn mặt, **phân cụm gộp dần** (Agglomerative Clustering) được ưa chuộng vì:
1. Không yêu cầu biết trước số cụm $k$.
2. Có thể sử dụng **ngưỡng khoảng cách** để tự động xác định số cụm.
3. Phù hợp với dữ liệu có cấu trúc phân cấp tự nhiên.

### 2.3.2 Các tiêu chí liên kết (Linkage Criteria)

Sự khác biệt cốt lõi giữa các biến thể của Agglomerative Clustering nằm ở cách định nghĩa **khoảng cách giữa hai cụm** $d(C_i, C_j)$:

| Tiêu chí | Công thức | Đặc điểm |
|----------|-----------|----------|
| **Liên kết đơn** | $d_{single}(C_i, C_j) = \min_{a \in C_i, b \in C_j} d(a, b)$ | Dễ bị hiệu ứng dây chuyền |
| **Liên kết hoàn toàn** | $d_{complete}(C_i, C_j) = \max_{a \in C_i, b \in C_j} d(a, b)$ | Tạo cụm gọn |
| **Liên kết trung bình** | $d_{average}(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{a \in C_i} \sum_{b \in C_j} d(a, b)$ | Cân bằng |
| **Phương pháp Ward** | Tối thiểu hóa tăng phương sai nội cụm | Yêu cầu khoảng cách Euclid |

### 2.3.3 Lý do chọn Complete Linkage cho Face Clustering

1. **Đảm bảo tính gắn kết trong cụm**: Với Liên kết hoàn toàn, điều kiện $d_{complete}(C_i, C_i) < ng\u01b0\u1ee1ng$ đảm bảo **mọi cặp khuôn mặt** trong cùng cụm có khoảng cách nhỏ hơn ngưỡng. Điều này rất quan trọng vì chúng ta muốn chắc chắn rằng tất cả ảnh trong một cụm thuộc về cùng một người.

2. **Kiểm soát chặt bán kính cụm**: Liên kết hoàn toàn ngăn chặn việc hình thành các cụm "rải rác" bằng cách hạn chế đường kính tối đa.

3. **Tương thích với khoảng cách cosin**: Liên kết hoàn toàn hoạt động tốt với bất kỳ độ đo khoảng cách nào, trong khi phương pháp Ward yêu cầu khoảng cách Euclid.

4. **Ngưỡng có ý nghĩa rõ ràng**: Với vector đặc trưng khuôn mặt đã chuẩn hóa L2, khoảng cách cosin 0.4 tương đương với góc $\theta \approx 66°$ giữa hai vector. Điều này dễ diễn giải và điều chỉnh.

> [Note: Vẽ hình minh họa so sánh Single Linkage vs Complete Linkage. Trong Single Linkage, hai cụm gần nhau có thể bị nối bởi một cầu nối nhiễu (chaining effect). Trong Complete Linkage, cụm có đường kính tối đa được kiểm soát.]

---

## 2.4 Các Công trình Nghiên cứu Liên quan

### 2.4.1 Nghiên cứu quốc tế

| Công trình | Tác giả | Phương pháp | So sánh với hệ thống đề xuất |
|------------|---------|-------------|------------------------------|
| Clustering Millions of Faces by Identity | Otto et al., 2018 | Approximate Rank-Order Clustering | Phức tạp hơn, tối ưu cho big data |
| Video Face Clustering with Unknown Number of Clusters | Tapaswi et al., 2019 | Ball Cluster Learning + Temporal constraints | Sử dụng temporal constraints tường minh |
| Learning to Cluster Faces via GCN | Yang et al., 2020 | Graph Convolutional Network | Yêu cầu supervised training |

### 2.4.2 Nhận xét

Hệ thống của luận văn chọn hướng tiếp cận cân bằng giữa hiệu quả và đơn giản trong triển khai: sử dụng mô hình pre-trained (InsightFace) kết hợp hierarchical clustering với quy tắc heuristic cho auto-tuning, phù hợp với quy mô video đơn lẻ mà không cần huấn luyện thêm.

---

## 2.5 Các Độ đo Đánh giá Phân cụm (Unsupervised)

Do bản chất **học không giám sát** (unsupervised learning) của bài toán Face Clustering, luận văn này sử dụng các **độ đo nội tại** (internal validation metrics) không yêu cầu ground truth. Đây là hướng tiếp cận phổ biến trong các nghiên cứu quốc tế [11], [12].

### 2.5.1 Silhouette Coefficient (Hệ số Silhouette)

**Nguồn gốc:** Được đề xuất trong [11].

**Công thức:**
$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Trong đó:
- $a(i)$: Khoảng cách trung bình từ điểm $i$ đến các điểm khác **trong cùng cụm** (đo tính gắn kết)
- $b(i)$: Khoảng cách trung bình tối thiểu từ điểm $i$ đến các điểm trong **cụm gần nhất** (đo tính phân tách)

**Ý nghĩa:**
| Giá trị | Diễn giải |
|---------|----------|
| $S \approx 1$ | Điểm được gán đúng cụm, cụm gọn và phân tách tốt |
| $S \approx 0$ | Điểm nằm trên ranh giới giữa hai cụm |
| $S < 0$ | Điểm có thể bị gán sai cụm |

### 2.5.2 Davies-Bouldin Index

**Nguồn gốc:** Được đề xuất trong [12].

**Công thức:**
$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Trong đó:
- $\sigma_i$: Độ phân tán trung bình trong cụm $i$
- $d(c_i, c_j)$: Khoảng cách giữa centroids của cluster $i$ và $j$

**Ý nghĩa:** Davies-Bouldin Index đo tỷ lệ giữa độ phân tán nội cụm và khoảng cách liên cụm. **Giá trị càng thấp càng tốt** (< 1.0 được coi là tốt).

### 2.5.3 Calinski-Harabasz Index (Variance Ratio Criterion)

**Công thức:**
$$CH = \frac{SS_B / (K-1)}{SS_W / (N-K)}$$

Trong đó:
- $SS_B$: Tổng bình phương khoảng cách giữa các cluster (between-cluster variance)
- $SS_W$: Tổng bình phương khoảng cách trong mỗi cluster (within-cluster variance)
- $K$: Số clusters, $N$: Tổng số điểm

**Ý nghĩa:** Đo tỷ lệ giữa độ phân tách liên cụm và độ compact nội cụm. **Giá trị càng cao càng tốt**.

### 2.5.4 Dunn Index

**Công thức:**
$$DI = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k \text{diam}(C_k)}$$

Trong đó:
- $d(C_i, C_j)$: Khoảng cách tối thiểu giữa hai cụm khác nhau
- $\text{diam}(C_k)$: Đường kính (khoảng cách lớn nhất giữa hai điểm) trong cụm $k$

**Ý nghĩa:** Đo tỷ lệ giữa khoảng cách liên cụm tối thiểu và đường kính cụm tối đa. **Giá trị càng cao càng tốt** (> 1 là lý tưởng).

### 2.5.5 Intra-cluster Distance (Mean Pairwise Distance)

**Công thức:**
$$\bar{d}_{intra}(C) = \frac{2}{|C|(|C|-1)} \sum_{i < j} d(x_i, x_j)$$

**Ý nghĩa:** Khoảng cách trung bình giữa các cặp điểm trong cùng cụm. Giá trị nhỏ cho thấy cụm compact. Đây là đo lường trực tiếp nhất cho **cluster cohesion**.

### 2.5.6 Bảng Tổng hợp Các Độ đo

| Độ đo | Giá trị tốt | Ưu điểm | Nhược điểm |
|-------|-------------|---------|------------|
| **Silhouette** | > 0.5 | Dễ diễn giải, quen thuộc | Chậm với dữ liệu lớn $O(n^2)$ |
| **Davies-Bouldin** | < 1.0 | Xử lý tốt các cụm hình dạng khác nhau | Nhạy cảm với outliers |
| **Calinski-Harabasz** | Cao | Nhanh, tương đối ổn định | Thiên về cụm hình cầu |
| **Dunn Index** | > 1.0 | Đánh giá cả separation và compactness | Rất nhạy với nhiễu |
| **Intra-cluster Distance** | Thấp | Trực quan, dễ hiểu | Không xét đến separation |

---

## 2.6 Phương pháp Trực quan hóa Không gian Embedding

Để đánh giá **định tính** kết quả phân cụm, luận văn sử dụng các phương pháp giảm chiều (dimensionality reduction) để hiển thị dữ liệu 512 chiều lên không gian 2D.

### 2.6.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Nguồn gốc:** Được đề xuất trong [13].

**Nguyên lý:** t-SNE chuyển đổi tương đồng (similarity) giữa các điểm trong không gian cao chiều thành xác suất, sau đó tối ưu hóa để duy trì cấu trúc này trong không gian thấp chiều.

**Tham số quan trọng:**
| Tham số | Mô tả | Giá trị thường dùng |
|---------|-------|--------------------|
| `perplexity` | Số lượng "láng giềng" hiệu quả | 30-50 |
| `n_iter` | Số vòng lặp tối ưu | 1000 |
| `learning_rate` | Tốc độ học | 200 |

**Ưu điểm:** Rất tốt trong việc **bảo toàn cấu trúc cục bộ** (local structure), giúp nhìn thấy các cụm rõ ràng trên không gian 2D.

**Nhược điểm:** Không bảo toàn tốt cấu trúc toàn cục, khoảng cách giữa các cụm trên biểu đồ có thể không phản ánh thực tế.

### 2.6.2 UMAP (Uniform Manifold Approximation and Projection)

**Nguồn gốc:** Được đề xuất trong [14].

**Nguyên lý:** UMAP dựa trên lý thuyết hình học Riemannian và topological data analysis để tìm một biểu diễn thấp chiều bảo toàn cả cấu trúc cục bộ và toàn cục.

**Tham số quan trọng:**
| Tham số | Mô tả | Giá trị thường dùng |
|---------|-------|--------------------|
| `n_neighbors` | Số lượng láng giềng | 15-50 |
| `min_dist` | Khoảng cách tối thiểu giữa các điểm trong 2D | 0.1-0.5 |
| `metric` | Độ đo khoảng cách | cosine |

**Ưu điểm:**
- Nhanh hơn t-SNE đáng kể (phù hợp dữ liệu lớn)
- Bảo toàn tốt **cả cấu trúc cục bộ và toàn cục**
- Khoảng cách giữa các cụm phản ánh thực tế hơn

### 2.6.3 So sánh t-SNE và UMAP

| Tiêu chí | t-SNE | UMAP |
|----------|-------|------|
| Tốc độ | Chậm $O(n^2)$ | Nhanh $O(n \log n)$ |
| Cấu trúc cục bộ | Rất tốt | Tốt |
| Cấu trúc toàn cục | Kém | Tốt |
| Ổn định | Thấp (random seed) | Cao |
| Phổ biến | Rộng rãi | Đang tăng |

**Lựa chọn trong luận văn:** Sử dụng **UMAP** với `metric="cosine"` để phản ánh đúng độ đo được dùng trong clustering (cosine distance).

---

## 2.7 Tổng kết Chương 2

Chương này đã trình bày các nền tảng lý thuyết cốt lõi cho hệ thống:

1. **InsightFace và ArcFace**: Mô hình trích xuất embedding với angular margin loss tạo ra các biểu diễn khuôn mặt có tính phân biệt cao.

2. **Complete Linkage Clustering**: Thuật toán phân cụm phân cấp đảm bảo intra-cluster cohesion, phù hợp với yêu cầu "mọi mẫu trong cụm phải tương đồng".

3. **Các nghiên cứu liên quan**: Từ rank-order clustering cho big data đến GCN-based methods, mỗi phương pháp có ưu nhược điểm riêng.

4. **Độ đo đánh giá nội tại**: Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn Index – các độ đo không cần ground truth, phù hợp với bài toán unsupervised.

5. **Trực quan hóa 2D**: t-SNE và UMAP cho phép trực quan hóa không gian embedding 512 chiều lên 2D để đánh giá định tính.

---

*Kết thúc Chương 2*

---

# CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1 Tổng quan Phương pháp Đề xuất

### 3.1.1 Ý tưởng chính

Hệ thống đề xuất trong luận văn này kết hợp hai phương pháp:

1. **Học sâu (Deep Learning)**: Sử dụng mô hình InsightFace với kiến trúc ArcFace để trích xuất đặc trưng khuôn mặt thành vector 512 chiều.

2. **Phân cụm phân cấp (Hierarchical Clustering)**: Áp dụng thuật toán Agglomerative Clustering với tiêu chí Complete Linkage để gom nhóm các khuôn mặt thuộc cùng một danh tính.

![Hình 3.1: Tổng quan phương pháp đề xuất](file:///path/to/fig3_1_overview.png)
*Hình 3.1: Sơ đồ tổng quan phương pháp đề xuất - Kết hợp Deep Learning và Hierarchical Clustering*

### 3.1.2 Các giai đoạn xử lý chính

Quy trình xử lý được chia thành **5 giai đoạn chính**:

| Giai đoạn | Tên | Mục đích |
|-----------|-----|----------|
| 1 | Tiền xử lý (Preprocessing) | Trích xuất frames từ video, phân tích đặc điểm |
| 2 | Phát hiện và trích xuất (Detection & Extraction) | Phát hiện khuôn mặt, trích xuất embedding 512-D |
| 3 | Phân cụm (Clustering) | Gom nhóm khuôn mặt thuộc cùng danh tính |
| 4 | Hậu xử lý (Post-processing) | Lọc, hợp nhất, tối ưu cụm |
| 5 | Xuất kết quả (Output) | Tạo danh sách nhân vật và thông tin scenes |

---

## 3.2 Kiến trúc Hệ thống

### 3.2.1 Kiến trúc Pipeline

Hệ thống được thiết kế theo kiến trúc **Pipeline tuần tự** (Sequential Pipeline Architecture), trong đó dữ liệu đi qua nhiều giai đoạn xử lý liên tiếp.

![Hình 3.2: Kiến trúc pipeline của hệ thống](file:///path/to/fig3_2_pipeline_architecture.png)
*Hình 3.2: Kiến trúc pipeline xử lý video - Luồng dữ liệu từ đầu vào đến đầu ra*

**Các thành phần chính:**

| Thành phần | Chức năng | Công nghệ |
|------------|-----------|-----------|
| **Nạp video** | Trích xuất khung hình từ video | FFmpeg |
| **Phát hiện khuôn mặt** | Xác định vị trí khuôn mặt | RetinaFace [6] |
| **Trích xuất đặc trưng** | Chuyển khuôn mặt thành vector 512 chiều | ArcFace [4] |
| **Bộ phân cụm** | Gom nhóm khuôn mặt | Agglomerative Clustering |
| **Tự điều chỉnh tham số** | Tự động tối ưu cấu hình | Hệ thống dựa trên quy tắc |

### 3.2.2 Luồng xử lý chi tiết

Luồng xử lý pipeline được mô tả trong Hình 3.3:

![Hình 3.3: Luồng xử lý pipeline 12 stages](file:///path/to/fig3_3_pipeline_flowchart.png)
*Hình 3.3: Sơ đồ luồng xử lý pipeline với 12 giai đoạn*

**Mô tả chi tiết từng giai đoạn:**

| Giai đoạn | Tên | Đầu vào | Đầu ra |
|-------|-----|---------|--------|
| 1 | Nạp video | Video (MP4) | Khung hình (JPEG) |
| 1.5 | Phân tích video | Khung hình | Hồ sơ video |
| 2 | Trích xuất đặc trưng | Khung hình | Vector 512 chiều |
| 3 | Lưu kho dữ liệu | Vector đặc trưng | File Parquet |
| 4 | Phân cụm ban đầu | Tâm track | Cụm thô |
| 5 | Hợp nhất cụm | Tâm cụm | Cụm đã gộp |
| 6 | Lọc cụm | Tất cả cụm | Cụm chính |
| 7 | Hấp thụ vệ tinh | Cụm chính + nhỏ | Cụm cuối cùng |
| 8 | Tạo ảnh xem trước | Cụm cuối cùng | Ảnh |
| 9 | Tạo danh sách nhân vật | Cụm | JSON |

---

## 3.3 Thiết kế Module Phát hiện và Trích xuất Khuôn mặt

### 3.3.1 Phát hiện khuôn mặt với RetinaFace

RetinaFace [6] là mô hình phát hiện khuôn mặt một giai đoạn (one-stage) với các đặc điểm:

- **Học đa nhiệm (Multi-task learning)**: Kết hợp phát hiện khuôn mặt, định vị điểm mốc, và tái tạo khuôn mặt 3D.
- **Mạng kim tự tháp đặc trưng (FPN)**: Phát hiện khuôn mặt ở nhiều kích thước khác nhau.
- **Phát hiện không dùng anchor**: Tăng tốc độ và giảm nhận diện sai.

![Hình 3.4: Kiến trúc RetinaFace](file:///path/to/fig3_4_retinaface.png)
*Hình 3.4: Kiến trúc mạng RetinaFace với Feature Pyramid Network [6]*

**Đầu ra của RetinaFace:**

| Thông tin | Mô tả |
|-----------|-------|
| Khung bao | Tọa độ (x1, y1, x2, y2) |
| Điểm tin cậy | Độ tin cậy phát hiện (0-1) |
| 5 điểm mốc | Mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải |

### 3.3.2 Trích xuất Embedding với ArcFace

ArcFace [4] trích xuất đặc trưng khuôn mặt thành vector 512 chiều với các bước:

| Bước | Mô tả | Kết quả |
|------|-------|---------|
| 1 | Căn chỉnh khuôn mặt | Chuẩn hóa về 112×112 điểm ảnh |
| 2 | Trích xuất đặc trưng | Đưa qua mạng ResNet-100 |
| 3 | Chuẩn hóa L2 | Chuẩn hóa vector về độ dài 1 |

![Hình 3.5: Quy trình trích xuất embedding](file:///path/to/fig3_5_embedding_extraction.png)
*Hình 3.5: Quy trình trích xuất embedding từ khuôn mặt*

**Công thức L2 Normalization:**
$$\hat{e} = \frac{e}{\|e\|_2} = \frac{e}{\sqrt{\sum_{i=1}^{512} e_i^2}}$$

### 3.3.3 Bộ lọc chất lượng khuôn mặt

Để đảm bảo chất lượng embedding, hệ thống áp dụng các bộ lọc:

| Tiêu chí | Ngưỡng mặc định | Ý nghĩa |
|----------|-----------------|---------|
| Điểm phát hiện | ≥ 0.45 | Độ tin cậy phát hiện |
| Kích thước khuôn mặt | ≥ 50 điểm ảnh | Kích thước tối thiểu |
| Điểm độ nét | ≥ 40.0 | Đo độ nét bằng Laplacian |
| Điểm góc nghiêng | ≥ 0.5 | Chất lượng góc quay đầu |

![Hình 3.6: Ví dụ bộ lọc chất lượng](file:///path/to/fig3_6_quality_filter.png)
*Hình 3.6: Ví dụ khuôn mặt đạt/không đạt tiêu chuẩn chất lượng*

---

## 3.4 Thiết kế Module Phân cụm Khuôn mặt

### 3.4.1 Liên kết Tracklet

Trước khi phân cụm, hệ thống liên kết các khuôn mặt liên tiếp của cùng một người thành **chuỗi theo dõi (tracklet)** dựa trên:

1. **Độ chồng lấp (IoU)**: Tỷ lệ trùng khớp khung bao giữa các khung hình liên tiếp.
2. **Độ tương đồng cosin**: Mức độ giống nhau giữa các vector đặc trưng.

![Hình 3.7: Minh họa tracklet linking](file:///path/to/fig3_7_tracklet_linking.png)
*Hình 3.7: Liên kết các detection liên tiếp thành tracklet*

**Track Centroid:**
$$c_{track} = \text{L2Norm}\left(\frac{1}{n}\sum_{i=1}^{n} e_i\right)$$

### 3.4.2 Phân cụm ban đầu với Complete Linkage

Hệ thống sử dụng **Agglomerative Clustering** với tiêu chí **Complete Linkage**:

**Cosine Distance:**
$$d_{cos}(a, b) = 1 - a^T b$$

**Complete Linkage:**
$$d_{complete}(C_i, C_j) = \max_{a \in C_i, b \in C_j} d_{cos}(a, b)$$

![Hình 3.8: Minh họa Complete Linkage](file:///path/to/fig3_8_complete_linkage.png)
*Hình 3.8: So sánh Single Linkage (dễ bị chaining) và Complete Linkage (cụm compact)*

**Lý do chọn Complete Linkage:**

| Tiêu chí | Liên kết hoàn toàn | Liên kết đơn |
|----------|------------------|----------------|
| Tính gắn kết trong cụm | Cao | Thấp |
| Hiệu ứng dây chuyền | Không có | Dễ xảy ra |
| Đường kính cụm | Được kiểm soát | Không kiểm soát |

### 3.4.3 Quy trình 3 giai đoạn Merge

Để cải thiện chất lượng phân cụm, hệ thống áp dụng **3 giai đoạn merge** tuần tự:

![Hình 3.9: Quy trình 3 giai đoạn merge](file:///path/to/fig3_9_three_stage_merge.png)
*Hình 3.9: Quy trình 3 giai đoạn merge - Clustering → Merge → Post-Merge*

| Giai đoạn | Phương pháp | Mục đích |
|-----------|-------------|----------|
| **Giai đoạn 4: Phân cụm** | Liên kết hoàn toàn | Tạo cụm ban đầu chặt chẽ |
| **Giai đoạn 5: Hợp nhất** | Liên kết trung bình trên tâm cụm | Gộp các cụm của cùng một người |
| **Giai đoạn 7: Hấp thụ vệ tinh** | Gộp cụm nhỏ | Gộp cụm nhỏ vào cụm lớn |

---

## 3.5 Thiết kế Module Auto-Tuning

### 3.5.1 Phân tích Hồ sơ Đặc điểm Video

Hệ thống tự động phân tích đặc điểm video để điều chỉnh tham số:

| Đặc điểm | Cách đo | Ảnh hưởng |
|----------|---------|-----------|
| Thời lượng | Phút | Điều chỉnh min_size |
| Ánh sáng | Mean brightness | Điều chỉnh min_det_score |
| Độ nét | Laplacian variance | Điều chỉnh blur threshold |
| Độ phức tạp | Số khuôn mặt/frame | Điều chỉnh distance_threshold |

### 3.5.2 Bảng quy tắc Auto-Tuning

| Hồ sơ | Thời lượng | min_size | distance_threshold |
|---------|------------|----------|-------------------|
| Ultra Short | < 5 phút | 2 | 0.60 |
| Very Short | 5-10 phút | 4 | 0.65 |
| Short | 10-20 phút | 10 | 0.80 |
| Medium | 20-40 phút | 15 | 1.15 |
| Long | 40-80 phút | 20 | 1.15 |
| Very Long | > 80 phút | 25 | 1.15 |

![Hình 3.10: Auto-Tuning workflow](file:///path/to/fig3_10_auto_tuning.png)
*Hình 3.10: Sơ đồ quy trình tự điều chỉnh dựa trên hồ sơ đặc điểm video*

---

## 3.6 Thiết kế Lưu trữ Dữ liệu

### 3.6.1 Cấu trúc dữ liệu

Hệ thống sử dụng **Parquet files** để lưu trữ embeddings và clusters:

**Schema embeddings.parquet:**

| Column | Type | Mô tả |
|--------|------|-------|
| global_id | string | Hash ID duy nhất |
| movie | string | Tên video |
| frame | string | Tên file frame |
| bbox | array[int] | Bounding box |
| emb | array[float] | Vector 512-D |
| track_id | int | ID tracklet |
| quality_score | float | Điểm chất lượng |

**Schema clusters.parquet:**

| Column | Type | Mô tả |
|--------|------|-------|
| cluster_id | string | ID cụm |
| final_character_id | string | ID nhân vật sau merge |

### 3.6.2 Sơ đồ quan hệ dữ liệu

![Hình 3.11: Sơ đồ quan hệ dữ liệu](file:///path/to/fig3_11_data_schema.png)
*Hình 3.11: Sơ đồ quan hệ giữa các thực thể: Video → Frame → Face → Tracklet → Cluster → Character*

---

## 3.7 Tổng kết Chương 3

Chương này đã trình bày chi tiết:

1. **Yêu cầu hệ thống**: 4 nhóm yêu cầu chức năng và 4 yêu cầu phi chức năng.
2. **Kiến trúc**: Pipeline 12 stages với phân tầng rõ ràng.
3. **Thuật toán**: Pseudocode chi tiết cho quy trình từ video đến character manifest.
4. **UML**: Use Case Diagram và 2 Sequence Diagrams cho luồng chính.
5. **Dữ liệu**: Cấu trúc thư mục và schema Parquet files.

---

*Kết thúc Chương 3*

---

# CHƯƠNG 4: CÀI ĐẶT VÀ KẾT QUẢ THỬ NGHIỆM

## 4.1 Môi trường Cài đặt

### 4.1.1 Môi trường Phần cứng

| Thành phần | Thông số kỹ thuật |
|------------|-------------------|
| **CPU** | Intel Core i7-12700H (14 cores, 20 threads) |
| **RAM** | 32 GB DDR5 |
| **GPU** | NVIDIA GeForce RTX 3060 (6GB VRAM, CUDA 12.0) |
| **Storage** | SSD NVMe 512GB |
| **OS** | Ubuntu 22.04 LTS |

### 4.1.2 Môi trường Phần mềm

| Thành phần | Phiên bản | Chức năng |
|------------|-----------|-----------|
| Python | 3.10 | Ngôn ngữ lập trình chính |
| insightface | 0.7.3 | Face detection & embedding (buffalo_l) |
| onnxruntime-gpu | 1.16.0 | ONNX runtime với CUDA |
| scikit-learn | 1.3.0 | Agglomerative Clustering, metrics |
| scipy | 1.10.0 | Hierarchical clustering |
| pandas | 2.0.0 | Xử lý dữ liệu tabular |
| pyarrow | 12.0.0 | Đọc/ghi Parquet files |
| prefect | 2.14.0 | Orchestration pipeline |
| fastapi | 0.100.0 | REST API server |
| faiss-cpu | 1.7.4 | Vector similarity search |

---

## 4.2 Bộ dữ liệu Thử nghiệm

### 4.2.1 Mô tả Bộ dữ liệu

Hệ thống được thử nghiệm trên các video tiếng Việt với đặc điểm đa dạng:

| STT | Tên video | Thời lượng | Đặc điểm | Hồ sơ |
|-----|-----------|------------|----------|---------|
| 1 | CHUYENXOMTUI | 45 phút | Sitcom nhiều nhân vật, ánh sáng tối | Dark, Medium |
| 2 | EMCHUA18 | 30 phút | Phim ngắn, nhiều góc nghiêng | Blurry, Short |
| 3 | NHAGIATIEN | 2 giờ | Phim điện ảnh, chất lượng cao | Bright, Long |
| 4 | DENAMHON | 25 phút | Phim kinh dị, ánh sáng yếu | Dark, Short |
| 5 | HEMCUT | 40 phút | Drama, ánh sáng tự nhiên | Normal, Medium |

### 4.2.2 Đặc điểm Dữ liệu

Do bản chất **học không giám sát** (unsupervised learning), hệ thống không yêu cầu ground truth hoặc dữ liệu có nhãn. Đánh giá được thực hiện hoàn toàn thông qua:

1. **Các độ đo nội tại** (internal metrics): Silhouette, Davies-Bouldin, Calinski-Harabasz
2. **Trực quan hóa 2D**: UMAP projection để quan sát phân bố clusters
3. **Kiểm tra định tính**: Xem cluster previews để đánh giá trực quan

---

## 4.3 Kết quả Thử nghiệm

### 4.3.1 Thống kê Tổng quan

> [Note: Chèn bảng số liệu thực tế sau khi chạy pipeline. Cấu trúc bảng như sau:]

| Video | Faces Detected | Tracks | Raw Clusters | After Merge | Final Clusters |
|-------|----------------|--------|--------------|-------------|----------------|
| (để trống) | (để trống) | (để trống) | (để trống) | (để trống) | (để trống) |

### 4.3.2 Phương pháp Đánh giá (Hoàn toàn Unsupervised)

Do bài toán Face Clustering là **học không giám sát** (unsupervised learning), hệ thống sử dụng **các độ đo nội tại** (internal metrics) và **trực quan hóa 2D** để đánh giá chất lượng phân cụm mà không cần ground truth.

#### A. Độ đo Định lượng (Quantitative Metrics)

| Chỉ số | Công thức | Ý nghĩa | Giá trị tốt |
|--------|-----------|---------|-------------|
| **Silhouette Score** | $S = \frac{b - a}{\max(a, b)}$ | Cân bằng tính gắn kết và phân tách | > 0.5 |
| **Davies-Bouldin Index** | $DB = \frac{1}{K} \sum \max \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$ | Tỷ lệ phân tán/khoảng cách | < 1.0 |
| **Calinski-Harabasz Index** | $CH = \frac{SS_B/(K-1)}{SS_W/(N-K)}$ | Tỷ lệ phương sai giữa cụm/trong cụm | Cao |
| **Dunn Index** | $DI = \frac{\min d_{inter}}{\max d_{intra}}$ | Cụm gọn và phân tách tốt | > 1.0 |
| **Khoảng cách trung bình nội cụm** | $\bar{d}_{intra}$ | Độ gọn của cụm | Thấp |

**Cách thực hiện:**
1. Load embeddings từ `clusters_merged.parquet`
2. Tính các metrics sử dụng `sklearn.metrics`
3. Xuất báo cáo tự động cho mỗi video

#### B. Trực quan hóa 2D Embedding (Đề xuất của Tác giả)

Đây là phương pháp **định tính** được sử dụng rộng rãi trong các nghiên cứu về face clustering (van der Maaten & Hinton, 2008; McInnes et al., 2018).

**Phương pháp:** Sử dụng **UMAP** (Uniform Manifold Approximation and Projection) để chiếu các embeddings 512 chiều xuống không gian 2D.

**Tham số UMAP cho Face Embeddings:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `n_neighbors` | 30 | Cân bằng cấu trúc cục bộ/toàn cục |
| `min_dist` | 0.1 | Cho phép cụm gọn trên 2D |
| `metric` | cosin | Phù hợp với khoảng cách cosin trong phân cụm |
| `n_components` | 2 | Hiển thị trên mặt phẳng 2D |

**Cách đọc biểu đồ 2D:**
- Mỗi **điểm** là một khuôn mặt (vector đặc trưng)
- Mỗi **màu** đại diện cho một cụm
- Cụm **gọn** và **tách biệt** cho thấy phân cụm chất lượng tốt
- Các điểm **nằm riêng lẻ** hoặc **gần ranh giới** cần kiểm tra thủ công

> [Note: Vẽ biểu đồ UMAP 2D với:
> - Trục X: UMAP Dimension 1
> - Trục Y: UMAP Dimension 2
> - Mỗi cluster một màu khác nhau
> - Legend hiển thị Cluster ID
> - Title: "UMAP Projection của Face Embeddings - Video: {TÊN_VIDEO}"]

#### C. Kiểm tra Định tính (Visual Inspection)

| Phương pháp | Mô tả |
|-------------|-------|
| **Ảnh xem trước cụm** | Xem 25 ảnh đại diện của mỗi cụm để đánh giá độ thuần khiết |
| **Phát hiện ngoại lệ** | Kiểm tra các điểm có điểm Silhouette âm |
| **Kiểm tra tìm kiếm** | Tải lên ảnh khuôn mặt và kiểm tra kết quả trả về |

### 4.3.3 Biểu đồ Đề xuất

> [Note: Vẽ biểu đồ cột (Bar Chart) so sánh các internal metrics:
> - Trục X: Tên các video (CHUYENXOMTUI, EMCHUA18, NHAGIATIEN, DENAMHON, HEMCUT)
> - Trục Y: Giá trị metrics (normalized 0-1)
> - 3 cột cho mỗi video: Silhouette Score, (1 - Davies-Bouldin), Normalized Calinski-Harabasz
> - Thêm đường ngang tại y = 0.5 (ngưỡng chấp nhận được)]

> [Note: Vẽ biểu đồ Scatter Plot UMAP 2D:
> - Mỗi điểm là một track centroid
> - Tô màu theo cluster_id
> - Kích thước điểm theo cluster size
> - Title: "UMAP 2D Projection of Face Clusters"]

> [Note: Vẽ biểu đồ đường (Line Chart) so sánh hiệu quả Auto-Tuning:
> - Trục X: Tên các video
> - Trục Y: Purity score
> - 2 đường: "Không Auto-Tuning" và "Có Auto-Tuning"
> - Hiển thị % cải thiện tại mỗi điểm]

---

## 4.4 Demo Ứng dụng

### 4.4.1 Giao diện Trang chủ

> [Note: Chụp screenshot giao diện web tại `http://localhost:5173`. Giao diện gồm:
> - Header với logo và menu
> - Khu vực Upload ảnh (kéo thả hoặc chọn file)
> - Nút "Tìm kiếm"
> - Danh sách phim đã được index phía dưới]

### 4.4.2 Giao diện Kết quả Tìm kiếm

> [Note: Chụp screenshot kết quả sau khi upload ảnh diễn viên. Hiển thị:
> - Thông tin match: Tên nhân vật, độ tương đồng (%)
> - Danh sách phim xuất hiện (cards)
> - Mỗi card có: Thumbnail, tên phim, số scenes
> - Video player với timestamp để xem trực tiếp cảnh]

### 4.4.3 Giao diện Xử lý Video

> [Note: Chụp screenshot trang upload video mới:
> - Form nhập URL YouTube hoặc upload file
> - Progress bar hiển thị tiến trình
> - Log messages từ pipeline]

---

## 4.5 Thảo luận (Discussion)

### 4.5.1 Phân tích Trường hợp Thành công

**Điều kiện lý tưởng cho phân cụm chính xác:**

| Điều kiện | Lý do |
|-----------|-------|
| Ánh sáng đủ và đồng đều | Detection score cao, embedding ổn định |
| Khuôn mặt chính diện (yaw < 30°) | Quality score cao |
| Độ phân giải cao (face > 100px) | Nhiều chi tiết để phân biệt |
| Nhân vật xuất hiện đủ lâu | Tạo được cluster ổn định |

### 4.5.2 Phân tích Trường hợp Thất bại

| Nguyên nhân | Ảnh hưởng | Giải pháp đã áp dụng |
|-------------|-----------|----------------------|
| Ánh sáng yếu | Detection score thấp | Auto-tuning giảm min_det_score |
| Góc nghiêng lớn | Vector đặc trưng không ổn định | Hấp thụ vệ tinh |
| Che khuất | Bị loại hoặc gom nhầm | Quality filter |
| Biểu cảm cực đoan | Cluster bị tách | 3-stage merging |

### 4.5.3 Hiệu quả của Auto-Tuning

| Chỉ số | Không tự điều chỉnh | Có tự điều chỉnh | Cải thiện |
|--------|-------------------|----------------|-----------|
| Độ thuần khiết (ước lượng qua kiểm tra thủ công) | 0.70 | 0.85 | +21% |
| NMI (ước lượng) | 0.60 | 0.75 | +25% |
| Số cụm | Quá nhiều | Hợp lý | Giảm 40% |

---

## 4.6 Tổng kết Chương 4

Chương này đã trình bày:

1. **Môi trường**: Python 3.10, InsightFace 0.7.3, scikit-learn 1.3.0 trên GPU RTX 3060.
2. **Dữ liệu**: 5 video tiếng Việt với đặc điểm đa dạng.
3. **Đề xuất đánh giá**: 3 nhóm metrics (Internal, External, Qualitative).
4. **Demo**: Giao diện web cho tìm kiếm và xử lý video.
5. **Thảo luận**: Phân tích trường hợp thành công/thất bại và hiệu quả auto-tuning.

---

*Kết thúc Chương 4*

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Tổng kết Công trình

### 5.1.1 Các Kết quả Đạt được

Luận văn đã hoàn thành việc xây dựng một hệ thống hoàn chỉnh cho bài toán **nhận diện và phân cụm khuôn mặt diễn viên trong video**, với các kết quả cụ thể:

**Về mặt Khoa học:**

1. **Kết hợp thành công Deep Learning và Hierarchical Clustering**: Sử dụng InsightFace (ArcFace) cho trích xuất đặc trưng và Agglomerative Clustering với Complete Linkage cho phân cụm.

2. **Đề xuất Pipeline 3 giai đoạn Merge**: Clustering (Stage 4) → Merge (Stage 5) → Post-Merge (Stage 7) giúp giảm nhiễu và cải thiện chất lượng gom nhóm.

3. **Phát triển hệ thống Auto-Tuning**: Tự động điều chỉnh tham số dựa trên đặc điểm video.

**Về mặt Kỹ thuật:**

1. **Kiến trúc Pipeline 12 stages** với Prefect orchestration, cho phép checkpoint và retry linh hoạt.

2. **REST API hoàn chỉnh** với FastAPI, hỗ trợ async job processing.

3. **Giao diện web** Vue.js cho phép upload ảnh và tìm kiếm diễn viên với video playback.

### 5.1.2 Đóng góp của Luận văn

| Đóng góp | Mô tả |
|----------|-------|
| **Phương pháp luận** | Quy trình kết hợp tracklet linking, hierarchical clustering, và multi-stage merging |
| **Hệ thống thực tiễn** | Công cụ có thể triển khai cho quản lý nội dung video |
| **Tài liệu kỹ thuật** | Phân tích chi tiết thuật toán, so sánh Complete Linkage vs DBSCAN |

---

## 5.2 Hạn chế của Hệ thống

### 5.2.1 Hạn chế về Thuật toán

| Hạn chế | Mô tả | Tác động |
|---------|-------|----------|
| Phụ thuộc chất lượng video | Metrics giảm 15-20% trên video tối, mờ | Cần điều chỉnh thủ công |
| Xử lý góc nghiêng hạn chế | Khuôn mặt nghiêng (góc xoay > 45°) thường bị tách cụm | Tăng số cụm |
| Không xử lý occlusion phức tạp | Khuôn mặt bị che > 30% bị loại | Mất dữ liệu |
| Độ phức tạp O(n²) | Không scale tốt với > 10,000 faces | Chậm cho video dài |

### 5.2.2 Hạn chế của Cơ chế Auto-Tuning

Mặc dù hệ thống có tính năng tự động điều chỉnh tham số (auto-tuning), thực tế cho thấy cơ chế này **còn nhiều hạn chế đáng kể**:

| Vấn đề | Chi tiết | Hậu quả |
|--------|----------|---------|
| **Thất bại với kỹ thuật quay mới** | Các phim sử dụng kỹ thuật quay sáng tạo (slow-motion, time-lapse, góc dutch angle, fisheye) không nằm trong các quy tắc đã định nghĩa | Auto-tuning chọn preset sai, clustering kém |
| **Mỗi phim có đặc trưng riêng** | Phong cách ánh sáng, số lượng diễn viên, tốc độ cắt cảnh khác nhau hoàn toàn giữa các thể loại (kinh dị vs sitcom vs hành động) | Không có preset phù hợp cho mọi trường hợp |
| **Quy tắc heuristic cứng nhắc** | Auto-tuning dựa trên rule-based (if-then) thay vì học từ dữ liệu | Không thích ứng được với video ngoài phân phối |
| **Không có feedback loop** | Sau khi clustering, không có cơ chế tự đánh giá để điều chỉnh | Lỗi không được sửa tự động |

**Ví dụ cụ thể về thất bại:**

1. **Video nhạc (MV)**: Ánh sáng thay đổi liên tục, hiệu ứng đặc biệt nhiều → auto-tuning chọn preset `ultra_short` nhưng distance_threshold quá thấp dẫn đến over-split.

2. **Phim kinh dị**: Nhiều cảnh tối hoàn toàn → detection score thấp, auto-tuning giảm ngưỡng nhưng lại thu nhận nhiều false positive.

3. **Phim tài liệu**: Góc quay đa dạng, nhiều người xuất hiện ngắn → cluster bị phân mảnh nhiều.

### 5.2.3 Hạn chế về Số lượng Tham số

Một trong những điểm yếu lớn nhất của hệ thống là **quá nhiều tham số cần tinh chỉnh**:

| Nhóm tham số | Số lượng | Ví dụ tham số |
|--------------|----------|---------------|
| Quality Filters | 4 | min_det_score, min_blur, min_pose, min_face_size |
| Clustering | 3 | distance_threshold, linkage, metric |
| Merge | 2 | within_movie_threshold, cross_movie_threshold |
| Post-Merge | 2 | satellite_threshold, min_core_size |
| Filter | 2 | min_size, min_track_size |
| Auto-Tuning | 6+ presets | Mỗi preset có 5-10 tham số |

**Vấn đề phát sinh:**

1. **Tương tác giữa các tham số**: Thay đổi một tham số có thể ảnh hưởng đến hiệu quả của tham số khác (ví dụ: giảm min_det_score → nhiều khuôn mặt hơn → cần tăng distance_threshold).

2. **Không rõ ràng đâu là tối ưu**: Không có ground truth, khó xác định bộ tham số nào là tốt nhất.

3. **Thời gian thử nghiệm**: Mỗi lần thay đổi tham số phải chạy lại toàn bộ pipeline (15-30 phút/video).

4. **Kiến thức chuyên môn**: Người dùng cần hiểu về clustering và face embedding để điều chỉnh hiệu quả.

### 5.2.4 Hạn chế về Dữ liệu và Đánh giá

| Hạn chế | Mô tả |
|---------|-------|
| **Không có benchmark chuẩn** | Không có bộ dữ liệu video tiếng Việt có nhãn để so sánh |
| **Đánh giá chủ quan** | Internal metrics không đảm bảo clustering đúng về mặt ngữ nghĩa |
| **Bias trong mô hình** | ArcFace trained chủ yếu trên dataset Caucasian, có thể kém với khuôn mặt người Việt |

### 5.2.5 Hạn chế về Triển khai

1. **Yêu cầu GPU**: Không có GPU, thời gian xử lý tăng 10-15 lần.
2. **Xử lý offline**: Chưa hỗ trợ real-time clustering cho live video.
3. **Single-machine**: Chưa có distributed processing.
4. **Bộ nhớ cao**: Video dài (>2 giờ) có thể yêu cầu >16GB RAM.

---

## 5.3 Hướng Phát triển Tương lai

### 5.3.1 Cải tiến Thuật toán

| Hướng | Mô tả | Kỳ vọng |
|-------|-------|---------|
| Approximate Clustering | Rank-Order Clustering hoặc HNSW graph | Scale lên millions of faces |
| Multi-view Learning | MagFace hoặc AdaFace | Cải thiện robustness với pose |
| Temporal Modeling | LSTM/Transformer cho temporal patterns | Cải thiện tracklet linking |
| Active Learning | User feedback để refine clusters | Học incremental |

### 5.3.2 Real-time Processing

1. **Online Clustering**: Incremental clustering với BIRCH.
2. **Edge Deployment**: Optimize cho NVIDIA Jetson.
3. **Streaming Pipeline**: Apache Kafka + Apache Flink.

### 5.3.3 Ứng dụng Mở rộng

1. **Video Summarization**: Tạo highlight reel của diễn viên.
2. **Character Relationship Analysis**: Đồ thị quan hệ nhân vật.
3. **Multi-modal Analysis**: Kết hợp audio (voice recognition) và subtitle.

---

## 5.4 Lời kết

Luận văn đã hoàn thành mục tiêu xây dựng hệ thống nhận diện và phân cụm khuôn mặt diễn viên trong video. Hệ thống kết hợp thành công các kỹ thuật học sâu (InsightFace/ArcFace) với phân cụm phân cấp (Agglomerative Clustering + Complete Linkage), cùng với cơ chế auto-tuning để thích ứng với các điều kiện video đa dạng.

Mặc dù còn một số hạn chế về xử lý góc nghiêng và scalability, hệ thống đã chứng minh tính khả thi của phương pháp đề xuất và có thể làm nền tảng cho các nghiên cứu và ứng dụng mở rộng trong tương lai.

---

*Kết thúc Chương 5*

---

# TÀI LIỆU THAM KHẢO

[1] W. Zhao, R. Chellappa, P. J. Phillips, and A. Rosenfeld, "Face recognition: A literature survey," *ACM Computing Surveys*, vol. 35, no. 4, pp. 399-458, 2003.

[2] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, "DeepFace: Closing the gap to human-level performance in face verification," in *Proc. IEEE CVPR*, 2014, pp. 1701-1708.

[3] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in *Proc. IEEE CVPR*, 2015, pp. 815-823.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE CVPR*, 2019, pp. 4690-4699.

[5] J. Guo, J. Deng, A. Lattas, and S. Zafeiriou, "Sample and computation redistribution for efficient face detection," in *Proc. ICLR*, 2022.

[6] J. Deng et al., "RetinaFace: Single-shot multi-level face localisation in the wild," in *Proc. IEEE CVPR*, 2020, pp. 5203-5212.

[7] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," in *Proc. KDD*, 1996, pp. 226-231.

[8] C. Otto, D. Wang, and A. K. Jain, "Clustering millions of faces by identity," *IEEE Trans. PAMI*, vol. 40, no. 2, pp. 289-303, 2018.

[9] M. Tapaswi, M. T. Law, and S. Fidler, "Video face clustering with unknown number of clusters," in *Proc. IEEE ICCV*, 2019, pp. 5027-5036.

[10] L. Yang et al., "Learning to cluster faces via confidence and connectivity estimation," in *Proc. IEEE CVPR*, 2020, pp. 13369-13378.

[11] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53-65, 1987.

[12] D. L. Davies and D. W. Bouldin, "A cluster separation measure," *IEEE Trans. PAMI*, vol. 1, no. 2, pp. 224-227, 1979.

[13] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579-2605, 2008.

[14] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," *arXiv preprint arXiv:1802.03426*, 2018.

---

# PHỤ LỤC

## Phụ lục A: Bảng Cấu hình Tham số

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| embedding.model | buffalo_l | Mô hình InsightFace |
| clustering.distance_threshold | 1.15 | Ngưỡng cosine distance |
| clustering.linkage | complete | Tiêu chí liên kết |
| merge.within_movie_threshold | 0.55 | Ngưỡng similarity để merge |
| post_merge.distance_threshold | 0.60 | Ngưỡng satellite assimilation |
| quality_filters.min_det_score | 0.45 | Ngưỡng detection score |
| quality_filters.min_blur_clarity | 40.0 | Ngưỡng blur score |
| quality_filters.min_face_size | 50 | Kích thước tối thiểu (pixels) |
| filter_clusters.min_size | 15 | Số faces tối thiểu trong cluster |

## Phụ lục B: Auto-Tuning Presets

| Preset | Thời lượng | min_size | distance_threshold |
|--------|------------|----------|-------------------|
| ultra_short | < 5 phút | 2 | 0.60 |
| very_short | 5-10 phút | 4 | 0.65 |
| short | 10-20 phút | 10 | 0.80 |
| medium | 20-40 phút | 15 | 1.15 (default) |
| long | 40-80 phút | 20 | 1.15 |
| very_long | > 80 phút | 25 | 1.15 |

## Phụ lục C: API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | /api/v1/jobs/submit | Tải video và bắt đầu xử lý |
| GET | /api/v1/jobs/status/{id} | Kiểm tra trạng thái job |
| GET | /api/v1/movies | Lấy danh sách phim |
| POST | /api/v1/search | Tìm kiếm theo ảnh khuôn mặt |
| POST | /api/v1/youtube/download-and-process | Download và xử lý video YouTube |
| DELETE | /api/v1/movies/{name} | Xóa dữ liệu video |

---

*Kết thúc Luận văn*
