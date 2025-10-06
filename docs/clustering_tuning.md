# Gợi ý tinh chỉnh gom cụm nhân vật

Tài liệu này tổng hợp các bước giúp làm sạch đầu vào, điều chỉnh tham số và tự đánh giá chất lượng gom cụm trong hệ thống hiện tại.

## 1. Làm sạch dữ liệu trước khi gom cụm
- Giữ lại các track đủ dài và cụm đủ lớn bằng cách chỉnh `filter.min_track_size` và `filter.min_cluster_size`. Các tham số này loại bỏ nhiễu do track ngắn hoặc cụm chỉ có vài khung hình.【F:configs/config.yaml†L53-L62】【F:tasks/cluster_task.py†L12-L36】
- Bật `quality_filters` để bỏ khung hình quá tối, mờ hoặc khuôn mặt quá nhỏ. Có thể tăng `min_det_score`, `min_face_ratio` hoặc ngưỡng độ nét/độ sáng khi dữ liệu nhiều nhiễu.【F:configs/config.yaml†L63-L79】

## 2. Tối ưu bước gom cụm theo từng phim
- Thuật toán mặc định là Agglomerative với `metric` và `linkage` có thể chỉnh trong `clustering`. Thử `linkage=average` hoặc `metric=cosine` nếu embeddings bị co cụm theo phương nhất định.【F:configs/config.yaml†L32-L45】【F:tasks/cluster_task.py†L54-L90】
- `auto_distance_percentile` cho phép suy ra ngưỡng từ phân bố khoảng cách thực tế. Khi dữ liệu giữa các phim khác nhau, có thể bật tùy chọn này thay vì cố định `distance_threshold`.【F:configs/config.yaml†L36-L41】【F:tasks/cluster_task.py†L68-L75】
- Với các phim khó, cấu hình `distance_threshold.per_movie` để đặt ngưỡng riêng cho từng `movie_id`. Điều này hữu ích khi có phim ít nhân vật (ngưỡng cao) và phim đông nhân vật (ngưỡng thấp).【F:configs/config.yaml†L36-L45】【F:tasks/cluster_task.py†L76-L88】
- Nếu muốn loại bỏ nhiều nhiễu hơn, bật `algo: auto` hoặc `hdbscan` để pipeline tự so sánh silhouette-score giữa Agglomerative và HDBSCAN, đồng thời loại track bị xem là outlier.【F:configs/config.yaml†L32-L37】【F:tasks/cluster_task.py†L91-L134】

## 3. Hợp nhất nhân vật toàn cục
- Sau khi gom cụm trong từng phim, `merge_clusters_task` kết nối các cụm có vector tâm đủ giống nhau. Chỉnh `merge.global_threshold` (cosine similarity) và `merge.knn` để kiểm soát mức độ gom. Ngưỡng càng cao thì ít trường hợp gộp nhầm nhưng dễ tách cùng một người thành nhiều cụm.【F:configs/config.yaml†L47-L52】【F:tasks/merge_clusters_task.py†L35-L118】
- `post_merge.distance_threshold` xử lý hậu kỳ trong từng phim sau bước merge. Giảm giá trị này nếu vẫn còn nhiều cụm trùng người; tăng lên để tránh gộp nhầm.【F:configs/config.yaml†L48-L52】

## 4. Đánh giá và tự dò tham số
- Chạy `prefect` flow hoặc riêng `validate_clusters_task` để sinh báo cáo `reports/cluster_metrics.csv`. File này chứa silhouette-score, kích thước cụm và tỷ lệ outlier giúp đánh giá độ chặt của từng cấu hình.【F:tasks/validation_task.py†L32-L83】
- Khi có nhãn tay cho một phần khung hình, dùng script `scripts/evaluate_clusters.py` để tính precision/recall/F1 giữa kết quả gom cụm và nhãn thật. Điều này cho phép so sánh các cấu hình khác nhau một cách định lượng.【F:scripts/evaluate_clusters.py†L1-L87】
- Để tự dò tham số, bạn có thể viết script lặp qua các cấu hình (ví dụ quét `distance_threshold` hoặc `merge.global_threshold`), chạy lại `cluster_task` và `merge_clusters_task`, rồi đánh giá bằng các báo cáo trên. Vì cấu hình đọc từ `configs/config.yaml`, có thể nhân bản file này theo từng biến thể và chạy pipeline theo từng bản cấu hình.

## 5. Một số mẹo bổ sung
- Dùng PCA (`pca.enable`) để giảm chiều và chuẩn hóa phân bố khoảng cách trước khi gom cụm, đặc biệt khi embeddings ban đầu có chiều cao.【F:configs/config.yaml†L24-L31】
- Nếu quan sát thấy cụm quá lớn gom nhiều người, xem lại bước chất lượng ảnh và cân nhắc tăng `filter_clusters.min_size` để loại bỏ cụm chỉ có vài track nhỏ (thường là nhiễu).【F:configs/config.yaml†L53-L62】
- Sau mỗi lần chỉnh tham số, kiểm tra các ảnh preview (`warehouse/cluster_previews`) để xác nhận trực quan các cụm quan trọng trước khi sử dụng kết quả.