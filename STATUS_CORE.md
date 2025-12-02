# 🧠 AI CORE - STATUS LOG

## ℹ️ Current State

- **Last Updated:** 2025-12-02
- **Engine Version:** v1.6 (Stable/Refactored)
- **Active Model:** Buffalo_L (InsightFace)

## 📊 Data Pipeline Status

| Stage           | Status   | Logic Health | Note                                                                 |
| :-------------- | :------- | :----------- | :------------------------------------------------------------------- |
| Ingestion       | ✅ Done  | 🟢 Good      | Đã fix dependencies (`requirements.txt`).                            |
| Tracklet        | ✅ Done  | 🟢 Good      | Logic nối track IoU/Cosine ổn định.                                  |
| Embedding       | ✅ Done  | 🟢 Good      | GPU Batch size: 32.                                                  |
| Clustering      | ✅ Done  | 🟢 Good      | Auto-tuning Silhouette Score active.                                 |
| Merging         | ✅ Done  | 🟢 Good      | Đã verify `utils.vector_utils` (l2_normalize).                       |
| Filter          | ✅ Done  | 🟢 Good      | Logic lọc dựa trên `high_quality_size`.                              |
| Post-Merge      | ✅ Done  | 🟢 Good      | Đã tối ưu hóa logic đồng bộ cột ID.                                  |
| Preview         | ✅ Done  | 🟢 Good      | Logic crop ảnh và fallback video ổn.                                 |
| Highlights      | ✅ Done  | 🟢 Good      | Ready for Backend integration.                                       |
| **Manifest** | ✅ Done  | 🟢 Good      | **[FIXED]** Đã thêm cơ chế bảo lưu nhãn (Label Persistence).         |
| **Labeling** | ✅ Done  | 🟢 Good      | **[FIXED]** Tối ưu hóa tính toán (dùng track_centroid có sẵn).       |
| Recognition API | ✅ Done  | 🟢 Good      | Đã verify `indexer` & `search_actor`.                                |

## 📝 Fixed Issues (Resolved)

- [x] **Dependency Crash:** Đã thêm `insightface`, `onnxruntime`, `opencv` vào `requirements.txt`.
- [x] **Data Loss Risk:** Sửa `character_task.py` để không overwrite nhãn cũ khi chạy lại pipeline.
- [x] **Broken Imports:** Đã xác nhận sự tồn tại của `vector_utils.py` và `indexer.py`.
- [x] **Redundant Compute:** Sửa `assign_labels_task.py` để dùng lại dữ liệu đã tính toán từ parquet.

## 🚀 Next Steps

1.  **Deployment:** Cài đặt môi trường theo `requirements.txt`.
2.  **Integration:** Backend có thể gọi `services.recognition.recognize()` an toàn.
3.  **Tuning:** Theo dõi độ chính xác của Auto-Labeling với ngưỡng `similarity_threshold: 0.55`.

## 🐛 Known Issues

- Hiện tại chưa phát hiện lỗi Critical nào. Hệ thống đã sẵn sàng cho Integration Test.