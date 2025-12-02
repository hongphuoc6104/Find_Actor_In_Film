# ⚙️ AI BACKEND - STATUS LOG

## ℹ️ Server Info

- **Host:** localhost:8000
- **Docs:** /docs
- **Celery Broker:** Redis (db=0)

## 🔌 API Endpoints Status

| Method | Endpoint              | Status     | Mode     | Contract Valid? |
| ------ | --------------------- | ---------- | -------- | --------------- |
| POST   | `/api/v1/jobs/submit` | ✅ Ready   | Real     | N/A             |
| GET    | `/api/v1/jobs/{id}`   | ✅ Ready   | Real     | N/A             |
| GET    | `/api/v1/characters`  | ⚠️ Dev     | **MOCK** | ✅ Yes          |
| POST   | `/api/v1/search`      | ⏳ Pending | N/A      | ❌ No           |

## 📝 Next Tasks (Priority)

- [ ] Viết API `GET /api/v1/characters` trả về Mock Data.
- [ ] Implement `celery_worker` để gọi pipeline `flows/pipeline.py`.
- [ ] Chuyển đổi API Characters từ Mock sang đọc file `warehouse/characters.json`.
