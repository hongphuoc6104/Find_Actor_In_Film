# test_recognition_flow.py
import argparse
import sys
import time
from typing import Optional

# Import hàm nhận diện chính từ lớp service của bạn
from services.recognition import recognize


def format_time(seconds: Optional[float]) -> str:
    """Chuyển đổi số giây thành định dạng MM:SS.ms cho dễ đọc."""
    if seconds is None:
        return "N/A"
    try:
        sec = float(seconds)
        minutes = int(sec // 60)
        remaining_seconds = sec % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"
    except (ValueError, TypeError):
        return "Invalid Time"


def main():
    """
    Hàm chính để thực thi luồng test nhận diện.
    """
    parser = argparse.ArgumentParser(
        description="""
        Script test nhanh luồng nhận diện hoàn chỉnh.
        Nhận vào một ảnh, trả về các phim và cảnh mà diễn viên đó xuất hiện.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn đến file ảnh của diễn viên cần nhận diện."
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f" Bắt đầu nhận diện cho ảnh: {args.image}")
    print("=" * 70)

    start_time = time.time()

    # Gọi hàm nhận diện chính từ service của bạn
    # Hàm này sẽ tự động xử lý việc tìm kiếm, lọc và định dạng kết quả
    results = recognize(image_path=args.image)

    end_time = time.time()
    print(f"\n Nhận diện hoàn tất sau {end_time - start_time:.2f} giây.\n")

    # --- Phân tích và in kết quả ---

    if not results or results.get("is_unknown"):
        print(" Không tìm thấy nhân vật nào phù hợp trong kho dữ liệu.")
        sys.exit(0)

    # Lặp qua từng phim có kết quả
    for movie_info in results.get("movies", []):
        movie_title = movie_info.get("movie", "Không rõ tên phim")
        print(f"🎬 Phim: {movie_title}")
        print("-" * 50)

        # Lặp qua từng nhân vật tìm thấy trong phim đó
        for char_info in movie_info.get("characters", []):
            char_id = char_info.get("character_id", "N/A")
            # --- CẬP NHẬT HIỂN THỊ TÊN ---
            name = char_info.get("name")

            # Nếu có tên thì hiển thị tên, nếu không thì hiển thị "Chưa gán nhãn"
            display_name = name if name else "Chưa gán nhãn"

            score = char_info.get("score", 0.0)
            match_label = char_info.get("match_label", "")

            # Hiển thị Tên trước, ID sau
            print(f"   Nhân vật: {display_name} (ID: {char_id})")
            print(f"     - Điểm tương đồng: {score:.4f} ({match_label})")

            # Lấy và in thông tin các cảnh xuất hiện
            scenes = char_info.get("scenes", [])
            if not scenes:
                print("     -  Không có thông tin về cảnh xuất hiện.")
            else:
                print("     -  Các cảnh xuất hiện:")
                for i, scene in enumerate(scenes):
                    start = scene.get("start_time")
                    end = scene.get("end_time")
                    print(f"       - Cảnh {i + 1}: Từ {format_time(start)} đến {format_time(end)}")
        print("\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[LỖI] Không tìm thấy file dữ liệu cần thiết: {e}", file=sys.stderr)
        print("Vui lòng đảm bảo bạn đã chạy pipeline thành công và các file trong 'warehouse/' đã được tạo.",
              file=sys.stderr)
    except Exception as e:
        print(f"\n[LỖI] Một lỗi không mong muốn đã xảy ra: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()