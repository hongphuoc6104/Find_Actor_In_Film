import streamlit as st
import requests

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="FaceLook Cinema", page_icon="🎬", layout="wide")

# API Configuration
BACKEND_BASE_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{BACKEND_BASE_URL}/api/v1/search"
MOVIES_ENDPOINT = f"{BACKEND_BASE_URL}/api/v1/movies"

# CSS Custom: Tối giản hóa thẻ phim
st.markdown("""
<style>
    div.stButton > button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    .char-card { background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #ff4b4b; }

    /* Style cho ô phim trong kho */
    .movie-box-simple { 
        padding: 5px; 
        border: 1px solid #333; 
        border-radius: 5px; 
        text-align: left; 
        background-color: #1e1e1e;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .movie-title-simple { font-weight: bold; color: #fff; font-size: 1em; }
    .movie-duration { font-size: 0.9em; color: #ff4b4b; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --- 2. KHỞI TẠO STATE & HÀM HỖ TRỢ ---

if 'search_response' not in st.session_state:
    st.session_state.search_response = None

# State lưu danh sách phim để không phải gọi API liên tục
if 'movie_library' not in st.session_state:
    st.session_state.movie_library = []
if 'is_library_loaded' not in st.session_state:
    st.session_state.is_library_loaded = False


def reset_search_state():
    """Xóa kết quả tìm kiếm cũ khi đổi ảnh"""
    st.session_state.search_response = None


def load_movies_automatically():
    """Hàm tự động gọi API lấy danh sách phim khi khởi động"""
    # Chỉ gọi nếu chưa tải lần nào
    if not st.session_state.is_library_loaded:
        try:
            response = requests.get(MOVIES_ENDPOINT)
            if response.status_code == 200:
                data = response.json()
                st.session_state.movie_library = data.get("movies", [])
                st.session_state.is_library_loaded = True  # Đánh dấu đã tải
            else:
                st.error(f"Không thể tải danh sách phim. Server code: {response.status_code}")
        except Exception as e:
            st.error(f"Lỗi kết nối Backend: {e}")


# --- 3. GỌI HÀM AUTO-LOAD NGAY KHI APP CHẠY ---
load_movies_automatically()

st.title("🎬 FaceLook Cinema")

# --- 4. TẠO TABS ---
tab_search, tab_gallery = st.tabs(["🔍 TÌM KIẾM DIỄN VIÊN", "📂 KHO PHIM ĐANG CÓ"])

# ================= TAB 1: TÌM KIẾM =================
with tab_search:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.header("📸 Upload ảnh")
        uploaded_file = st.file_uploader(
            "Chọn ảnh khuôn mặt...",
            type=['jpg', 'png', 'jpeg'],
            key="search_uploader",
            on_change=reset_search_state
        )

        if st.button("🚀 QUÉT AI (CẬP NHẬT)"):
            if uploaded_file:
                st.session_state.search_response = None
                with st.spinner("Đang phân tích..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file}
                        response = requests.post(SEARCH_ENDPOINT, files=files)

                        if response.status_code == 200:
                            st.session_state.search_response = response.json()
                            matches = st.session_state.search_response.get("matches", [])
                            if not matches:
                                st.warning("✅ Đã quét xong nhưng KHÔNG TÌM THẤY diễn viên.")
                            else:
                                st.success(f"✅ Tìm thấy trong {len(matches)} phim!")
                        else:
                            st.error(f"Lỗi Server: {response.text}")
                    except Exception as e:
                        st.error(f"Lỗi kết nối: {e}")
            else:
                st.warning("Vui lòng chọn ảnh trước.")

    with col_right:
        st.header("🍿 Kết quả phân tích")
        data = st.session_state.search_response

        if data:
            matches = data.get("matches", [])
            if not matches:
                st.info("🤷‍♂️ Không tìm thấy kết quả phù hợp.")
            else:
                for m_idx, match in enumerate(matches):
                    movie_name = match.get("movie", "Unknown Movie")
                    raw_video_url = match.get("video_url", "")
                    final_video_url = raw_video_url if raw_video_url.startswith(
                        "http") else f"{BACKEND_BASE_URL}{raw_video_url}"

                    with st.container(border=True):
                        st.subheader(f"🎞️ {movie_name}")
                        characters = match.get("characters", [])
                        for c_idx, char in enumerate(characters):
                            char_name = char.get("name", "Unknown")
                            score_display = char.get("score_display", "N/A")
                            scenes = char.get("scenes", [])

                            st.markdown(
                                f'''<div class="char-card">Nhân vật: <b>{char_name}</b> | Độ tin cậy: <span style="color:#4CAF50">{score_display}</span></div>''',
                                unsafe_allow_html=True)

                            if scenes:
                                scene_map = {}
                                for s_idx, s in enumerate(scenes):
                                    start = s.get("start_time", 0)
                                    end = s.get("end_time", 0)
                                    label = f"Cảnh {s_idx + 1}: {start}s - {end}s"
                                    scene_map[label] = start

                                unique_key = f"sel_{m_idx}_{c_idx}"
                                selected_label = st.selectbox(f"Chọn đoạn ({char_name}):",
                                                              options=list(scene_map.keys()), key=unique_key)
                                seek_time = scene_map[selected_label]
                                st.video(final_video_url, start_time=int(seek_time))
                            else:
                                st.warning("Không có phân đoạn cụ thể.")
                                st.video(final_video_url)
        else:
            if uploaded_file:
                st.info("👈 Bấm 'QUÉT AI' để xem kết quả.")
            else:
                st.info("👈 Upload ảnh bên trái để bắt đầu.")

# ================= TAB 2: KHO PHIM (AUTO LOAD) =================
with tab_gallery:
    col_head, col_reload = st.columns([4, 1])
    with col_head:
        st.header("📂 Danh sách phim")
    with col_reload:
        # Nút reload nhỏ gọn nếu muốn tải lại thủ công
        if st.button("🔄 Refresh"):
            st.session_state.is_library_loaded = False
            st.rerun()

    # Lấy danh sách từ state (đã được auto-load ở đầu file)
    movie_list = st.session_state.movie_library

    if movie_list:
        st.success(f"Đang hiển thị {len(movie_list)} phim.")

        # Hiển thị Grid 3 cột
        cols = st.columns(3)
        for idx, movie in enumerate(movie_list):
            name = movie.get("movie_name", "Unknown")

            # [LOGIC] Thời lượng: Backend cần trả về field 'duration',
            # nếu chưa có thì tạm hiển thị Placeholder hoặc size.
            duration = movie.get("duration", "N/A")

            raw_url = movie.get("video_url", "")
            full_url = raw_url if raw_url.startswith("http") else f"{BACKEND_BASE_URL}{raw_url}"

            with cols[idx % 3]:
                # Giao diện thẻ phim đơn giản: Tên + Thời lượng
                st.markdown(f"""
                <div class="movie-box-simple">
                    <span class="movie-title-simple">🎬 {name}</span>
                    <span class="movie-duration">⏱️ {duration}</span>
                </div>
                """, unsafe_allow_html=True)

                # Player
                st.video(full_url)
    else:
        # Trường hợp mới mở app, đang load hoặc lỗi
        if not st.session_state.is_library_loaded:
            st.spinner("Đang kết nối kho phim...")
        else:
            st.info("Kho phim hiện tại đang trống hoặc không kết nối được.")