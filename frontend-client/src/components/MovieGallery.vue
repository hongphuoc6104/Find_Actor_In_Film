<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const API_BASE = "http://localhost:8000";
const movies = ref([]);
const isLoading = ref(false);

// --- [NEW] TỪ ĐIỂN TÊN PHIM (Đồng bộ với SearchSection) ---
const MOVIE_NAMES = {
    "HEMCUT": "Hẻm Cụt",
    "GAIGIALAMCHIEU": "Gái Già Lắm Chiêu",
    "NHAGIATIEN": "Nhà Gia Tiên",
    "BOGIA": "Bố Già",
    "CHUYENXOMTUI": "Chuyện Xóm Tui",
    "DENAMHON": "Đèn Âm Hồn",
    "EMCHUA18": "Em Chưa 18",
    "NGUOIVOCUOICUNG": "Người Vợ Cuối Cùng",
    "KEANDANH": "Kẻ Ẩn Danh",
    "SIEULAYGAPSIEULUA": "Siêu Lầy Gặp Siêu Lừa",
    "TAMCAM": "Tấm Cám",
    "NANG2": "Nắng 2"
};

// Helper xử lý URL
const getFullUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    const cleanPath = url.startsWith('/') ? url.slice(1) : url;
    return `${API_BASE}/${cleanPath}`;
};

// [NEW] Hàm lấy tên hiển thị tiếng Việt
const getMovieDisplayName = (rawName) => {
    if (!rawName) return "";
    const key = rawName.toUpperCase();
    return MOVIE_NAMES[key] || rawName;
};

const fetchMovies = async () => {
    isLoading.value = true;
    try {
        const response = await axios.get(`${API_BASE}/api/v1/movies`);
        movies.value = response.data.movies || [];
    } catch (error) {
        console.error("Lỗi tải kho phim:", error);
    } finally {
        isLoading.value = false;
    }
};

onMounted(() => {
    fetchMovies();
});
</script>

<template>
    <div class="animate-fade-in">
        <div class="flex justify-between items-end mb-8 border-b border-[#333] pb-4">
            <h2 class="text-lg font-serif text-white tracking-wide border-l-2 border-[#E50914] pl-3">
                DANH SÁCH PHIM HIỆN CÓ
            </h2>
            <button @click="fetchMovies" class="text-xs text-gray-500 hover:text-white uppercase tracking-widest transition-colors">
                Làm mới
            </button>
        </div>

        <div v-if="isLoading" class="flex justify-center p-20">
                <p class="text-gray-500 text-xs uppercase tracking-widest animate-pulse">Đang tải dữ liệu...</p>
        </div>

        <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <div v-for="(movie, index) in movies" :key="index" class="group bg-[#121212] border border-[#333] hover:border-[#E50914] transition-all duration-300 cursor-pointer">
                <!-- Video Thumbnail/Player -->
                <div class="aspect-video bg-black relative overflow-hidden">
                    <video controls class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" :src="getFullUrl(movie.video_url)"></video>
                </div>

                <!-- Info -->
                <div class="p-6">
                    <!-- [EDITED] Áp dụng hàm getMovieDisplayName tại đây -->
                    <h3 class="font-serif text-xl text-white truncate mb-2 group-hover:text-[#E50914] transition-colors">
                        {{ getMovieDisplayName(movie.movie_name) }}
                    </h3>
                    <div class="flex justify-between items-center border-t border-[#333] pt-4 mt-2">
                        <span class="text-xs text-gray-500 uppercase tracking-widest">Thời lượng: {{ movie.duration || 'N/A' }}</span>
                        <span class="text-xs font-bold text-white bg-[#E50914] px-2 py-0.5">HD</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.animate-fade-in {
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>