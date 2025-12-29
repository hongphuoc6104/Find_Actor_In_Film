<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const API_BASE = "http://localhost:8000";
const movies = ref([]);
const isLoading = ref(false);

// Delete confirmation state
const showDeleteConfirm = ref(false);
const movieToDelete = ref(null);
const isDeleting = ref(false);
const deleteMessage = ref('');

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

// Check if movie is protected (has custom config)
const isProtectedMovie = (movieName) => {
    const protectedMovies = Object.keys(MOVIE_NAMES);
    return protectedMovies.includes(movieName.toUpperCase());
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

// Open delete confirmation
const confirmDelete = (movie) => {
    movieToDelete.value = movie;
    showDeleteConfirm.value = true;
    deleteMessage.value = '';
};

// Cancel delete
const cancelDelete = () => {
    showDeleteConfirm.value = false;
    movieToDelete.value = null;
    deleteMessage.value = '';
};

// Perform delete
const performDelete = async () => {
    if (!movieToDelete.value) return;
    
    isDeleting.value = true;
    deleteMessage.value = '';
    
    try {
        const response = await axios.delete(`${API_BASE}/api/v1/movies/${movieToDelete.value.movie_name}`);
        deleteMessage.value = `✅ Đã xóa ${response.data.deleted_files} files, ${response.data.deleted_folders} folders`;
        
        // Refresh list after 1.5s
        setTimeout(() => {
            showDeleteConfirm.value = false;
            movieToDelete.value = null;
            fetchMovies();
        }, 1500);
    } catch (error) {
        deleteMessage.value = `❌ Lỗi: ${error.response?.data?.detail || error.message}`;
    } finally {
        isDeleting.value = false;
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
            <div v-for="(movie, index) in movies" :key="index" class="group bg-[#121212] border border-[#333] hover:border-[#E50914] transition-all duration-300 relative">
                <!-- Video Thumbnail/Player -->
                <div class="aspect-video bg-black relative overflow-hidden">
                    <video controls class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" :src="getFullUrl(movie.video_url)"></video>
                </div>

                <!-- Info -->
                <div class="p-6">
                    <h3 class="font-serif text-xl text-white truncate mb-2 group-hover:text-[#E50914] transition-colors">
                        {{ getMovieDisplayName(movie.movie_name) }}
                    </h3>
                    <div class="flex justify-between items-center border-t border-[#333] pt-4 mt-2">
                        <span class="text-xs text-gray-500 uppercase tracking-widest">Thời lượng: {{ movie.duration || 'N/A' }}</span>
                        <div class="flex gap-2 items-center">
                            <span class="text-xs font-bold text-white bg-[#E50914] px-2 py-0.5">HD</span>
                            <!-- Delete button - only show for non-protected movies -->
                            <button 
                                v-if="!isProtectedMovie(movie.movie_name)"
                                @click.stop="confirmDelete(movie)"
                                class="text-xs text-gray-500 hover:text-red-500 transition-colors px-2 py-0.5 border border-gray-600 hover:border-red-500 rounded"
                                title="Xóa video và dữ liệu"
                            >
                                🗑️
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Delete Confirmation Modal -->
        <div v-if="showDeleteConfirm" class="fixed inset-0 bg-black/80 flex items-center justify-center z-50" @click.self="cancelDelete">
            <div class="bg-[#1a1a1a] border border-[#333] rounded-xl p-6 max-w-md w-full mx-4">
                <h3 class="text-xl font-bold text-white mb-4">⚠️ Xác nhận xóa</h3>
                
                <p class="text-gray-300 mb-4">
                    Bạn có chắc muốn xóa <span class="text-[#E50914] font-bold">{{ movieToDelete?.movie_name }}</span>?
                </p>
                
                <div class="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-sm text-red-300">
                    <p class="font-bold mb-1">⚠️ Hành động này sẽ xóa:</p>
                    <ul class="list-disc list-inside text-xs space-y-1">
                        <li>Video gốc (.mp4)</li>
                        <li>Tất cả frames đã trích xuất</li>
                        <li>Tất cả face crops</li>
                        <li>Embeddings & clusters</li>
                        <li>Preview images</li>
                    </ul>
                </div>
                
                <!-- Status message -->
                <div v-if="deleteMessage" class="mb-4 text-sm" :class="deleteMessage.startsWith('✅') ? 'text-green-400' : 'text-red-400'">
                    {{ deleteMessage }}
                </div>
                
                <div class="flex gap-3 justify-end">
                    <button 
                        @click="cancelDelete"
                        :disabled="isDeleting"
                        class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50"
                    >
                        Hủy
                    </button>
                    <button 
                        @click="performDelete"
                        :disabled="isDeleting"
                        class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                        <span v-if="isDeleting" class="animate-spin">⏳</span>
                        <span>{{ isDeleting ? 'Đang xóa...' : 'Xóa vĩnh viễn' }}</span>
                    </button>
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