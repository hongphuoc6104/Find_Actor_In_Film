<script setup>
import { ref, reactive, computed } from 'vue';
import axios from 'axios';

const API_BASE = "http://localhost:8000";
const isLoading = ref(false);
const errorMessage = ref('');

// State Upload
const selectedFile = ref(null);
const previewImage = ref(null);

// State Kết quả
const searchResults = ref(null);
const activeMatchIndex = ref(null); // Index phim đang chọn
const videoRefs = reactive({});
// showAllScenes không còn cần thiết vì dùng Dropdown

// --- HELPER FUNCTIONS ---
const formatTime = (seconds) => {
    if (!seconds && seconds !== 0) return "00:00";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return h > 0 ? `${h}:${m}:${s}` : `${m}:${s}`;
};

const getFullUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    const cleanPath = url.startsWith('/') ? url.slice(1) : url;
    return `${API_BASE}/${cleanPath}`;
};

// --- COMPUTED PROPERTIES ---
const detectedActorName = computed(() => {
    if (!searchResults.value?.matches?.length) return "DIỄN VIÊN";
    const firstMatch = searchResults.value.matches[0];
    if (firstMatch.characters?.length) {
        return firstMatch.characters[0].name.toUpperCase();
    }
    return "DIỄN VIÊN";
});

// Lấy thông tin phim đang chọn để hiển thị bên cột trái
const currentMovieInfo = computed(() => {
    if (activeMatchIndex.value === null || !searchResults.value) return null;
    const match = searchResults.value.matches[activeMatchIndex.value];
    return {
        title: match.movie,
        score: match.characters[0].score_display,
        actor: match.characters[0].name
    };
});

// Lấy danh sách cảnh (Dùng cho Dropdown)
const displayedScenes = computed(() => {
    if (activeMatchIndex.value === null) return [];
    return searchResults.value.matches[activeMatchIndex.value].characters[0].scenes;
});

// --- LOGIC XỬ LÝ DỮ LIỆU ---
const processSearchResults = (data) => {
    if (!data || !data.matches) return data;
    const validMatches = [];

    data.matches.forEach(match => {
        if (!match.characters || match.characters.length === 0) return;

        match.characters.forEach(c => {
            if (c.name && c.name.toLowerCase() === 'unknown') c.name = 'Chưa rõ';
            if (c.score === undefined && c.score_display) {
                const num = parseInt(c.score_display.replace('%', ''));
                c.score = isNaN(num) ? 0 : num / 100;
            }
            c.score_display = `${Math.round((c.score || 0) * 100)}%`;
        });

        match.characters.sort((a, b) => (b.score || 0) - (a.score || 0));
        const bestChar = match.characters[0];

        if ((bestChar.score || 0) < 0.35) return;

        let allScenes = [...(bestChar.scenes || [])];
        for (let i = 1; i < match.characters.length; i++) {
            const otherChar = match.characters[i];
            if ((otherChar.score || 0) >= 0.35 && otherChar.scenes) {
                allScenes = allScenes.concat(otherChar.scenes);
            }
        }
        allScenes.sort((a, b) => a.start_time - b.start_time);
        bestChar.scenes = allScenes;
        match.characters = [bestChar];

        validMatches.push(match);
    });

    data.matches = validMatches;
    return data;
};

// --- EVENTS ---
const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    selectedFile.value = file;
    previewImage.value = URL.createObjectURL(file);
    searchResults.value = null;
    activeMatchIndex.value = null;
    errorMessage.value = '';
};

const searchFace = async () => {
    if (!selectedFile.value) return;
    isLoading.value = true;
    errorMessage.value = '';
    activeMatchIndex.value = null;

    const formData = new FormData();
    formData.append('file', selectedFile.value);

    try {
        const response = await axios.post(`${API_BASE}/api/v1/search`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        searchResults.value = processSearchResults(response.data);
    } catch (error) {
        if (error.code === "ERR_NETWORK") errorMessage.value = "Lỗi kết nối Backend.";
        else errorMessage.value = "Lỗi: " + (error.response?.data?.detail || error.message);
    } finally {
        isLoading.value = false;
    }
};

const selectMovie = (index) => {
    activeMatchIndex.value = index;
    // Scroll nhẹ lên đầu cột phải để người dùng thấy player ngay
    setTimeout(() => {
        const rightCol = document.getElementById('right-column-container');
        if (rightCol) rightCol.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
};

const backToList = () => {
    activeMatchIndex.value = null;
};

const seekTo = (seconds) => {
    const videoEl = videoRefs['active'];
    if (videoEl) {
        videoEl.currentTime = seconds;
        videoEl.play();
    }
};

// Xử lý sự kiện Dropdown thay đổi
const onSceneChange = (event) => {
    const seconds = parseFloat(event.target.value);
    if (!isNaN(seconds)) {
        seekTo(seconds);
    }
};
</script>

<template>
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-fade-in items-start">

        <!-- CỘT TRÁI: UPLOAD & THÔNG TIN -->
        <div class="lg:col-span-4 xl:col-span-3 space-y-6 sticky top-6">

            <!-- 1. Box Upload -->
            <div class="bg-[#121212] p-6 border border-[#333]">
                <h2 class="text-sm font-bold text-gray-400 mb-4 tracking-widest uppercase border-b border-[#333] pb-2">
                    ẢNH ĐỐI TƯỢNG
                </h2>

                <div class="relative group cursor-pointer aspect-[3/4] bg-black border border-[#333] flex items-center justify-center overflow-hidden transition-colors hover:border-[#E50914]">
                    <img v-if="previewImage" :src="previewImage" class="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity">
                    <div v-else class="text-center p-6">
                        <span class="text-xs font-bold text-gray-500 uppercase tracking-widest group-hover:text-white transition-colors">
                            Chọn ảnh
                        </span>
                    </div>
                    <input type="file" @change="handleFileUpload" class="absolute inset-0 opacity-0 cursor-pointer" accept="image/*">
                </div>

                <button
                    @click="searchFace"
                    :disabled="!selectedFile || isLoading"
                    class="w-full mt-4 bg-[#E50914] hover:bg-[#B20710] disabled:bg-[#333] disabled:text-gray-600 disabled:cursor-not-allowed text-white text-xs font-bold py-3 uppercase tracking-widest transition-all">
                    <span v-if="isLoading">ĐANG QUÉT...</span>
                    <span v-else>QUÉT AI</span>
                </button>
                <p v-if="errorMessage" class="text-[#E50914] text-xs mt-2 text-center">{{ errorMessage }}</p>
            </div>

            <!-- 2. Box Thông tin Phụ -->
            <div v-if="searchResults && !isLoading" class="bg-[#121212] p-6 border border-[#333] animate-slide-up">

                <!-- Trạng thái 1: Đang xem danh sách -->
                <div v-if="activeMatchIndex === null">
                    <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Đã tìm thấy</h3>
                    <div class="text-3xl text-white font-serif font-bold mb-1">{{ detectedActorName }}</div>
                    <div class="text-sm text-gray-400">Xuất hiện trong <span class="text-[#E50914] font-bold">{{ searchResults.matches.length }}</span> phim.</div>
                </div>

                <!-- Trạng thái 2: Đang xem chi tiết phim -->
                <div v-else>
                    <h3 class="text-xs font-bold text-[#E50914] uppercase tracking-widest mb-2">ĐANG PHÂN TÍCH</h3>
                    <div class="text-2xl text-white font-serif font-bold mb-2 leading-tight">{{ currentMovieInfo.title }}</div>

                    <div class="flex items-center gap-2 mt-4 pt-4 border-t border-[#333]">
                         <div class="text-right">
                             <div class="text-3xl font-bold text-green-500">{{ currentMovieInfo.score }}</div>
                             <div class="text-[10px] text-gray-500 uppercase tracking-widest">Độ chính xác</div>
                         </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- CỘT PHẢI: KẾT QUẢ CHÍNH -->
        <div id="right-column-container" class="lg:col-span-8 xl:col-span-9">

            <!-- Loading -->
            <div v-if="isLoading" class="h-96 flex flex-col items-center justify-center border border-[#333] bg-[#121212]">
                <div class="w-16 h-1 bg-[#333] overflow-hidden mb-4">
                    <div class="h-full bg-[#E50914] w-1/2 animate-slide"></div>
                </div>
                <p class="text-gray-500 text-xs uppercase tracking-widest animate-pulse">Đang tìm kiếm trong kho dữ liệu...</p>
            </div>

            <!-- Empty State -->
            <div v-else-if="!searchResults" class="h-96 flex items-center justify-center border border-[#333] bg-[#121212]/30 border-dashed">
                <p class="text-gray-600 text-sm font-medium tracking-wide">KẾT QUẢ SẼ HIỂN THỊ TẠI ĐÂY</p>
            </div>

            <!-- Content -->
            <div v-else class="animate-fade-in">

                <!-- VIEW 1: GRID VIEW (DANH SÁCH) -->
                <div v-if="activeMatchIndex === null">
                    <div class="bg-[#121212] border border-[#333] p-6 mb-6 flex justify-between items-center">
                        <h2 class="text-sm font-bold text-gray-400 tracking-widest uppercase">DANH SÁCH PHIM TÌM THẤY</h2>
                        <span class="text-xs text-[#E50914] font-bold">{{ searchResults.matches.length }} KẾT QUẢ</span>
                    </div>

                    <div v-if="searchResults.matches.length === 0" class="p-6 bg-[#1A1A1A] border border-yellow-900/30 text-yellow-600 text-center">
                        Không tìm thấy kết quả phù hợp.
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div
                            v-for="(match, index) in searchResults.matches"
                            :key="index"
                            @click="selectMovie(index)"
                            class="bg-[#121212] border border-[#333] hover:border-[#E50914] cursor-pointer transition-all p-4 flex flex-col group h-full relative overflow-hidden"
                        >
                            <!-- [EDITED] Nhãn XEM NGAY nhỏ gọn hơn, không che tên phim -->
                            <div class="absolute top-0 right-0 bg-[#E50914] text-white text-[9px] font-bold px-1.5 py-0.5 transform translate-x-full group-hover:translate-x-0 transition-transform">
                                XEM NGAY
                            </div>

                            <!-- [EDITED] Thêm padding phải (pr-8) để tránh chữ chạm vào mép/nhãn -->
                            <h3 class="font-serif text-lg text-white mb-2 group-hover:text-[#E50914] transition-colors line-clamp-2 pr-8 mt-1">
                                {{ match.movie }}
                            </h3>
                            <div class="mt-auto pt-3 border-t border-[#333] flex justify-between items-center">
                                <span class="text-xs text-gray-500 uppercase">{{ match.characters[0].scenes.length }} cảnh</span>
                                <span class="text-green-500 font-bold text-sm">{{ match.characters[0].score_display }}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- VIEW 2: DETAIL VIEW (PLAYER & DROPDOWN) -->
                <div v-else class="bg-[#121212] border border-[#333]">
                    <!-- Toolbar -->
                    <div class="flex items-center justify-between px-4 py-3 border-b border-[#333] bg-[#1A1A1A]">
                        <button @click="backToList" class="flex items-center gap-2 text-xs font-bold text-gray-400 hover:text-white transition-colors uppercase tracking-widest">
                            <span class="text-lg leading-none">←</span> Chọn phim khác
                        </button>
                        <div class="text-xs text-gray-500 uppercase tracking-widest">
                            Chi tiết phân cảnh
                        </div>
                    </div>

                    <!-- Player Area -->
                    <div class="grid grid-cols-1">
                        <!-- Video Player -->
                        <div class="relative bg-black w-full aspect-video border-b border-[#333]">
                             <video
                                :ref="(el) => { if (el) videoRefs['active'] = el }"
                                controls
                                class="w-full h-full object-contain"
                                :src="getFullUrl(searchResults.matches[activeMatchIndex].video_url)">
                            </video>
                        </div>

                        <!-- [EDITED] Timeline Dropdown (Thay cho list cuộn) -->
                        <div class="p-6 bg-[#0a0a0a]">
                             <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-2">
                                <h4 class="text-gray-400 text-xs uppercase tracking-widest font-bold">
                                    Tìm thấy {{ displayedScenes.length }} phân đoạn:
                                </h4>

                                <!-- DROPDOWN CHỌN CẢNH -->
                                <div class="relative w-full sm:w-2/3 md:w-1/2">
                                    <select
                                        @change="onSceneChange"
                                        class="w-full appearance-none bg-[#1A1A1A] border border-[#333] text-white text-sm py-2 px-4 pr-8 rounded-sm focus:outline-none focus:border-[#E50914] transition-colors cursor-pointer"
                                    >
                                        <option value="" disabled selected>-- Chọn thời gian xuất hiện --</option>
                                        <option
                                            v-for="(scene, sIndex) in displayedScenes"
                                            :key="sIndex"
                                            :value="scene.start_time"
                                        >
                                            Cảnh {{ sIndex + 1 }}: {{ formatTime(scene.start_time) }} - {{ formatTime(scene.end_time) }}
                                        </option>
                                    </select>

                                    <!-- Mũi tên Dropdown tùy chỉnh -->
                                    <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-400">
                                        <svg class="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
                                    </div>
                                </div>
                             </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
</template>

<style scoped>
@keyframes slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(200%); }
}
.animate-slide {
    animation: slide 1.5s infinite linear;
}

.animate-fade-in {
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.animate-slide-up {
    animation: slideUp 0.6s ease-out;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>