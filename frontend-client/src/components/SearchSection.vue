<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from 'vue';
import axios from 'axios';

const API_BASE = "http://localhost:8000";
const isLoading = ref(false);
const errorMessage = ref('');

// Ngưỡng tìm kiếm (người dùng có thể chỉnh)
const searchThreshold = ref(0.35);

// State Upload
const selectedFile = ref(null);
const previewImage = ref(null);

// State Kết quả
const searchResults = ref(null);
const activeMatchIndex = ref(null); // Index phim đang chọn
const videoRefs = reactive({});

// --- LAYOUT SYNC STATE (Đồng bộ chiều cao) ---
const leftColumnRef = ref(null);
const rightColumnHeight = ref('auto');
let resizeObserver = null;

// --- TỪ ĐIỂN TÊN PHIM ---
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

const getMovieDisplayName = (rawName) => {
    if (!rawName) return "";
    const key = rawName.toUpperCase();
    return MOVIE_NAMES[key] || rawName;
};

// --- COMPUTED PROPERTIES ---

const detectedActorName = computed(() => {
    const matches = searchResults.value?.matches;
    if (!matches || matches.length === 0) return "---";

    const firstMatch = matches[0];
    if (firstMatch.characters?.length) {
        let name = firstMatch.characters[0].name;
        if (!name || name.toLowerCase() === 'unknown' || name.toLowerCase() === 'chưa rõ') {
            return "DIỄN VIÊN";
        }
        return name.toUpperCase();
    }
    return "DIỄN VIÊN";
});

const statusLabel = computed(() => {
    const count = searchResults.value?.matches?.length || 0;
    return count > 0 ? "ĐÃ TÌM THẤY" : "KHÔNG TÌM THẤY";
});

const currentMovieInfo = computed(() => {
    if (activeMatchIndex.value === null || !searchResults.value) return null;
    const match = searchResults.value.matches[activeMatchIndex.value];
    return {
        title: getMovieDisplayName(match.movie),
        score: match.characters[0].score_display,
        actor: match.characters[0].name
    };
});

const displayedScenes = computed(() => {
    if (activeMatchIndex.value === null) return [];
    return searchResults.value.matches[activeMatchIndex.value].characters[0].scenes;
});

// --- LOGIC XỬ LÝ DỮ LIỆU ---
const processSearchResults = (data) => {
    if (!data || !data.matches) return data;
    const validMatches = [];
    const threshold = searchThreshold.value;
    data.matches.forEach(match => {
        if (!match.characters || match.characters.length === 0) return;
        match.characters.forEach(c => {
            if (c.score === undefined && c.score_display) {
                const num = parseInt(c.score_display.replace('%', ''));
                c.score = isNaN(num) ? 0 : num / 100;
            }
            c.score_display = `${Math.round((c.score || 0) * 100)}%`;
        });
        match.characters.sort((a, b) => (b.score || 0) - (a.score || 0));
        const bestChar = match.characters[0];
        if ((bestChar.score || 0) < threshold) return;
        let allScenes = [...(bestChar.scenes || [])];
        for (let i = 1; i < match.characters.length; i++) {
            const otherChar = match.characters[i];
            if ((otherChar.score || 0) >= threshold && otherChar.scenes) {
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
    // Scroll nhẹ
    setTimeout(() => {
        const rightCol = document.getElementById('right-column-container');
        if (rightCol) rightCol.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
};

const backToList = () => {
    activeMatchIndex.value = null;
};

// [UPDATED] Hàm seekTo an toàn hơn, tránh lỗi khi thao tác nhanh
const seekTo = (seconds) => {
    const videoEl = videoRefs['active'];
    if (videoEl) {
        const attemptPlay = () => {
            videoEl.currentTime = seconds;
            videoEl.play().catch(err => {
                // Bắt lỗi nếu play bị gián đoạn (do user click quá nhanh) để không crash
                console.warn("Auto-play interrupted:", err);
            });
        };

        // Kiểm tra xem video đã load metadata chưa (readyState >= 1)
        if (videoEl.readyState >= 1) {
            attemptPlay();
        } else {
            // Nếu chưa, chờ sự kiện loadedmetadata rồi mới tua
            videoEl.addEventListener('loadedmetadata', attemptPlay, { once: true });
        }
    }
};

const onSceneChange = (event) => {
    const seconds = parseFloat(event.target.value);
    if (!isNaN(seconds)) seekTo(seconds);
};

// --- LIFECYCLE & RESIZE OBSERVER (Xử lý đồng bộ chiều cao) ---
onMounted(() => {
    if (leftColumnRef.value) {
        resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                // Cập nhật chiều cao cột phải bằng chiều cao cột trái
                rightColumnHeight.value = entry.contentRect.height;
            }
        });
        resizeObserver.observe(leftColumnRef.value);
    }
});

onUnmounted(() => {
    if (resizeObserver) resizeObserver.disconnect();
});
</script>

<template>
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-fade-in items-start">

        <!-- CỘT TRÁI: UPLOAD & THÔNG TIN (LÀM CHUẨN HEIGHT) -->
        <div ref="leftColumnRef" class="lg:col-span-4 xl:col-span-3 space-y-6 sticky top-6">

            <!-- 1. Box Upload -->
            <div class="bg-[#121212] p-6 border border-[#333]">
                <h2 class="text-base font-bold text-gray-400 mb-4 tracking-widest uppercase border-b border-[#333] pb-2">
                    ẢNH DIỄN VIÊN TẢI LÊN
                </h2>

                <div class="relative group cursor-pointer aspect-[3/4] bg-black border border-[#333] flex items-center justify-center overflow-hidden transition-colors hover:border-[#E50914]">
                    <img v-if="previewImage" :src="previewImage" class="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity">
                    <div v-else class="text-center p-6">
                        <span class="text-sm font-bold text-gray-500 uppercase tracking-widest group-hover:text-white transition-colors">
                            Chọn ảnh
                        </span>
                    </div>
                    <input type="file" @change="handleFileUpload" class="absolute inset-0 opacity-0 cursor-pointer" accept="image/*">
                </div>

                <!-- Nguong tim kiem -->
                <div class="mt-4 p-3 bg-[#1a1a1a] border border-[#333]">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-sm text-gray-400">Ngưỡng tìm kiếm</span>
                        <span class="text-sm font-bold text-white">{{ Math.round(searchThreshold * 100) }}%</span>
                    </div>
                    <input 
                        type="range" 
                        v-model.number="searchThreshold" 
                        min="0.2" max="0.7" step="0.05"
                        class="w-full h-2 bg-[#333] rounded-lg appearance-none cursor-pointer accent-[#E50914]"
                    >
                    <div class="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Nhiều KQ</span>
                        <span>Chính xác</span>
                    </div>
                </div>

                <button
                    @click="searchFace"
                    :disabled="!selectedFile || isLoading"
                    class="w-full mt-4 bg-[#E50914] hover:bg-[#B20710] disabled:bg-[#333] disabled:text-gray-600 disabled:cursor-not-allowed text-white text-sm font-bold py-3 uppercase tracking-widest transition-all">
                    <span v-if="isLoading">ĐANG QUÉT...</span>
                    <span v-else>QUÉT AI</span>
                </button>
                <p v-if="errorMessage" class="text-[#E50914] text-sm mt-2 text-center">{{ errorMessage }}</p>
            </div>

            <!-- 2. Box Thông tin Phụ -->
            <div v-if="searchResults && !isLoading" class="bg-[#121212] p-6 border border-[#333] animate-slide-up min-h-[150px] flex flex-col justify-center">
                <Transition name="smooth-fade" mode="out-in">
                    <!-- Trạng thái 1: Kết quả tìm kiếm (Tổng quan) -->
                    <div v-if="activeMatchIndex === null" key="summary">
                        <h3 class="text-xs font-bold uppercase tracking-widest mb-2"
                            :class="searchResults.matches.length > 0 ? 'text-gray-500' : 'text-[#E50914]'">
                            {{ statusLabel }}
                        </h3>

                        <div class="text-3xl text-white font-serif font-bold mb-1">{{ detectedActorName }}</div>

                        <div class="text-sm text-gray-400">
                            Xuất hiện trong <span class="font-bold" :class="searchResults.matches.length > 0 ? 'text-[#E50914]' : 'text-gray-600'">
                                {{ searchResults.matches.length }}
                            </span> phim.
                        </div>
                    </div>

                    <!-- Trạng thái 2: Đang xem chi tiết phim -->
                    <div v-else key="detail">
                        <h3 class="text-xs font-bold text-[#E50914] uppercase tracking-widest mb-2">ĐANG PHÂN TÍCH</h3>
                        <div class="text-2xl text-white font-serif font-bold mb-2 leading-tight">{{ currentMovieInfo.title }}</div>
                        <div class="flex items-center gap-2 mt-4 pt-4 border-t border-[#333]">
                             <div class="text-right">
                                 <div class="text-3xl font-bold text-green-500">{{ currentMovieInfo.score }}</div>
                                 <div class="text-[10px] text-gray-500 uppercase tracking-widest">Độ chính xác</div>
                             </div>
                        </div>
                    </div>
                </Transition>
            </div>
        </div>

        <!-- CỘT PHẢI: KẾT QUẢ CHÍNH -->
        <div id="right-column-container" class="lg:col-span-8 xl:col-span-9 scroll-mt-6">

            <!-- Loading -->
            <div v-if="isLoading" class="h-96 flex flex-col items-center justify-center border border-[#333] bg-[#121212]">
                <div class="w-16 h-1 bg-[#333] overflow-hidden mb-4">
                    <div class="h-full bg-[#E50914] w-1/2 animate-slide"></div>
                </div>
                <p class="text-gray-500 text-xs uppercase tracking-widest animate-pulse">Đang tìm kiếm trong kho dữ liệu...</p>
            </div>

            <!-- Empty State -->
            <div v-else-if="!searchResults" class="h-96 flex items-center justify-center border border-[#333] bg-[#121212]/30 border-dashed">
                <p class="text-gray-600 text-sm font-medium tracking-wide">VUI LÒNG CHỌN HÌNH ẢNH</p>
            </div>

            <!-- Content -->
            <div v-else class="animate-fade-in">

                <Transition name="smooth-fade" mode="out-in">

                    <!-- VIEW 1: GRID VIEW -->
                    <div v-if="activeMatchIndex === null" key="grid-view">
                        <div class="bg-[#121212] border border-[#333] p-6 mb-6 flex justify-between items-center">
                            <h2 class="text-sm font-bold text-gray-400 tracking-widest uppercase">DANH SÁCH PHIM TÌM THẤY</h2>
                            <span class="text-xs text-[#E50914] font-bold">{{ searchResults.matches.length }} KẾT QUẢ</span>
                        </div>

                        <div v-if="searchResults.matches.length === 0" class="h-64 flex flex-col items-center justify-center bg-[#1A1A1A] border border-[#333] text-gray-500">
                             <p class="uppercase tracking-widest text-xs font-bold">Không tìm thấy phim nào phù hợp</p>
                             <p class="text-[10px] mt-2 opacity-60">Hãy thử tải lên một hình ảnh rõ nét hơn</p>
                        </div>

                        <div v-else class="grid grid-cols-2 md:grid-cols-3 gap-4">
                            <div
                                v-for="(match, index) in searchResults.matches"
                                :key="index"
                                @click="selectMovie(index)"
                                class="bg-[#121212] border border-[#333] hover:border-[#E50914] cursor-pointer transition-all p-4 flex flex-col group h-full relative overflow-hidden"
                            >
                                <div class="absolute top-0 right-0 bg-[#E50914] text-white text-[9px] font-bold px-1.5 py-0.5 transform translate-x-full group-hover:translate-x-0 transition-transform">
                                    XEM NGAY
                                </div>
                                <h3 class="font-serif text-lg text-white mb-2 group-hover:text-[#E50914] transition-colors line-clamp-2 pr-8 mt-1">
                                    {{ getMovieDisplayName(match.movie) }}
                                </h3>
                                <div class="mt-auto pt-3 border-t border-[#333] flex justify-between items-center">
                                    <span class="text-xs text-gray-500 uppercase">{{ match.characters[0].scenes.length }} cảnh</span>
                                    <span class="text-green-500 font-bold text-sm">{{ match.characters[0].score_display }}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- VIEW 2: DETAIL VIEW (PLAYER & DROPDOWN) -->
                    <div v-else
                         class="bg-[#121212] border border-[#333] flex flex-col transition-all duration-300"
                         :style="{ height: rightColumnHeight ? rightColumnHeight + 'px' : 'auto' }"
                         key="detail-view">

                        <!-- Toolbar -->
                        <div class="flex-none flex items-center justify-between px-4 py-3 border-b border-[#333] bg-[#1A1A1A]">
                            <button @click="backToList" class="flex items-center gap-2 text-xs font-bold text-gray-400 hover:text-white transition-colors uppercase tracking-widest">
                                <span class="text-lg leading-none">←</span> Chọn phim khác
                            </button>
                            <div class="text-xs text-gray-500 uppercase tracking-widest">
                                Chi tiết phân cảnh
                            </div>
                        </div>

                        <!-- Player Area -->
                        <div class="flex-1 min-h-0 relative bg-black border-b border-[#333] flex items-center justify-center">
                             <!-- [UPDATED] Thêm :key="activeMatchIndex" để buộc Vue tạo mới Player khi đổi phim -->
                             <video
                                :key="activeMatchIndex"
                                :ref="(el) => { if (el) videoRefs['active'] = el }"
                                controls
                                class="max-w-full max-h-full object-contain"
                                :src="getFullUrl(searchResults.matches[activeMatchIndex].video_url)">
                            </video>
                        </div>

                        <!-- Timeline Dropdown -->
                        <div class="flex-none p-6 bg-[#0a0a0a]">
                             <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-2">
                                <h4 class="text-gray-400 text-xs uppercase tracking-widest font-bold">
                                    Tìm thấy {{ displayedScenes.length }} phân đoạn:
                                </h4>

                                <!-- DROPDOWN -->
                                <div class="relative w-64">
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
                                </div>
                             </div>
                        </div>
                    </div>
                </Transition>

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

.smooth-fade-enter-active,
.smooth-fade-leave-active {
    transition: opacity 0.4s ease, transform 0.4s ease;
}

.smooth-fade-enter-from,
.smooth-fade-leave-to {
    opacity: 0;
    transform: translateY(10px);
}
</style>