<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

// Form data
const youtubeUrl = ref('');
const videoName = ref('');

// State
const isLoading = ref(false);
const status = ref('');
const message = ref('');
const stage = ref('');
const jobId = ref(null);
let pollInterval = null;

// Video library
const videos = ref([]);
const loadingVideos = ref(false);

const emit = defineEmits(['video-added']);

// Load video library
const loadVideos = async () => {
  loadingVideos.value = true;
  try {
    const response = await axios.get(`${API_URL}/api/v1/videos`);
    videos.value = response.data.videos || [];
  } catch (error) {
    console.error('Error loading videos:', error);
  } finally {
    loadingVideos.value = false;
  }
};

// Poll download status
const pollJobStatus = async () => {
  if (!jobId.value) return;
  
  try {
    const response = await axios.get(`${API_URL}/api/v1/youtube/status/${jobId.value}`);
    const jobStatus = response.data.status;
    stage.value = response.data.stage || '';
    
    if (jobStatus === 'COMPLETED') {
      status.value = 'success';
      message.value = '✅ Tải xong!';
      isLoading.value = false;
      clearInterval(pollInterval);
      loadVideos(); // Refresh list
      emit('video-added');
    } else if (jobStatus === 'FAILED' || jobStatus === 'CANCELLED') {
      status.value = 'error';
      message.value = response.data.stage || 'Tải thất bại';
      isLoading.value = false;
      clearInterval(pollInterval);
    } else if (jobStatus === 'DOWNLOADING') {
      status.value = 'downloading';
      message.value = 'Đang tải video...';
    }
  } catch (error) {
    console.error('Poll error:', error);
  }
};

// Download video (ONLY download, no processing)
const downloadVideo = async () => {
  if (!youtubeUrl.value.trim()) {
    status.value = 'error';
    message.value = 'Vui lòng nhập URL YouTube';
    return;
  }

  isLoading.value = true;
  status.value = 'downloading';
  message.value = 'Đang kiểm tra video...';
  stage.value = '';

  try {
    const formData = new FormData();
    formData.append('url', youtubeUrl.value);
    if (videoName.value.trim()) {
      formData.append('movie_title', videoName.value.trim().toUpperCase().replace(/\s+/g, '_'));
    }

    // Use download-only endpoint (NOT download-and-process)
    const response = await axios.post(`${API_URL}/api/v1/youtube/download`, formData, {
      timeout: 600000  // 10 minutes for download
    });

    if (response.data.status === 'SUCCESS' || response.data.status === 'EXISTS') {
      status.value = 'success';
      message.value = response.data.status === 'EXISTS' 
        ? '✅ Video đã tồn tại trong kho!' 
        : '✅ Tải xong!';
      isLoading.value = false;
      loadVideos();
      emit('video-added');
    } else {
      status.value = 'error';
      message.value = 'Tải thất bại';
      isLoading.value = false;
    }
    
  } catch (error) {
    status.value = 'error';
    isLoading.value = false;
    if (error.response?.data?.detail) {
      message.value = error.response.data.detail;
    } else if (error.code === 'ECONNABORTED') {
      message.value = 'Timeout - video quá lớn, thử lại sau';
    } else {
      message.value = 'Không thể kết nối server';
    }
  }
};

// Delete video
const deleteVideo = async (movieName) => {
  if (!confirm(`Xóa video "${movieName}" và tất cả dữ liệu liên quan?`)) return;
  
  try {
    await axios.delete(`${API_URL}/api/v1/movies/${movieName}`);
    loadVideos();
  } catch (error) {
    alert('Lỗi xóa video: ' + (error.response?.data?.detail || error.message));
  }
};

onMounted(() => {
  loadVideos();
});

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval);
});
</script>

<template>
  <div class="max-w-4xl mx-auto">
    <!-- Title -->
    <div class="text-center mb-8">
      <h2 class="text-3xl font-bold text-white mb-2">Tải Video từ YouTube</h2>
      <p class="text-gray-400 text-base">Nhập URL video YouTube - Tự động tải về kho phim</p>
    </div>

    <!-- Download Form -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333] mb-8">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <!-- URL Input -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">YouTube URL *</label>
          <input
            v-model="youtubeUrl"
            type="text"
            placeholder="https://www.youtube.com/watch?v=..."
            :disabled="isLoading"
            class="w-full px-4 py-3 bg-[#0a0a0a] border border-[#444] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#E50914] transition-colors disabled:opacity-50"
            @keyup.enter="!isLoading && downloadVideo()"
          />
        </div>
        
        <!-- Video Name -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Tên Video (tùy chọn)</label>
          <input
            v-model="videoName"
            type="text"
            placeholder="VD: PHIM_HAY_2024"
            :disabled="isLoading"
            class="w-full px-4 py-3 bg-[#0a0a0a] border border-[#444] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#E50914] transition-colors disabled:opacity-50"
          />
          <p class="text-xs text-gray-500 mt-1">Để trống = tự động từ tiêu đề</p>
        </div>
      </div>

      <!-- Download Button -->
      <button
        @click="downloadVideo"
        :disabled="isLoading"
        :class="[
          'w-full py-3 rounded-lg font-bold text-white transition-all duration-300',
          isLoading 
            ? 'bg-gray-600 cursor-not-allowed' 
            : 'bg-blue-600 hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-500/25'
        ]"
      >
        <span v-if="isLoading" class="flex items-center justify-center gap-2">
          <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          Đang tải...
        </span>
        <span v-else>Tải Video</span>
      </button>

      <!-- Status Message -->
      <div v-if="status" class="mt-4">
        <div v-if="status === 'success'" class="bg-green-900/30 border border-green-700 rounded-lg p-4">
          <p class="text-green-400 font-medium text-base">{{ message }}</p>
        </div>
        <div v-else-if="status === 'downloading'" class="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
          <div class="flex items-center gap-3">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
            <div>
              <p class="text-blue-400 font-medium text-base">{{ message }}</p>
              <p v-if="stage" class="text-base text-gray-400 mt-1">{{ stage }}</p>
            </div>
          </div>
        </div>
        <div v-else-if="status === 'error'" class="bg-red-900/30 border border-red-700 rounded-lg p-4">
          <p class="text-red-400 text-base">Lỗi: {{ message }}</p>
        </div>
      </div>
    </div>

    <!-- Video Library -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333]">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-xl font-bold text-white">Kho Phim</h3>
        <button @click="loadVideos" class="text-base text-gray-400 hover:text-white">Làm mới</button>
      </div>

      <div v-if="loadingVideos" class="text-center py-8 text-gray-400 text-base">
        Đang tải...
      </div>
      
      <div v-else-if="videos.length === 0" class="text-center py-8 text-gray-500 text-base">
        Chưa có video. Hãy tải video đầu tiên!
      </div>
      
      <div v-else class="space-y-2">
        <div 
          v-for="video in videos" 
          :key="video.movie_name"
          class="flex items-center justify-between p-3 bg-[#0a0a0a] rounded-lg border border-[#333] hover:border-[#444]"
        >
          <div class="flex items-center gap-3">
            <span class="text-xl font-bold text-[#E50914]">V</span>
            <div>
              <p class="font-medium text-white text-base">{{ video.movie_name }}</p>
              <p class="text-sm text-gray-500">
                {{ video.file_size_mb }} MB
                <span v-if="video.is_processed" class="ml-2 text-green-500">[Đã xử lý]</span>
                <span v-else class="ml-2 text-yellow-500">[Chưa xử lý]</span>
              </p>
            </div>
          </div>
          <button 
            @click="deleteVideo(video.movie_name)"
            class="p-2 text-red-400 hover:text-red-300 hover:bg-red-900/30 rounded text-base"
          >
            Xóa
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
