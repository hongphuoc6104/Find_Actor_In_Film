<script setup>
import { ref, onUnmounted } from 'vue';
import axios from 'axios';

const youtubeUrl = ref('');
const isLoading = ref(false);
const status = ref(''); // 'success', 'error', 'downloading', 'processing'
const message = ref('');
const stage = ref('');
const downloadedMovie = ref(null);
const jobId = ref(null);
let pollInterval = null;

const API_URL = 'http://localhost:8000';

const emit = defineEmits(['switch-to-search', 'set-processing']);

const pollJobStatus = async () => {
  if (!jobId.value) return;
  
  try {
    const response = await axios.get(`${API_URL}/api/v1/youtube/status/${jobId.value}`);
    const jobStatus = response.data.status;
    stage.value = response.data.stage || '';
    
    if (jobStatus === 'COMPLETED') {
      status.value = 'success';
      message.value = '✅ Xử lý hoàn tất! Bạn có thể tìm kiếm diễn viên.';
      isLoading.value = false;
      emit('set-processing', false, '');
      clearInterval(pollInterval);
    } else if (jobStatus === 'FAILED' || jobStatus === 'CANCELLED') {
      status.value = 'error';
      message.value = response.data.stage || 'Xử lý thất bại';
      isLoading.value = false;
      emit('set-processing', false, '');
      clearInterval(pollInterval);
    } else if (jobStatus === 'PROCESSING') {
      status.value = 'processing';
      message.value = 'Đang xử lý video...';
      emit('set-processing', true, `⚙️ ${stage.value}`);
    } else if (jobStatus === 'DOWNLOADING') {
      status.value = 'downloading';
      message.value = 'Đang tải video...';
      emit('set-processing', true, `📥 ${stage.value}`);
    }
  } catch (error) {
    console.error('Poll error:', error);
  }
};

const downloadAndProcess = async () => {
  if (!youtubeUrl.value.trim()) {
    status.value = 'error';
    message.value = 'Vui lòng nhập URL YouTube';
    return;
  }

  isLoading.value = true;
  status.value = 'downloading';
  message.value = 'Đang kiểm tra video...';
  stage.value = '';
  downloadedMovie.value = null;
  jobId.value = null;
  
  emit('set-processing', true, '📥 Đang kiểm tra video...');

  try {
    const formData = new FormData();
    formData.append('url', youtubeUrl.value);

    const response = await axios.post(`${API_URL}/api/v1/youtube/download-and-process`, formData, {
      timeout: 120000 // 2 minutes for getting info
    });

    jobId.value = response.data.job_id;
    downloadedMovie.value = response.data;
    
    status.value = 'downloading';
    message.value = 'Đang tải video trong nền...';
    emit('set-processing', true, '📥 Đang tải video...');
    // Start polling for status
    pollInterval = setInterval(pollJobStatus, 3000);
    
  } catch (error) {
    status.value = 'error';
    isLoading.value = false;
    emit('set-processing', false, '');
    if (error.response?.data?.detail) {
      message.value = error.response.data.detail;
    } else if (error.code === 'ECONNABORTED') {
      message.value = 'Timeout - Thử lại với video ngắn hơn';
    } else {
      message.value = 'Không thể kết nối đến server';
    }
  }
};

const cancelJob = async () => {
  if (!jobId.value) return;
  
  try {
    await axios.post(`${API_URL}/api/v1/youtube/cancel/${jobId.value}`);
    status.value = 'error';
    message.value = 'Đã hủy';
    isLoading.value = false;
    emit('set-processing', false, '');
    clearInterval(pollInterval);
  } catch (error) {
    console.error('Cancel error:', error);
  }
};

const goToSearch = () => {
  emit('switch-to-search');
};

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval);
});
</script>

<template>
  <div class="max-w-2xl mx-auto">
    <!-- Title -->
    <div class="text-center mb-8">
      <h2 class="text-2xl font-bold text-white mb-2">📺 Tải & Xử Lý Video YouTube</h2>
      <p class="text-gray-400 text-sm">Nhập URL video YouTube → Tự động download + train</p>
    </div>

    <!-- Input Form -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333]">
      <div class="flex flex-col gap-4">
        <!-- URL Input -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">
            YouTube URL
          </label>
          <input
            v-model="youtubeUrl"
            type="text"
            placeholder="https://www.youtube.com/watch?v=..."
            :disabled="isLoading"
            class="w-full px-4 py-3 bg-[#0a0a0a] border border-[#444] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-[#E50914] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            @keyup.enter="!isLoading && downloadAndProcess()"
          />
          <p class="text-xs text-gray-500 mt-1">💡 URL sẽ tự động được làm sạch (bỏ &list=... và các params thừa)</p>
        </div>

        <!-- Buttons -->
        <div class="flex gap-3">
          <!-- Download Button -->
          <button
            @click="downloadAndProcess"
            :disabled="isLoading"
            :class="[
              'flex-1 py-3 rounded-lg font-bold text-white transition-all duration-300',
              isLoading 
                ? 'bg-gray-600 cursor-not-allowed' 
                : 'bg-[#E50914] hover:bg-[#ff1a1a] hover:shadow-lg hover:shadow-red-500/25'
            ]"
          >
            <span v-if="isLoading" class="flex items-center justify-center gap-2">
              <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              {{ status === 'downloading' ? 'Đang tải...' : 'Đang xử lý...' }}
            </span>
            <span v-else>🚀 Tải & Xử Lý</span>
          </button>
          
          <!-- Cancel Button -->
          <button
            v-if="isLoading"
            @click="cancelJob"
            class="px-4 py-3 rounded-lg font-bold text-white bg-gray-700 hover:bg-gray-600 transition-all"
          >
            ✕ Hủy
          </button>
        </div>
      </div>

      <!-- Status Message -->
      <div v-if="status" class="mt-4">
        <!-- Success -->
        <div v-if="status === 'success'" class="bg-green-900/30 border border-green-700 rounded-lg p-4">
          <div class="flex items-start gap-3">
            <span class="text-2xl">✅</span>
            <div class="flex-1">
              <p class="text-green-400 font-medium">{{ message }}</p>
              <div v-if="downloadedMovie" class="mt-2 text-sm text-gray-300">
                <p>📁 Movie: <span class="text-white font-mono">{{ downloadedMovie.movie_title }}</span></p>
                <p v-if="downloadedMovie.duration_display">⏱️ Thời lượng: {{ downloadedMovie.duration_display }}</p>
              </div>
              <button 
                @click="goToSearch"
                class="mt-4 px-4 py-2 bg-[#E50914] hover:bg-[#ff1a1a] rounded-lg font-medium text-sm transition-all"
              >
                🔍 Tìm kiếm diễn viên ngay
              </button>
            </div>
          </div>
        </div>

        <!-- Downloading -->
        <div v-else-if="status === 'downloading'" class="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
          <div class="flex items-center gap-3">
            <div class="animate-bounce text-2xl">📥</div>
            <div class="flex-1">
              <p class="text-blue-400 font-medium">{{ message }}</p>
              <p v-if="stage" class="text-sm text-gray-400 mt-1">{{ stage }}</p>
            </div>
          </div>
        </div>

        <!-- Processing -->
        <div v-else-if="status === 'processing'" class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4">
          <div class="flex items-center gap-3">
            <div class="animate-pulse text-2xl">⚙️</div>
            <div class="flex-1">
              <p class="text-yellow-400 font-medium">{{ message }}</p>
              <p v-if="stage" class="text-sm text-gray-400 mt-1">{{ stage }}</p>
              <p class="text-xs text-gray-500 mt-2">Quá trình này có thể mất 5-15 phút...</p>
            </div>
          </div>
        </div>

        <!-- Error -->
        <div v-else-if="status === 'error'" class="bg-red-900/30 border border-red-700 rounded-lg p-4">
          <div class="flex items-start gap-3">
            <span class="text-2xl">❌</span>
            <p class="text-red-400">{{ message }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Info -->
    <div class="mt-6 text-center text-xs text-gray-500 space-y-1">
      <p>💡 Video ngắn (~5-10 phút) sẽ xử lý nhanh hơn • Chất lượng: 1080p</p>
    </div>
  </div>
</template>
