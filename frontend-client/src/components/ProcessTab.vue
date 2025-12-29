<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

// Videos list
const videos = ref([]);
const loadingVideos = ref(false);

// Form data
const selectedVideo = ref('');
const mode = ref('full'); // 'full' or 'recluster'

// Parameters
const params = ref({
  // Stage 1-2 (require full)
  min_det_score: null,
  min_face_size: null,
  min_blur_clarity: null,
  landmark_hard_cutoff: null,
  landmark_core: null,
  // Stage 4+ (can recluster)
  distance_threshold: null,
  merge_threshold: null,
  min_cluster_size: null,
  min_track_size: null,
  post_merge_threshold: null
});

// Processing state
const isProcessing = ref(false);
const jobId = ref(null);
const status = ref('');
const stage = ref('');
let pollInterval = null;

// Parameter definitions for UI
const parameterGroups = [
  {
    title: 'Stage 1-2: Trích Xuất & Nhận Diện',
    warning: 'Thay đổi cần chạy "Xử lý từ đầu"',
    params: [
      { key: 'min_det_score', label: 'Độ tin cậy nhận diện', default: 0.45, min: 0.2, max: 0.9, step: 0.05, 
        desc: 'Thấp = phát hiện nhiều mặt hơn nhưng có thể nhận nhầm' },
      { key: 'min_face_size', label: 'Kích thước mặt tối thiểu (px)', default: 50, min: 20, max: 120, step: 5,
        desc: 'Cao = bỏ qua mặt nhỏ/xa camera' },
      { key: 'min_blur_clarity', label: 'Độ rõ nét tối thiểu', default: 40, min: 15, max: 80, step: 5,
        desc: 'Cao = loại bỏ ảnh mờ' },
      { key: 'landmark_hard_cutoff', label: 'Ngưỡng landmark cứng', default: 0.55, min: 0.3, max: 0.8, step: 0.05,
        desc: 'Thấp = chấp nhận mặt nghiêng. Cao = chỉ lấy mặt thẳng' },
      { key: 'landmark_core', label: 'Ngưỡng landmark core', default: 0.70, min: 0.5, max: 0.9, step: 0.05,
        desc: 'Ngưỡng để chọn ảnh đại diện chất lượng cao' }
    ]
  },
  {
    title: 'Stage 4-7: Phân Cụm & Gộp',
    warning: 'Có thể chạy "Gom nhóm lại"',
    params: [
      { key: 'distance_threshold', label: 'Ngưỡng clustering', default: 1.15, min: 0.4, max: 1.5, step: 0.05,
        desc: 'Thấp = chặt chẽ (ít gom nhầm). Cao = gom nhiều hơn' },
      { key: 'merge_threshold', label: 'Ngưỡng merge cụm', default: 0.55, min: 0.35, max: 0.75, step: 0.05,
        desc: 'Độ giống nhau tối thiểu để gộp 2 cụm' },
      { key: 'min_track_size', label: 'Số frame tối thiểu/track', default: 3, min: 1, max: 10, step: 1,
        desc: 'Số lần xuất hiện liên tục tối thiểu. 1 = giữ mọi detection' },
      { key: 'min_cluster_size', label: 'Số ảnh tối thiểu/cụm', default: 15, min: 1, max: 50, step: 1,
        desc: 'Cao = chỉ giữ nhân vật chính (xuất hiện nhiều)' },
      { key: 'post_merge_threshold', label: 'Ngưỡng post-merge', default: 0.60, min: 0.40, max: 0.80, step: 0.05,
        desc: 'Ngưỡng để hấp thụ cụm nhỏ vào cụm lớn' }
    ]
  }
];

// Presets
const presets = [
  { name: 'Phim dài (>40 phút)', values: { min_cluster_size: 20, distance_threshold: 1.15 } },
  { name: 'Phim ngắn (10-40 phút)', values: { min_cluster_size: 10, distance_threshold: 0.80 } },
  { name: 'MV/Clip (<10 phút)', values: { min_cluster_size: 2, distance_threshold: 0.60 } }
];

// Load videos
const loadVideos = async () => {
  loadingVideos.value = true;
  try {
    const response = await axios.get(`${API_URL}/api/v1/videos`);
    videos.value = response.data.videos || [];
    if (videos.value.length > 0 && !selectedVideo.value) {
      selectedVideo.value = videos.value[0].movie_name;
    }
  } catch (error) {
    console.error('Error loading videos:', error);
  } finally {
    loadingVideos.value = false;
  }
};

// Apply preset
const applyPreset = (preset) => {
  Object.entries(preset.values).forEach(([key, value]) => {
    params.value[key] = value;
  });
};

// Reset to defaults
const resetParams = () => {
  parameterGroups.forEach(group => {
    group.params.forEach(p => {
      params.value[p.key] = null;
    });
  });
};

// Check if video has embeddings (for recluster mode)
const selectedVideoInfo = computed(() => {
  return videos.value.find(v => v.movie_name === selectedVideo.value);
});

// Poll job status
const pollJobStatus = async () => {
  if (!jobId.value) return;
  
  try {
    const response = await axios.get(`${API_URL}/api/v1/process/status/${jobId.value}`);
    const jobStatus = response.data.status;
    stage.value = response.data.stage || '';
    
    if (jobStatus === 'COMPLETED') {
      status.value = 'success';
      isProcessing.value = false;
      clearInterval(pollInterval);
      loadVideos();
    } else if (jobStatus === 'FAILED') {
      status.value = 'error';
      isProcessing.value = false;
      clearInterval(pollInterval);
    } else {
      status.value = 'processing';
    }
  } catch (error) {
    console.error('Poll error:', error);
  }
};

// Start processing
const startProcessing = async () => {
  if (!selectedVideo.value) {
    alert('Vui lòng chọn video');
    return;
  }

  isProcessing.value = true;
  status.value = 'processing';
  stage.value = 'Đang khởi động...';
  jobId.value = null;

  try {
    const formData = new FormData();
    formData.append('movie_name', selectedVideo.value);
    formData.append('mode', mode.value);
    
    // Add non-null params
    Object.entries(params.value).forEach(([key, value]) => {
      if (value !== null) {
        formData.append(key, value);
      }
    });

    const response = await axios.post(`${API_URL}/api/v1/process/start`, formData);
    jobId.value = response.data.job_id;
    
    // Start polling
    pollInterval = setInterval(pollJobStatus, 3000);
    
  } catch (error) {
    status.value = 'error';
    stage.value = error.response?.data?.detail || 'Lỗi không xác định';
    isProcessing.value = false;
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
      <h2 class="text-3xl font-bold text-white mb-2">Xử Lý Video</h2>
      <p class="text-gray-400 text-base">Chọn video, điều chỉnh tham số, và chạy pipeline</p>
    </div>

    <!-- Video Selection -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333] mb-6">
      <h3 class="text-xl font-bold text-white mb-4">1. Chọn Video</h3>
      
      <div v-if="loadingVideos" class="text-gray-400 text-base">Đang tải...</div>
      <div v-else-if="videos.length === 0" class="text-yellow-500 text-base">
        Chưa có video. Hãy tải video trước ở tab "Tải Video"
      </div>
      <div v-else>
        <select 
          v-model="selectedVideo"
          class="w-full px-4 py-3 bg-[#0a0a0a] border border-[#444] rounded-lg text-white text-base focus:outline-none focus:border-[#E50914]"
        >
          <option v-for="video in videos" :key="video.movie_name" :value="video.movie_name">
            {{ video.movie_name }} 
            ({{ video.file_size_mb }}MB)
            {{ video.is_processed ? '[Đã xử lý]' : '[Chưa xử lý]' }}
          </option>
        </select>
      </div>
    </div>

    <!-- Processing Mode -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333] mb-6">
      <h3 class="text-xl font-bold text-white mb-4">2. Chế Độ Xử Lý</h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <label 
          :class="[
            'p-4 border-2 rounded-lg cursor-pointer transition-all',
            mode === 'full' ? 'border-[#E50914] bg-[#E50914]/10' : 'border-[#444] hover:border-[#666]'
          ]"
        >
          <input type="radio" v-model="mode" value="full" class="hidden" />
          <div class="flex items-start gap-3">
            <span class="text-xl font-bold text-[#E50914]">MỚI</span>
            <div>
              <p class="font-bold text-white text-base">Xử lý từ đầu</p>
              <p class="text-base text-gray-400">Chạy full pipeline: trích xuất - nhận diện - phân cụm</p>
            </div>
          </div>
        </label>
        
        <label 
          :class="[
            'p-4 border-2 rounded-lg cursor-pointer transition-all',
            mode === 'recluster' ? 'border-[#E50914] bg-[#E50914]/10' : 'border-[#444] hover:border-[#666]',
            !selectedVideoInfo?.is_processed ? 'opacity-50 cursor-not-allowed' : ''
          ]"
        >
          <input 
            type="radio" 
            v-model="mode" 
            value="recluster" 
            class="hidden"
            :disabled="!selectedVideoInfo?.is_processed"
          />
          <div class="flex items-start gap-3">
            <span class="text-xl font-bold text-blue-400">LẠI</span>
            <div>
              <p class="font-bold text-white text-base">Gom nhóm lại</p>
              <p class="text-base text-gray-400">Bỏ qua trích xuất, chỉ chạy lại clustering</p>
              <p v-if="!selectedVideoInfo?.is_processed" class="text-sm text-red-400 mt-1">
                Cần xử lý từ đầu trước
              </p>
            </div>
          </div>
        </label>
      </div>
    </div>

    <!-- Parameters -->
    <div class="bg-[#1a1a1a] rounded-xl p-6 border border-[#333] mb-6">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-xl font-bold text-white">3. Tham Số</h3>
        <button @click="resetParams" class="text-base text-gray-400 hover:text-white">Reset</button>
      </div>

      <!-- Presets -->
      <div class="flex flex-wrap gap-2 mb-6">
        <button 
          v-for="preset in presets" 
          :key="preset.name"
          @click="applyPreset(preset)"
          class="px-3 py-1.5 bg-[#333] hover:bg-[#444] rounded-lg text-sm text-gray-300 transition-colors"
        >
          {{ preset.name }}
        </button>
      </div>

      <!-- Parameter Groups -->
      <div class="space-y-6">
        <div v-for="group in parameterGroups" :key="group.title">
          <div class="flex items-center gap-2 mb-3">
            <h4 class="font-medium text-gray-300">{{ group.title }}</h4>
            <span class="text-xs px-2 py-0.5 rounded" 
                  :class="group.warning.includes('⚠️') ? 'bg-yellow-900/50 text-yellow-400' : 'bg-blue-900/50 text-blue-400'">
              {{ group.warning }}
            </span>
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div v-for="param in group.params" :key="param.key" class="p-3 bg-[#0a0a0a] rounded-lg">
              <div class="flex justify-between items-center mb-2">
                <label class="text-sm text-gray-300">{{ param.label }}</label>
                <span class="text-xs text-gray-500">
                  {{ params[param.key] !== null ? params[param.key] : `(${param.default})` }}
                </span>
              </div>
              <input 
                type="range"
                v-model.number="params[param.key]"
                :min="param.min"
                :max="param.max"
                :step="param.step"
                class="w-full accent-[#E50914]"
              />
              <p class="text-xs text-gray-500 mt-1">{{ param.desc }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Start Button -->
    <button
      @click="startProcessing"
      :disabled="isProcessing || !selectedVideo"
      :class="[
        'w-full py-4 rounded-lg font-bold text-white transition-all duration-300 text-lg',
        isProcessing || !selectedVideo
          ? 'bg-gray-600 cursor-not-allowed' 
          : 'bg-[#E50914] hover:bg-[#ff1a1a] hover:shadow-lg hover:shadow-red-500/25'
      ]"
    >
      <span v-if="isProcessing" class="flex items-center justify-center gap-2">
        <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
        Đang xử lý...
      </span>
      <span v-else>Bắt Đầu Xử Lý</span>
    </button>

    <!-- Status -->
    <div v-if="status" class="mt-4">
      <div v-if="status === 'success'" class="bg-green-900/30 border border-green-700 rounded-lg p-4">
        <p class="text-green-400 font-medium text-base mb-4">Xử lý hoàn tất!</p>
        
        <!-- UMAP Chart -->
        <div v-if="selectedVideo" class="mt-4 bg-[#0a0a0a] rounded-lg p-4 border border-[#333]">
          <h4 class="text-base font-bold text-white mb-3">Biểu đồ Clustering UMAP</h4>
          <img 
            :src="`http://localhost:8000/static/evaluation/${selectedVideo}/umap_projection.png`" 
            alt="UMAP Projection"
            class="w-full rounded-lg"
            @error="(e) => e.target.style.display='none'"
          />
          <p class="text-sm text-gray-400 mt-2">
            Mỗi màu đại diện 1 cụm (nhân vật). Các điểm gần nhau = giống nhau.
          </p>
        </div>
      </div>
      <div v-else-if="status === 'processing'" class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4">
        <div class="flex items-center gap-3">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400"></div>
          <div>
            <p class="text-yellow-400 font-medium text-base">Đang xử lý...</p>
            <p class="text-base text-gray-400 mt-1">{{ stage }}</p>
          </div>
        </div>
      </div>
      <div v-else-if="status === 'error'" class="bg-red-900/30 border border-red-700 rounded-lg p-4">
        <p class="text-red-400 text-base">Lỗi: {{ stage }}</p>
      </div>
    </div>
  </div>
</template>
