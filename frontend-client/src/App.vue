<script setup>
import { ref, provide } from 'vue';
import MovieGallery from './components/MovieGallery.vue';
import SearchSection from './components/SearchSection.vue';
import YouTubeDownload from './components/YouTubeDownload.vue';

const currentTab = ref('search'); // 'search', 'gallery', 'youtube'

// Global processing state - chỉ hiển thị banner, KHÔNG khóa tabs
const isProcessing = ref(false);
const processingMessage = ref('');

// Provide to child components
provide('isProcessing', isProcessing);
provide('processingMessage', processingMessage);

const setProcessing = (value, message = '') => {
  isProcessing.value = value;
  processingMessage.value = message;
};
</script>

<template>
  <div class="min-h-screen bg-[#050505] text-[#E0E0E0] font-sans selection:bg-[#E50914] selection:text-white">
    <div class="max-w-[1800px] mx-auto px-8 py-5">

        <!-- HEADER -->
        <header class="mb-6 border-b border-[#333] pb-4 flex flex-row justify-between items-center gap-8">
            <div class="flex items-baseline gap-4">
                <h1 class="font-serif text-3xl md:text-4xl text-white font-bold tracking-tighter">
                    FACELOOK <span class="text-[#E50914]">CINEMA</span>
                </h1>
            </div>

            <!-- Navigation - KHÔNG khóa, cho phép đổi tab tự do -->
            <div class="flex gap-6">
                <button
                    @click="currentTab = 'search'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'search' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    🔍 Tìm kiếm
                </button>
                <button
                    @click="currentTab = 'gallery'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'gallery' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    🎬 Kho phim
                </button>
                <button
                    @click="currentTab = 'youtube'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300 relative',
                             currentTab === 'youtube' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    📺 YouTube
                    <!-- Processing indicator dot -->
                    <span v-if="isProcessing && currentTab !== 'youtube'" class="absolute -top-1 -right-2 w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></span>
                </button>
            </div>
        </header>

        <!-- Processing Banner - hiển thị ở mọi tab -->
        <div v-if="isProcessing" class="mb-4 bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 flex items-center gap-3 cursor-pointer hover:bg-yellow-900/50 transition-colors" @click="currentTab = 'youtube'">
          <div class="animate-spin h-5 w-5 border-2 border-yellow-400 border-t-transparent rounded-full"></div>
          <span class="text-yellow-400 text-sm font-medium">{{ processingMessage || 'Đang xử lý video...' }}</span>
          <span class="text-gray-400 text-xs ml-auto">Click để xem chi tiết</span>
        </div>

        <!-- MAIN CONTENT AREA - use keep-alive to preserve state -->
        <main>
            <keep-alive>
                <component 
                    :is="currentTab === 'search' ? SearchSection : 
                         currentTab === 'gallery' ? MovieGallery : YouTubeDownload"
                    @switch-to-search="currentTab = 'search'"
                    @set-processing="setProcessing"
                />
            </keep-alive>
        </main>
    </div>
  </div>
</template>