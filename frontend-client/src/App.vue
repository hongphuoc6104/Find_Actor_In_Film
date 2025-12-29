<script setup>
import { ref, provide } from 'vue';
import SearchSection from './components/SearchSection.vue';
import DownloadTab from './components/DownloadTab.vue';
import ProcessTab from './components/ProcessTab.vue';

const currentTab = ref('download'); // 'download', 'process', 'search'

// Global processing state
const isProcessing = ref(false);
const processingMessage = ref('');

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

            <!-- Navigation -->
            <div class="flex gap-6">
                <button
                    @click="currentTab = 'download'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'download' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    📥 Tải Video
                </button>
                <button
                    @click="currentTab = 'process'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300 relative',
                             currentTab === 'process' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    ⚙️ Xử Lý
                    <span v-if="isProcessing && currentTab !== 'process'" class="absolute -top-1 -right-2 w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></span>
                </button>
                <button
                    @click="currentTab = 'search'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'search' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    🔍 Tìm Kiếm
                </button>
            </div>
        </header>

        <!-- Processing Banner -->
        <div v-if="isProcessing" class="mb-4 bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 flex items-center gap-3 cursor-pointer hover:bg-yellow-900/50 transition-colors" @click="currentTab = 'process'">
          <div class="animate-spin h-5 w-5 border-2 border-yellow-400 border-t-transparent rounded-full"></div>
          <span class="text-yellow-400 text-sm font-medium">{{ processingMessage || 'Đang xử lý video...' }}</span>
          <span class="text-gray-400 text-xs ml-auto">Click để xem chi tiết</span>
        </div>

        <!-- MAIN CONTENT -->
        <main>
            <keep-alive>
                <component 
                    :is="currentTab === 'download' ? DownloadTab : 
                         currentTab === 'process' ? ProcessTab : SearchSection"
                    @switch-to-search="currentTab = 'search'"
                    @video-added="currentTab = 'process'"
                    @set-processing="setProcessing"
                />
            </keep-alive>
        </main>
    </div>
  </div>
</template>