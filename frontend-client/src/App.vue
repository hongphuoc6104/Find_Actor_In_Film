<script setup>
import { ref } from 'vue';
import MovieGallery from './components/MovieGallery.vue';
import SearchSection from './components/SearchSection.vue';

const currentTab = ref('search'); // 'search' hoặc 'gallery'
</script>

<template>
  <div class="min-h-screen bg-[#050505] text-[#E0E0E0] font-sans selection:bg-[#E50914] selection:text-white">
    <div class="max-w-[1600px] mx-auto p-6 md:p-12">

        <!-- HEADER -->
        <header class="mb-12 border-b border-[#333] pb-6 flex flex-col md:flex-row justify-between items-end gap-6">
            <div>
                <h1 class="font-serif text-5xl md:text-6xl text-white font-bold tracking-tighter">
                    FACELOOK <span class="text-[#E50914]">CINEMA</span>
                </h1>
                <p class="text-xs text-gray-500 mt-3 uppercase tracking-[0.3em] font-medium">Hệ thống phân tích video AI</p>
            </div>

            <!-- Navigation -->
            <div class="flex gap-8">
                <button
                    @click="currentTab = 'search'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'search' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    Tìm kiếm diễn viên
                </button>
                <button
                    @click="currentTab = 'gallery'"
                    :class="['text-sm font-bold uppercase tracking-widest pb-1 border-b-2 transition-all duration-300',
                             currentTab === 'gallery' ? 'border-[#E50914] text-white' : 'border-transparent text-gray-500 hover:text-white']">
                    Kho phim
                </button>
            </div>
        </header>

        <!-- MAIN CONTENT AREA -->
        <main>
            <!-- TAB 1: TÌM KIẾM -->
            <div v-if="currentTab === 'search'">
                <SearchSection />
            </div>

            <!-- TAB 2: KHO PHIM -->
            <div v-else-if="currentTab === 'gallery'">
                <MovieGallery />
            </div>
        </main>
    </div>
  </div>
</template>