<script setup>
import { ref } from 'vue';
import MovieGallery from './components/MovieGallery.vue';
import SearchSection from './components/SearchSection.vue';

const currentTab = ref('search'); // 'search' hoặc 'gallery'
</script>

<template>
  <div class="min-h-screen bg-[#050505] text-[#E0E0E0] font-sans selection:bg-[#E50914] selection:text-white">
    <!-- [EDITED] Tăng padding container: px-8 py-5 (lớn hơn bản Compact cũ) -->
    <div class="max-w-[1800px] mx-auto px-8 py-5">

        <!-- HEADER -->
        <header class="mb-6 border-b border-[#333] pb-4 flex flex-row justify-between items-center gap-8">
            <div class="flex items-baseline gap-4">
                <!-- [EDITED] Tăng size Logo lên text-4xl (cân đối với nội dung to hơn) -->
                <h1 class="font-serif text-3xl md:text-4xl text-white font-bold tracking-tighter">
                    FACELOOK <span class="text-[#E50914]">CINEMA</span>
                </h1>
                <!-- [EDITED] Hiển thị lại slogan với size text-xs -->
            </div>

            <!-- Navigation -->
            <div class="flex gap-8">
                <!-- [EDITED] Tăng size menu lên text-sm (thay vì text-xs) -->
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