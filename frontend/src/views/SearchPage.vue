<template>
  <!-- App page shell -->
  <div class="w-full">
    <!-- Page heading -->
    <div class="mb-4 sm:mb-6">
      <h2 class="text-xl sm:text-2xl font-semibold tracking-tight">Tìm kiếm khuôn mặt trong phim</h2>
      <p class="mt-1 text-sm text-slate-600">
        Upload ảnh một người, hệ thống sẽ tìm phim & các đoạn xuất hiện. Giao diện tối giản để test nhanh.
      </p>
    </div>

    <!-- 2-col responsive layout: left controls + right viewer -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
      <!-- Left: Face search & results -->
      <section
        class="bg-white rounded-2xl shadow-soft border border-slate-200 p-3 sm:p-4 lg:p-5 flex flex-col min-h-[360px]"
      >
        <header class="flex items-center gap-2 pb-3 border-b border-slate-200">
          <div class="shrink-0 w-9 h-9 rounded-xl bg-slate-100 grid place-items-center">
            <span class="text-slate-600 text-sm">1</span>
          </div>
          <div class="min-w-0">
            <h3 class="text-sm sm:text-base font-medium">Chọn ảnh & tìm kiếm</h3>
            <p class="text-xs text-slate-500">Kéo-thả ảnh hoặc chọn file. Có thể thử nhiều ảnh.</p>
          </div>
          <div class="ml-auto flex items-center gap-2">
            <!-- chỗ để đặt nút refresh/clear nếu component có emit -->
          </div>
        </header>

        <!-- Nội dung FaceSearch -->
        <div class="mt-3 sm:mt-4 flex-1 min-h-0">
          <!--
            Component hiện có: FaceSearch.vue
            Không đổi API nội bộ. Tailwind lo chuyện bố cục co giãn.
          -->
          <FaceSearch />
        </div>
      </section>

      <!-- Right: Scene viewer -->
      <section
        class="bg-white rounded-2xl shadow-soft border border-slate-200 p-3 sm:p-4 lg:p-5 flex flex-col min-h-[360px]"
      >
        <header class="flex items-center gap-2 pb-3 border-b border-slate-200">
          <div class="shrink-0 w-9 h-9 rounded-xl bg-slate-100 grid place-items-center">
            <span class="text-slate-600 text-sm">2</span>
          </div>
          <div class="min-w-0">
            <h3 class="text-sm sm:text-base font-medium">Xem & tua cảnh</h3>
            <p class="text-xs text-slate-500">Chọn phim/nhân vật ở panel trái để hiển thị timeline.</p>
          </div>
        </header>

        <!-- Viewer/body: chống tràn bằng flex + min-h-0 -->
        <div class="mt-3 sm:mt-4 flex-1 min-h-0">
          <!--
            Component hiện có: SceneViewer.vue
            Không đổi props/emit để tương thích code cũ.
          -->
          <SceneViewer />
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
// Dùng đường dẫn tương đối để tránh phụ thuộc alias '@'
import FaceSearch from '../components/FaceSearch.vue'
import SceneViewer from '../components/SceneViewer.vue'

/**
 * Lưu ý:
 * - Không thêm CSS thủ công; mọi thứ dùng Tailwind utility classes.
 * - Bố cục: grid 1 cột (mobile) → 2 cột từ lg: trở lên.
 * - Các section có shadow nhẹ, border, padding hợp lý; dùng min-h & min-h-0 để tránh overflow.
 * - Không thay đổi logic nghiệp vụ trong component con.
 */
</script>

<!-- Không dùng <style>; Tailwind CDN trong index.html đảm nhiệm -->
