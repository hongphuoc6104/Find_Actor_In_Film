<template>
  <div class="mx-auto max-w-5xl p-4 space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-semibold text-slate-800">Quản lý phim đã xử lý</h1>
      <div class="flex items-center gap-2">
        <input
          v-model="filterText"
          type="text"
          placeholder="Lọc theo tên phim…"
          class="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm focus:outline-none"
        />
        <button
          class="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm hover:bg-slate-50"
          @click="fetchMovies"
          :disabled="loading"
        >
          Làm mới
        </button>
      </div>
    </div>

    <!-- Error -->
    <div v-if="error" class="rounded-lg border border-rose-200 bg-rose-50 p-3 text-rose-700 text-sm">
      {{ error }}
    </div>

    <!-- Table -->
    <div class="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <table class="min-w-full text-sm">
        <thead class="bg-slate-50 text-slate-500 uppercase text-xs">
          <tr>
            <th class="px-4 py-3 text-left">Phim</th>
            <th class="px-4 py-3 text-left">Movie ID</th>
            <th class="px-4 py-3 text-left">Số nhân vật</th>
            <th class="px-4 py-3 text-left">Số cảnh</th>
            <th class="px-4 py-3 text-left">Ảnh preview</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="loading">
            <td colspan="5" class="px-4 py-6 text-center text-slate-500">Đang tải…</td>
          </tr>

          <tr
            v-for="m in filteredMovies"
            :key="m.movie_id"
            class="border-t border-slate-100 hover:bg-slate-50"
          >
            <!-- TÊN PHIM (HIỂN THỊ THEO BACKEND) -->
            <td class="px-4 py-3 font-medium text-slate-800">
              {{ m.movie || m.movie_title || ('Phim ' + m.movie_id) }}
            </td>

            <!-- ID (để tra cứu, không dùng cho hiển thị chính) -->
            <td class="px-4 py-3 text-slate-600">{{ m.movie_id }}</td>

            <td class="px-4 py-3 text-slate-600">{{ m.character_count ?? 0 }}</td>
            <td class="px-4 py-3 text-slate-600">{{ m.scene_count ?? 0 }}</td>
            <td class="px-4 py-3 text-slate-600">
              {{ m.preview_count ?? 0 }}
            </td>
          </tr>

          <tr v-if="!loading && !filteredMovies.length">
            <td colspan="5" class="px-4 py-6 text-center text-slate-500">
              Không có phim nào.
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const API_BASE =
  (typeof window !== 'undefined' && window.__API_BASE__) || 'http://127.0.0.1:8000'

const movies = ref([])       // [{ movie_id, movie (tên), character_count, ... }]
const loading = ref(false)
const error = ref('')
const filterText = ref('')

const filteredMovies = computed(() => {
  const q = filterText.value.trim().toLowerCase()
  if (!q) return movies.value
  return movies.value.filter(m => {
    const name = String(m.movie || m.movie_title || '').toLowerCase()
    return name.includes(q)
  })
})

async function fetchMovies() {
  loading.value = true
  error.value = ''
  try {
    const url = new URL('/movies', API_BASE)
    const res = await fetch(url.toString())
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    const ct = res.headers.get('content-type') || ''
    if (!ct.includes('application/json')) {
      const t = await res.text().catch(() => '')
      throw new SyntaxError(`Expected JSON, got: ${t.slice(0, 120)}...`)
    }
    const data = await res.json()
    // Backend có thể trả {movies: [...] } hoặc mảng trực tiếp
    const arr = Array.isArray(data) ? data : (data.movies || [])
    // Chuẩn hóa để luôn có 'movie' là tên phim
    movies.value = arr.map(x => ({
      movie_id: String(x.movie_id ?? ''),
      movie: x.movie || x.movie_title || '',   // ← tên phim dùng để hiển thị
      character_count: x.character_count ?? 0,
      scene_count: x.scene_count ?? 0,
      preview_count: x.preview_count ?? 0,
    }))
  } catch (e) {
    console.error(e)
    error.value = e?.message || 'Không tải được danh sách phim'
  } finally {
    loading.value = false
  }
}

onMounted(fetchMovies)
</script>

<style scoped>
/* Tailwind đang dùng CDN trong index.html cho bản prototype */
</style>
