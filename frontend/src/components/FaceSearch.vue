<template>
  <div class="mx-auto max-w-5xl p-4 space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-semibold text-slate-800">Tìm người trong phim</h1>
      <button
        class="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm hover:bg-slate-50"
        @click="refreshMovies"
        :disabled="catalog.loading.value"
      >
        Làm mới danh sách phim
      </button>
    </div>

    <!-- Catalog status -->
    <div v-if="catalog.error.value" class="rounded-lg border border-rose-200 bg-rose-50 p-3 text-rose-700 text-sm">
      {{ catalog.error.value }}
    </div>
    <div v-else class="text-xs text-slate-500">
      {{ catalog.movieCount.value }} phim đã xử lý.
    </div>

    <!-- Upload + Search -->
    <div class="rounded-xl border border-slate-200 p-4 bg-white space-y-3">
      <div class="flex items-center gap-3">
        <input type="file" accept="image/*" @change="onFile" class="block w-full text-sm text-slate-600" />
        <button
          class="rounded-lg bg-slate-900 px-4 py-2 text-white text-sm disabled:opacity-60"
          :disabled="!imageFile || loading"
          @click="doSearch"
        >
          {{ loading ? 'Đang nhận diện...' : 'Tìm trong kho' }}
        </button>
      </div>

      <div v-if="imageUrl" class="flex items-center gap-4 pt-2">
        <img :src="imageUrl" alt="query" class="h-28 w-28 rounded-lg object-cover border" />
        <div class="text-xs text-slate-500">
          {{ imageFile?.name }} — {{ (imageFile?.size/1024).toFixed(1) }} KB
        </div>
      </div>
    </div>

    <!-- Error -->
    <div v-if="error" class="rounded-lg border border-rose-200 bg-rose-50 p-3 text-rose-700 text-sm">
      {{ error }}
    </div>

    <!-- Empty result -->
    <div
      v-if="result && (result.is_unknown || !result.movies?.length)"
      class="rounded-lg border border-amber-200 bg-amber-50 p-3 text-amber-800 text-sm"
    >
      Không tìm thấy người trong ảnh xuất hiện ở bất kỳ phim nào.
    </div>

    <!-- Results -->
    <div v-if="result && result.movies?.length" class="space-y-4">
      <div
        v-for="m in result.movies"
        :key="m.movie_id ?? m.movie"
        class="rounded-xl border border-slate-200 bg-white"
      >
        <!-- Movie header -->
        <div class="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <div class="font-medium text-slate-800">
            Phim:
            <span class="font-mono">
              {{ displayMovieName(m) }}
            </span>
          </div>
          <div class="text-xs text-slate-500">
            {{ m.characters?.length || 0 }} nhân vật phù hợp
          </div>
        </div>

        <!-- Characters -->
        <div class="p-4">
          <div v-for="c in m.characters" :key="c.character_id" class="mb-6 last:mb-0">
            <div class="flex items-center gap-3">
              <div class="font-semibold text-slate-700">
                Nhân vật: <span class="font-mono">{{ c.character_id }}</span>
              </div>
              <span
                class="rounded-full px-2 py-0.5 text-xs"
                :class="(c.match_status || m.match_status) === 'present'
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-sky-50 text-sky-700 border border-sky-200'"
              >
                {{ c.match_label || m.match_label || c.match_status || m.match_status || 'Gần giống' }}
              </span>
              <span class="text-xs text-slate-400">điểm: {{ (c.score ?? 0).toFixed(3) }}</span>
            </div>

            <!-- Scenes -->
            <div v-if="!c.scenes?.length" class="mt-2 text-xs text-slate-500">
              Không có cảnh để xem trước.
            </div>

            <div v-else class="mt-3 space-y-2">
              <!-- Nếu nhiều cảnh, hiển thị dropdown chọn nhanh -->
              <details class="rounded-lg border border-slate-200">
                <summary class="cursor-pointer px-3 py-2 text-sm text-slate-700">
                  {{ Math.min(c.scenes.length, maxScenes) }} cảnh tiêu biểu (từ {{ c.scenes.length }})
                </summary>
                <div class="p-3 grid gap-2 sm:grid-cols-2">
                  <div
                    v-for="(s, idx) in limitedScenes(c.scenes)"
                    :key="idx"
                    class="rounded-lg border border-slate-200 p-3"
                  >
                    <div class="text-sm text-slate-700">Cảnh {{ idx + 1 }}</div>
                    <div class="mt-1 text-xs text-slate-500">
                      Thời lượng: {{ sceneDuration(s) }}
                    </div>
                    <div class="mt-2 flex items-center gap-2">
                      <a
                        v-if="s.video_url"
                        class="rounded-lg bg-slate-900 px-3 py-1.5 text-white text-xs"
                        :href="toVideoHref(s, m)"
                        target="_blank"
                        rel="noopener"
                      >
                        Mở video
                      </a>
                      <span v-else class="text-xs text-slate-500">Thiếu video_url</span>
                      <span class="text-xs text-slate-400">score: {{ s.score?.toFixed(3) ?? '—' }}</span>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </div>
      </div> <!-- movie card -->
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import useMovieCatalog from '@/composables/useMovieCatalog.js'

/** API base: ưu tiên window.__API_BASE__ rồi fallback localhost */
const API_BASE =
  (typeof window !== 'undefined' && window.__API_BASE__) || 'http://127.0.0.1:8000'

const catalog = useMovieCatalog()

const imageFile = ref(null)
const imageUrl = ref(null)
const loading = ref(false)
const error = ref('')
const result = ref(null)

// chỉ hiển thị tối đa N cảnh/nhân vật để giao diện gọn
const maxScenes = 10

function onFile(e) {
  const f = e.target.files?.[0]
  imageFile.value = f || null
  imageUrl.value = f ? URL.createObjectURL(f) : null
  error.value = ''
  result.value = null
}

async function refreshMovies() {
  await catalog.refresh()
}

function displayMovieName(m) {
  // Ưu tiên tên kèm theo payload nhận diện
  const direct =
    (m?.movie && String(m.movie).trim()) ||
    (m?.movie_title && String(m.movie_title).trim()) ||
    (m?.label && String(m.label).trim())
  if (direct) return direct

  // Tra theo catalog map (id -> title)
  const id = (m?.movie_id ?? '').toString()
  const fromMap = id && catalog.idToName.value[id]
  if (fromMap) return fromMap

  // Cuối cùng: string hóa id
  return id || 'unknown'
}

function sceneDuration(s) {
  const st = Number(s.start_time || 0)
  const et = Number(s.end_time || 0)
  const d = Math.max(0, et - st)
  return `${d.toFixed(1)}s`
}

function limitedScenes(scenes) {
  return (scenes || []).slice(0, maxScenes)
}

function toVideoHref(scene, movieItem) {
  // Ưu tiên video_url từ scene; nếu không có, dựng từ movie title
  const baseUrl = scene.video_url
    ? new URL(scene.video_url, API_BASE).toString()
    : new URL(`/videos/${displayMovieName(movieItem)}`, API_BASE).toString()

  const t = Math.max(0, Number(scene.start_time || 0))
  return `${baseUrl}#t=${t.toFixed(1)}`
}

async function doSearch() {
  if (!imageFile.value) return
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const form = new FormData()
    form.append('image', imageFile.value)

    const url = new URL('/recognize', API_BASE)
    const res = await fetch(url.toString(), { method: 'POST', body: form })
    const contentType = res.headers.get('content-type') || ''
    if (!contentType.includes('application/json')) {
      const txt = await res.text().catch(() => '')
      throw new SyntaxError(`Expected JSON, got: ${txt.slice(0, 120)}...`)
    }
    const data = await res.json()
    // đảm bảo luôn có m.movie (title) cho FE hiển thị
    if (Array.isArray(data?.movies)) {
      for (const m of data.movies) {
        const name = displayMovieName(m)
        if (!m.movie) m.movie = name
      }
    }
    result.value = data
  } catch (e) {
    console.error(e)
    error.value = e?.message || 'Nhận diện thất bại'
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await catalog.ensureLoaded()
})
</script>

<style scoped>
/* Tailwind dùng CDN trong index.html cho prototyping — giữ nguyên */
</style>
