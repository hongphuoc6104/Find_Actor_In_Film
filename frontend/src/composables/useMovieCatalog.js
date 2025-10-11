// Composable quản lý danh mục phim và map id <-> title
// Dùng chung cho FaceSearch + MovieManagementPage

import { ref, computed } from 'vue'

// API base: ưu tiên window.__API_BASE__ (được BE inject) rồi tới config.js, cuối cùng fallback
let API_BASE =
  (typeof window !== 'undefined' && window.__API_BASE__) ||
  (import.meta?.env?.VITE_API_BASE || 'http://127.0.0.1:8000')

// Singleton để toàn bộ app dùng chung 1 nguồn dữ liệu
let _singleton

function createCatalog() {
  const loading = ref(false)
  const error = ref('')
  const movies = ref([]) // [{ movie_id: 0, movie: 'EMCHUA18', label: 'EMCHUA18', video_path: '...' }, ...]

  // Map thuận tiện
  const idToName = ref({})    // {'0': 'EMCHUA18', '1': 'GAIGIALAMCHIEU', ...}
  const nameToId = ref({})    // {'EMCHUA18': '0', ...}

  const movieCount = computed(() => movies.value.length)

  function _rebuildMaps(list) {
    const id2 = {}
    const name2id = {}
    for (const m of list) {
      // Chuẩn hóa
      const id = (m?.movie_id ?? '').toString()
      const title = (m?.movie ?? m?.label ?? '').toString().trim()
      if (!title) continue

      if (id && id !== 'null' && id !== 'undefined' && id !== '') {
        id2[id] = title
      }
      // nameToId chỉ set khi có id hợp lệ
      if (title && id && id !== 'null' && id !== 'undefined' && id !== '') {
        name2id[title] = id
      }
    }
    idToName.value = id2
    nameToId.value = name2id
  }

  async function refresh() {
    loading.value = true
    error.value = ''
    try {
      const url = new URL('/movies', API_BASE).toString()
      const res = await fetch(url, { method: 'GET' })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`GET /movies thất bại: ${res.status} ${txt.slice(0, 120)}`)
      }
      const data = await res.json()
      const list = Array.isArray(data?.movies) ? data.movies : []

      movies.value = list
      _rebuildMaps(list)
    } catch (e) {
      console.error(e)
      error.value = e?.message || 'Tải danh sách phim thất bại'
    } finally {
      loading.value = false
    }
  }

  async function ensureLoaded() {
    if (movies.value.length === 0 && !loading.value) {
      await refresh()
    }
  }

  // Trả về tên phim từ:
  // - movie object (có thể có movie_id, movie, movie_title...)
  // - movie_id (số hoặc chuỗi)
  // - title (chuỗi)
  function getMovieTitle(any) {
    // 1) Nếu là object trả về trực tiếp tên nếu đã có
    if (any && typeof any === 'object') {
      const direct =
        (any.movie && String(any.movie).trim()) ||
        (any.movie_title && String(any.movie_title).trim()) ||
        (any.label && String(any.label).trim())
      if (direct) return direct

      const id = (any.movie_id ?? '').toString()
      if (id && idToName.value[id]) return idToName.value[id]
    }

    // 2) Nếu là số/chuỗi id
    const s = (any ?? '').toString().trim()
    if (!s) return ''
    if (idToName.value[s]) return idToName.value[s]

    // 3) Có thể chính là title
    return s
  }

  // Trả về id từ title (nếu có), else ''
  function getMovieIdByTitle(title) {
    const t = (title ?? '').toString().trim()
    return nameToId.value[t] ?? ''
  }

  return {
    // state
    loading,
    error,
    movies,
    movieCount,

    // maps
    idToName,
    nameToId,

    // actions
    refresh,
    ensureLoaded,

    // helpers
    getMovieTitle,
    getMovieIdByTitle,
  }
}

export default function useMovieCatalog() {
  if (!_singleton) _singleton = createCatalog()
  return _singleton
}
