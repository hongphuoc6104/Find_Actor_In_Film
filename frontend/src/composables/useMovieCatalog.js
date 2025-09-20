import { computed, reactive } from 'vue'
import axios from 'axios'

import { API_BASE_URL } from '../config.js'

const state = reactive({
  movies: [],
  isLoading: false,
  error: '',
  lastFetched: null,
})

const normaliseNumber = (value, fallback = 0) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const normaliseMovies = (payload) => {
  if (!Array.isArray(payload)) {
    return []
  }

  return payload.map((movie) => ({
    movie_id: String(movie?.movie_id ?? ''),
    movie: movie?.movie ?? movie?.title ?? null,
    character_count: normaliseNumber(movie?.character_count),
    scene_count: normaliseNumber(movie?.scene_count),
    preview_count: normaliseNumber(movie?.preview_count),
  }))
}

const fetchMovies = async () => {
  state.isLoading = true
  state.error = ''

  try {
    const { data } = await axios.get(`${API_BASE_URL}/movies`)
    state.movies = normaliseMovies(data)
    state.lastFetched = new Date().toISOString()
  } catch (error) {
    const responseMessage =
      error?.response?.data?.detail ??
      error?.response?.data?.message ??
      error?.message
    state.movies = []
    state.error = responseMessage ?? 'Không thể tải danh sách phim.'
  } finally {
    state.isLoading = false
  }
}

export const useMovieCatalog = () => ({
  movies: computed(() => state.movies),
  isLoading: computed(() => state.isLoading),
  error: computed(() => state.error),
  lastFetched: computed(() => state.lastFetched),
  fetchMovies,
})
