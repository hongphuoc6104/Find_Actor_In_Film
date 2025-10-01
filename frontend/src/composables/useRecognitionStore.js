import { computed, reactive } from 'vue'
import axios from 'axios'

import { API_BASE_URL } from '../config.js'
import { toAbsoluteAssetUrl } from '../utils/assetUrls.js'
const defaultInfoMessage = 'Tải ảnh khuôn mặt để bắt đầu tìm kiếm.'
import { filterHighlights } from '../utils/highlights.js'
const state = reactive({
  movies: [],
  resultMeta: null,
  isSearching: false,
  searchError: '',
  searchInfo: defaultInfoMessage,
  hasSearched: false,
  selectedMovieId: null,
  selectedCharacterId: null,
  isSceneLoading: false,
  sceneError: '',
  scenes: {},
})

const sceneKey = (movieId, characterId) => `${movieId ?? ''}::${characterId ?? ''}`

const currentMovie = computed(() => {
  if (!state.selectedMovieId) {
    return null
  }
  return state.movies.find((movie) => movie.movie_id === state.selectedMovieId) ?? null
})

const currentCharacter = computed(() => {
  const movie = currentMovie.value
  if (!movie || !state.selectedCharacterId) {
    return null
  }
  return (
    movie.characters.find(
      (character) => character.character_id === state.selectedCharacterId,
    ) ?? null
  )
})

const currentSceneEntry = computed(() => {
  const key = sceneKey(state.selectedMovieId, state.selectedCharacterId)
  return state.scenes[key] ?? null
})

const currentScene = computed(() => {
  if (currentSceneEntry.value?.scene) {
    return currentSceneEntry.value.scene
  }
  return currentCharacter.value?.scene ?? null
})

const movieProgress = (movieId) => {
  const summary = { confirmed: 0, rejected: 0, pending: 0, total: 0 }
  if (!movieId) {
    return summary
  }

  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return summary
  }

  movie.characters.forEach((character) => {
    const status = character.verificationStatus
    if (status === 'confirmed') {
      summary.confirmed += 1
    } else if (status === 'rejected') {
      summary.rejected += 1
    } else {
      summary.pending += 1
    }
  })

  summary.total = summary.confirmed + summary.rejected + summary.pending
  return summary
}

const overallProgress = computed(() => {
  const aggregate = { confirmed: 0, rejected: 0, pending: 0, total: 0 }
  state.movies.forEach((movie) => {
    const summary = movieProgress(movie.movie_id)
    aggregate.confirmed += summary.confirmed
    aggregate.rejected += summary.rejected
    aggregate.pending += summary.pending
  })
  aggregate.total = aggregate.confirmed + aggregate.rejected + aggregate.pending
  return aggregate
})

const normaliseNumber = (value, fallback = 0) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const assetFieldKeys = [
  'frame',
  'frame_url',
  'framePath',
  'frame_path',
  'framePreview',
  'frame_preview',
  'frameUrl',
  'preview',
  'preview_url',
  'preview_path',
  'previewImage',
  'preview_image',
  'previewUrl',
  'previewPath',
  'image',
  'thumbnail',
  'clip',
  'clip_url',
  'clip_path',
  'clipUrl',
  'clipPath',
  'video',
  'video_url',
  'video_path',
  'videoUrl',
  'videoPath',
]


const ensureFrameMetadata = (entry) => {
  if (!entry || typeof entry !== 'object') {
    return entry ?? null
  }

  const copy = { ...entry }

  if (Array.isArray(entry.timeline)) {
    copy.timeline = entry.timeline.map((item) =>
      item && typeof item === 'object' ? ensureFrameMetadata(item) : item,
    )
  }
  // Giữ nguyên highlights từ BE
  const rawHighlights = Array.isArray(entry.highlights) ? entry.highlights : []
  copy.highlights = rawHighlights

  // Ưu tiên danh sách filtered_highlights từ backend/store nếu có, fallback sang tự lọc
  const highlightOptions = {
    support: entry?.highlight_support,
    settings: entry?.highlight_settings,
  }
  const filteredSource = Array.isArray(entry.filtered_highlights)
    ? entry.filtered_highlights
    : rawHighlights
  const filteredHighlightsResult = filterHighlights(filteredSource, highlightOptions)
  const filteredHighlights = filteredHighlightsResult.items ?? []
  copy.filtered_highlights = filteredHighlights

  // highlight_total chỉ lấy từ BE, không overwrite
  const highlightTotalValue =
    entry?.highlight_total !== undefined
      ? normaliseNumber(entry.highlight_total, null)
      : null
  copy.highlight_total = highlightTotalValue

  // highlight_display_count là số highlight sau lọc
  const filteredCount = filteredHighlightsResult.stats?.inCount ?? filteredHighlights.length
  copy.highlight_display_count = filteredCount
  if (filteredHighlightsResult.stats) {
    copy.highlight_filter_stats = filteredHighlightsResult.stats
  }


  if (entry.scene_index !== undefined) {
    copy.scene_index = normaliseNumber(entry.scene_index, null)
  }
  if (entry.highlight_index !== undefined) {
    copy.highlight_index = normaliseNumber(entry.highlight_index, null)
  }
  if (entry.source_scene_index !== undefined) {
    copy.source_scene_index = normaliseNumber(entry.source_scene_index, null)
  }

  const frameSources = [
    entry.frame_url,
    entry.frame,
    entry.frameUrl,
    entry.frame_path,
    entry.framePath,
  ]
  const frameUrl = frameSources.find((value) => typeof value === 'string' && value) || ''
  const absoluteFrameUrl = toAbsoluteAssetUrl(frameUrl)

  if (absoluteFrameUrl) {
    copy.frame_url = absoluteFrameUrl
    copy.frame = absoluteFrameUrl
  }

  const frameLabelSource =
    typeof entry.frame_name === 'string' && entry.frame_name
      ? entry.frame_name
      : frameUrl

  if (frameLabelSource) {
    const parts = String(frameLabelSource).split(/[\\/]/)
    copy.frame_name = parts[parts.length - 1] || frameLabelSource
  }
  assetFieldKeys.forEach((key) => {
    const value = copy[key] ?? entry[key]
    if (typeof value === 'string' && value) {
      copy[key] = toAbsoluteAssetUrl(value)
    }
  })

  return copy
}

const buildProgressMessage = () => {
  if (!state.movies.length) {
    return defaultInfoMessage
  }
  const progress = overallProgress.value
  if (progress.total === 0) {
    return 'Chọn một phim để xem các cảnh gợi ý.'
  }
  const reviewed = progress.confirmed + progress.rejected
  return `Đã duyệt ${reviewed}/${progress.total} nhân vật.`
}

const resetScenes = () => {
  state.scenes = {}
  state.sceneError = ''
}

const resetResults = () => {
  state.movies = []
  state.resultMeta = null
  state.selectedMovieId = null
  state.selectedCharacterId = null
  resetScenes()
}

const resetSearch = () => {
  resetResults()
  state.searchError = ''
  state.searchInfo = defaultInfoMessage
  state.hasSearched = false
}

const normaliseCharacter = (character) => {
  const rawNextCursor =
    character?.next_scene_cursor ?? character?.scene_cursor ?? null
  const nextCursor =
    rawNextCursor === null || rawNextCursor === undefined
      ? null
      : normaliseNumber(rawNextCursor, null)

  const repImage =
    character?.rep_image && typeof character?.rep_image === 'object'
      ? ensureFrameMetadata(character.rep_image)
      : character?.rep_image ?? null

  const previews = Array.isArray(character?.previews)
    ? character.previews.map((item) =>
        item && typeof item === 'object' ? ensureFrameMetadata(item) : item,
      )
    : []

  const sceneEntry =
    character?.scene && typeof character?.scene === 'object'
      ? ensureFrameMetadata(character.scene)
      : character?.scene ?? null

  const highlightTotal =
    sceneEntry && typeof sceneEntry.highlight_total === 'number'
      ? sceneEntry.highlight_total
      : null

  const highlightDisplayCount =
    sceneEntry && typeof sceneEntry.highlight_display_count === 'number'
      ? sceneEntry.highlight_display_count
      : Array.isArray(sceneEntry?.highlights)
      ? sceneEntry.highlights.length
      : 0

  const totalScenes =
    highlightTotal !== null
      ? highlightTotal
      : normaliseNumber(character?.total_scenes, null)


  return {
    movie_id: String(character?.movie_id ?? ''),
    movie: character?.movie ?? null,
    character_id: String(character?.character_id ?? ''),
    score: normaliseNumber(character?.score ?? character?.distance),
    distance: normaliseNumber(character?.distance ?? character?.score),
    count: normaliseNumber(character?.count),
    track_count: normaliseNumber(
      character?.track_count,
      normaliseNumber(character?.count),
    ),
    rep_image: repImage,
    previews,
    preview_paths: Array.isArray(character?.preview_paths)
      ? character.preview_paths
      : [],
    raw_cluster_ids: Array.isArray(character?.raw_cluster_ids)
      ? character.raw_cluster_ids
      : [],
    movies: Array.isArray(character?.movies) ? character.movies : [],
    scene: sceneEntry,
    scene_index:
      character?.scene_index !== undefined
        ? normaliseNumber(character.scene_index, null)
        : null,
    next_scene_cursor: nextCursor,
    total_scenes: totalScenes,
    highlight_total: highlightTotal ?? totalScenes ?? 0,
    highlight_display_count: highlightDisplayCount,
    has_more_scenes:
      character?.has_more_scenes ?? (nextCursor !== null && nextCursor !== undefined),
    verificationStatus: null,
    decisionHistory: [],
    match_status: character?.match_status ?? null,
    match_label: character?.match_label ?? null,
  }
}

const normaliseMovies = (payload) => {
  if (!payload) {
    return []
  }

  const rawMovies = Array.isArray(payload) ? payload : payload.movies
  if (!Array.isArray(rawMovies)) {
    return []
  }

  return rawMovies
    .map((movie) => {
      const characters = Array.isArray(movie?.characters)
        ? movie.characters.map(normaliseCharacter).filter((item) => item.character_id)
        : []

      return {
        movie_id: String(movie?.movie_id ?? ''),
        movie: movie?.movie ?? null,
        score: normaliseNumber(movie?.score),
        characters,
        match_status: movie?.match_status ?? null,
        match_label: movie?.match_label ?? null,
      }
    })
    .filter((movie) => movie.movie_id && movie.characters.length)
}

const updateSceneEntry = (payload) => {
  if (!payload) {
    return
  }

  const movieId = String(payload.movie_id ?? '')
  const characterId = String(payload.character_id ?? '')
  if (!movieId || !characterId) {
    return
  }

  const sceneData =
    payload?.scene && typeof payload.scene === 'object'
      ? ensureFrameMetadata(payload.scene)
      : payload?.scene ?? null

  const sceneHighlightTotal =
    sceneData && typeof sceneData.highlight_total === 'number'
      ? sceneData.highlight_total
      : null


  const sceneHighlightDisplayCount =
    sceneData && typeof sceneData.highlight_display_count === 'number'
      ? sceneData.highlight_display_count
      : Array.isArray(sceneData?.highlights)
      ? sceneData.highlights.length
      : 0


  if (
    sceneData &&
    typeof sceneData === 'object' &&
    payload?.scene_index !== null &&
    payload?.scene_index !== undefined
  ) {
    sceneData.scene_index = normaliseNumber(payload.scene_index, null)
  }


const entry = {
  movie_id: movieId,
  character_id: characterId,
  scene_index:
    payload.scene_index !== undefined
      ? normaliseNumber(payload.scene_index, null)
      : null,
  scene: sceneData,
  next_cursor:
    payload.next_cursor !== undefined
      ? normaliseNumber(payload.next_cursor, null)
      : null,
  highlight_total: sceneHighlightTotal ?? 0,
  highlight_display_count: sceneHighlightDisplayCount,
  total_scenes:
    sceneHighlightTotal !== null
      ? sceneHighlightTotal
      : payload.total_scenes !== undefined
      ? normaliseNumber(payload.total_scenes, null)
      : null,
  has_more:
    payload.has_more !== undefined
      ? payload.has_more
      : payload.next_cursor !== null && payload.next_cursor !== undefined,
  // ✅ copy thêm support/settings từ sceneData để SceneViewer dùng
  highlight_support:
    sceneData?.highlight_support && typeof sceneData.highlight_support === 'object'
      ? { ...sceneData.highlight_support }
      : null,
  highlight_settings:
    sceneData?.highlight_settings && typeof sceneData.highlight_settings === 'object'
      ? { ...sceneData.highlight_settings }
      : null,
}


  const key = sceneKey(movieId, characterId)
  state.scenes[key] = entry

  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return
  }

  const character = movie.characters.find(
    (item) => item.character_id === characterId,
  )
  if (!character) {
    return
  }

  character.scene = entry.scene
  character.scene_index = entry.scene_index
  character.next_scene_cursor = entry.next_cursor
  character.total_scenes = entry.total_scenes
  character.has_more_scenes = entry.has_more
  character.highlight_total = entry.highlight_total
  character.highlight_display_count = sceneHighlightDisplayCount
}

const selectMovie = (movieId) => {
  if (!movieId) {
    state.selectedMovieId = null
    state.selectedCharacterId = null
    return
  }

  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return
  }

  state.selectedMovieId = movieId
  const pendingCharacter = movie.characters.find(
    (item) => item.verificationStatus !== 'confirmed' && item.verificationStatus !== 'rejected',
  )
  const fallback = movie.characters[0]
  state.selectedCharacterId = pendingCharacter?.character_id ?? fallback?.character_id ?? null
  state.sceneError = ''
}

const selectCharacter = (movieId, characterId) => {
  if (!movieId || !characterId) {
    return
  }

  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return
  }

  const exists = movie.characters.some(
    (item) => item.character_id === characterId,
  )
  if (!exists) {
    return
  }

  state.selectedMovieId = movieId
  state.selectedCharacterId = characterId
  state.sceneError = ''
}

const advanceToNextCharacter = (currentMovieId, currentCharacterId) => {
  if (!currentMovieId || !currentCharacterId) {
    state.selectedCharacterId = null
    return false
  }

  const movie = state.movies.find((item) => item.movie_id === currentMovieId)
  if (!movie) {
    state.selectedCharacterId = null
    return false
  }

  const remaining = movie.characters.find(
    (character) =>
      character.character_id !== currentCharacterId &&
      character.verificationStatus !== 'confirmed' &&
      character.verificationStatus !== 'rejected',
  )

  if (remaining) {
    state.selectedCharacterId = remaining.character_id
    return true
  }

  const nextMovie = state.movies.find(
    (item) =>
      item.movie_id !== currentMovieId &&
      item.characters.some(
        (character) =>
          character.verificationStatus !== 'confirmed' &&
          character.verificationStatus !== 'rejected',
      ),
  )

  if (nextMovie) {
    state.selectedMovieId = nextMovie.movie_id
    const nextCharacter = nextMovie.characters.find(
      (character) =>
        character.verificationStatus !== 'confirmed' &&
        character.verificationStatus !== 'rejected',
    )
    state.selectedCharacterId = nextCharacter?.character_id ?? null
    return true
  }

  state.selectedCharacterId = null
  return false
}

const recordDecision = (movieId, characterId, status) => {
  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return
  }

  const character = movie.characters.find(
    (item) => item.character_id === characterId,
  )
  if (!character) {
    return
  }

  character.verificationStatus = status
  const historyEntry = {
    at: new Date().toISOString(),
    status,
    scene_index:
      state.scenes[sceneKey(movieId, characterId)]?.scene_index ??
      character.scene_index ??
      null,
  }

  if (Array.isArray(character.decisionHistory)) {
    character.decisionHistory = [...character.decisionHistory, historyEntry]
  } else {
    character.decisionHistory = [historyEntry]
  }

  state.searchInfo = buildProgressMessage()
}

const applyDecision = (status) => {
  const movie = currentMovie.value
  const character = currentCharacter.value
  if (!movie || !character) {
    return false
  }

  recordDecision(movie.movie_id, character.character_id, status)
  const advanced = advanceToNextCharacter(movie.movie_id, character.character_id)
  if (!advanced) {
    state.sceneError = ''
  }
  return advanced
}

const ensureInitialScene = async (movieId, characterId) => {
  if (!movieId || !characterId) {
    return null
  }

  const key = sceneKey(movieId, characterId)
  const cached = state.scenes[key]
  if (cached && cached.scene_index === 0) {
    return cached
  }

  return loadScene(movieId, characterId, 0)
}

const loadScene = async (movieId, characterId, cursor = 0) => {
  if (!movieId || !characterId) {
    return null
  }

  state.isSceneLoading = true
  state.sceneError = ''

  try {
    const { data } = await axios.post(`${API_BASE_URL}/scene`, {
      movie_id: movieId,
      character_id: characterId,
      cursor,
    })
    updateSceneEntry(data)
    return data
  } catch (error) {
    const responseMessage =
      error?.response?.data?.detail ??
      error?.response?.data?.message ??
      error?.message
    state.sceneError =
      responseMessage ?? 'Không thể tải cảnh cho nhân vật đã chọn.'
    return null
  } finally {
    state.isSceneLoading = false
  }
}

const loadNextSceneForCurrent = async () => {
  const movie = currentMovie.value
  const character = currentCharacter.value
  if (!movie || !character) {
    return null
  }

  const key = sceneKey(movie.movie_id, character.character_id)
  const cached = state.scenes[key]
  const cursor =
    cached?.next_cursor ??
    cached?.scene_index === 0
      ? cached?.next_cursor
      : character.next_scene_cursor

  if (cursor === null || cursor === undefined) {
    state.sceneError = 'Không còn cảnh khác cho nhân vật này.'
    return null
  }

  return loadScene(movie.movie_id, character.character_id, cursor)
}

const recogniseFace = async (file) => {
  if (!file) {
    state.searchError = 'Vui lòng chọn một tệp ảnh để tìm kiếm.'
    return
  }

  state.isSearching = true
  state.searchError = ''
  state.searchInfo = ''
  state.hasSearched = true

  const formData = new FormData()
  formData.append('image', file)

  try {
    const { data } = await axios.post(`${API_BASE_URL}/recognize`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })

    const movies = normaliseMovies(data)
    state.movies = movies
    state.resultMeta = data ?? null
    resetScenes()

    if (movies.length) {
      const firstMovie = movies[0]
      state.selectedMovieId = firstMovie.movie_id
      state.selectedCharacterId = firstMovie.characters[0]?.character_id ?? null
      state.searchInfo = 'Chọn một phim để duyệt các cảnh nhận diện.'
    } else {
      state.selectedMovieId = null
      state.selectedCharacterId = null
      state.searchInfo = 'Không tìm thấy phim phù hợp với khuôn mặt này.'
    }
  } catch (error) {
    const responseMessage =
      error?.response?.data?.detail ??
      error?.response?.data?.message ??
      error?.message
    state.searchError =
      responseMessage ?? 'Đã xảy ra lỗi khi liên hệ dịch vụ nhận diện.'
    resetResults()
  } finally {
    state.isSearching = false
    state.sceneError = ''
    state.searchInfo = state.searchInfo || buildProgressMessage()
  }
}

const hasPendingCharacters = computed(() => {
  const progress = overallProgress.value
  return progress.pending > 0
})

export const useRecognitionStore = () => ({
  state,
  movies: computed(() => state.movies),
  isSearching: computed(() => state.isSearching),
  searchError: computed(() => state.searchError),
  searchInfo: computed(() => state.searchInfo),
  hasSearched: computed(() => state.hasSearched),
  selectedMovieId: computed(() => state.selectedMovieId),
  selectedCharacterId: computed(() => state.selectedCharacterId),
  isSceneLoading: computed(() => state.isSceneLoading),
  sceneError: computed(() => state.sceneError),
  currentMovie,
  currentCharacter,
  currentScene,
  currentSceneEntry,
  overallProgress,
  movieProgress,
  hasPendingCharacters,
  recogniseFace,
  resetSearch,
  selectMovie,
  selectCharacter,
  ensureInitialScene,
  loadScene,
  loadNextSceneForCurrent,
  applyDecision,
  updateSceneEntry,
})

export const __test__ = {
  normaliseCharacter,
  normaliseMovies,
  ensureFrameMetadata,
}