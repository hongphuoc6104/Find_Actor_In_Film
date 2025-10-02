import { computed, reactive } from 'vue'
import axios from 'axios'

import { API_BASE_URL } from '../config.js'
import { toAbsoluteAssetUrl } from '../utils/assetUrls.js'
const defaultInfoMessage = 'Tải ảnh khuôn mặt để bắt đầu tìm kiếm.'
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

const createSceneCache = () => ({
  cursor: null,
  scene_index: null,
  next_cursor: null,
  entries: {},
  highlight_total: null,
  highlight_display_count: 0,
  total_scenes: null,
  has_more: false,
})

const applySceneEntryToCharacter = (
  movieId,
  characterId,
  entry,
  nextCursor,
  hasMore,
  totals = {},
) => {
  const movie = state.movies.find((item) => item.movie_id === movieId)
  if (!movie) {
    return
  }

  const character = movie.characters.find((item) => item.character_id === characterId)
  if (!character) {
    return
  }

  const resolvedNextCursor =
    nextCursor !== undefined ? nextCursor : entry?.next_cursor ?? null
  const resolvedHasMore =
    hasMore !== undefined && hasMore !== null
      ? Boolean(hasMore)
      : Boolean(
          resolvedNextCursor !== null && resolvedNextCursor !== undefined,
        )

  const resolvedTotalScenes =
    totals.total_scenes ?? entry?.total_scenes ?? null
  const resolvedHighlightTotal =
    totals.highlight_total ?? entry?.highlight_total ?? resolvedTotalScenes ?? 0
  const resolvedHighlightDisplay =
    totals.highlight_display_count ?? entry?.highlight_display_count ?? 0

  character.scene = entry?.scene ?? null
  character.scene_index =
    entry?.scene_index !== undefined && entry?.scene_index !== null
      ? entry.scene_index
      : null
  character.next_scene_cursor = resolvedNextCursor ?? null
  character.total_scenes = resolvedTotalScenes
  character.has_more_scenes = resolvedHasMore
  character.highlight_total = resolvedHighlightTotal
  character.highlight_display_count = resolvedHighlightDisplay
}

const resolveActiveSceneEntry = (cache) => {
  if (!cache || typeof cache !== 'object') {
    return null
  }

  const cursor = cache.cursor
  if (cursor === null || cursor === undefined) {
    return null
  }

  const entries = cache.entries
  if (!entries || typeof entries !== 'object') {
    return null
  }

  return entries[cursor] ?? null
}

const firstNonNull = (...values) => {
  for (const value of values) {
    if (value !== null && value !== undefined) {
      return value
    }
  }
  return null
}

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
  const cache = state.scenes[key]
  const entry = resolveActiveSceneEntry(cache)
  if (!entry) {
    return null
  }
  return entry
})

const currentScene = computed(() => {
  const entry = currentSceneEntry.value
  if (entry?.scene) {
    return entry.scene
  }
  return currentCharacter.value?.scene ?? null
})

const currentSceneNavigation = computed(() => {
  const movieId = state.selectedMovieId
  const characterId = state.selectedCharacterId

  const fallback = {
    index: null,
    total: null,
    displayCount: 0,
    knownCount: 0,
    hasMore: false,
  }

  if (!movieId || !characterId) {
    return fallback
  }

  const key = sceneKey(movieId, characterId)
  const cache = state.scenes[key]
  const entry = resolveActiveSceneEntry(cache)
  const character = currentCharacter.value

  const indexCandidates = [
    cache?.cursor,
    entry?.scene_index,
    character?.scene_index,
  ]

  let index = null
  for (const candidate of indexCandidates) {
    const parsed = parseSceneIndex(candidate)
    if (parsed !== null) {
      index = parsed
      break
    }
  }

  const knownFromEntries = computeKnownEntryCount(cache?.entries)
  const displayCandidate = firstNonNull(
    cache?.highlight_display_count,
    entry?.highlight_display_count,
    character?.highlight_display_count,
    knownFromEntries,
    0,
  )
  const displayCount = normaliseNumber(displayCandidate, 0)

  const totalValue = firstNonNull(
    cache?.total_scenes,
    cache?.highlight_total,
    entry?.total_scenes,
    entry?.highlight_total,
    character?.total_scenes,
    character?.highlight_total,
    null,
  )

  const total = (() => {
    const parsed = parseSceneIndex(totalValue)
    if (parsed === null) {
      return null
    }
    return parsed + 1
  })()

  const knownCount = Math.max(
    displayCount,
    knownFromEntries,
    index !== null ? index + 1 : 0,
  )

  return {
    index,
    total,
    displayCount,
    knownCount,
    hasMore: Boolean(firstNonNull(cache?.has_more, character?.has_more_scenes, false)),
  }
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


const parseSceneIndex = (value) => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null
  }
  return Math.trunc(parsed)
}

const computeKnownEntryCount = (entries) => {
  if (!entries || typeof entries !== 'object') {
    return 0
  }
  const indexes = Object.keys(entries)
    .map((key) => Number(key))
    .filter((value) => Number.isInteger(value) && value >= 0)
  if (!indexes.length) {
    return 0
  }
  return Math.max(...indexes) + 1
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

  const rawHighlights = Array.isArray(entry.highlights) ? entry.highlights : []
  copy.highlights = rawHighlights.map((item) =>
    item && typeof item === 'object' ? { ...item } : item,
  )


  if (typeof console !== 'undefined' && typeof console.debug === 'function') {

    try {
       console.debug('DEBUG_HL ensureFrameMetadata scene highlights', {
        highlightCount: copy.highlights.length,
        reportedTotal:
          entry?.highlight_total !== undefined ? entry.highlight_total : null,
        reportedDisplay:
          entry?.highlight_display_count !== undefined
            ? entry.highlight_display_count
            : null,
      })
    } catch (error) {
      // Ignore logging errors
    }
  }



  const highlightTotalValue =
    entry?.highlight_total !== undefined
      ? normaliseNumber(entry.highlight_total, null)
      : normaliseNumber(copy.highlights.length, null)
  copy.highlight_total = highlightTotalValue

  const highlightDisplayValue =
    entry?.highlight_display_count !== undefined
      ? normaliseNumber(entry.highlight_display_count, copy.highlights.length)
      : copy.highlights.length
  copy.highlight_display_count = highlightDisplayValue

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
    sceneEntry && sceneEntry.highlight_total !== null && sceneEntry.highlight_total !== undefined
      ? normaliseNumber(sceneEntry.highlight_total, null)

      : Array.isArray(sceneEntry?.highlights)
      ? sceneEntry.highlights.length
      : 0

  const highlightDisplayCount =
    sceneEntry && sceneEntry.highlight_display_count !== undefined
      ? normaliseNumber(
          sceneEntry.highlight_display_count,
          Array.isArray(sceneEntry?.highlights) ? sceneEntry.highlights.length : 0,
        )
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

  let sceneIndex =
    payload.scene_index !== undefined
      ? normaliseNumber(payload.scene_index, null)
      : null
  const nextCursor =
    payload.next_cursor !== undefined
      ? normaliseNumber(payload.next_cursor, null)
      : null

  let sceneData =
    payload?.scene && typeof payload.scene === 'object'
      ? ensureFrameMetadata(payload.scene)
      : payload?.scene ?? null


  if (
    sceneIndex === null &&
    sceneData &&
    typeof sceneData === 'object' &&
    sceneData.scene_index !== null &&
    sceneData.scene_index !== undefined
  ) {
    sceneIndex = normaliseNumber(sceneData.scene_index, null)
  }

  const key = sceneKey(movieId, characterId)
  const existingCache = state.scenes[key] ?? createSceneCache()

  const entries =
    existingCache.entries && typeof existingCache.entries === 'object'
      ? { ...existingCache.entries }
      : {}

  const previousEntry =
    sceneIndex !== null && sceneIndex !== undefined ? entries[sceneIndex] ?? null : null

  const freshScene =
    sceneData && typeof sceneData === 'object' ? sceneData : null
  const previousScene =
    previousEntry?.scene && typeof previousEntry.scene === 'object'
      ? previousEntry.scene
      : null
  const baseScene =
    freshScene ?? (previousScene ? { ...previousScene } : null)

  const sceneHighlights = Array.isArray(baseScene?.highlights)
    ? baseScene.highlights
    : []


  const highlightTotalFromScene =
    baseScene &&
    baseScene.highlight_total !== null &&
    baseScene.highlight_total !== undefined
      ? normaliseNumber(baseScene.highlight_total, null)
      : null

  const highlightDisplayFromScene =
    baseScene &&
    baseScene.highlight_display_count !== null &&
    baseScene.highlight_display_count !== undefined
      ? normaliseNumber(baseScene.highlight_display_count, sceneHighlights.length)
      : null

  const payloadTotalScenes =
    payload.total_scenes !== undefined
      ? normaliseNumber(payload.total_scenes, sceneHighlights.length)
      : null

  const payloadHighlightTotal =
    payload.highlight_total !== undefined
      ? normaliseNumber(payload.highlight_total, null)
      : null

  const payloadHighlightDisplay =
    payload.highlight_display_count !== undefined
      ? normaliseNumber(payload.highlight_display_count, null)
      : null


  const resolvedHighlightTotal = firstNonNull(
    highlightTotalFromScene,
    payloadHighlightTotal,
    normaliseNumber(previousEntry?.highlight_total, null),
    normaliseNumber(existingCache.highlight_total, null),
    payloadTotalScenes,
    sceneHighlights.length,
    0,
  )

  const resolvedHighlightDisplayCount = firstNonNull(
    highlightDisplayFromScene,
    payloadHighlightDisplay,
    normaliseNumber(previousEntry?.highlight_display_count, null),
    normaliseNumber(existingCache.highlight_display_count, null),
    sceneHighlights.length,
    0,
  )


  const resolvedTotalScenes = firstNonNull(
    payloadTotalScenes,
    resolvedHighlightTotal,
    normaliseNumber(existingCache.total_scenes, null),
  )

  if (baseScene && typeof baseScene === 'object') {
    if (sceneIndex !== null && sceneIndex !== undefined) {
      baseScene.scene_index = sceneIndex
    }
    baseScene.highlight_total = resolvedHighlightTotal
    baseScene.highlight_display_count = resolvedHighlightDisplayCount
  }

  const hasMore =
    payload.has_more !== undefined

      ? Boolean(payload.has_more)
      : nextCursor !== null && nextCursor !== undefined

  let activeEntry = previousEntry ?? null


  if (sceneIndex !== null && sceneIndex !== undefined) {
    const mergedEntry = {
      ...(previousEntry ?? {}),
      movie_id: movieId,
      character_id: characterId,
      scene_index: sceneIndex,
      scene: baseScene,
      next_cursor: nextCursor,
      highlight_total: resolvedHighlightTotal,
      highlight_display_count: resolvedHighlightDisplayCount,
      total_scenes: resolvedTotalScenes,

      has_more: hasMore,
    }
    entries[sceneIndex] = mergedEntry
    activeEntry = mergedEntry
  }

  const updatedCache = {
    ...existingCache,
    cursor:
      sceneIndex !== null && sceneIndex !== undefined
        ? sceneIndex
        : existingCache.cursor,
    scene_index:
      sceneIndex !== null && sceneIndex !== undefined
        ? sceneIndex
        : existingCache.scene_index ?? existingCache.cursor ?? null,
    next_cursor: nextCursor,
    entries,
    highlight_total:
      sceneIndex !== null && sceneIndex !== undefined
        ? resolvedHighlightTotal
        : firstNonNull(existingCache.highlight_total, resolvedHighlightTotal, 0),
    highlight_display_count:
      sceneIndex !== null && sceneIndex !== undefined
        ? resolvedHighlightDisplayCount
        : firstNonNull(
            existingCache.highlight_display_count,
            resolvedHighlightDisplayCount,
            0,
          ),
    total_scenes:
      sceneIndex !== null && sceneIndex !== undefined
        ? resolvedTotalScenes
        : firstNonNull(existingCache.total_scenes, resolvedTotalScenes, null),
    has_more: Boolean(hasMore),
  }

  state.scenes[key] = updatedCache

  const characterEntry =
    sceneIndex !== null && sceneIndex !== undefined
      ? activeEntry
      : resolveActiveSceneEntry(updatedCache)

  const finalHighlightTotal = firstNonNull(
    updatedCache.highlight_total,
    characterEntry?.highlight_total,
    0,
  )

  applySceneEntryToCharacter(
    movieId,
    characterId,
    characterEntry,
    updatedCache.next_cursor,
    updatedCache.has_more,
    {
      total_scenes: updatedCache.total_scenes ?? null,
      highlight_total: finalHighlightTotal,
      highlight_display_count: firstNonNull(
        updatedCache.highlight_display_count,
        characterEntry?.highlight_display_count,
        0,
      ),
    },
  )
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

      state.scenes[sceneKey(movieId, characterId)]?.cursor ??
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

  const cachedEntry =
    cached?.entries && typeof cached.entries === 'object'
      ? cached.entries[0] ?? null
      : null

  if (cachedEntry) {
    const baseCache = cached ?? createSceneCache()
    const entries =
      baseCache.entries && typeof baseCache.entries === 'object'
        ? { ...baseCache.entries }
        : {}
    entries[0] = cachedEntry

    const nextCursor = firstNonNull(
      cachedEntry.next_cursor,
      baseCache.next_cursor,
      null,
    )
    const resolvedHasMore =
      cachedEntry.has_more !== undefined && cachedEntry.has_more !== null
        ? Boolean(cachedEntry.has_more)
        : Boolean(nextCursor !== null && nextCursor !== undefined)

    const updatedCache = {
      ...baseCache,
      cursor: 0,
      next_cursor: nextCursor,
      entries,
      highlight_total: firstNonNull(
        cachedEntry.highlight_total,
        baseCache.highlight_total,
        0,
      ),
      highlight_display_count: firstNonNull(
        cachedEntry.highlight_display_count,
        baseCache.highlight_display_count,
        0,
      ),
      total_scenes: firstNonNull(
        cachedEntry.total_scenes,
        baseCache.total_scenes,
        null,
      ),
      has_more: resolvedHasMore,
    }

    state.scenes[key] = updatedCache

    applySceneEntryToCharacter(
      movieId,
      characterId,
      cachedEntry,
      updatedCache.next_cursor,
      updatedCache.has_more,
      {
        total_scenes: updatedCache.total_scenes ?? cachedEntry.total_scenes ?? null,
        highlight_total: firstNonNull(
          updatedCache.highlight_total,
          cachedEntry.highlight_total,
          0,
        ),
        highlight_display_count: firstNonNull(
          updatedCache.highlight_display_count,
          cachedEntry.highlight_display_count,
          0,
        ),
      },
    )

    return cachedEntry
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
    const payload = { ...data }
    if (payload.scene_index === undefined || payload.scene_index === null) {
      payload.scene_index = cursor
    }
    updateSceneEntry(payload)
    return payload
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
  const activeEntry = resolveActiveSceneEntry(cached)
  const nextCursor = firstNonNull(
    cached?.next_cursor,
    activeEntry?.next_cursor,
    character.next_scene_cursor,
  )

  if (nextCursor === null || nextCursor === undefined) {
    state.sceneError = 'Không còn cảnh khác cho nhân vật này.'
    return null
  }

  return loadScene(movie.movie_id, character.character_id, nextCursor)
}

const loadSceneAtIndex = async (movieId, characterId, targetIndex) => {
  const index = parseSceneIndex(targetIndex)
  if (!movieId || !characterId || index === null) {
    state.sceneError = 'Chỉ số highlight không hợp lệ.'
    state.isSceneLoading = false
    return null
  }

  const key = sceneKey(movieId, characterId)
  const cache = state.scenes[key] ?? createSceneCache()
  const entries =
    cache.entries && typeof cache.entries === 'object' ? cache.entries : {}
  const existingEntry = entries[index]

  if (existingEntry && typeof existingEntry === 'object') {
    const updatedCache = {
      ...cache,
      cursor: index,
      scene_index: index,
      entries,
    }
    state.scenes[key] = updatedCache

    const totals = {
      total_scenes: firstNonNull(
        updatedCache.total_scenes,
        existingEntry.total_scenes,
        existingEntry.highlight_total,
        null,
      ),
      highlight_total: firstNonNull(
        updatedCache.highlight_total,
        existingEntry.highlight_total,
        null,
      ),
      highlight_display_count: firstNonNull(
        updatedCache.highlight_display_count,
        existingEntry.highlight_display_count,
        null,
      ),
    }

    applySceneEntryToCharacter(
      movieId,
      characterId,
      existingEntry,
      updatedCache.next_cursor,
      updatedCache.has_more,
      totals,
    )

    state.sceneError = ''
    state.isSceneLoading = false
    return existingEntry
  }

  const navigation = currentSceneNavigation.value
  const knownMaxIndex = Math.max(navigation.knownCount - 1, 0)

  if (!cache.has_more && navigation.total !== null && index >= navigation.total) {
    state.sceneError = `Highlight #${index + 1} không tồn tại.`
    state.isSceneLoading = false
    return null
  }

  if (!cache.has_more && navigation.total === null && index > knownMaxIndex) {
    state.sceneError = 'Không còn highlight nào sau vị trí hiện tại.'
    state.isSceneLoading = false
    return null
  }

  return loadScene(movieId, characterId, index)
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
  currentSceneNavigation,
  overallProgress,
  movieProgress,
  hasPendingCharacters,
  recogniseFace,
  resetSearch,
  selectMovie,
  selectCharacter,
  ensureInitialScene,
  loadScene,
  loadSceneAtIndex,
  loadNextSceneForCurrent,
  applyDecision,
  updateSceneEntry,
})

export const __test__ = {
  normaliseCharacter,
  normaliseMovies,
  ensureFrameMetadata,
}