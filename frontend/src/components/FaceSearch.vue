<template>
  <section class="face-search">
    <div class="face-search__layout">
      <aside class="face-search__column face-search__column--upload">
        <form class="face-search__form" @submit.prevent="submitSearch">
          <div class="face-search__form-grid">
            <label class="face-search__file">
              <span>Ảnh khuôn mặt</span>
              <input ref="fileInput" type="file" accept="image/*" @change="onFileChange" />
            </label>
            <div class="face-search__form-actions">
              <button type="submit" :disabled="!selectedFile || isSearching">
                <span v-if="isSearching">Đang tìm…</span>
                <span v-else>Tìm kiếm</span>
              </button>
              <button
                type="button"
                class="secondary"
                @click="resetSearch"
                :disabled="isSearching && !selectedFile"
              >
                Đặt lại
              </button>
            </div>
          </div>

          <p v-if="selectedFile" class="face-search__file-name">{{ selectedFile.name }}</p>
        </form>

        <details v-if="previewUrl" class="face-search__collapse" open>
          <summary>Ảnh đã chọn</summary>
          <div class="face-search__preview">
            <img :src="previewUrl" alt="Xem trước ảnh khuôn mặt" />
          </div>
        </details>

        <div class="face-search__feedback">
          <p v-if="searchError" class="face-search__feedback-error">{{ searchError }}</p>
          <p v-else-if="searchInfo" class="face-search__feedback-info">{{ searchInfo }}</p>
        </div>
      </aside>

      <section class="face-search__column face-search__column--movies face-search__movies">
        <template v-if="movies.length">
          <header>
            <h2>Phim phù hợp</h2>
            <p class="face-search__progress" v-if="overallProgress.total">
              Đã duyệt {{ overallProgress.confirmed + overallProgress.rejected }}/{{ overallProgress.total }} nhân vật.
            </p>
          </header>

          <details class="face-search__movie-toggle" open>
            <summary>Danh sách phim</summary>
            <ul>
              <li v-for="movie in movies" :key="movie.movie_id">
                <button
                  type="button"
                  :class="['face-search__movie-button', { active: movie.movie_id === selectedMovieId }]"
                  @click="handleSelectMovie(movie.movie_id)"
                >
                  <span class="face-search__movie-title">{{ movie.movie || `Phim #${movie.movie_id}` }}</span>
                  <span
                    v-if="movie.match_label"
                    class="face-search__movie-label"
                    :data-match-status="movie.match_status || ''"
                  >
                    {{ movie.match_label }}
                  </span>
                  <span class="face-search__movie-score" v-if="movie.score !== null">
                    {{ formatScore(movie.score) }}
                  </span>
                  <span class="face-search__movie-progress">
                    {{ movieProgress(movie.movie_id).confirmed }}/{{
                      movieProgress(movie.movie_id).total || movie.characters.length
                    }} xác nhận
                  </span>
                </button>
              </li>
            </ul>
          </details>
        </template>
        <p v-else-if="hasSearched && !isSearching" class="face-search__no-results">
          Không có phim nào khớp với ảnh đã tải lên.
        </p>
        <p v-else class="face-search__placeholder">Tải ảnh khuôn mặt để xem các phim phù hợp.</p>
      </section>

      <section class="face-search__column face-search__column--viewer face-search__details">
        <template v-if="movies.length">
          <template v-if="currentMovie && currentCharacter">
            <header class="face-search__details-header">
              <div>
                <h2>{{ currentMovie.movie || `Phim #${currentMovie.movie_id}` }}</h2>
                <p class="face-search__character-label">
                  Nhân vật: <strong>{{ currentCharacter.character_id }}</strong>
                </p>
                <p
                  v-if="currentMovie.match_label"
                  class="face-search__movie-match"
                  :data-match-status="currentMovie.match_status || ''"
                >
                  {{ currentMovie.match_label }}
                </p>
              </div>
              <div class="face-search__character-meta">
                <span
                  v-if="currentCharacter.match_label"
                  class="face-search__character-match"
                  :data-match-status="currentCharacter.match_status || ''"
                >
                  {{ currentCharacter.match_label }}
                </span>
                <span v-if="currentCharacter.score !== null">Điểm: {{ formatScore(currentCharacter.score) }}</span>
                <span v-if="currentCharacter.count">Số lần xuất hiện: {{ currentCharacter.count }}</span>
              </div>
            </header>

            <div class="face-search__characters">
              <button
                v-for="character in currentMovie.characters"
                :key="character.character_id"
                type="button"
                :class="[
                  'face-search__character-button',
                  {
                    active: character.character_id === selectedCharacterId,
                    confirmed: character.verificationStatus === 'confirmed',
                    rejected: character.verificationStatus === 'rejected',
                  },
                ]"
                @click="handleSelectCharacter(currentMovie.movie_id, character.character_id)"
              >
                <span class="face-search__character-id">{{ character.character_id }}</span>
                <span
                  v-if="character.match_label"
                  class="face-search__character-badge"
                  :data-match-status="character.match_status || ''"
                >
                  {{ character.match_label }}
                </span>
                <span class="face-search__character-status" v-if="character.verificationStatus === 'confirmed'">✔</span>
                <span class="face-search__character-status" v-else-if="character.verificationStatus === 'rejected'">✖</span>
              </button>
            </div>

            <div class="face-search__viewer">
              <nav v-if="availableDetailTabs.length > 1" class="face-search__tabs" aria-label="Chi tiết cảnh">
                <button
                  v-for="tab in availableDetailTabs"
                  :key="tab.id"
                  type="button"
                  :class="['face-search__tab', { active: tab.id === activeDetailTab }]"
                  @click="activeDetailTab = tab.id"
                >
                  {{ tab.label }}
                </button>
              </nav>

              <div class="face-search__tab-panel">
                <div
                  v-if="activeDetailTab === 'scene' && hasHighlightNavigation"
                  class="face-search__highlight-nav"
                >
                  <button
                    type="button"
                    class="face-search__highlight-button"
                    @click="goToPreviousHighlight"
                    :disabled="!canGoPreviousHighlight"
                  >
                    ⟵ Trước
                  </button>
                  <span v-if="highlightLabel" class="face-search__highlight-count">
                    {{ highlightLabel }}
                  </span>
                  <button
                    type="button"
                    class="face-search__highlight-button"
                    @click="goToNextHighlight"
                    :disabled="!canGoNextHighlight"
                  >
                    Tiếp theo ⟶
                  </button>
                </div>
                <p
                  v-if="sceneStatusMessage && activeDetailTab === 'scene'"
                  :class="['face-search__highlight-status', { 'is-error': sceneError && !isSceneLoading }]"
                >
                  {{ sceneStatusMessage }}
                </p>
                <SceneViewer
                  v-if="activeDetailTab === 'scene'"
                  :scene="currentScene"
                  :meta="currentSceneEntry"
                  :movie-title="currentMovie.movie || `Phim #${currentMovie.movie_id}`"
                  :character-id="currentCharacter.character_id"
                  :is-loading="isSceneLoading"
                  :highlight-index="resolvedHighlightIndex"
                  :highlight-total="resolvedHighlightTotal"
                  @highlight-change="handleViewerHighlightChange"
                />

                <section v-else-if="activeDetailTab === 'history'" class="face-search__history">
                  <h3>Lịch sử xác nhận</h3>
                  <ul>
                    <li v-for="(entry, index) in currentCharacter.decisionHistory" :key="index">
                      <span class="face-search__history-status">{{ historyLabel(entry.status) }}</span>
                      <span class="face-search__history-meta">
                        lúc {{ formatTime(entry.at) }}
                        <template v-if="entry.scene_index !== null">— Cảnh #{{ entry.scene_index + 1 }}</template>
                      </span>
                    </li>
                  </ul>
                </section>

                <section v-else class="face-search__info">
                  <h3>Thông tin nhân vật</h3>
                  <dl>
                    <div v-for="item in detailInfo" :key="item.label">
                      <dt>{{ item.label }}</dt>
                      <dd>{{ item.value }}</dd>
                    </div>
                  </dl>
                </section>
              </div>
              <p v-if="sceneError && activeDetailTab === 'scene'" class="face-search__feedback-error">
                {{ sceneError }}
              </p>
            </div>


          </template>
          <p v-else class="face-search__placeholder">Chọn một phim để bắt đầu kiểm tra các cảnh.</p>
        </template>
        <p v-else-if="hasSearched && !isSearching" class="face-search__placeholder">
          Không có phim nào khớp với ảnh đã tải lên.
        </p>
        <p v-else class="face-search__placeholder">Tải ảnh khuôn mặt để xem chi tiết cảnh quay.</p>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'

import { useRecognitionStore } from '../composables/useRecognitionStore.js'
import SceneViewer from './SceneViewer.vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref('')

const store = useRecognitionStore()

const movies = computed(() => store.movies.value)
const isSearching = computed(() => store.isSearching.value)
const searchError = computed(() => store.searchError.value)
const searchInfo = computed(() => store.searchInfo.value)
const hasSearched = computed(() => store.hasSearched.value)
const selectedMovieId = computed(() => store.selectedMovieId.value)
const selectedCharacterId = computed(() => store.selectedCharacterId.value)
const isSceneLoading = computed(() => store.isSceneLoading.value)
const sceneError = computed(() => store.sceneError.value)
const currentMovie = computed(() => store.currentMovie.value)
const currentCharacter = computed(() => store.currentCharacter.value)
const currentScene = computed(() => store.currentScene.value)
const currentSceneEntry = computed(() => store.currentSceneEntry.value)
const overallProgress = computed(() => store.overallProgress.value)
const highlightNavigation = computed(() => store.currentSceneNavigation.value)

const highlightEventSnapshot = ref({ index: null, total: null })

const parseIndex = (value) => {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null
}

const parseCount = (value) => {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

const resolvedHighlightIndex = computed(() => {
  const navIndex = parseIndex(highlightNavigation.value.index)
  if (navIndex !== null) {
    return navIndex
  }
  return parseIndex(highlightEventSnapshot.value.index)
})

const resolvedHighlightTotal = computed(() => {
  const navTotal = parseCount(highlightNavigation.value.total)
  if (navTotal !== null) {
    return navTotal
  }
  const snapshotTotal = parseCount(highlightEventSnapshot.value.total)
  if (snapshotTotal !== null) {
    return snapshotTotal
  }
  const fallback = Math.max(
    highlightNavigation.value.displayCount ?? 0,
    highlightNavigation.value.knownCount ?? 0,
  )
  return parseCount(fallback)
})

const hasHighlightNavigation = computed(() => {
  const nav = highlightNavigation.value
  return (
    resolvedHighlightIndex.value !== null ||
    parseCount(nav.total) !== null ||
    (nav.displayCount ?? 0) > 0 ||
    (nav.knownCount ?? 0) > 0 ||
    parseCount(highlightEventSnapshot.value.total) !== null
  )
})

const highlightLabel = computed(() => {
  if (!hasHighlightNavigation.value) {
    return ''
  }
  const index = resolvedHighlightIndex.value
  const nav = highlightNavigation.value
  const total = resolvedHighlightTotal.value
  if (index === null) {
    if (total) {
      return `Highlight 0/${total}`
    }
    const fallbackCount = Math.max(
      nav.displayCount ?? 0,
      nav.knownCount ?? 0,
      parseCount(highlightEventSnapshot.value.total) ?? 0,
    )
    if (fallbackCount > 0) {
      return `Highlight 0/${fallbackCount}`
    }
    return 'Highlight'
  }

  const current = index + 1
  if (total) {
    return `Highlight ${current}/${total}`
  }

  const fallbackTotal = Math.max(nav.displayCount ?? 0, nav.knownCount ?? 0)
  if (nav.hasMore && fallbackTotal <= current) {
    return `Highlight ${current}+`
  }
  if (nav.hasMore) {
    return fallbackTotal > 0
      ? `Highlight ${current}/${fallbackTotal}+`
      : `Highlight ${current}+`
  }
  if (fallbackTotal > 0) {
    return `Highlight ${current}/${fallbackTotal}`
  }
  return `Highlight ${current}`
})

const canGoPreviousHighlight = computed(() => {
  const index = resolvedHighlightIndex.value
  return index !== null && index > 0 && !isSceneLoading.value
})

const canGoNextHighlight = computed(() => {
  if (!hasHighlightNavigation.value || isSceneLoading.value) {
    return false
  }
  const nav = highlightNavigation.value
  const index = resolvedHighlightIndex.value
  const total = resolvedHighlightTotal.value
  if (total) {
    if (index === null) {
      return total > 0
    }
    return index + 1 < total
  }
  if (nav.hasMore) {
    return true
  }
  const fallbackTotal = Math.max(nav.knownCount ?? 0, nav.displayCount ?? 0)
  if (fallbackTotal === 0) {
    return false
  }
  if (index === null) {
    return true
  }
  return index + 1 < fallbackTotal
})

const sceneStatusMessage = computed(() => {
  if (isSceneLoading.value) {
    return 'Đang tải highlight…'
  }
  if (sceneError.value) {
    return sceneError.value
  }
  return ''
})

const DETAIL_TAB_OPTIONS = [
  { id: 'scene', label: 'Khung cảnh' },
  { id: 'history', label: 'Lịch sử xác nhận' },
  { id: 'info', label: 'Thông tin' },
]

const activeDetailTab = ref('scene')

const availableDetailTabs = computed(() => {
  const character = currentCharacter.value
  return DETAIL_TAB_OPTIONS.filter((option) => {
    if (option.id === 'scene') {
      return true
    }
    if (option.id === 'history') {
      return Boolean(character?.decisionHistory?.length)
    }
    return Boolean(character)
  })
})

const movieProgress = (movieId) => store.movieProgress(movieId)

const cleanupPreview = () => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

const onFileChange = (event) => {
  const file = event.target.files && event.target.files[0]
  selectedFile.value = file ?? null
  cleanupPreview()
  if (file) {
    previewUrl.value = URL.createObjectURL(file)
  }
}

const resetSearch = () => {
  selectedFile.value = null
  cleanupPreview()
  if (fileInput.value) {
    fileInput.value.value = ''
  }
  store.resetSearch()
}

const submitSearch = async () => {
  if (!selectedFile.value) {
    return
  }
  await store.recogniseFace(selectedFile.value)
}

const handleSelectMovie = (movieId) => {
  store.selectMovie(movieId)
}

const handleSelectCharacter = (movieId, characterId) => {
  store.selectCharacter(movieId, characterId)
}


const goToHighlight = async (targetIndex) => {
  const movie = currentMovie.value
  const character = currentCharacter.value
  if (!movie || !character) {
    return
  }
  await store.loadSceneAtIndex(movie.movie_id, character.character_id, targetIndex)
}

const goToPreviousHighlight = async () => {
  if (!canGoPreviousHighlight.value) {
    return
  }
  const target = (resolvedHighlightIndex.value ?? 0) - 1
  await goToHighlight(target)
}

const goToNextHighlight = async () => {
  if (!canGoNextHighlight.value) {
    return
  }
  const current = resolvedHighlightIndex.value ?? -1
  await goToHighlight(current + 1)
}

const handleViewerHighlightChange = (payload) => {
  const index = parseIndex(payload?.index)
  const total = parseCount(payload?.total)
  highlightEventSnapshot.value = {
    index,
    total,
  }
}

const ensureScene = async () => {
  const movieId = store.selectedMovieId.value
  const characterId = store.selectedCharacterId.value
  if (!movieId || !characterId) {
    return
  }
  await store.ensureInitialScene(movieId, characterId)
}

const historyLabel = (status) => {
  if (status === 'confirmed') {
    return 'Đúng'
  }
  if (status === 'rejected') {
    return 'Không phải'
  }
  return status
}

const formatTime = (isoString) => {
  if (!isoString) {
    return 'không rõ thời gian'
  }
  try {
    const date = new Date(isoString)
    if (Number.isNaN(date.getTime())) {
      return isoString
    }
    return date.toLocaleString()
  } catch (error) {
    return isoString
  }
}

const formatScore = (value) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(3)
  }
  return value
}

const verificationLabel = (status) => {
  if (!status) {
    return 'Chưa xác định'
  }
  if (status === 'pending') {
    return 'Đang chờ xác nhận'
  }
  return historyLabel(status)
}

const detailInfo = computed(() => {
  const items = []
  const movie = currentMovie.value
  const character = currentCharacter.value

  if (movie) {
    items.push({ label: 'Phim', value: movie.movie || `Phim #${movie.movie_id}` })
    if (movie.match_label) {
      items.push({ label: 'Đánh giá phim', value: movie.match_label })
    }
    if (movie.score !== undefined && movie.score !== null) {
      items.push({ label: 'Điểm phim', value: formatScore(movie.score) })
    }
    const progressState = movieProgress(movie.movie_id)
    if (progressState.total) {
      items.push({
        label: 'Tiến độ xác nhận',
        value: `${progressState.confirmed}/${progressState.total} nhân vật`,
      })
    }
  }

  if (character) {
    items.push({ label: 'Nhân vật', value: character.character_id })
    if (character.match_label) {
      items.push({ label: 'Kết quả nhận diện', value: character.match_label })
    }
    if (character.score !== undefined && character.score !== null) {
      items.push({ label: 'Điểm giống nhau', value: formatScore(character.score) })
}
    if (character.count) {
      items.push({ label: 'Số lần xuất hiện', value: character.count })
    }
    if (character.verificationStatus) {
      items.push({
        label: 'Trạng thái kiểm tra',
        value: verificationLabel(character.verificationStatus),
      })
    }
  }

  if (overallProgress.value?.total) {
    items.push({
      label: 'Tổng số nhân vật đã duyệt',
      value: `${overallProgress.value.confirmed + overallProgress.value.rejected}/${overallProgress.value.total}`,
    })
  }

  return items
})

watch(
  () => [store.selectedMovieId.value, store.selectedCharacterId.value],
  async () => {
    highlightEventSnapshot.value = { index: null, total: null }
    await ensureScene()
  },
  { immediate: true },
)

watch(
  availableDetailTabs,
  (tabs) => {
    if (!tabs.length) {
      activeDetailTab.value = 'scene'
      return
    }
    if (!tabs.some((tab) => tab.id === activeDetailTab.value)) {
      activeDetailTab.value = tabs[0].id
    }
  },
  { immediate: true },
)

watch(
  () => store.selectedCharacterId.value,
  () => {
    activeDetailTab.value = 'scene'
  },
)

onBeforeUnmount(() => {
  cleanupPreview()
})
</script>

<style scoped>
.face-search {
  --surface-bg: #ffffff;
  --surface-border: #e2e8f0;
  --surface-radius: 0.75rem;
  --surface-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.face-search__layout {
  display: grid;
  grid-template-columns: clamp(300px, 24vw, 340px) minmax(0, 1fr) minmax(0, 1fr);
  gap: clamp(1.5rem, 3vw, 2.5rem);
  align-items: start;
}

.face-search__column {
  background: var(--surface-bg);
  border: 1px solid var(--surface-border);
  border-radius: var(--surface-radius);
  box-shadow: var(--surface-shadow);
  padding: clamp(1.25rem, 2vw, 1.75rem);
  display: grid;
  gap: clamp(1.25rem, 2vw, 1.75rem);
}

.face-search__column--upload {
  position: sticky;
  top: clamp(1rem, 2vw, 1.5rem);
  align-self: start;
}

.face-search__column--movies,
.face-search__column--viewer {
  align-self: start;
}

.face-search__column--viewer {
  gap: 1.5rem;
}

.face-search__form {
  display: grid;
  gap: 1rem;
}

.face-search__form-grid {
  display: grid;
  gap: 1rem;
}

.face-search__file {
  display: grid;
  gap: 0.5rem;
  font-weight: 600;
}

.face-search__file input[type='file'] {
  border: 1px solid var(--surface-border);
  border-radius: 0.65rem;
  padding: 0.6rem 0.75rem;
  font-size: 0.95rem;
  cursor: pointer;
  background: #f8fafc;
}

.face-search__form-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

button[type='submit'],
button.secondary {
  border-radius: 999px;
  font-weight: 600;
  padding: 0.6rem 1.5rem;
  border: 1px solid transparent;
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
}

button[type='submit'] {
  background: #2563eb;
  color: #f8fafc;
}

button[type='submit']:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

button.secondary {
  background: #e2e8f0;
  color: #1f2937;
}

.face-search__form-actions button:not(:disabled):hover {
  filter: brightness(0.98);
}

.face-search__file-name {
  margin: 0;
  font-size: 0.9rem;
  color: #475569;
}

.face-search__collapse {
  border: 1px solid var(--surface-border);
  border-radius: var(--surface-radius);
  background: #f8fafc;
  padding: 0.75rem 1rem;
}

.face-search__collapse summary {
  cursor: pointer;
  font-weight: 600;
  color: #1e293b;
  list-style: none;
}

.face-search__collapse[open] > summary {
  margin-bottom: 0.75rem;
}

.face-search__collapse summary::-webkit-details-marker {
  display: none;
}

.face-search__preview img {
  width: 100%;
  border-radius: 0.6rem;
  object-fit: cover;
}

.face-search__feedback p {
  margin: 0;
}

.face-search__feedback-error {
  color: #b91c1c;
  font-weight: 600;
}

.face-search__feedback-info {
  color: #1d4ed8;
}

.face-search__movies {
  display: grid;
  gap: 1.25rem;
}

.face-search__movies header {
  display: grid;
  gap: 0.25rem;
}

.face-search__movies h2 {
  margin: 0;
  font-size: 1.05rem;
}

.face-search__progress {
  margin: 0;
  font-size: 0.85rem;
  color: #475569;
}

.face-search__movie-toggle {
  border: 1px solid var(--surface-border);
  border-radius: calc(var(--surface-radius) - 0.15rem);
  background: #f8fafc;
  padding: 0.75rem 1rem;
}

.face-search__movie-toggle summary {
  list-style: none;
  font-weight: 600;
  cursor: pointer;
  color: #1e293b;
}

.face-search__movie-toggle summary::-webkit-details-marker {
  display: none;
}

.face-search__movie-toggle[open] > summary {
  margin-bottom: 0.75rem;
}

.face-search__movie-toggle ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.75rem;
}

.face-search__movie-button {
  width: 100%;
  text-align: left;
  background: #ffffff;
  border-radius: 0.65rem;
  padding: 0.85rem 1rem;
  display: grid;
  gap: 0.25rem;
  border: 1px solid #e2e8f0;
  font: inherit;
  cursor: pointer;
  transition: border-color 150ms ease, background 150ms ease;
}

.face-search__movie-button:hover {
  border-color: #94a3b8;
}

.face-search__movie-button.active {
  border-color: #2563eb;
  background: #e0ecff;
}

.face-search__movie-title {
  font-weight: 600;
}

.face-search__movie-label,
.face-search__movie-match,
.face-search__character-match {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: #0f766e;
}

.face-search__character-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: #0f766e;
}

[data-match-status='near_match'] {
  color: #b45309;
}

[data-match-status='possible'] {
  color: #2563eb;
}

.face-search__movie-score {
  font-size: 0.85rem;
  color: #2563eb;
}

.face-search__movie-progress {
  font-size: 0.8rem;
  color: #475569;
}

.face-search__details {
  display: grid;
  gap: 1.5rem;
}

.face-search__details-header {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}

.face-search__details-header h2 {
  margin: 0;
}

.face-search__character-label {
  margin: 0.35rem 0 0;
  color: #475569;
}

.face-search__character-meta {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  align-items: center;
  color: #334155;
  font-size: 0.9rem;
}

.face-search__characters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.face-search__character-button {
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  border: 1px solid #cbd5f5;
  background: #ffffff;
  cursor: pointer;
  transition: border-color 150ms ease, background 150ms ease;
  display: flex;
  gap: 0.35rem;
  align-items: center;
}

.face-search__character-button.active {
  border-color: #2563eb;
  background: #e0ecff;
}

.face-search__character-button.confirmed {
  border-color: #0ea5e9;
}

.face-search__character-button.rejected {
  border-color: #f97316;
}

.face-search__viewer {
  display: grid;
  gap: 1.25rem;
}

.face-search__highlight-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  background: #f8fafc;
  border: 1px solid var(--surface-border);
  border-radius: 0.75rem;
  padding: 0.5rem 0.75rem;
}

.face-search__highlight-button {
  border: 1px solid #cbd5f5;
  background: #ffffff;
  border-radius: 999px;
  padding: 0.4rem 0.95rem;
  font-weight: 600;
  color: #1e293b;
  cursor: pointer;
  transition: background 150ms ease, border-color 150ms ease, color 150ms ease;
}

.face-search__highlight-button:not(:disabled):hover {
  background: rgba(37, 99, 235, 0.08);
  border-color: #93c5fd;
  color: #1d4ed8;
}

.face-search__highlight-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.face-search__highlight-count {
  font-weight: 700;
  color: #1d4ed8;
  font-size: 0.95rem;
}

.face-search__highlight-status {
  margin: 0;
  font-size: 0.9rem;
  color: #1e293b;
}

.face-search__highlight-status.is-error {
  color: #dc2626;
}

.face-search__tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  background: #f8fafc;
  border-radius: 999px;
  padding: 0.25rem;
  border: 1px solid var(--surface-border);
}

.face-search__tab {
  border: none;
  background: transparent;
  border-radius: 999px;
  padding: 0.45rem 1rem;
  font-weight: 600;
  color: #1e293b;
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease;
}

.face-search__tab:hover {
  background: rgba(37, 99, 235, 0.12);
}
.face-search__info h3 {
  margin: 0;
}

.face-search__info dl {
  margin: 0;
  display: grid;
  gap: 0.75rem;
}

.face-search__info div {
  display: grid;
  grid-template-columns: minmax(120px, max-content) 1fr;
  gap: 0.75rem;
  align-items: baseline;
  font-size: 0.92rem;
}

.face-search__info dt {
  font-weight: 600;
  color: #1e293b;
}

face-search__info dd {
  margin: 0;
  color: #334155;
}

.face-search__placeholder,
.face-search__no-results {
  margin: 0;
  color: #475569;
  font-size: 1rem;
  background: #f8fafc;
  border: 1px solid var(--surface-border);
  border-radius: var(--surface-radius);
  padding: 1.25rem 1.5rem;
  text-align: center;
}

@media (max-width: 1280px) {
  .face-search__layout {
    grid-template-columns: clamp(260px, 32vw, 320px) minmax(0, 1fr);
    gap: clamp(1.25rem, 3vw, 2.25rem);
  }

  .face-search__column--viewer {
    grid-column: 1 / -1;
  }
}

@media (max-width: 960px) {
  .face-search__layout {
    grid-template-columns: 1fr;
  }

  .face-search__column--upload {
    position: static;
  }
}

@media (max-width: 640px) {
  .face-search__column {
    padding: 1.25rem;
  }

  .face-search__tabs {
    border-radius: 0.75rem;
  }


}
</style>
