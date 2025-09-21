<template>
  <section class="face-search">
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
          <button type="button" class="secondary" @click="resetSearch" :disabled="isSearching && !selectedFile">
            Đặt lại
          </button>
        </div>
      </div>

      <p v-if="selectedFile" class="face-search__file-name">{{ selectedFile.name }}</p>
    </form>

    <div v-if="previewUrl" class="face-search__preview">
      <h2>Ảnh đã chọn</h2>
      <img :src="previewUrl" alt="Xem trước ảnh khuôn mặt" />
    </div>

    <div class="face-search__feedback">
      <p v-if="searchError" class="face-search__feedback-error">{{ searchError }}</p>
      <p v-else-if="searchInfo" class="face-search__feedback-info">{{ searchInfo }}</p>
    </div>

    <section v-if="movies.length" class="face-search__results">
      <aside class="face-search__movies">
        <header>
          <h2>Phim phù hợp</h2>
          <p class="face-search__progress" v-if="overallProgress.total">
            Đã duyệt {{ overallProgress.confirmed + overallProgress.rejected }}/{{ overallProgress.total }} nhân vật.
          </p>
        </header>

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
              <span class="face-search__movie-score" v-if="movie.score !== null">{{ formatScore(movie.score) }}</span>
              <span class="face-search__movie-progress">
                {{ movieProgress(movie.movie_id).confirmed }}/{{ movieProgress(movie.movie_id).total || movie.characters.length }} xác nhận
              </span>
            </button>
          </li>
        </ul>
      </aside>

      <section class="face-search__details">
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
              :class="['face-search__character-button', { active: character.character_id === selectedCharacterId, confirmed: character.verificationStatus === 'confirmed', rejected: character.verificationStatus === 'rejected' }]"
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

          <SceneViewer
            :scene="currentScene"
            :meta="currentSceneEntry"
            :movie-title="currentMovie.movie || `Phim #${currentMovie.movie_id}`"
            :character-id="currentCharacter.character_id"
            :is-loading="isSceneLoading"
          />

          <p v-if="sceneError" class="face-search__feedback-error">{{ sceneError }}</p>

          <footer class="face-search__actions">
            <button type="button" class="confirm" @click="handleDecision('confirmed')" :disabled="!currentCharacter">
              Đúng
            </button>
            <button type="button" class="reject" @click="handleDecision('rejected')" :disabled="!currentCharacter">
              Không phải
            </button>
            <button type="button" class="secondary" @click="loadAnotherScene" :disabled="isSceneLoading || !canLoadAnotherScene">
              Cảnh khác
            </button>
          </footer>

          <section v-if="currentCharacter.decisionHistory?.length" class="face-search__history">
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
        </template>
        <p v-else class="face-search__placeholder">Chọn một phim để bắt đầu kiểm tra các cảnh.</p>
      </section>
    </section>

    <p v-else-if="hasSearched && !isSearching" class="face-search__no-results">
      Không có phim nào khớp với ảnh đã tải lên.
    </p>
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

const movieProgress = (movieId) => store.movieProgress(movieId)

const canLoadAnotherScene = computed(() => {
  const meta = currentSceneEntry.value
  const character = currentCharacter.value
  if (!meta && !character) {
    return false
  }
  const nextCursor = meta?.next_cursor ?? character?.next_scene_cursor
  return nextCursor !== null && nextCursor !== undefined
})

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

const loadAnotherScene = async () => {
  await store.loadNextSceneForCurrent()
}

const handleDecision = async (status) => {
  const advanced = store.applyDecision(status)
  if (advanced) {
    await ensureScene()
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

watch(
  () => [store.selectedMovieId.value, store.selectedCharacterId.value],
  async () => {
    await ensureScene()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  cleanupPreview()
})
</script>

<style scoped>
.face-search {
  display: grid;
  gap: 2rem;
}

.face-search__form {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1rem;
}

.face-search__form-grid {
  display: grid;
  gap: 1.25rem;
}

.face-search__file {
  display: grid;
  gap: 0.5rem;
  font-weight: 600;
}

.face-search__file input[type='file'] {
  padding: 0.5rem;
  border-radius: 0.75rem;
  border: 1px solid #cbd5f5;
  font-size: 0.95rem;
  cursor: pointer;
}

.face-search__form-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

button[type='submit'],
button.secondary,
.face-search__actions button {
  border-radius: 999px;
  font-weight: 600;
  padding: 0.6rem 1.5rem;
  border: none;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease;
}

button[type='submit'] {
  background: #2563eb;
  color: #f8fafc;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.35);
}

button[type='submit']:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  box-shadow: none;
}

button.secondary {
  background: #e2e8f0;
  color: #1e293b;
}

.face-search__actions button {
  min-width: 128px;
}

.face-search__actions button.confirm {
  background: #22c55e;
  color: #0f172a;
}

.face-search__actions button.reject {
  background: #ef4444;
  color: #fff7ed;
}

.face-search__form-actions button:not(:disabled):hover,
.face-search__actions button:not(:disabled):hover {
  transform: translateY(-1px);
}

.face-search__file-name {
  margin: 0;
  color: #475569;
  font-size: 0.9rem;
}

.face-search__preview {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1rem;
  max-width: 360px;
}

.face-search__preview h2 {
  margin: 0;
  font-size: 1.1rem;
}

.face-search__preview img {
  width: 100%;
  border-radius: 0.75rem;
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

.face-search__results {
  display: grid;
  grid-template-columns: minmax(220px, 260px) 1fr;
  gap: 2rem;
  align-items: start;
}

.face-search__movies {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1rem;
}

.face-search__movies header {
  display: grid;
  gap: 0.25rem;
}

.face-search__movies h2 {
  margin: 0;
  font-size: 1.15rem;
}

.face-search__progress {
  margin: 0;
  font-size: 0.9rem;
  color: #475569;
}

.face-search__movies ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 0.75rem;
}

.face-search__movie-button {
  width: 100%;
  text-align: left;
  background: #f8fafc;
  border-radius: 0.9rem;
  padding: 0.85rem 1rem;
  display: grid;
  gap: 0.25rem;
  border: 1px solid transparent;
  font: inherit;
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease;
}

.face-search__movie-button.active {
  background: #e0ecff;
  border-color: #2563eb;
}

.face-search__movie-button:hover {
  border-color: #94a3b8;
}

.face-search__movie-title {
  font-weight: 600;
}

.face-search__movie-label,
.face-search__movie-match,
.face-search__character-match {
  display: block;
  font-size: 0.85rem;
  font-weight: 600;
  color: #0f766e;
}

.face-search__character-badge {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  margin-top: 0.15rem;
  color: #0f766e;
}

.face-search__movie-label[data-match-status='near_match'],
.face-search__movie-match[data-match-status='near_match'],
.face-search__character-match[data-match-status='near_match'],
.face-search__character-badge[data-match-status='near_match'] {
  color: #b45309;
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
  margin: 0;
  color: #475569;
}

.face-search__character-meta {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: center;
  color: #334155;
}

.face-search__characters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.face-search__character-button {
  padding: 0.5rem 0.9rem;
  border-radius: 999px;
  border: 1px solid #cbd5f5;
  background: #ffffff;
  cursor: pointer;
  transition: transform 120ms ease, border-color 120ms ease;
  display: flex;
  gap: 0.35rem;
  align-items: center;
}

.face-search__character-button.active {
  border-color: #2563eb;
  background: #e0ecff;
}

.face-search__character-button.confirmed {
  border-color: #22c55e;
}

.face-search__character-button.rejected {
  border-color: #ef4444;
}

.face-search__character-button:hover {
  transform: translateY(-1px);
}

.face-search__character-id {
  font-weight: 600;
}

.face-search__character-status {
  font-size: 0.9rem;
}

.face-search__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.face-search__history {
  display: grid;
  gap: 0.75rem;
}

.face-search__history h3 {
  margin: 0;
  font-size: 1.05rem;
}

.face-search__history ul {
  margin: 0;
  padding-left: 1.1rem;
  display: grid;
  gap: 0.35rem;
}

.face-search__history-status {
  font-weight: 600;
}

.face-search__history-meta {
  color: #475569;
  margin-left: 0.35rem;
}

.face-search__placeholder,
.face-search__no-results {
  margin: 0;
  color: #475569;
  font-size: 1rem;
}

@media (max-width: 960px) {
  .face-search__results {
    grid-template-columns: 1fr;
  }

  .face-search__movies {
    position: sticky;
    top: 1rem;
  }
}

@media (max-width: 640px) {
  .face-search__form,
  .face-search__preview,
  .face-search__movies {
    padding: 1.25rem;
  }

  .face-search__actions button {
    flex: 1 1 auto;
  }
}
</style>
