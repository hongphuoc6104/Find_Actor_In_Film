<template>
  <section class="management-page">
    <header class="management-page__header">
      <div>
        <h1>Quản lý phim</h1>
        <p>Theo dõi trạng thái xử lý và tải lên phim mới cho hệ thống nhận diện.</p>
      </div>
      <button type="button" class="management-page__refresh" @click="refreshMovies" :disabled="isLoading">
        Làm mới danh sách
      </button>
    </header>

    <section class="management-page__upload">
      <h2>Tải phim mới</h2>
      <form @submit.prevent="submitUpload" class="upload-form">
        <label>
          <span>Tệp video</span>
          <input ref="videoInput" type="file" accept="video/*" @change="onVideoChange" />
        </label>

        <div class="upload-form__grid">
          <label>
            <span>Mã phim (tùy chọn)</span>
            <input type="text" v-model="form.movieId" placeholder="movie_001" />
          </label>
          <label>
            <span>Nguồn</span>
            <input type="text" v-model="form.source" placeholder="Blu-ray rip" />
          </label>
        </div>

        <label class="upload-form__checkbox">
          <input type="checkbox" v-model="form.refresh" />
          <span>Làm mới dữ liệu nhận diện sau khi tải lên</span>
        </label>

        <button type="submit" :disabled="!form.file || isSubmitting">
          <span v-if="isSubmitting">Đang tải…</span>
          <span v-else>Gửi yêu cầu xử lý</span>
        </button>
      </form>

      <p v-if="uploadError" class="upload-form__error">{{ uploadError }}</p>
      <p v-else-if="uploadMessage" class="upload-form__info">{{ uploadMessage }}</p>
    </section>

    <section class="management-page__list">
      <header>
        <h2>Danh sách phim</h2>
        <p v-if="lastFetched" class="management-page__timestamp">Cập nhật lần cuối: {{ formatTimestamp(lastFetched) }}</p>
      </header>

      <div v-if="isLoading" class="management-page__placeholder">Đang tải danh sách phim…</div>
      <div v-else-if="error" class="management-page__placeholder management-page__placeholder--error">{{ error }}</div>
      <div v-else-if="!movies.length" class="management-page__placeholder">Chưa có phim nào trong hệ thống.</div>
      <table v-else class="management-page__table">
        <thead>
          <tr>
            <th>Tên phim</th>
            <th>Nhân vật</th>
            <th>Cảnh</th>
            <th>Ảnh preview</th>
            <th>Xác nhận</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="movie in movies" :key="movie.movie_id">
            <td>
              <div class="management-page__movie-name">
                <strong>{{ movie.movie || `Phim #${movie.movie_id}` }}</strong>
                <span class="management-page__movie-id">ID: {{ movie.movie_id }}</span>
              </div>
            </td>
            <td>{{ movie.character_count }}</td>
            <td>{{ movie.scene_count }}</td>
            <td>{{ movie.preview_count }}</td>
            <td>
              <span v-if="progress(movie.movie_id).total">
                {{ progress(movie.movie_id).confirmed }}/{{ progress(movie.movie_id).total }} đã xác nhận
              </span>
              <span v-else>Chưa xác nhận</span>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import axios from 'axios'

import { API_BASE_URL } from '../config.js'
import { useMovieCatalog } from '../composables/useMovieCatalog.js'
import { useRecognitionStore } from '../composables/useRecognitionStore.js'

const catalog = useMovieCatalog()
const recognitionStore = useRecognitionStore()

const movies = computed(() => catalog.movies.value)
const isLoading = computed(() => catalog.isLoading.value)
const error = computed(() => catalog.error.value)
const lastFetched = computed(() => catalog.lastFetched.value)

const progress = (movieId) => recognitionStore.movieProgress(movieId)

const videoInput = ref(null)

const form = reactive({
  file: null,
  movieId: '',
  source: '',
  refresh: false,
})

const isSubmitting = ref(false)
const uploadMessage = ref('')
const uploadError = ref('')

const onVideoChange = (event) => {
  const file = event.target.files && event.target.files[0]
  form.file = file ?? null
}

const submitUpload = async () => {
  if (!form.file) {
    uploadError.value = 'Vui lòng chọn tệp video để tải lên.'
    uploadMessage.value = ''
    return
  }

  const payload = new FormData()
  payload.append('video', form.file)
  if (form.movieId) {
    payload.append('movie_id', form.movieId)
  }
  if (form.source) {
    payload.append('source', form.source)
  }
  if (form.refresh) {
    payload.append('refresh', 'true')
  }

  isSubmitting.value = true
  uploadError.value = ''
  uploadMessage.value = ''

  try {
    const { data } = await axios.post(`${API_BASE_URL}/upload`, payload, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    uploadMessage.value =
      data?.detail ?? data?.message ?? 'Đã gửi yêu cầu xử lý phim thành công.'
    uploadError.value = ''
    form.file = null
    form.movieId = ''
    form.source = ''
    form.refresh = false
    if (videoInput.value) {
      videoInput.value.value = ''
    }
    await catalog.fetchMovies()
  } catch (error) {
    const responseMessage =
      error?.response?.data?.detail ??
      error?.response?.data?.message ??
      error?.message
    uploadError.value = responseMessage ?? 'Không thể tải video lên.'
    uploadMessage.value = ''
  } finally {
    isSubmitting.value = false
  }
}

const refreshMovies = () => {
  catalog.fetchMovies()
}

const formatTimestamp = (isoString) => {
  if (!isoString) {
    return 'Chưa xác định'
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

onMounted(() => {
  catalog.fetchMovies()
})
</script>

<style scoped>
.management-page {
  display: grid;
  gap: 2.5rem;
}

.management-page__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  flex-wrap: wrap;
}

.management-page__header h1 {
  margin: 0;
  font-size: clamp(1.75rem, 2.8vw, 2.3rem);
}

.management-page__header p {
  margin: 0.35rem 0 0;
  color: #475569;
  max-width: 38rem;
}

.management-page__refresh {
  border: none;
  border-radius: 999px;
  background: #2563eb;
  color: #f8fafc;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 10px 35px rgba(37, 99, 235, 0.35);
  transition: transform 120ms ease, box-shadow 120ms ease;
}

.management-page__refresh:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  box-shadow: none;
}

.management-page__refresh:not(:disabled):hover {
  transform: translateY(-1px);
}

.management-page__upload {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.75rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1.25rem;
}

.management-page__upload h2 {
  margin: 0;
}

.upload-form {
  display: grid;
  gap: 1rem;
}

.upload-form label {
  display: grid;
  gap: 0.5rem;
  font-weight: 600;
}

.upload-form input[type='file'],
.upload-form input[type='text'] {
  border: 1px solid #cbd5f5;
  border-radius: 0.75rem;
  padding: 0.6rem 0.75rem;
  font-size: 0.95rem;
}

.upload-form__grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}

.upload-form__checkbox {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
}

.upload-form button[type='submit'] {
  justify-self: start;
  border: none;
  border-radius: 999px;
  background: #16a34a;
  color: #f0fdf4;
  padding: 0.6rem 1.5rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease;
  box-shadow: 0 10px 35px rgba(22, 163, 74, 0.35);
}

.upload-form button[type='submit']:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  box-shadow: none;
}

.upload-form button[type='submit']:not(:disabled):hover {
  transform: translateY(-1px);
}

.upload-form__error {
  margin: 0;
  color: #b91c1c;
  font-weight: 600;
}

.upload-form__info {
  margin: 0;
  color: #2563eb;
}

.management-page__list {
  display: grid;
  gap: 1.5rem;
}

.management-page__list header {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: baseline;
}

.management-page__list h2 {
  margin: 0;
}

.management-page__timestamp {
  margin: 0;
  color: #475569;
  font-size: 0.9rem;
}

.management-page__placeholder {
  padding: 1.25rem 1rem;
  border-radius: 0.9rem;
  background: #ffffff;
  color: #475569;
  text-align: center;
  box-shadow: 0 15px 35px rgba(15, 23, 42, 0.06);
}

.management-page__placeholder--error {
  color: #b91c1c;
}

.management-page__table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border-radius: 1rem;
  overflow: hidden;
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
}

.management-page__table th,
.management-page__table td {
  padding: 1rem 1.1rem;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
  font-size: 0.95rem;
}

.management-page__table th {
  background: #f1f5f9;
  font-weight: 600;
  color: #1e293b;
}

.management-page__table tr:last-child td {
  border-bottom: none;
}

.management-page__movie-name {
  display: grid;
  gap: 0.35rem;
}

.management-page__movie-id {
  color: #475569;
  font-size: 0.85rem;
}

@media (max-width: 768px) {
  .management-page__table th,
  .management-page__table td {
    padding: 0.75rem;
  }
}

@media (max-width: 640px) {
  .management-page__upload {
    padding: 1.25rem;
  }
}
</style>
