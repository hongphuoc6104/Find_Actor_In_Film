<template>
  <section class="face-search">
    <form class="upload" @submit.prevent="submitSearch">
      <label class="file-picker" for="face-image">
        <span class="file-picker__label">Face image</span>
        <input
          id="face-image"
          ref="fileInput"
          type="file"
          name="face"
          accept="image/*"
          @change="onFileChange"
        />
      </label>

      <div class="actions">
        <button type="submit" :disabled="!selectedFile || isLoading">
          <span v-if="isLoading">Searching…</span>
          <span v-else>Search</span>
        </button>
        <button type="button" class="secondary" @click="resetForm" :disabled="isLoading && !selectedFile">
          Clear
        </button>
      </div>

      <p v-if="selectedFile" class="file-name">{{ selectedFile.name }}</p>
    </form>

    <div v-if="previewUrl" class="upload-preview">
      <h2>Preview</h2>
      <img :src="previewUrl" alt="Selected face preview" />
    </div>

    <p v-if="errorMessage" class="feedback error">{{ errorMessage }}</p>
    <p v-else-if="infoMessage" class="feedback info">{{ infoMessage }}</p>

    <section v-if="candidates.length" class="results">
      <h2>Matches</h2>
      <div class="candidate-grid">
        <article v-for="candidate in candidates" :key="candidate.character_id ?? candidate.id" class="candidate-card">
          <div class="candidate-card__media" v-if="candidateImage(candidate)">
            <img :src="candidateImage(candidate)" :alt="candidateAlt(candidate)" />
          </div>
          <div class="candidate-card__body">
            <h3>
              {{ candidate.character_id ?? candidate.id ?? 'Unknown character' }}
            </h3>
            <p class="distance" v-if="candidate.distance !== undefined">
              Distance: <strong>{{ formatDistance(candidate.distance) }}</strong>
            </p>
            <div class="movies" v-if="formatMovies(candidate.movies).length">
              <h4>Movies</h4>
              <ul>
                <li v-for="(movie, index) in formatMovies(candidate.movies)" :key="index">{{ movie }}</li>
              </ul>
            </div>
          </div>
        </article>
      </div>
    </section>

    <p v-else-if="hasSearched && !isLoading && !errorMessage" class="feedback subtle">
      No matches were returned for this image.
    </p>
  </section>
</template>

<script setup>
import { onBeforeUnmount, ref } from 'vue'
import axios from 'axios'
import { API_BASE_URL } from '../config.js'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref('')
const isLoading = ref(false)
const candidates = ref([])
const hasSearched = ref(false)
const errorMessage = ref('')
const infoMessage = ref('Upload a clear face image to start searching.')

const cleanupPreview = () => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

const resetForm = () => {
  selectedFile.value = null
  cleanupPreview()
  candidates.value = []
  hasSearched.value = false
  errorMessage.value = ''
  infoMessage.value = 'Upload a clear face image to start searching.'
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const onFileChange = (event) => {
  const file = event.target.files && event.target.files[0]
  selectedFile.value = file ?? null
  cleanupPreview()
  if (file) {
    previewUrl.value = URL.createObjectURL(file)
    infoMessage.value = 'Ready to search. Submit to see matching characters.'
    errorMessage.value = ''
  } else {
    infoMessage.value = 'Upload a clear face image to start searching.'
  }
}

const candidateImage = (candidate) => {
  const possible = candidate?.preview_image ?? candidate?.image_url ?? candidate?.image
  if (!possible || typeof possible !== 'string') {
    return ''
  }
  if (possible.startsWith('data:') || possible.startsWith('http://') || possible.startsWith('https://')) {
    return possible
  }
  return `data:image/jpeg;base64,${possible}`
}

const candidateAlt = (candidate) => {
  const identifier = candidate?.character_id ?? candidate?.id ?? 'candidate'
  return `Preview for ${identifier}`
}

const formatMovies = (movies) => {
  if (!movies) {
    return []
  }
  if (Array.isArray(movies)) {
    return movies.map((movie) => {
      if (movie == null) return ''
      if (typeof movie === 'string') return movie
      if (typeof movie === 'object') {
        return movie.title ?? movie.name ?? JSON.stringify(movie)
      }
      return String(movie)
    }).filter(Boolean)
  }
  if (typeof movies === 'string') {
    return [movies]
  }
  if (typeof movies === 'object') {
    return Object.values(movies)
      .map((value) => (typeof value === 'string' ? value : ''))
      .filter(Boolean)
  }
  return [String(movies)]
}

const formatDistance = (distance) => {
  if (typeof distance === 'number' && Number.isFinite(distance)) {
    return distance.toFixed(4)
  }
  return distance
}

const submitSearch = async () => {
  if (!selectedFile.value) {
    errorMessage.value = 'Please select an image file before searching.'
    return
  }

  isLoading.value = true
  errorMessage.value = ''
  infoMessage.value = ''
  hasSearched.value = false

  const formData = new FormData()
  formData.append('image', selectedFile.value)

  try {
    const { data } = await axios.post(`${API_BASE_URL}/recognize`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })

    let payload = []
    if (Array.isArray(data)) {
      payload = data
    } else if (Array.isArray(data?.candidates)) {
      payload = data.candidates
    } else if (Array.isArray(data?.results)) {
      payload = data.results
    } else if (data) {
      payload = [data]
    }

    candidates.value = payload
    hasSearched.value = true

    if (!payload.length) {
      infoMessage.value = ''
    }
  } catch (error) {
    const responseMessage = error?.response?.data?.detail ?? error?.response?.data?.message
    errorMessage.value = responseMessage ?? error?.message ?? 'Unexpected error while contacting the recognition service.'
    candidates.value = []
    hasSearched.value = true
  } finally {
    isLoading.value = false
  }
}

onBeforeUnmount(() => {
  cleanupPreview()
})

</script>

<style scoped>
.face-search {
  display: grid;
  gap: 2rem;
}

.upload {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1rem;
}

.file-picker {
  display: grid;
  gap: 0.5rem;
}

.file-picker__label {
  font-weight: 600;
}

.file-picker input[type='file'] {
  font-size: 0.95rem;
  padding: 0.35rem;
  border-radius: 0.65rem;
  border: 1px solid #cbd5f5;
  cursor: pointer;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

button[type='submit'],
button.secondary {
  border-radius: 999px;
  font-weight: 600;
  padding: 0.6rem 1.4rem;
  border: none;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease;
}

button[type='submit'] {
  background: #2563eb;
  color: #f8fafc;
  box-shadow: 0 10px 35px rgba(37, 99, 235, 0.35);
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

button:not(:disabled):hover {
  transform: translateY(-1px);
}

.file-name {
  font-size: 0.9rem;
  color: #475569;
  margin: 0;
}

.upload-preview {
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 1rem;
  max-width: 360px;
}

.upload-preview h2 {
  margin: 0;
  font-size: 1.1rem;
}

.upload-preview img {
  width: 100%;
  border-radius: 0.75rem;
  object-fit: cover;
}

.feedback {
  margin: 0;
  font-size: 0.95rem;
}

.feedback.error {
  color: #b91c1c;
}

.feedback.info {
  color: #1d4ed8;
}

.feedback.subtle {
  color: #475569;
}

.results {
  display: grid;
  gap: 1.5rem;
}

.results h2 {
  margin: 0;
}

.candidate-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1.5rem;
}

.candidate-card {
  background: #ffffff;
  border-radius: 1rem;
  overflow: hidden;
  box-shadow: 0 20px 35px rgba(15, 23, 42, 0.08);
  display: flex;
  flex-direction: column;
}

.candidate-card__media {
  width: 100%;
  padding-top: 65%;
  position: relative;
  background: #e2e8f0;
}

.candidate-card__media img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.candidate-card__body {
  display: grid;
  gap: 0.75rem;
  padding: 1rem 1.25rem 1.5rem;
}

.candidate-card__body h3 {
  margin: 0;
  font-size: 1.1rem;
}

.candidate-card__body .distance {
  margin: 0;
  color: #1f2937;
}

.candidate-card__body .movies {
  display: grid;
  gap: 0.5rem;
}

.candidate-card__body .movies h4 {
  margin: 0;
  font-size: 0.95rem;
  color: #334155;
}

.candidate-card__body .movies ul {
  list-style: disc;
  margin: 0 0 0 1.2rem;
  padding: 0;
  display: grid;
  gap: 0.25rem;
}

.candidate-card__body .movies li {
  font-size: 0.9rem;
  color: #475569;
}

@media (max-width: 640px) {
  .upload,
  .upload-preview {
    padding: 1.25rem;
  }

  .candidate-grid {
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  }
}
</style>