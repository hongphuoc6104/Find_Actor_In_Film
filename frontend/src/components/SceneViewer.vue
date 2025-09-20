<template>
  <section class="scene-viewer">
    <header class="scene-viewer__header">
      <h3>Cảnh hiện tại</h3>
      <p v-if="sceneIndexLabel" class="scene-viewer__meta">{{ sceneIndexLabel }}</p>
    </header>

    <div class="scene-viewer__content" :class="{ 'scene-viewer__content--loading': isLoading }">
      <div v-if="isLoading" class="scene-viewer__loading">Đang tải cảnh…</div>
      <template v-else>
        <div v-if="sceneImage" class="scene-viewer__frame">
          <img :src="sceneImage" alt="Khung hình đề xuất" @load="onImageLoad" />
          <div v-for="(box, index) in overlayBoxes" :key="index" class="scene-viewer__bbox" :style="box" />
        </div>
        <p v-else class="scene-viewer__placeholder">Không có cảnh nào để hiển thị.</p>
      </template>
    </div>

    <dl v-if="sceneDetails.length" class="scene-viewer__details">
      <div v-for="item in sceneDetails" :key="item.label" class="scene-viewer__details-row">
        <dt>{{ item.label }}</dt>
        <dd>{{ item.value }}</dd>
      </div>
    </dl>
  </section>
</template>

<script setup>
import { computed, reactive, watch } from 'vue'

const props = defineProps({
  scene: {
    type: Object,
    default: null,
  },
  meta: {
    type: Object,
    default: null,
  },
  movieTitle: {
    type: String,
    default: '',
  },
  characterId: {
    type: String,
    default: '',
  },
  isLoading: {
    type: Boolean,
    default: false,
  },
})

const imageSize = reactive({ width: 0, height: 0 })

const resetImageSize = () => {
  imageSize.width = 0
  imageSize.height = 0
}

watch(
  () => props.scene,
  () => {
    resetImageSize()
  },
)

const sceneImage = computed(() => {
  if (!props.scene) {
    return ''
  }
  return props.scene.frame || props.scene.image || props.scene.preview_image || ''
})

const sceneIndexLabel = computed(() => {
  if (!props.meta) {
    return ''
  }
  const index = props.meta.scene_index
  const total = props.meta.total_scenes
  if (typeof index === 'number' && typeof total === 'number') {
    return `Cảnh ${index + 1}/${total}`
  }
  if (typeof index === 'number') {
    return `Cảnh thứ ${index + 1}`
  }
  return ''
})

const toBox = (entry) => {
  if (!entry) {
    return null
  }
  if (Array.isArray(entry) && entry.length >= 4) {
    const [x1, y1, x2, y2] = entry.map((value) => Number(value) || 0)
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
  }
  if (typeof entry === 'object') {
    if (
      ['x', 'y', 'width', 'height'].every((key) => typeof entry[key] === 'number')
    ) {
      return {
        x: entry.x,
        y: entry.y,
        width: entry.width,
        height: entry.height,
      }
    }
    if (
      ['x1', 'y1', 'x2', 'y2'].every(
        (key) => typeof entry[key] === 'number' || typeof entry[key] === 'string',
      )
    ) {
      const x1 = Number(entry.x1) || 0
      const y1 = Number(entry.y1) || 0
      const x2 = Number(entry.x2) || 0
      const y2 = Number(entry.y2) || 0
      return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
    }
  }
  return null
}

const rawBoxes = computed(() => {
  if (!props.scene) {
    return []
  }
  const boxes = []
  if (Array.isArray(props.scene.boxes)) {
    boxes.push(...props.scene.boxes)
  }
  if (props.scene.bbox) {
    boxes.push(props.scene.bbox)
  }
  return boxes.map(toBox).filter(Boolean)
})

const overlayBoxes = computed(() => {
  if (!rawBoxes.value.length || !sceneImage.value) {
    return []
  }

  const width = imageSize.width || Number(props.scene?.width) || 0
  const height = imageSize.height || Number(props.scene?.height) || 0
  if (!width || !height) {
    return []
  }

  return rawBoxes.value.map((box) => ({
    left: `${Math.max((box.x / width) * 100, 0)}%`,
    top: `${Math.max((box.y / height) * 100, 0)}%`,
    width: `${Math.min((box.width / width) * 100, 100)}%`,
    height: `${Math.min((box.height / height) * 100, 100)}%`,
  }))
})

const formatTimestamp = (timestamp) => {
  if (typeof timestamp !== 'number' || !Number.isFinite(timestamp)) {
    return null
  }

  const totalSeconds = Math.max(timestamp, 0)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  const milliseconds = Math.round((totalSeconds % 1) * 1000)
  if (milliseconds) {
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds
      .toString()
      .padStart(3, '0')}`
  }
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

const sceneDetails = computed(() => {
  if (!props.scene) {
    return []
  }

  const details = []
  if (props.movieTitle) {
    details.push({ label: 'Phim', value: props.movieTitle })
  }
  if (props.characterId) {
    details.push({ label: 'Nhân vật', value: props.characterId })
  }
  if (props.scene.timestamp !== undefined && props.scene.timestamp !== null) {
    const formatted = formatTimestamp(Number(props.scene.timestamp))
    if (formatted) {
      details.push({ label: 'Timestamp', value: formatted })
    }
  }
  if (props.scene.frame) {
    details.push({ label: 'Khung hình', value: props.scene.frame })
  }
  if (
    props.scene.det_score !== undefined &&
    props.scene.det_score !== null &&
    props.scene.det_score !== ''
  ) {
    const score = Number(props.scene.det_score)
    const formatted = Number.isFinite(score)
      ? score.toFixed(3)
      : String(props.scene.det_score)
    details.push({ label: 'Điểm phát hiện', value: formatted })
  }
  return details
})

const onImageLoad = (event) => {
  if (!event?.target) {
    return
  }
  const { naturalWidth, naturalHeight } = event.target
  if (naturalWidth && naturalHeight) {
    imageSize.width = naturalWidth
    imageSize.height = naturalHeight
  }
}
</script>

<style scoped>
.scene-viewer {
  display: grid;
  gap: 1rem;
  background: #ffffff;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 15px 45px rgba(15, 23, 42, 0.08);
}

.scene-viewer__header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
}

.scene-viewer__header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.scene-viewer__meta {
  margin: 0;
  color: #475569;
  font-size: 0.9rem;
}

.scene-viewer__content {
  position: relative;
  min-height: 240px;
  border-radius: 0.75rem;
  background: #f1f5f9;
  overflow: hidden;
}

.scene-viewer__content--loading {
  display: grid;
  place-items: center;
}

.scene-viewer__loading {
  font-weight: 600;
  color: #1e293b;
}

.scene-viewer__frame {
  position: relative;
  width: 100%;
  height: 100%;
}

.scene-viewer__frame img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #0f172a;
}

.scene-viewer__bbox {
  position: absolute;
  border: 3px solid rgba(59, 130, 246, 0.9);
  border-radius: 0.25rem;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35);
}

.scene-viewer__placeholder {
  margin: 0;
  padding: 2.5rem 1rem;
  text-align: center;
  color: #475569;
}

.scene-viewer__details {
  margin: 0;
  display: grid;
  gap: 0.5rem;
}

.scene-viewer__details-row {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.75rem;
  font-size: 0.9rem;
}

.scene-viewer__details-row dt {
  font-weight: 600;
  color: #1e293b;
}

.scene-viewer__details-row dd {
  margin: 0;
  color: #334155;
}

@media (max-width: 640px) {
  .scene-viewer {
    padding: 1.25rem;
  }
}
</style>
