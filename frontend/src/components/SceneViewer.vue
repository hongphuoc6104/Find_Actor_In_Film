<template>
  <section class="scene-viewer">
    <header class="scene-viewer__header">
      <h3>Cảnh hiện tại</h3>
      <p v-if="sceneIndexLabel" class="scene-viewer__meta">{{ sceneIndexLabel }}</p>
    </header>

    <div class="scene-viewer__content" :class="{ 'scene-viewer__content--loading': isLoading }">
      <div v-if="isLoading" class="scene-viewer__loading">Đang tải cảnh…</div>
      <template v-else>
        <div
          v-if="sceneClip"
          class="scene-viewer__frame scene-viewer__frame--video"
        >
          <video
            ref="videoRef"
            class="scene-viewer__video"
            :src="sceneClip"
            controls
            @loadedmetadata="onVideoLoadedMetadata"
            @loadeddata="onVideoLoadedMetadata"
            @timeupdate="onVideoTimeUpdate"
            @seeked="onVideoTimeUpdate"
          />
          <div
            v-for="(box, index) in overlayBoxes"
            :key="`video-box-${index}`"
            class="scene-viewer__bbox"
            :style="box"
          />
        </div>
        <div v-else-if="sceneImage" class="scene-viewer__frame">
          <img :src="sceneImage" alt="Khung hình đề xuất" @load="onImageLoad" />
          <div
            v-for="(box, index) in overlayBoxes"
            :key="`image-box-${index}`"
            class="scene-viewer__bbox"
            :style="box"
          />
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
import { computed, reactive, ref, watch } from 'vue'

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
const videoSize = reactive({ width: 0, height: 0 })
const videoRef = ref(null)
const videoTime = ref(0)

const parseDimension = (value) => {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? number : 0
}

const resetImageSize = () => {
  imageSize.width = 0
  imageSize.height = 0
}

const resetVideoState = () => {
  videoSize.width = parseDimension(props.scene?.width)
  videoSize.height = parseDimension(props.scene?.height)
  videoTime.value = 0
  if (videoRef.value) {
    try {
      videoRef.value.pause()
      videoRef.value.currentTime = 0
      videoRef.value.load()
    } catch (error) {
      // ignore reset errors for browsers without media support
    }
  }
}

const sceneClip = computed(() => {
  if (!props.scene) {
    return ''
  }
  const clip =
    props.scene.clip || props.scene.clip_url || props.scene.clipPath || props.scene.clip_path
  return typeof clip === 'string' ? clip : ''
})

watch(
  () => props.scene,
  () => {
    resetImageSize()
    resetVideoState()
  },
)

watch(
  () => sceneClip.value,
  () => {
    resetVideoState()
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

const timelineEntries = computed(() => {
  if (!props.scene || !Array.isArray(props.scene.timeline)) {
    return []
  }
  return props.scene.timeline
    .map((entry) => (entry && typeof entry === 'object' ? entry : null))
    .filter(Boolean)
})

const clipFps = computed(() => {
  const value = Number(props.scene?.clip_fps)
  return Number.isFinite(value) && value > 0 ? value : null
})

const activeTimelineEntry = computed(() => {
  const timeline = timelineEntries.value
  if (!timeline.length) {
    return null
  }
  if (!sceneClip.value) {
    return timeline[0]
  }

  const fps = clipFps.value
  const epsilon = fps ? 1 / fps : 0.05
  const current = videoTime.value ?? 0

  let candidate = timeline[0]
  timeline.forEach((entry, index) => {
    let offset = Number(
      entry?.clip_offset ?? entry?.offset ?? entry?.relative_time ?? entry?.timestamp,
    )
    if (!Number.isFinite(offset)) {
      offset = fps ? index / fps : index * epsilon
    }
    if (offset <= current + epsilon) {
      candidate = entry
    }
  })

  return candidate
})

const activeBoxes = computed(() => {
  if (sceneClip.value && activeTimelineEntry.value) {
    const entry = activeTimelineEntry.value
    const boxes = []
    if (Array.isArray(entry.boxes)) {
      boxes.push(...entry.boxes)
    }
    if (entry.bbox) {
      boxes.push(entry.bbox)
    }
    return boxes.map(toBox).filter(Boolean)
  }
  return rawBoxes.value
})

const baseDimensions = computed(() => {
  if (sceneClip.value) {
    const width = videoSize.width || parseDimension(props.scene?.width)
    const height = videoSize.height || parseDimension(props.scene?.height)
    return { width, height }
  }
  const width = imageSize.width || parseDimension(props.scene?.width)
  const height = imageSize.height || parseDimension(props.scene?.height)
  return { width, height }
})

const overlayBoxes = computed(() => {
  const boxes = activeBoxes.value
  if (!boxes.length) {
    return []
  }

  const { width, height } = baseDimensions.value
  if (!width || !height) {
    return []
  }

  return boxes.map((box) => ({
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
  if (sceneClip.value) {
    const clipDuration = formatTimestamp(Number(props.scene?.duration))
    if (clipDuration) {
      details.push({ label: 'Độ dài clip', value: clipDuration })
    }
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

const onVideoLoadedMetadata = (event) => {
  const video = event?.target ?? videoRef.value
  if (!video) {
    return
  }
  if (video.videoWidth && video.videoHeight) {
    videoSize.width = video.videoWidth
    videoSize.height = video.videoHeight
  }
  videoTime.value = video.currentTime ?? 0
}

const onVideoTimeUpdate = (event) => {
  const video = event?.target ?? videoRef.value
  if (!video) {
    return
  }
  videoTime.value = video.currentTime ?? 0
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

.scene-viewer__frame img,
.scene-viewer__frame video {
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
