<template>
  <section class="scene-viewer">
    <header class="scene-viewer__header">
      <div>
        <h3>Cảnh hiện tại</h3>
        <p v-if="sceneIndexLabel" class="scene-viewer__meta">{{ sceneIndexLabel }}</p>
      </div>
    </header>

    <div class="scene-viewer__layout">
      <div class="scene-viewer__stage">
        <div
          class="scene-viewer__canvas"
          :class="{ 'scene-viewer__canvas--loading': isLoading }"
        >
          <div v-if="isLoading" class="scene-viewer__loading">Đang tải cảnh…</div>
          <template v-else>
            <div v-if="sceneClip" class="scene-viewer__frame scene-viewer__frame--video">
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
      </div>

      <aside v-if="hasSidebarContent" class="scene-viewer__sidebar">
        <section v-if="sceneDetails.length" class="scene-viewer__panel">
          <h4>Thông tin cảnh</h4>
          <dl class="scene-viewer__details">
            <div v-for="item in sceneDetails" :key="item.label" class="scene-viewer__details-row">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </div>
          </dl>
        </section>

        <details v-if="timelineSegments.length" class="scene-viewer__panel scene-viewer__timeline" open>
          <summary>Dòng thời gian</summary>
          <ol>
            <li
              v-for="segment in timelineSegments"
              :key="segment.id"
              :class="['scene-viewer__timeline-item', { active: segment.active }]"
            >
              <div class="scene-viewer__timeline-header">
                <span class="scene-viewer__timeline-index">{{ segment.order }}</span>
                <div class="scene-viewer__timeline-text">
                  <span class="scene-viewer__timeline-label">{{ segment.label }}</span>
                  <span v-if="segment.range" class="scene-viewer__timeline-range">{{ segment.range }}</span>
                </div>
              </div>
              <p v-if="segment.description" class="scene-viewer__timeline-description">
                {{ segment.description }}
              </p>
            </li>
          </ol>
        </details>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'

import {
  collectBoxesFromScene,
  collectBoxesFromTimelineEntry,
  pickActiveTimelineEntry,
  scaleBoxes,
} from '../utils/sceneTimeline.js'
import { toAbsoluteAssetUrl } from '../utils/assetUrls.js'

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
  const sources = [
    props.scene.clip_url,
    props.scene.clip,
    props.scene.clipUrl,
    props.scene.clipPath,
    props.scene.clip_path,
  ]
  const clip = sources.find((value) => typeof value === 'string' && value)
  return clip ? toAbsoluteAssetUrl(clip) : ''
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
  const sources = [
    props.scene.frame_url,
    props.scene.frame,
    props.scene.image,
    props.scene.preview_image,
    props.scene.thumbnail,
  ]
  const source = sources.find((item) => typeof item === 'string' && item)
  return source ? toAbsoluteAssetUrl(source) : ''
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

const rawBoxes = computed(() => {
  return collectBoxesFromScene(props.scene)
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

  return pickActiveTimelineEntry(timeline, videoTime.value ?? 0, clipFps.value)
})

const activeBoxes = computed(() => {
  if (sceneClip.value && activeTimelineEntry.value) {
    return collectBoxesFromTimelineEntry(activeTimelineEntry.value)
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

  return scaleBoxes(boxes, width, height)
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

const extractFileName = (value) => {
  if (!value || typeof value !== 'string') {
    return ''
  }
  const segments = value.split(/[\\/]/)
  return segments[segments.length - 1] || value
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
  const frameLabel =
    typeof props.scene.frame_name === 'string' && props.scene.frame_name
      ? props.scene.frame_name
      : extractFileName(
          typeof props.scene.frame === 'string' && props.scene.frame
            ? props.scene.frame
            : props.scene.frame_url,
        )
  if (frameLabel) {
    details.push({ label: 'Khung hình', value: frameLabel })
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

const timelineSegments = computed(() => {
  return timelineEntries.value
    .map((entry, index) => {
      const start = formatTimestamp(Number(entry.start ?? entry.time ?? entry.timestamp))
      const end = formatTimestamp(Number(entry.end ?? entry.until))
      let range = ''
      if (start && end) {
        range = `${start} → ${end}`
      } else if (start) {
        range = start
      } else if (end) {
        range = `Đến ${end}`
      }
      const label =
        entry.label || entry.event || entry.state || entry.status || `Khoảng #${index + 1}`
      const description = entry.description || entry.note || ''
      return {
        id: entry.id ?? index,
        order: index + 1,
        label,
        range,
        description,
        active: entry === activeTimelineEntry.value,
      }
    })
    .filter((segment) => segment.label || segment.range || segment.description)
})

const hasSidebarContent = computed(() => {
  return Boolean(sceneDetails.value.length || timelineSegments.value.length)
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
  --surface-border: #e2e8f0;
  --surface-radius: 0.75rem;
  --surface-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
  background: #ffffff;
  border: 1px solid var(--surface-border);
  border-radius: var(--surface-radius);
  padding: 1.5rem;
  box-shadow: var(--surface-shadow);
  display: grid;
  gap: 1.5rem;
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

.scene-viewer__layout {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
  align-items: start;
}

.scene-viewer__stage {
  display: grid;
  gap: 1rem;
}

.scene-viewer__canvas {
  position: relative;
  min-height: 260px;
  border: 1px solid var(--surface-border);
  border-radius: calc(var(--surface-radius) - 0.1rem);
  background: #0f172a;
  overflow: hidden;
}

.scene-viewer__canvas--loading {
  display: grid;
  place-items: center;
  background: #111827;
}

.scene-viewer__loading {
  font-weight: 600;
  color: #f8fafc;
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
  background: transparent;
}

.scene-viewer__video {
  background: transparent;
}

.scene-viewer__bbox {
  position: absolute;
  border: 2px solid rgba(14, 165, 233, 0.9);
  border-radius: 0.35rem;
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.3);
}

.scene-viewer__placeholder {
  margin: 0;
  color: #cbd5f5;
  text-align: center;
  padding: 2.5rem 1rem;
}

.scene-viewer__sidebar {
  display: grid;
  gap: 1rem;
}

.scene-viewer__panel {
  background: #f8fafc;
  border: 1px solid var(--surface-border);
  border-radius: var(--surface-radius);
  padding: 1rem 1.25rem;
  display: grid;
  gap: 0.75rem;
}

.scene-viewer__panel h4 {
  margin: 0;
  font-size: 1rem;
  color: #1e293b;
}

.scene-viewer__details {
  margin: 0;
  display: grid;
  gap: 0.5rem;
}

.scene-viewer__details-row {
  display: grid;
  grid-template-columns: minmax(100px, max-content) 1fr;
  gap: 0.75rem;
  font-size: 0.9rem;
  align-items: baseline;
}

.scene-viewer__details-row dt {
  font-weight: 600;
  color: #1e293b;
}

.scene-viewer__details-row dd {
  margin: 0;
  color: #334155;
}

.scene-viewer__timeline {
  background: #f1f5f9;
}

.scene-viewer__timeline summary {
  cursor: pointer;
  font-weight: 600;
  color: #1e293b;
  list-style: none;
}

.scene-viewer__timeline summary::-webkit-details-marker {
  display: none;
}

.scene-viewer__timeline[open] > summary {
  margin-bottom: 0.75rem;
}

.scene-viewer__timeline ol {
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.75rem;
}

.scene-viewer__timeline-item {
  list-style: none;
  border: 1px solid transparent;
  border-radius: 0.65rem;
  padding: 0.5rem 0.75rem;
  background: #ffffff;
  display: grid;
  gap: 0.35rem;
  transition: border-color 150ms ease, background 150ms ease;
}

.scene-viewer__timeline-item.active {
  border-color: #2563eb;
  background: rgba(37, 99, 235, 0.08);
}

.scene-viewer__timeline-header {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
}

.scene-viewer__timeline-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.75rem;
  height: 1.75rem;
  border-radius: 999px;
  background: #2563eb;
  color: #f8fafc;
  font-size: 0.85rem;
  font-weight: 600;
}

.scene-viewer__timeline-text {
  display: grid;
  gap: 0.25rem;
}

.scene-viewer__timeline-label {
  font-weight: 600;
  color: #1e293b;
}

.scene-viewer__timeline-range {
  font-size: 0.85rem;
  color: #475569;
}

.scene-viewer__timeline-description {
  margin: 0;
  font-size: 0.85rem;
  color: #475569;
}

@media (max-width: 900px) {
  .scene-viewer__layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .scene-viewer {
    padding: 1.25rem;
  }
}
</style>
