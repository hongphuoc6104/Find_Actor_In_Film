<template>
  <section class="scene-viewer">
    <header class="scene-viewer__header">
      <div>
        <h3>Highlight nổi bật</h3>
        <p v-if="highlightSummary" class="scene-viewer__meta">{{ highlightSummary }}</p>
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
            <!-- Ưu tiên video -->
            <div v-if="sceneVideo" class="scene-viewer__frame scene-viewer__frame--video">
              <video
                ref="videoRef"
                class="scene-viewer__video"
                :src="sceneVideo"
                controls
                @loadedmetadata="onVideoLoadedMetadata"
                @timeupdate="onVideoTimeUpdate"
              />
            </div>

            <!-- Fallback ảnh -->
            <div v-else-if="sceneImage" class="scene-viewer__frame">
              <img :src="sceneImage" alt="Khung hình đề xuất" @load="onImageLoad" />
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
          <summary>Những đoạn highlight có nhân vật xuất hiện nổi bật</summary>
          <ol>
            <li
              v-for="segment in timelineSegments"
              :key="segment.highlight.id"
              :class="['scene-viewer__timeline-item', { active: segment.active }]"
              @click="seekToSegment(segment.highlight)"
            >
              <div class="scene-viewer__timeline-header">
                <span class="scene-viewer__timeline-index">{{ segment.order }}</span>
                <div class="scene-viewer__timeline-text">
                  <span class="scene-viewer__timeline-label">{{ segment.label }}</span>
                  <span v-if="segment.range" class="scene-viewer__timeline-range">{{ segment.range }}</span>
                </div>
              </div>
            </li>
          </ol>
        </details>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { toAbsoluteAssetUrl } from '../utils/assetUrls.js'
import { filterHighlights } from '../utils/highlights.js'
import { PAUSE_TOLERANCE_SEC, SEEK_PAD_SEC } from '../utils/config.js'

const props = defineProps({
  scene: { type: Object, default: null },
  meta: { type: Object, default: null },
  movieTitle: { type: String, default: '' },
  characterId: { type: String, default: '' },
  isLoading: { type: Boolean, default: false },
})

const imageSize = reactive({ width: 0, height: 0 })
const videoSize = reactive({ width: 0, height: 0 })
const videoRef = ref(null)
const videoTime = ref(0)
const pendingSeekTime = ref(null)
const activeSegment = ref(null)
const DEFAULT_HIGHLIGHT_SUPPORT = Object.freeze({
  det_score_threshold: 0.75,
  min_duration: 4,
})

const highlightFilterOptions = computed(() => {
  const scene = props.scene
  const options = {}
  if (scene && typeof scene.highlight_settings === 'object') {
    options.settings = scene.highlight_settings
  }
  if (scene && typeof scene.highlight_support === 'object') {
    options.support = scene.highlight_support
  } else {
    options.support = DEFAULT_HIGHLIGHT_SUPPORT
  }
  return options
})

const filteredHighlightsResult = computed(() => {
  const options = highlightFilterOptions.value
  if (!props.scene) {
    return filterHighlights([], options)
  }
  const source = Array.isArray(props.scene.filtered_highlights)
    ? props.scene.filtered_highlights
    : Array.isArray(props.scene.highlights)
      ? props.scene.highlights
      : []
  return filterHighlights(source, options)
})
const filteredHighlights = computed(() => filteredHighlightsResult.value?.items ?? [])

const parseCountValue = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const highlightStats = computed(() => {
  const total = parseCountValue(props.scene?.highlight_total)
  const display = parseCountValue(props.scene?.highlight_display_count)
  const filterStats = filteredHighlightsResult.value?.stats ?? {
    inCount: filteredHighlightsResult.value?.items?.length ?? 0,
    outCount: 0,
    reasons: { det_score: null, score: null, duration: null },
  }
  const filteredCount = filterStats.inCount ?? filteredHighlights.value.length
  return {
    total,
    display: display !== null ? display : filteredCount,
    filteredCount,
    stats: filterStats,
  }
})

const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value)
const isAfterSegment = (time, segment) => {
  if (!isFiniteNumber(time) || !segment) return false
  const end = isFiniteNumber(segment.end) ? segment.end : null
  if (end === null) return false
  const tolerance = Number.isFinite(PAUSE_TOLERANCE_SEC) ? PAUSE_TOLERANCE_SEC : 0
  return time > end + tolerance
}
const isWithinSegmentWindow = (time, segment) => {
  if (!isFiniteNumber(time) || !segment) return false
  const start = isFiniteNumber(segment.start) ? segment.start : null
  const end = isFiniteNumber(segment.end) ? segment.end : null
  if (start === null || end === null) return false
  const tolerance = Number.isFinite(PAUSE_TOLERANCE_SEC) ? PAUSE_TOLERANCE_SEC : 0
  return time >= start && time <= end + tolerance
}


const parseDimension = (value) => {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : 0
}
const parseTimeValue = (value) => {
  const n = Number(value)
  return Number.isFinite(n) && n >= 0 ? n : null
}

const getSeekPad = () => (Number.isFinite(SEEK_PAD_SEC) ? SEEK_PAD_SEC : 0)
const computeSegmentSeekStart = (segment) => {
  if (!segment) return null
  const rawStart = parseTimeValue(typeof segment === 'number' ? segment : segment.start)
  if (rawStart === null) return null
  return Math.max(rawStart - getSeekPad(), 0)
}

const sceneStartTime = computed(() => {
  const activeStart = computeSegmentSeekStart(activeSegment.value)
  if (activeStart !== null) return activeStart

  if (!props.scene) return null

  const highlightStart = filteredHighlights.value.length
    ? computeSegmentSeekStart(filteredHighlights.value[0])
    : null

  const candidates = [
    highlightStart,
    parseTimeValue(props.scene.start_time),
    parseTimeValue(props.scene.video_start_timestamp),
    parseTimeValue(props.scene.timestamp),
  ]

  for (const candidate of candidates) {
    if (candidate !== null) return candidate
  }
  return null
})

/* --- Video state --- */
const resetImageSize = () => { imageSize.width = 0; imageSize.height = 0 }
const resetVideoState = () => {
  activeSegment.value = filteredHighlights.value[0] ?? null
  videoSize.width = parseDimension(props.scene?.width)
  videoSize.height = parseDimension(props.scene?.height)
  const safeStart = sceneStartTime.value ?? 0
  videoTime.value = safeStart
  pendingSeekTime.value = safeStart
  if (videoRef.value) {
    try {
      videoRef.value.pause()
      if (Number.isFinite(safeStart)) videoRef.value.currentTime = safeStart
    } catch {}
  }
}

const sceneVideo = computed(() => {
  if (!props.scene) return ''
  const sources = [
    props.scene.video_url,
    props.scene.video,
    props.scene.video_path,
  ]
  const src = sources.find(s => typeof s === 'string' && s)
  return src ? toAbsoluteAssetUrl(src) : ''
})

const sceneImage = computed(() => {
  if (!props.scene) return ''
  const sources = [props.scene.frame_url, props.scene.frame, props.scene.thumbnail]
  const src = sources.find(s => typeof s === 'string' && s)
  return src ? toAbsoluteAssetUrl(src) : ''
})

watch(() => props.scene, () => { resetImageSize(); resetVideoState() })
watch(() => sceneVideo.value, () => { resetVideoState() })
watch(filteredHighlights, (segments) => {
  if (!segments.length) {
    activeSegment.value = null
    return
  }
  if (!activeSegment.value) {
    activeSegment.value = segments[0]
    return
  }
  const existing = segments.find((segment) => segment.id === activeSegment.value.id)
  activeSegment.value = existing ?? segments[0]
})
/* --- Timeline --- */
const timelineSegments = computed(() => {
  const format = (ts) => {
    if (typeof ts !== 'number' || !Number.isFinite(ts)) return ''
    const m = Math.floor(ts / 60)
    const s = Math.floor(ts % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const currentActiveId = activeSegment.value?.id ?? null
  const totalHighlights = highlightStats.value.total ?? highlightStats.value.filteredCount
  return filteredHighlights.value.map((highlight, index) => {
    const orderValue = Number.isFinite(highlight.order)
      ? Number(highlight.order)
      : index + 1
    const scoreValue = highlight.effective_score ?? highlight.score ?? 0
    const labelScore = Number.isFinite(scoreValue)
      ? `${Math.round(scoreValue * 100)}%`
      : ''
    const durationValue = Number.isFinite(highlight.duration)
      ? `${highlight.duration.toFixed(1)}s`
      : ''
    const range =
      highlight.start !== null && highlight.end !== null
        ? `${format(highlight.start)} → ${format(highlight.end)}${
            durationValue ? ` (${durationValue})` : ''
          }`
        : ''

    const baseLabel = Number.isFinite(totalHighlights) && totalHighlights > 0
      ? `Highlight #${orderValue}/${totalHighlights}`
      : `Highlight #${orderValue}`
    const label = labelScore ? `${baseLabel} — ${labelScore}` : baseLabel


    return {
      order: orderValue,
      label,
      range,
      start: highlight.start,
      end: highlight.end,
      highlight,
      active:
        currentActiveId !== null
          ? currentActiveId === highlight.id
          : isWithinSegmentWindow(videoTime.value, {
              start: highlight.start,
              end: highlight.end,
            }),
    }
  })
})


const seekToSegment = (segment) => {
  if (!segment) return

  const targetTime = computeSegmentSeekStart(segment)
  if (targetTime === null) return

  pendingSeekTime.value = targetTime
  activeSegment.value = segment
  videoTime.value = targetTime

  const video = videoRef.value
  if (!video) return

  try {
    video.currentTime = targetTime
  } catch {}
  pendingSeekTime.value = null
  video.play()
}

const onVideoTimeUpdate = (e) => {
  const video = e?.target ?? videoRef.value
  if (!video) return
  const currentTime = video.currentTime ?? 0
  videoTime.value = currentTime

  const segments = filteredHighlights.value
  if (!segments.length) {
    return
  }

  const currentSegment = segments.find((item) =>
    isWithinSegmentWindow(currentTime, item),
  )

  if (currentSegment && (!activeSegment.value || activeSegment.value.id !== currentSegment.id)) {
    activeSegment.value = currentSegment
  }

  const segment = activeSegment.value

  if (!segment) {
    return
  }


  if (!isFiniteNumber(segment.start) || !isFiniteNumber(segment.end)) {

    return
  }


  if (!isAfterSegment(currentTime, segment)) {
    return
  }

  const nextSegment = (() => {
    const index = segments.findIndex((item) => item.id === segment.id)
    if (index === -1) return segments[0] ?? null
    return segments[index + 1] ?? null
  })()

  if (nextSegment) {
    seekToSegment(nextSegment)
  } else {
    video.pause()
    activeSegment.value = null
  }
}

const sceneDetails = computed(() => {
  if (!props.scene) return []
  const details = []
  if (props.movieTitle) details.push({ label: 'Phim', value: props.movieTitle })
  if (props.characterId) details.push({ label: 'Nhân vật', value: props.characterId })
  const { display, total } = highlightStats.value
  if (display) {
    const labelValue = total !== null ? `${display}/${total}` : display
    details.push({ label: 'Highlight đạt chuẩn', value: labelValue })
  }
  return details
})

const hasSidebarContent = computed(() => Boolean(sceneDetails.value.length || timelineSegments.value.length))

const onImageLoad = (e) => {
  const { naturalWidth, naturalHeight } = e.target || {}
  if (naturalWidth && naturalHeight) {
    imageSize.width = naturalWidth
    imageSize.height = naturalHeight
  }
}

const onVideoLoadedMetadata = (e) => {
  const video = e?.target ?? videoRef.value
  if (!video) return
  videoSize.width = video.videoWidth
  videoSize.height = video.videoHeight
  const targetTime = pendingSeekTime.value ?? sceneStartTime.value
  if (Number.isFinite(targetTime)) video.currentTime = targetTime
  pendingSeekTime.value = null
  videoTime.value = video.currentTime ?? 0
}

const highlightSummary = computed(() => {
  const { display, total } = highlightStats.value
  const displayValue = Number(display)
  const effectiveDisplay = Number.isFinite(displayValue) ? displayValue : 0
  if (!effectiveDisplay) {
    return 'Không có highlight đạt chuẩn'
  }
  const totalValue = Number(total)
  if (Number.isFinite(totalValue) && totalValue > 0) {
    return `${effectiveDisplay}/${totalValue} highlight đạt chuẩn`
  }
  return effectiveDisplay === 1
    ? '1 highlight đạt chuẩn'
    : `${effectiveDisplay} highlight đạt chuẩn`
})
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
