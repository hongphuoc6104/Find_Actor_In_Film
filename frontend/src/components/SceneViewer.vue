<template>
  <section class="scene-viewer">
    <header class="scene-viewer__header">
      <div>
        <h3>Highlight nổi bật</h3>
        <p v-if="highlightSummary" class="scene-viewer__meta">{{ highlightSummary }}</p>
      </div>
      <div v-if="highlightCounter" class="scene-viewer__counter">{{ highlightCounter }}</div>
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

        <p v-if="filterDiagnosticMessage" class="scene-viewer__notice">
          {{ filterDiagnosticMessage }}
        </p>

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
  highlightIndex: { type: Number, default: null },
  highlightTotal: { type: Number, default: null },
})

const emit = defineEmits(['highlight-change'])

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

const parseIndexValue = (value) => {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null
}


const resolveSceneIdentifier = (scene) => {
  if (!scene || typeof scene !== 'object') {
    return null
  }
  return (
    scene.id ??
    scene.scene_id ??
    scene.sceneId ??
    scene.uuid ??
    scene.key ??
    scene.slug ??
    null
  )
}


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

const buildFallbackStats = (items = []) => ({
  inCount: items.length,
  outCount: 0,
  reasons: { det_score: null, score: null, duration: null },
})

const filteredHighlightsRes = computed(() => {
  const scene = props.scene
  const options = highlightFilterOptions.value
  if (!scene) {
    return filterHighlights([], options)
  }

  if (Array.isArray(scene.filtered_highlights)) {
    const items = scene.filtered_highlights
    const stats =
      scene.highlight_filter_stats && typeof scene.highlight_filter_stats === 'object'
        ? scene.highlight_filter_stats
        : buildFallbackStats(items)
    return { items, stats }
  }

  const source = Array.isArray(scene.highlights) ? scene.highlights : []
  return filterHighlights(source, {
    settings: options.settings,
    support: options.support,
  })
})

const rawHighlights = computed(() => {
  if (!Array.isArray(props.scene?.highlights)) {
    return []
  }
  return props.scene.highlights
})

const filteredHighlights = computed(() => filteredHighlightsRes.value?.items ?? [])

const filterStats = computed(() => {
  const stats = filteredHighlightsRes.value?.stats
  return stats && typeof stats === 'object' ? stats : buildFallbackStats(filteredHighlights.value)
})

const effectiveHighlights = computed(() => {
  if (filteredHighlights.value.length) {
    return filteredHighlights.value
  }
  return rawHighlights.value.map((item, index) => {
    if (!item || typeof item !== 'object') {
      return item
    }
    if (item.id !== undefined && item.id !== null && item.id !== '') {
      return item
    }
    return { ...item, id: `raw-highlight-${index}` }
  })
})

const filterDiagnosticMessage = computed(() => {
  if (filteredHighlights.value.length || !rawHighlights.value.length) {
    return ''
  }
  const outCount = Number(filterStats.value?.outCount ?? 0)
  if (!Number.isFinite(outCount) || outCount <= 0) {
    return ''
  }

  const reasons = filterStats.value?.reasons ?? {}
  const parts = []
  const formatReason = (value, label) => {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) {
      parts.push(`${label} ≥ ${parsed}`)
    }
  }
  formatReason(reasons.score, 'Điểm')
  formatReason(reasons.det_score, 'Điểm phát hiện')
  formatReason(reasons.duration, 'Độ dài')

  const thresholdsText = parts.length ? ` (ngưỡng: ${parts.join(', ')})` : ''
  return `Không có highlight đạt chuẩn sau khi áp dụng bộ lọc${thresholdsText}. Đang hiển thị danh sách highlight gốc (${outCount} mục bị loại).`
  return `Không có highlight đạt chuẩn sau khi áp dụng bộ lọc${thresholdsText}. Đang hiển thị danh sách highlight gốc (${outCount} mục bị loại).`
})


const parseCountValue = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const highlightStats = computed(() => {
  const total = parseCountValue(props.scene?.highlight_total)
  const display = parseCountValue(props.scene?.highlight_display_count)
  const filteredCount = filterStats.value?.inCount ?? filteredHighlights.value.length
  const effectiveCount = effectiveHighlights.value.length
  const usingRescuePath = filteredHighlights.value.length === 0 && effectiveCount > 0
  const resolvedDisplay = display !== null ? display : filteredCount
  return {
    total,
    display: usingRescuePath ? effectiveCount : resolvedDisplay,
    filteredCount,
    effectiveCount,
    rescueCount: usingRescuePath ? effectiveCount : 0,
    stats: filterStats.value,
  }
})


const resolvedHighlightIndex = computed(() => {
  const metaIndex = parseIndexValue(props.meta?.scene_index)
  if (metaIndex !== null) {
    return metaIndex
  }
  return parseIndexValue(props.highlightIndex)
})

const resolvedHighlightTotal = computed(() => {
  const stats = highlightStats.value
  const candidates = [
    props.meta?.highlight_total,
    props.meta?.total_scenes,
    props.meta?.scene_count,
    props.meta?.highlight_display_count,
    props.highlightTotal,
    stats?.display,
    stats?.total,
  ]
  for (const candidate of candidates) {
    const parsed = parseCountValue(candidate)
    if (parsed !== null) {
      return parsed
    }
  }
  return null
})

const highlightCounter = computed(() => {
  const index = resolvedHighlightIndex.value
  const total = resolvedHighlightTotal.value
  if (index === null && total === null) {
    return ''
  }
  if (index === null) {
    return total !== null ? `Highlight 0/${total}` : ''
  }
  const current = index + 1
  if (total !== null) {
    return `Highlight ${current}/${total}`
  }
  return `Highlight ${current}`
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

  const highlightStart = effectiveHighlights.value.length
    ? computeSegmentSeekStart(effectiveHighlights.value[0])
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
  activeSegment.value = effectiveHighlights.value[0] ?? null
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
watch(effectiveHighlights, (segments) => {
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

let lastHighlightIndex = null
let lastHighlightTotal = null
watch(
  () => [resolvedHighlightIndex.value, resolvedHighlightTotal.value],
  ([index, total]) => {
    if (lastHighlightIndex === index && lastHighlightTotal === total) {
      return
    }
    lastHighlightIndex = index
    lastHighlightTotal = total
    emit('highlight-change', { index, total })
  },
  { immediate: true },
)

/* --- Timeline --- */
const timelineSegments = computed(() => {
  const format = (ts) => {
    if (typeof ts !== 'number' || !Number.isFinite(ts)) return ''
    const m = Math.floor(ts / 60)
    const s = Math.floor(ts % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const currentActiveId = activeSegment.value?.id ?? null
  const stats = highlightStats.value
  const totalHighlights = (() => {
    const parsedTotal = Number(stats.total)
    if (Number.isFinite(parsedTotal) && parsedTotal > 0) {
      return parsedTotal
    }
    if (stats.filteredCount) {
      return stats.filteredCount
    }
    if (stats.effectiveCount) {
      return stats.effectiveCount
    }
    return effectiveHighlights.value.length
  })()
  return effectiveHighlights.value.map((highlight, index) => {
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

  const segments = effectiveHighlights.value
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

watch(
  () => props.scene?.highlights ?? null,
  (highlights) => {
    if (!Array.isArray(highlights)) {
      return
    }
    const sceneId = resolveSceneIdentifier(props.scene)
    highlights.forEach((item, index) => {
      if (!item || typeof item !== 'object') {
        console.debug('SceneViewer: highlight entry is not an object', {
          sceneId,
          index,
          entry: item,
        })
        return
      }
      const missing = []
      if (item.start === undefined || item.start === null) missing.push('start')
      if (item.end === undefined || item.end === null) missing.push('end')
      if (!('id' in item) || item.id === null || item.id === '') missing.push('id')
      if (missing.length) {
        console.debug('SceneViewer: highlight missing payload fields', {
          sceneId,
          index,
          missing,
          highlight: item,
        })
      }
    })
  },
  { immediate: true },
)

const zeroHighlightLogKey = ref(null)
watch(
  () => effectiveHighlights.value.length,
  (length) => {
    if (!props.scene) {
      return
    }
    if (length > 0) {
      zeroHighlightLogKey.value = null
      return
    }
    const rawCount = rawHighlights.value.length
    if (!rawCount) {
      return
    }
    const filteredCount = filteredHighlights.value.length
    const stats = filterStats.value
    const sceneId = resolveSceneIdentifier(props.scene)
    const key = `${sceneId ?? 'unknown'}::${rawCount}::${filteredCount}`
    if (zeroHighlightLogKey.value === key) {
      return
    }
    zeroHighlightLogKey.value = key
    console.debug('SceneViewer: no visible highlights rendered, falling back to raw data', {
      sceneId,
      rawCount,
      filteredCount,
      filterStats: stats,
    })
  },
  { immediate: true },
)

const hasSidebarContent = computed(
  () => Boolean(sceneDetails.value.length || timelineSegments.value.length || filterDiagnosticMessage.value),
)

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

.scene-viewer__counter {
  margin-left: auto;
  font-weight: 600;
  color: #1d4ed8;
  font-size: 0.95rem;
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

.scene-viewer__notice {
  margin: 0 0 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 0.65rem;
  border: 1px solid rgba(37, 99, 235, 0.2);
  background: rgba(37, 99, 235, 0.08);
  color: #1d4ed8;
  font-size: 0.85rem;
  line-height: 1.4;
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
