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
              <div
                v-for="(box, index) in overlayBoxes"
                :key="`video-box-${index}`"
                class="scene-viewer__bbox"
                :style="box"
              />
            </div>

            <!-- Fallback ảnh -->
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
          <summary>Những khoảng có nhân vật xuất hiện (độ chính xác cao)</summary>
          <ol>
            <li
              v-for="segment in timelineSegments"
              :key="segment.id"
              :class="['scene-viewer__timeline-item', { active: segment.active }]"
              @click="seekToSegment(segment)"
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
import { collectBoxesFromScene, scaleBoxes } from '../utils/sceneTimeline.js'
import { toAbsoluteAssetUrl } from '../utils/assetUrls.js'

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

const parseDimension = (value) => {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : 0
}
const parseTimeValue = (value) => {
  const n = Number(value)
  return Number.isFinite(n) && n >= 0 ? n : null
}

const sceneStartTime = computed(() => {
  if (!props.scene) return null
  const cands = [props.scene.start_time, props.scene.video_start_timestamp, props.scene.timestamp]
  for (const c of cands) {
    const parsed = parseTimeValue(c)
    if (parsed !== null) return parsed
  }
  return null
})

/* --- Video state --- */
const resetImageSize = () => { imageSize.width = 0; imageSize.height = 0 }
const resetVideoState = () => {
  videoSize.width = parseDimension(props.scene?.width)
  videoSize.height = parseDimension(props.scene?.height)
  const safeStart = sceneStartTime.value ?? 0
  videoTime.value = safeStart
  pendingSeekTime.value = safeStart
  if (videoRef.value) {
    try { videoRef.value.pause() } catch {}
  }
}

const sceneVideo = computed(() => {
  if (!props.scene) return ''
  const sources = [
    props.scene.video_url, props.scene.video, props.scene.video_path,
    props.scene.clip_url, props.scene.clip_path
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

/* --- Timeline --- */
function mergeTimelineEntries(entries, maxGap = 1.0) {
  if (!entries.length) return []
  const merged = []
  let current = { start: entries[0].timestamp, end: entries[0].timestamp }
  for (let i = 1; i < entries.length; i++) {
    const ts = entries[i].timestamp
    if (ts - current.end <= maxGap) {
      current.end = ts
    } else {
      merged.push({ ...current })
      current = { start: ts, end: ts }
    }
  }
  merged.push({ ...current })
  return merged
}


const timelineSegments = computed(() => {
  if (!props.scene) return []

  // Nếu API đã trả về highlights thì dùng luôn
  if (Array.isArray(props.scene.highlights) && props.scene.highlights.length) {
  return props.scene.highlights.map((h, i) => {
    const format = (ts) => {
      if (typeof ts !== 'number' || !Number.isFinite(ts)) return ''
      const m = Math.floor(ts / 60), s = Math.floor(ts % 60)
      return `${m}:${s.toString().padStart(2, '0')}`
    }
    return {
      id: i,
      order: i + 1,
      label: `Cảnh ${i + 1} (score: ${(h.max_score*100).toFixed(0)}%)`,
      range: `${format(h.start)} → ${format(h.end)} (${h.duration.toFixed(1)}s)`,
      start: h.start,
      end: h.end,
      active: videoTime.value >= h.start && videoTime.value <= h.end,
    }
  })
}


  // Fallback: nếu chưa có highlights thì gộp từ timeline như trước
  if (Array.isArray(props.scene.timeline)) {
    const raw = props.scene.timeline
      .filter(e => e && typeof e === 'object' && (e.det_score ?? 0) >= 0.9)
    const merged = mergeTimelineEntries(raw, 1.0)
    return merged.map((seg, i) => {
      const format = (ts) => {
        if (typeof ts !== 'number' || !Number.isFinite(ts)) return ''
        const m = Math.floor(ts / 60), s = Math.floor(ts % 60)
        return `${m}:${s.toString().padStart(2, '0')}`
      }
      return {
        id: i,
        order: i + 1,
        label: `Khoảng #${i + 1}`,
        range: `${format(seg.start)} → ${format(seg.end)}`,
        start: seg.start,
        end: seg.end,
        active: videoTime.value >= seg.start && videoTime.value <= seg.end,
      }
    })
  }

  return []
})


const seekToSegment = (segment) => {
  if (videoRef.value) {
    videoRef.value.currentTime = segment.start
    videoRef.value.play()
    activeSegment.value = segment
  }
}

const onVideoTimeUpdate = (e) => {
  const video = e?.target ?? videoRef.value
  if (!video) return
  videoTime.value = video.currentTime ?? 0
  if (activeSegment.value && video.currentTime >= activeSegment.value.end) {
    video.pause()
    activeSegment.value = null
  }
}

/* --- Others --- */
const rawBoxes = computed(() => collectBoxesFromScene(props.scene))
const overlayBoxes = computed(() => {
  const w = videoSize.width || imageSize.width, h = videoSize.height || imageSize.height
  return (w && h) ? scaleBoxes(rawBoxes.value, w, h) : []
})

const sceneDetails = computed(() => {
  if (!props.scene) return []
  const details = []
  if (props.movieTitle) details.push({ label: 'Phim', value: props.movieTitle })
  if (props.characterId) details.push({ label: 'Nhân vật', value: props.characterId })
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

const sceneIndexLabel = computed(() => {
  if (!props.meta) return ''
  const i = props.meta.scene_index, total = props.meta.total_scenes
  if (typeof i === 'number' && typeof total === 'number') return `Cảnh ${i + 1}/${total}`
  if (typeof i === 'number') return `Cảnh thứ ${i + 1}`
  return ''
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
