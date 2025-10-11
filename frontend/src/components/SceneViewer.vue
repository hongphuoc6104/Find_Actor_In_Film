<template>
  <div class="flex flex-col gap-4">
    <!-- State hints -->
    <div
      v-if="!currentCharacter"
      class="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600"
    >
      Chưa có nhân vật nào được chọn. Hãy chọn ở panel trái.
    </div>

    <!-- Viewer -->
    <div v-else class="grid grid-cols-1 gap-4 lg:grid-cols-5">
      <!-- Media player -->
      <div class="lg:col-span-3">
        <div class="rounded-2xl border border-slate-200 bg-black/90 overflow-hidden">
          <div class="aspect-video bg-black">
            <!-- Ưu tiên video nếu có; fallback ảnh -->
            <video
              v-if="media.type === 'video'"
              ref="player"
              class="w-full h-full"
              :src="media.src"
              controls
              playsinline
            />
            <img
              v-else
              :src="media.src"
              alt="frame"
              class="w-full h-full object-contain bg-black"
            />
          </div>
        </div>

        <!-- Scene header / controls -->
        <div class="mt-3 flex items-center justify-between gap-2">
          <div class="min-w-0">
            <h4 class="text-sm font-medium truncate">
              Nhân vật: {{ currentCharacter?.name || ('#' + currentCharacter?.character_id) }}
            </h4>
            <p class="text-xs text-slate-500">
              Cảnh {{ (currentSceneIndex ?? 0) + 1 }} / {{ totalScenes || '?' }}
              <span v-if="sceneScore">
                · score: <span class="font-medium">{{ sceneScore }}</span>
              </span>
              <span v-if="currentScene?.start !== undefined && currentScene?.end !== undefined">
                · {{ toTime(currentScene.start) }} → {{ toTime(currentScene.end) }}
              </span>
            </p>
          </div>

          <div class="flex items-center gap-2">
            <button
              type="button"
              class="px-3 py-2 rounded-lg border border-slate-300 bg-white text-xs hover:bg-slate-100 disabled:opacity-50"
              :disabled="loadingScene || (currentSceneIndex ?? 0) <= 0"
              @click="goPrev"
            >
              ← Cảnh trước
            </button>
            <button
              type="button"
              class="px-3 py-2 rounded-lg bg-slate-800 text-white text-xs hover:bg-slate-700 disabled:opacity-50"
              :disabled="loadingScene || !hasMoreScene"
              @click="goNext"
            >
              Cảnh tiếp →
            </button>
          </div>
        </div>
      </div>

      <!-- Timeline list -->
      <div class="lg:col-span-2">
        <div class="rounded-2xl border border-slate-200 bg-white overflow-hidden">
          <div class="px-3 py-2 border-b border-slate-200">
            <h4 class="text-sm font-semibold">Timeline</h4>
            <p class="text-xs text-slate-500">Chọn một cảnh để nhảy tới.</p>
          </div>

          <div class="max-h-[420px] overflow-auto divide-y divide-slate-100">
            <button
              v-for="(hl, idx) in timeline"
              :key="idx"
              type="button"
              class="w-full text-left px-3 py-2 hover:bg-slate-50"
              :class="idx === currentSceneIndex ? 'bg-slate-100/70' : ''"
              @click="jumpTo(idx)"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="min-w-0">
                  <div class="text-sm font-medium">
                    {{ toTime(hl.start) }} → {{ toTime(hl.end) }}
                  </div>
                  <div class="text-xs text-slate-500">
                    dur {{ (hl.end - hl.start).toFixed(2) }}s
                    <span v-if="getScore(hl) !== null"> · score {{ getScore(hl) }}</span>
                  </div>
                </div>
                <div class="text-[10px] text-slate-500">
                  #{{ idx + 1 }}
                </div>
              </div>
            </button>

            <div v-if="!timeline.length" class="px-3 py-4 text-sm text-slate-500">
              Không có highlight hợp lệ cho nhân vật này.
            </div>
          </div>
        </div>

        <!-- Status -->
        <p v-if="loadingScene" class="mt-2 text-xs text-slate-500">
          Đang tải cảnh…
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useRecognitionStore } from '../composables/useRecognitionStore'
import { toAbsoluteAssetUrl } from '../utils/assetUrls'

// Store
const {
  currentMovieId,
  currentCharacter,
  currentScene,
  currentSceneIndex,
  hasMoreScene,
  loadingScene,
  loadNextSceneForCurrent,
  ensureInitialScene,
  // optional APIs (store cũ có thể không có)
  loadSceneAtIndex,
} = useRecognitionStore()

const player = ref(null)

// Media source cho player (ưu tiên video)
const media = computed(() => {
  const s = currentScene.value
  if (!s) return { type: 'image', src: '' }

  const video =
    s.video_url || s.video || s.clip_url || s.clip || s.asset?.video_url || s.asset?.video
  if (typeof video === 'string' && video.length) {
    return { type: 'video', src: toAbsoluteAssetUrl(video) }
  }
  const frame =
    s.frame_url || s.frame || s.preview || s.asset?.frame_url || s.asset?.frame
  return { type: 'image', src: frame ? toAbsoluteAssetUrl(frame) : '' }
})

// Tổng số scene (nếu store expose), fallback timeline.length
const totalScenes = computed(() => {
  const arr = timeline.value
  if (Array.isArray(arr)) return arr.length
  return null
})

// Timeline (danh sách highlight)
const timeline = computed(() => {
  const ch = currentCharacter.value
  if (!ch) return []
  const t = ch.scenesPrepared || ch.scenes_cached || ch.scenes || []
  return Array.isArray(t) ? t : []
})

// Score hiển thị
const sceneScore = computed(() => getScore(currentScene.value))

function getScore(hl) {
  if (!hl) return null
  const v =
    hl.effective_score ??
    hl.score ??
    hl.max_score ??
    hl.actor_similarity ??
    hl.avg_similarity ??
    null
  return typeof v === 'number' ? Number(v.toFixed(3)) : null
}

// Format time
function toTime(sec) {
  if (sec == null || isNaN(sec)) return '--:--'
  const s = Math.max(0, Number(sec))
  const m = Math.floor(s / 60)
  const r = (s % 60).toFixed(2).padStart(5, '0')
  return `${m}:${r}`
}

// Điều hướng scene
async function goNext() {
  if (!hasMoreScene.value) return
  await loadNextSceneForCurrent()
  await nextTick()
  playIfVideo()
}

async function goPrev() {
  if (typeof loadSceneAtIndex === 'function') {
    const idx = Math.max(0, (currentSceneIndex.value ?? 0) - 1)
    await loadSceneAtIndex(idx)
    await nextTick()
    playIfVideo()
  }
}

async function jumpTo(idx) {
  if (typeof loadSceneAtIndex === 'function') {
    await loadSceneAtIndex(idx)
    await nextTick()
    playIfVideo()
  } else {
    const cur = currentSceneIndex.value ?? 0
    if (idx === cur) return
    if (idx < cur) {
      await ensureInitialScene()
    }
    for (let i = (currentSceneIndex.value ?? 0); i < idx; i++) {
      if (!hasMoreScene.value) break
      await loadNextSceneForCurrent()
    }
    await nextTick()
    playIfVideo()
  }
}

function playIfVideo() {
  const el = player.value
  if (el && typeof el.play === 'function') {
    el.muted = true
    el.play().catch(() => {})
  }
}

watch(() => currentScene.value, () => {
  nextTick().then(playIfVideo)
})
</script>

