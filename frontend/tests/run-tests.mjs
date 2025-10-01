import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'
import { pickActiveTimelineEntry } from '../src/utils/sceneTimeline.js'
import { toAbsoluteAssetUrl, __test__ as assetUrlTest } from '../src/utils/assetUrls.js'
import { useRecognitionStore, __test__ } from '../src/composables/useRecognitionStore.js'
import { __test__ as configTest } from '../src/utils/config.js'
const { normaliseMovies, ensureFrameMetadata } = __test__

const samplePayload = [
  {
    movie_id: 'm1',
    movie: 'Movie 1',
    score: 0.52,
    match_status: 'present',
    match_label: 'Xuất hiện trong phim',
    characters: [
      {
        movie_id: 'm1',
        character_id: 'c1',
        score: 0.52,
        match_status: 'present',
        match_label: 'Xuất hiện trong phim',
      },
      {
        movie_id: 'm1',
        character_id: 'c2',
        score: 0.34,
        match_status: 'near_match',
        match_label: 'Có nhân vật gần giống',
      },
    ],
  },
]

const normalised = normaliseMovies(samplePayload)
assert.equal(normalised.length, 1, 'Expected one movie entry after normalisation')
assert.equal(normalised[0].match_status, 'present', 'Movie status should be preserved')
assert.equal(normalised[0].characters.length, 2, 'All characters above threshold should remain')
assert.equal(
  normalised[0].characters[1].match_label,
  'Có nhân vật gần giống',
  'Character match label should be preserved',
)

const componentPath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/components/FaceSearch.vue')
const componentSource = await readFile(componentPath, 'utf-8')

assert(
  componentSource.includes('face-search__movie-label') &&
    componentSource.includes('{{ movie.match_label }}'),
  'Movie label should be rendered in the template',
)
assert(
  componentSource.includes('face-search__character-match') &&
    componentSource.includes('{{ currentCharacter.match_label }}'),
  'Current character label should be rendered in the template',
)
assert(
  componentSource.includes('face-search__character-badge') &&
    componentSource.includes('{{ character.match_label }}'),
  'Character list should display match labels',
)

console.log('Frontend snapshot tests passed.')

const sceneViewerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/components/SceneViewer.vue')
const sceneViewerSource = await readFile(sceneViewerPath, 'utf-8')

assert(
  sceneViewerSource.includes('computeSegmentSeekStart(effectiveHighlights.value[0])') ||
    sceneViewerSource.includes('computeSegmentSeekStart(filteredHighlights.value[0])'),
  'Scene viewer should prioritise highlight starts using the padded seek helper when highlights are available',
)

assert(
  !sceneViewerSource.includes('clip_offset') && !sceneViewerSource.includes('clip_url'),
  'Scene viewer should not reference clip-based offsets when seeking in the video',
)

assert(
  sceneViewerSource.includes('const targetTime = computeSegmentSeekStart(segment)'),
  'Selecting a highlight should compute a padded seek target before seeking',
)

assert(
  sceneViewerSource.includes('Array.isArray(scene.filtered_highlights)'),
  'Scene viewer should prioritise pre-filtered highlights when provided by the backend',
)

assert(
  sceneViewerSource.includes('const filteredHighlightsRes = computed(() => {'),
  'Scene viewer should compute highlight filtering metadata via the helper result structure',
)

assert(
  sceneViewerSource.includes('const rawHighlights = computed(() => {'),
  'Scene viewer should expose raw highlights for rescue rendering when filtering removes all items',
)

assert(
  sceneViewerSource.includes('const effectiveHighlights = computed(() => {'),
  'Scene viewer should expose a fallback highlight list when filtered highlights are empty',
)

assert(
  sceneViewerSource.includes('const filterDiagnosticMessage = computed(() => {'),
  'Scene viewer should surface diagnostic messaging when highlights are filtered out',
)

assert(
  sceneViewerSource.includes('scene-viewer__notice') &&
    sceneViewerSource.includes('Không có highlight đạt chuẩn sau khi áp dụng bộ lọc'),
  'Scene viewer should render a diagnostic banner when no highlights survive filtering',
)

assert(
  sceneViewerSource.includes('Đang hiển thị danh sách highlight gốc'),
  'Scene viewer should explain that raw highlights are displayed when filters remove all items',
)

assert(
  sceneViewerSource.includes("formatReason(reasons.score, 'Điểm')") &&
    sceneViewerSource.includes("formatReason(reasons.det_score, 'Điểm phát hiện')") &&
    sceneViewerSource.includes("formatReason(reasons.duration, 'Độ dài')"),
  'Diagnostic banner should enumerate score, detection score, and duration thresholds',
)

assert(
  sceneViewerSource.includes("console.debug('SceneViewer: no visible highlights rendered, falling back to raw data'") &&
    sceneViewerSource.includes('filterStats: stats'),
  'Scene viewer should log diagnostic statistics when falling back to raw highlights',
)


assert(
  sceneViewerSource.includes('video.currentTime = targetTime'),
  'Video playback should seek to the computed padded timestamp',
)

assert(
  sceneViewerSource.includes('Math.max(rawStart - getSeekPad(), 0)'),
  'Scene viewer should subtract the configured seek pad when determining highlight starts',
)

assert(
  sceneViewerSource.includes('MIN_VIEWABLE_SEC') ||
    sceneViewerSource.includes('isWithinSegmentWindow'),
  'Scene viewer should skip auto-pausing highlights shorter than the configured threshold',
)

assert(
  sceneViewerSource.includes('PAUSE_TOLERANCE_SEC'),
  'Scene viewer should rely on the configured tolerance when tracking highlight windows',
)

console.log('Scene viewer QA checks passed.')

const { resolveConfigValue, toNumber, metaEnv, runtimeEnv, globalEnv } = configTest

metaEnv.VITE_FAKE_TOLERANCE = '0.4'
runtimeEnv.VITE_FAKE_TOLERANCE = '0.5'
globalEnv.FAKE_TOLERANCE = '0.6'

assert.equal(
  resolveConfigValue('FAKE_TOLERANCE'),
  '0.6',
  'Config loader should prioritise runtime overrides injected into window.__APP_CONFIG__',
)

delete globalEnv.FAKE_TOLERANCE
metaEnv.FAKE_TOLERANCE = '0.7'

assert.equal(
  resolveConfigValue('FAKE_TOLERANCE'),
  '0.7',
  'Config loader should read direct meta env keys before prefixed variants',
)

delete metaEnv.FAKE_TOLERANCE
delete metaEnv.VITE_FAKE_TOLERANCE
delete runtimeEnv.VITE_FAKE_TOLERANCE

assert.equal(toNumber('1.25', 0), 1.25, 'Config loader should parse numeric strings correctly')
assert.equal(toNumber('NaN', 0.3), 0.3, 'Config loader should fall back when values are invalid')


const timeline = [
  { timestamp: 10, clip_offset: 0, duration: 2.5, bbox: [0, 0, 100, 100] },
  { timestamp: 12.5, clip_offset: 2.5, duration: 2.5, bbox: [50, 50, 150, 150] },
]
const selectedEarly = pickActiveTimelineEntry(timeline, 10.6, { fps: 2, sceneStart: 10 })
assert.strictEqual(
  selectedEarly,
  timeline[0],
  'Timeline entry before transition should use the first segment',
)
const selectedLate = pickActiveTimelineEntry(timeline, 14.9, { fps: 2, sceneStart: 10 })
assert.strictEqual(
  selectedLate,
  timeline[1],
  'Timeline selection should advance when clip_offset increases',
)

const legacySelected = pickActiveTimelineEntry(timeline, 0.6, 2)
assert.strictEqual(
  legacySelected,
  timeline[0],
  'Legacy selection should still support clip-based offsets',
)

console.log('Scene timeline utility tests passed.')

const assetBase = assetUrlTest.computeApiAssetBase('https://api.example.com/api/')
assert.equal(
  assetBase,
  'https://api.example.com',
  'API asset base should drop a trailing /api segment',
)

const clipAsset = toAbsoluteAssetUrl('clips/sample.mp4', 'https://api.example.com/api')
assert.equal(
  clipAsset,
  'https://api.example.com/clips/sample.mp4',
  'Relative clip paths should resolve against the API host',
)

const frameAsset = toAbsoluteAssetUrl('/frames/sample.jpg', 'https://api.example.com/api/')
assert.equal(
  frameAsset,
  'https://api.example.com/frames/sample.jpg',
  'Leading slashes should be handled when resolving asset URLs',
)

const previousWindow = globalThis.window
globalThis.window = { location: { origin: 'https://frontend.example.com' } }

try {
  const relativeBase = assetUrlTest.computeApiAssetBase('/api')
  assert.equal(
    relativeBase,
    'https://frontend.example.com',
    'Relative API bases should resolve against the current origin',
  )

  const relativeClip = toAbsoluteAssetUrl('/clips/example.mp4', '/api')
  assert.equal(
    relativeClip,
    'https://frontend.example.com/clips/example.mp4',
    'Clip URLs should resolve against the origin when API base is relative',
  )

  const relativeFrame = toAbsoluteAssetUrl('frames/example.jpg', '/api/')
  assert.equal(
    relativeFrame,
    'https://frontend.example.com/frames/example.jpg',
    'Frame URLs should resolve against the origin when API base is relative',
  )
} finally {
  globalThis.window = previousWindow
}


const normalisedMetadata = ensureFrameMetadata({
  frame: '/frames/example.jpg',
  preview_image: 'previews/example.jpg',
  clip_url: '/clips/example.mp4',
  video_url: '/videos/example.mp4',
  timeline: [
    {
      frame_url: 'frames/nested.jpg',
      clip_path: '/clips/nested.mp4',
      preview_path: 'previews/nested.jpg',
    },
  ],
})

assert.equal(
  normalisedMetadata.frame,
  'http://localhost:8000/frames/example.jpg',
  'Frame URLs in metadata should be converted to absolute URLs',
)
assert.equal(
  normalisedMetadata.preview_image,
  'http://localhost:8000/previews/example.jpg',
  'Preview images should resolve to absolute API URLs',
)
assert.equal(
  normalisedMetadata.clip_url,
  'http://localhost:8000/clips/example.mp4',
  'Clip URLs should resolve to the API host',
)
assert.equal(
  normalisedMetadata.video_url,
  'http://localhost:8000/videos/example.mp4',
  'Video URLs should resolve to the API host',
)
assert.equal(
  normalisedMetadata.timeline[0].frame,
  'http://localhost:8000/frames/nested.jpg',
  'Nested timeline frames should inherit absolute URLs',
)
assert.equal(
  normalisedMetadata.timeline[0].clip_path,
  'http://localhost:8000/clips/nested.mp4',
  'Nested clip paths should be normalised to absolute URLs',
)
assert.equal(
  normalisedMetadata.timeline[0].preview_path,
  'http://localhost:8000/previews/nested.jpg',
  'Nested preview paths should be normalised to absolute URLs',
)

console.log('Asset URL helper tests passed.')

const clientFilteredScene = ensureFrameMetadata({
  highlights: [
    { id: 'h1', start: 10, end: 14, score: 0.8 },
  ],
  highlight_support: { det_score_threshold: 0.75, min_duration: 4, highlight_count: 1 },
})

assert.equal(
  clientFilteredScene.filtered_highlights.length,
  1,
  'Scenes with only raw highlights should be filtered on the client',
)
assert.equal(
  clientFilteredScene.highlight_display_count,
  1,
  'Filtered highlight counts should reflect the number of accepted segments',
)
assert.equal(
  clientFilteredScene.filtered_highlights[0].id,
  'h1',
  'Client-side filtering should retain highlight metadata',
)

assert.equal(
  clientFilteredScene.highlight_support.highlight_count,
  1,
  'Client-side filtering should preserve highlight support counts from the backend',
)
assert(clientFilteredScene.__fh_stats, 'Client-side filtering should expose highlight stats metadata')
assert.equal(
  clientFilteredScene.__fh_stats.reasons.duration,
  4,
  'Client-side stats should record the duration threshold',
)
assert.equal(
  clientFilteredScene.__fh_stats.reasons.score,
  0.75,
  'Client-side stats should record the score threshold',
)
assert.equal(
  clientFilteredScene.__fh_stats.reasons.det_score,
  0.75,
  'Client-side stats should record the detection score threshold',
)

const filterExhaustedScene = ensureFrameMetadata({
  highlights: [
    { id: 'raw-1', start: 0, end: 2.5, max_score: 0.5 },
    { id: 'raw-2', start: 4, end: 6, max_score: 0.55 },
  ],
  highlight_support: { det_score_threshold: 0.9, min_duration: 4, highlight_count: 2 },
})

assert.equal(
  filterExhaustedScene.filtered_highlights.length,
  0,
  'Scenes should report zero filtered highlights when entries do not meet thresholds',
)
assert.equal(
  filterExhaustedScene.highlight_display_count,
  0,
  'Scenes should surface a filtered count of zero when filters remove all highlights',
)
assert.equal(
  filterExhaustedScene.highlights.length,
  2,
  'Scenes should retain the raw highlight list when falling back to original data',
)
assert.equal(
  filterExhaustedScene.highlight_support.highlight_count,
  2,
  'Scenes should retain highlight support counts even when filtering removes all highlights',
)
assert(filterExhaustedScene.__fh_stats, 'Scenes should retain highlight filter statistics when filters return no entries')
assert.equal(
  filterExhaustedScene.__fh_stats.outCount,
  2,
  'Scenes should report the number of highlights removed by filtering',
)
assert.equal(
  filterExhaustedScene.__fh_stats.reasons.score,
  0.9,
  'Filter stats should expose the score threshold used when removing highlights',
)
assert.equal(
  filterExhaustedScene.__fh_stats.reasons.det_score,
  0.9,
  'Filter stats should expose the detection score threshold used when removing highlights',
)
assert.equal(
  filterExhaustedScene.__fh_stats.reasons.duration,
  4,
  'Filter stats should expose the duration threshold used when removing highlights',
)

const preferredFilteredScene = ensureFrameMetadata({
  highlights: [],
  filtered_highlights: [
    { id: 'h2', start: 5, end: 9, score: 0.82 },
  ],
  highlight_support: { det_score_threshold: 0.75, min_duration: 4 },
  highlight_total: 3,
})

assert.equal(
  preferredFilteredScene.highlights.length,
  0,
  'Scenes should preserve empty raw highlight lists when sent from the backend',
)
assert.equal(
  preferredFilteredScene.filtered_highlights.length,
  1,
  'Scenes should rely on backend-provided filtered highlight lists when available',
)
assert.equal(
  preferredFilteredScene.highlight_display_count,
  1,
  'Backend filtered highlight counts should surface through scene metadata',
)
assert.equal(
  preferredFilteredScene.highlight_total,
  3,
  'Scene metadata should preserve backend highlight totals when provided',
)
assert.equal(
  preferredFilteredScene.filtered_highlights[0].id,
  'h2',
  'Backend filtered highlight metadata should be passed through intact',
)

const highlightStore = useRecognitionStore()
highlightStore.resetSearch()
highlightStore.state.movies = [
  {
    movie_id: 'm1',
    movie: 'Movie 1',
    characters: [
      {
        movie_id: 'm1',
        character_id: 'c1',
        scene: null,
        scene_index: null,
        next_scene_cursor: null,
        total_scenes: null,
        has_more_scenes: false,
        verificationStatus: null,
        decisionHistory: [],
      },
    ],
  },
]
highlightStore.state.selectedMovieId = 'm1'
highlightStore.state.selectedCharacterId = 'c1'

const highlightKey = 'm1::c1'

highlightStore.updateSceneEntry({
  movie_id: 'm1',
  character_id: 'c1',
  scene_index: 0,
  next_cursor: 1,
  total_scenes: 2,
  scene: {
    highlights: [
      { start: 10, end: 15, max_score: 0.95 },
      { start: 20, end: 25, max_score: 0.4 },
    ],
    highlight_index: 0,
    highlight_total: 2,
    source_scene_index: 0,
  },
})

assert.equal(
  highlightStore.state.scenes[highlightKey].cursor,
  0,
  'First highlight should set cursor 0',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].next_cursor,
  1,
  'Next cursor should point to the second highlight',
)
const firstEntry = highlightStore.state.scenes[highlightKey].entries[0]
assert.ok(firstEntry, 'Scene cache should store the first highlight entry')
assert.equal(
  firstEntry.next_cursor,
  1,
  'First highlight entry should track the next cursor',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].total_scenes,
  2,
  'Total highlights should be tracked',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].highlight_total,
  2,
  'Scene cache should retain backend highlight totals',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].highlight_display_count,
  1,
  'Scene cache should expose filtered highlight counts',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlights.length,
    2,
  'Character scene should retain the raw highlight list from the backend',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.filtered_highlights.length,
  1,
  'Character scene should expose a single filtered highlight segment',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_total,
  2,
  'Character scene should retain highlight metadata',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_display_count,
  1,
  'Character scene should expose filtered highlight counts',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].has_more_scenes,
  true,
  'Character should report remaining highlights',
)

highlightStore.updateSceneEntry({
  movie_id: 'm1',
  character_id: 'c1',
  scene_index: 1,
  next_cursor: null,
  total_scenes: 2,
  scene: {
    highlights: [{ start: 30, end: 35.2, max_score: 0.88 }],
    highlight_index: 1,
    highlight_total: 2,
    source_scene_index: 0,
  },
  has_more: false,
})

assert.equal(
  highlightStore.state.scenes[highlightKey].scene_index,
  highlightStore.state.scenes[highlightKey].cursor,
  1,
  'Second highlight should update the cursor index',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].has_more,
  false,
  'Scene cache should report no further highlights',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].highlight_display_count,
  1,
  'Scene cache should continue reporting filtered highlight counts',
)
assert.equal(
  Object.keys(highlightStore.state.scenes[highlightKey].entries).length,
  2,
  'Scene cache should retain previous highlight entries',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_index,
  1,
  'Character scene should switch to the second highlight',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.scene_index,
  1,
  'Character scene should expose the updated cursor index',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].has_more_scenes,
  false,
  'Character should report no remaining highlights',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].highlight_display_count,
  1,
  'Character metadata should track filtered highlight counts',
)

highlightStore.resetSearch()

console.log('Highlight navigation tests passed.')

const backendHighlightMeta = ensureFrameMetadata({
  highlight_total: 3,
  highlight_support: { det_score_threshold: 0.75, min_duration: 4 },
  highlights: [
    { start: 0, end: 6, max_score: 0.92 },
    { start: 10, end: 15, max_score: 0.81 },
    { start: 20, end: 26, max_score: 0.5 },
  ],
})

assert.equal(
  backendHighlightMeta.highlight_total,
  3,
  'ensureFrameMetadata should keep backend highlight totals',
)
assert.equal(
  backendHighlightMeta.highlight_display_count,
  2,
  'ensureFrameMetadata should report filtered highlight counts',
)

const relaxedHighlightMeta = ensureFrameMetadata({
  highlight_total: 1,
  highlight_support: { det_score_threshold: 0.6, min_duration: 4 },
  highlights: [{ start: 30, end: 35, max_score: 0.65 }],
})

assert.equal(
  relaxedHighlightMeta.highlight_display_count,
  1,
  'Highlights meeting relaxed backend thresholds should be preserved',
)
assert.equal(
  relaxedHighlightMeta.highlight_total,
  1,
  'Relaxed threshold scenes should retain backend highlight totals',
)

console.log('Highlight threshold synchronisation tests passed.')