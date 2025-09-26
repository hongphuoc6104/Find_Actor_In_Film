import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'
import { pickActiveTimelineEntry } from '../src/utils/sceneTimeline.js'
import { toAbsoluteAssetUrl, __test__ as assetUrlTest } from '../src/utils/assetUrls.js'
import { useRecognitionStore, __test__ } from '../src/composables/useRecognitionStore.js'
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
    highlights: [{ start: 10, end: 12.5, max_score: 0.95 }],
    highlight_index: 0,
    highlight_total: 2,
    source_scene_index: 0,
  },
})

assert.equal(
  highlightStore.state.scenes[highlightKey].scene_index,
  0,
  'First highlight should set cursor 0',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].next_cursor,
  1,
  'Next cursor should point to the second highlight',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].total_scenes,
  2,
  'Total highlights should be tracked',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlights.length,
  1,
  'Character scene should use a single highlight segment',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.scene_index,
  0,
  'Character scene should report the highlight cursor index',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_total,
  2,
  'Character scene should retain highlight metadata',
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
    highlights: [{ start: 30, end: 33.2, max_score: 0.88 }],
    highlight_index: 1,
    highlight_total: 2,
    source_scene_index: 0,
  },
  has_more: false,
})

assert.equal(
  highlightStore.state.scenes[highlightKey].scene_index,
  1,
  'Second highlight should update the cursor index',
)
assert.equal(
  highlightStore.state.scenes[highlightKey].has_more,
  false,
  'Scene cache should report no further highlights',
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

highlightStore.resetSearch()

console.log('Highlight navigation tests passed.')