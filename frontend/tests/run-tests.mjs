import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import { JSDOM } from 'jsdom'
import { parse, compileScript } from '@vue/compiler-sfc'
import { pickActiveTimelineEntry } from '../src/utils/sceneTimeline.js'
import { toAbsoluteAssetUrl, __test__ as assetUrlTest } from '../src/utils/assetUrls.js'
import { __test__ as configTest } from '../src/utils/config.js'

const initialDom = new JSDOM('<!doctype html><html><body></body></html>')
globalThis.window = initialDom.window
globalThis.document = initialDom.window.document
globalThis.navigator = initialDom.window.navigator
globalThis.HTMLElement = initialDom.window.HTMLElement
globalThis.SVGElement = initialDom.window.SVGElement
globalThis.Element = initialDom.window.Element
globalThis.Node = initialDom.window.Node
globalThis.Document = initialDom.window.Document
globalThis.XMLSerializer = initialDom.window.XMLSerializer
globalThis.getComputedStyle = initialDom.window.getComputedStyle.bind(initialDom.window)
if (!globalThis.URL) {
  globalThis.URL = initialDom.window.URL
}
if (!globalThis.URL.createObjectURL) {
  globalThis.URL.createObjectURL = () => 'blob:mock'
}
if (!globalThis.URL.revokeObjectURL) {
  globalThis.URL.revokeObjectURL = () => {}
}

const { useRecognitionStore, __test__ } = await import(
  '../src/composables/useRecognitionStore.js'
)
const { normaliseMovies, ensureFrameMetadata } = __test__

const buildDataModule = (code) => {
  const base64 = Buffer.from(code, 'utf-8').toString('base64')
  return `data:text/javascript;base64,${base64}`
}

const require = createRequire(import.meta.url)
const vueRuntimeUrl = pathToFileURL(
  require.resolve('vue/dist/vue.runtime.esm-bundler.js'),
).href

const compileVueModuleUrl = async (filePath, replacements = {}) => {
  const source = await readFile(filePath, 'utf-8')
  const { descriptor } = parse(source, { filename: filePath })
  const script = compileScript(descriptor, { id: filePath, inlineTemplate: true })
  let code = script.content

  if (vueRuntimeUrl) {
    code = code.split("from 'vue'").join(`from '${vueRuntimeUrl}'`)
    code = code.split('from "vue"').join(`from "${vueRuntimeUrl}"`)
  }

  for (const [original, replacement] of Object.entries(replacements)) {
    const singleQuoted = `'${original}'`
    const doubleQuoted = `"${original}"`
    if (code.includes(singleQuoted)) {
      code = code.split(singleQuoted).join(`'${replacement}'`)
    }
    if (code.includes(doubleQuoted)) {
      code = code.split(doubleQuoted).join(`"${replacement}"`)
    }
  }

  code = code.replace(/from ['"](\.\.?(?:\/[\w.-]+)+)['"]/g, (match, relPath) => {
    const absolutePath = resolve(dirname(filePath), relPath)
    const fileUrl = pathToFileURL(absolutePath).href
    return match.replace(relPath, fileUrl)
  })

  return buildDataModule(code)
}

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

{
  const captured = []
  const originalDebug = console.debug
  console.debug = (...args) => {
    captured.push(args)
  }
  try {
    const entry = ensureFrameMetadata({
      duration: 50,
      highlights: [
        { id: 'a', start: 5, end: 5.4, duration: 0.4, order: 1 },
        { id: 'b', start: 7, end: 8, duration: 1, order: 2 },
        { id: 'c', start: 40, end: 40.5, duration: 0.5, order: 3 },
      ],
    })

    assert(Array.isArray(entry.highlights), 'Highlight array should be preserved from the backend')
    assert.equal(entry.highlights.length, 3, 'All backend highlights should remain available')
    assert.equal(entry.highlight_total, 3, 'Highlight totals should default to the backend highlight count when missing')
    assert.equal(
      entry.highlight_display_count,
      3,
      'Display counts should default to the backend highlight count when missing',
    )

    const debugInvocation = captured.find(
      (args) => Array.isArray(args) && typeof args[0] === 'string' && args[0].includes('DEBUG_HL'),
    )
    assert(debugInvocation, 'DEBUG_HL logging should be emitted when recording backend highlights')
    assert(debugInvocation?.[0]?.includes('ensureFrameMetadata'), 'Highlight debug log should tag ensureFrameMetadata')
    const payload = debugInvocation?.[1]
    assert.equal(payload?.highlightCount, 3, 'Debug log should include the highlight count received from the backend')
    assert.equal(payload?.reportedTotal ?? null, null, 'Debug log should include the reported total when provided')
    assert.equal(payload?.reportedDisplay ?? null, null, 'Debug log should include the reported display count when provided')
  } finally {
    console.debug = originalDebug
  }
}


console.log('Frontend snapshot tests passed.')

const sceneViewerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/components/SceneViewer.vue')
const sceneViewerSource = await readFile(sceneViewerPath, 'utf-8')

assert(
  sceneViewerSource.includes('const availableHighlights = computed(() => {'),
  'Scene viewer should read highlight data directly from the backend payload',
)

assert(
  sceneViewerSource.includes('computeSegmentSeekStart(availableHighlights.value[0])'),
  'Scene viewer should prioritise highlight starts using the padded seek helper when highlights are available',
)

assert(
  sceneViewerSource.includes("console.debug('DEBUG_HL SceneViewer menu'"),
  'Scene viewer should emit DEBUG_HL logs when rendering the highlight menu',
)

assert(
  sceneViewerSource.includes("console.debug('DEBUG_HL SceneViewer playback'"),
  'Scene viewer should emit DEBUG_HL logs for highlight playback decisions',
)


assert(
  !sceneViewerSource.includes('filtered_highlights') && !sceneViewerSource.includes('merged_highlights'),
  'Scene viewer should no longer reference client-side merged or filtered highlights',
)

assert(
  !sceneViewerSource.includes('scene-viewer__notice'),
  'Scene viewer should no longer render the filtered highlight diagnostic banner',
)

assert(
  !sceneViewerSource.includes('clip_offset') && !sceneViewerSource.includes('clip_url'),
  'Scene viewer should not reference clip-based offsets when seeking in the video',
)

assert(
  sceneViewerSource.includes("console.debug('SceneViewer: no visible highlights rendered, falling back to raw data'") &&
    sceneViewerSource.includes('filterStats: stats'),
  'Scene viewer should log diagnostic statistics when falling back to raw highlights',
)


assert(
  sceneViewerSource.includes('scene-viewer__next-button'),
  'Scene viewer should expose a manual control for playing the next highlight once playback pauses',
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
  sceneViewerSource.includes('PAUSE_TOLERANCE_SEC'),
  'Scene viewer should rely on the configured tolerance when tracking highlight windows',
)

{
  const sceneViewerModuleUrl = await compileVueModuleUrl(sceneViewerPath)
  const { createApp, nextTick } = await import(vueRuntimeUrl)
  const { default: SceneViewer } = await import(sceneViewerModuleUrl)

  let summary
  const props = {
    scene: {
      highlights: [
        { id: 'a', start: 1, end: 2 },
        { id: 'b', start: 3, end: 4 },
      ],
      highlight_total: '5',
    },
  }

  const container = document.createElement('div')
  const app = createApp(SceneViewer, props)
  app.mount(container)
  await nextTick()
  summary = container.querySelector('.scene-viewer__meta')?.textContent ?? ''
  app.unmount()

  assert.equal(
    summary,
    '2/5 highlight từ backend',
    'Highlight stats should fall back to the available highlight count when backend display totals are missing',
  )
}


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
  2,
  'Scene cache should surface backend highlight counts when they are not provided explicitly',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlights.length,
    2,
  'Character scene should retain the raw highlight list from the backend',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_total,
  2,
  'Character scene should retain highlight metadata',
)
assert.equal(
  highlightStore.state.movies[0].characters[0].scene.highlight_display_count,
  2,
  'Character scene should expose the backend highlight count',
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
  'Scene cache should reflect the highlight count reported for the active scene',
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
  'Character metadata should update when the backend reports fewer highlights',
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
  3,
  'ensureFrameMetadata should surface the backend highlight count when provided',
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

highlightStore.resetSearch()

highlightStore.state.movies = [
  {
    movie_id: 'nav-movie',
    movie: 'Navigation Movie',
    characters: [
      {
        movie_id: 'nav-movie',
        character_id: 'nav-character',
        scene: null,
        scene_index: null,
        next_scene_cursor: null,
        total_scenes: null,
        has_more_scenes: false,
        decisionHistory: [],
        verificationStatus: null,
        highlight_total: null,
        highlight_display_count: 0,
      },
    ],
  },
]

highlightStore.state.selectedMovieId = 'nav-movie'
highlightStore.state.selectedCharacterId = 'nav-character'

const navKey = 'nav-movie::nav-character'

const createSceneEntry = (index, nextCursor, hasMore) => ({
  movie_id: 'nav-movie',
  character_id: 'nav-character',
  scene_index: index,
  next_cursor: nextCursor,
  total_scenes: 3,
  highlight_total: 3,
  highlight_display_count: 3,
  has_more: hasMore,
  scene: {
    scene_index: index,
    highlight_index: index,
    highlight_total: 3,
    highlight_display_count: 3,
    source_scene_index: index,
    highlights: [
      { id: `h-${index}`, start: index * 10, end: index * 10 + 5, max_score: 0.9 },
    ],
  },
})

highlightStore.updateSceneEntry(createSceneEntry(0, 1, true))
highlightStore.updateSceneEntry(createSceneEntry(1, 2, true))
highlightStore.updateSceneEntry(createSceneEntry(2, null, false))

assert.equal(
  Object.keys(highlightStore.state.scenes[navKey].entries).length,
  3,
  'Scene cache should retain all loaded cursor entries',
)

assert.equal(
  highlightStore.currentSceneNavigation.value.knownCount,
  3,
  'Navigation metadata should report the number of cached highlights',
)

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  2,
  'Latest update should advance the active cursor index',
)

await highlightStore.loadSceneAtIndex('nav-movie', 'nav-character', 0)

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  0,
  'loadSceneAtIndex should jump to the requested cursor when cached',
)

await highlightStore.loadSceneAtIndex('nav-movie', 'nav-character', 1)

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  1,
  'Subsequent navigation should advance to the second highlight',
)

assert.equal(
  highlightStore.state.movies[0].characters[0].scene.scene_index,
  1,
  'Active character scene metadata should follow the navigation index',
)

await highlightStore.loadSceneAtIndex('nav-movie', 'nav-character', 2)

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  2,
  'Navigation should reach the final cached highlight entry',
)

assert.equal(
  highlightStore.currentSceneNavigation.value.hasMore,
  false,
  'Navigation metadata should report when no further highlights are available',
)

const missingHighlight = await highlightStore.loadSceneAtIndex(
  'nav-movie',
  'nav-character',
  3,
)

assert.equal(missingHighlight, null, 'Out-of-range navigation should not resolve a scene')
assert(
  highlightStore.state.sceneError,
  'Store should surface an error when navigating beyond known highlights',
)

highlightStore.resetSearch()

console.log('Highlight cursor navigation tests passed.')

const dom = new JSDOM('<!doctype html><html><body></body></html>')
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.navigator = dom.window.navigator
globalThis.HTMLElement = dom.window.HTMLElement
globalThis.SVGElement = dom.window.SVGElement
globalThis.Element = dom.window.Element
globalThis.Node = dom.window.Node
globalThis.Document = dom.window.Document
globalThis.XMLSerializer = dom.window.XMLSerializer
globalThis.getComputedStyle = dom.window.getComputedStyle.bind(dom.window)
globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0)
globalThis.cancelAnimationFrame = (id) => clearTimeout(id)
if (!globalThis.URL) {
  globalThis.URL = dom.window.URL
}
if (!globalThis.URL.createObjectURL) {
  globalThis.URL.createObjectURL = () => 'blob:mock'
}
if (!globalThis.URL.revokeObjectURL) {
  globalThis.URL.revokeObjectURL = () => {}
}

const sceneViewerStubModule = buildDataModule(`
  import { defineComponent, h } from '${vueRuntimeUrl}';
  export default defineComponent({
    name: 'SceneViewerStub',
    props: {
      scene: { type: Object, default: null },
      meta: { type: Object, default: null },
      highlightIndex: { type: Number, default: null },
      highlightTotal: { type: Number, default: null },
    },
    emits: ['highlight-change'],
    setup(props) {
      return () =>
        h('div', { class: 'scene-viewer-stub' }, [
          h('span', { class: 'scene-stub__index' }, props.highlightIndex ?? ''),
          h('span', { class: 'scene-stub__total' }, props.highlightTotal ?? ''),
          h(
            'span',
            { class: 'scene-stub__scene-index' },
            props.scene && props.scene.scene_index !== undefined
              ? String(props.scene.scene_index)
              : '',
          ),
        ])
    },
  })
`)

const faceSearchPath = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../src/components/FaceSearch.vue',
)

const faceSearchModuleUrl = await compileVueModuleUrl(faceSearchPath, {
  './SceneViewer.vue': sceneViewerStubModule,
})

const { default: FaceSearchComponent } = await import(faceSearchModuleUrl)
const { mount } = await import('@vue/test-utils')
const { nextTick } = await import(vueRuntimeUrl)

highlightStore.resetSearch()
highlightStore.state.movies = [
  {
    movie_id: 'movie-interaction',
    movie: 'Interaction Test Movie',
    characters: [
      {
        movie_id: 'movie-interaction',
        character_id: 'char-interaction',
        scene: null,
        scene_index: null,
        next_scene_cursor: null,
        total_scenes: null,
        has_more_scenes: false,
        decisionHistory: [],
        verificationStatus: null,
      },
    ],
  },
]

highlightStore.state.selectedMovieId = 'movie-interaction'
highlightStore.state.selectedCharacterId = 'char-interaction'

const interactionScene = (index, nextCursor, hasMore) => ({
  movie_id: 'movie-interaction',
  character_id: 'char-interaction',
  scene_index: index,
  next_cursor: nextCursor,
  total_scenes: 3,
  highlight_total: 3,
  highlight_display_count: 3,
  has_more: hasMore,
  scene: {
    scene_index: index,
    highlight_index: index,
    highlight_total: 3,
    highlight_display_count: 3,
    source_scene_index: index,
    highlights: [
      { id: `nav-${index}`, start: index * 5, end: index * 5 + 3, max_score: 0.85 },
    ],
  },
})

highlightStore.updateSceneEntry(interactionScene(0, 1, true))
highlightStore.updateSceneEntry(interactionScene(1, 2, true))
highlightStore.updateSceneEntry(interactionScene(2, null, false))

await highlightStore.ensureInitialScene('movie-interaction', 'char-interaction')

const wrapper = mount(FaceSearchComponent, { attachTo: document.body })

await nextTick()
await nextTick()

const labelNode = wrapper.element.querySelector('.face-search__highlight-count')
assert(labelNode, 'Highlight navigation label should render')
assert.equal(
  labelNode.textContent.trim(),
  'Highlight 1/4',
  'Initial highlight label should show the first scene out of four known items',
)

const initialSceneNode = wrapper.element.querySelector('.scene-stub__scene-index')
assert(initialSceneNode, 'Scene viewer stub should render an index placeholder')
assert.equal(
  initialSceneNode.textContent.trim(),
  '0',
  'Scene viewer stub should display the first scene index by default',
)

const highlightButtons = Array.from(
  wrapper.element.querySelectorAll('.face-search__highlight-button'),
)
assert.equal(highlightButtons.length, 2, 'Highlight navigation should render previous and next buttons')

highlightButtons[1].dispatchEvent(new dom.window.Event('click', { bubbles: true }))
await nextTick()

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  1,
  'Clicking the next button should advance the store cursor',
)
const nextLabelNode = wrapper.element.querySelector('.face-search__highlight-count')
assert(nextLabelNode, 'Highlight label should remain visible after advancing')
assert.equal(
  nextLabelNode.textContent.trim(),
  'Highlight 2/4',
  'Highlight label should update after moving forward',
)
const forwardSceneNode = wrapper.element.querySelector('.scene-stub__scene-index')
assert(forwardSceneNode, 'Scene viewer stub should expose an index after advancing')
assert.equal(
  forwardSceneNode.textContent.trim(),
  '1',
  'Scene viewer stub should receive the updated scene metadata',
)

highlightButtons[0].dispatchEvent(new dom.window.Event('click', { bubbles: true }))
await nextTick()

assert.equal(
  highlightStore.currentSceneNavigation.value.index,
  0,
  'Clicking the previous button should move the cursor backwards',
)
const previousLabelNode = wrapper.element.querySelector('.face-search__highlight-count')
assert(previousLabelNode, 'Highlight label should remain visible after navigating back')
assert.equal(
  previousLabelNode.textContent.trim(),
  'Highlight 1/4',
  'Highlight label should revert after navigating back',
)
const backwardSceneNode = wrapper.element.querySelector('.scene-stub__scene-index')
assert(backwardSceneNode, 'Scene viewer stub should expose an index after navigating back')
assert.equal(
  backwardSceneNode.textContent.trim(),
  '0',
  'Scene viewer stub should reflect the restored scene metadata',
)

wrapper.unmount()
highlightStore.resetSearch()

console.log('FaceSearch highlight interaction tests passed.')