import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'

import { __test__ } from '../src/composables/useRecognitionStore.js'
import {
  collectBoxesFromScene,
  collectBoxesFromTimelineEntry,
  computeOverlayBoxes,
  pickActiveTimelineEntry,
  scaleBoxes,
  toBox,
} from '../src/utils/sceneTimeline.js'

const { normaliseMovies } = __test__

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

const simpleBox = toBox([10, 20, 30, 60])
assert.deepEqual(
  simpleBox,
  { x: 10, y: 20, width: 20, height: 40 },
  'toBox should convert array coordinates to box objects',
)

const sceneBoxes = collectBoxesFromScene({ bbox: [0, 0, 100, 100], boxes: [[10, 10, 60, 90]] })
assert.equal(sceneBoxes.length, 2, 'Scene-level boxes should merge bbox and boxes array')

const timeline = [
  { clip_offset: 0, duration: 2.5, bbox: [0, 0, 100, 100] },
  { clip_offset: 2.5, duration: 2.5, bbox: [50, 50, 150, 150] },
]
const selectedEarly = pickActiveTimelineEntry(timeline, 0.6, 2)
assert.strictEqual(
  selectedEarly,
  timeline[0],
  'Timeline entry before transition should use the first segment',
)
const selectedLate = pickActiveTimelineEntry(timeline, 4.9, 2)
assert.strictEqual(
  selectedLate,
  timeline[1],
  'Timeline selection should advance when clip_offset increases',
)

const timelineBoxes = collectBoxesFromTimelineEntry(selectedLate)
assert.equal(timelineBoxes.length, 1, 'Timeline entry should expose bbox for overlays')
const scaled = scaleBoxes(timelineBoxes, 200, 200)
assert.deepEqual(
  scaled[0],
  { left: '25%', top: '25%', width: '50%', height: '50%' },
  'Scaled boxes should respect relative dimensions',
)

const overlay = computeOverlayBoxes(
  { timeline, clip_fps: 2 },
  200,
  200,
  4.9,
)
assert.equal(overlay.length, 1, 'Overlay helper should honour active timeline entries')
assert.equal(
  overlay[0].left,
  '25%',
  'Overlay helper should convert boxes to CSS percentages',
)

console.log('Scene timeline utility tests passed.')