import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'

import { __test__ } from '../src/composables/useRecognitionStore.js'

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