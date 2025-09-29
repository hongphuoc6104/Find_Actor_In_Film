const metaEnv = typeof import.meta !== 'undefined' && import.meta?.env ? import.meta.env : {}
const runtimeEnv = typeof process !== 'undefined' && process?.env ? process.env : {}
const globalEnv =
  typeof window !== 'undefined' && window?.__APP_CONFIG__ ? window.__APP_CONFIG__ : {}
const resolveConfigValue = (key) => {
  if (key in globalEnv) return globalEnv[key]
  if (key in metaEnv) return metaEnv[key]
  const viteKey = `VITE_${key}`
  if (viteKey in metaEnv) return metaEnv[viteKey]
  if (key in runtimeEnv) return runtimeEnv[key]
  if (viteKey in runtimeEnv) return runtimeEnv[viteKey]
  return undefined
}

const toNumber = (value, fallback) => {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

const toFiniteNumber = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const parseThresholdFromSource = (source, keys) => {
  if (!source || typeof source !== 'object') {
    return null
  }

  for (const key of keys) {
    if (key in source) {
      const parsed = toFiniteNumber(source[key])
      if (parsed !== null) {
        return parsed
      }
    }
  }
  return null
}


export const SEEK_PAD_SEC = toNumber(resolveConfigValue('SEEK_PAD_SEC'), 0)
export const PAUSE_TOLERANCE_SEC = toNumber(resolveConfigValue('PAUSE_TOLERANCE_SEC'), 0.2)
export const MIN_VIEWABLE_SEC = toNumber(resolveConfigValue('MIN_VIEWABLE_SEC'), 0.35)
export const DEFAULT_HIGHLIGHT_SCORE_THRESHOLD = toNumber(
  resolveConfigValue('HIGHLIGHT_SCORE_THRESHOLD'),
  0.75,
)
export const DEFAULT_HIGHLIGHT_MIN_DURATION_SEC = toNumber(
  resolveConfigValue('MIN_HL_DURATION_SEC'),
  4.0,
)


export const resolveHighlightThresholds = (...sources) => {
  const orderedSources = sources.flat().filter((item) => item && typeof item === 'object')

  let minScore = null
  let minDuration = null

  for (const source of orderedSources) {
    if (minScore === null) {
      minScore = parseThresholdFromSource(source, [
        'minScore',
        'min_score',
        'det_score_threshold',
        'MIN_SCORE',
      ])
    }
    if (minDuration === null) {
      minDuration = parseThresholdFromSource(source, [
        'minDuration',
        'min_duration',
        'MIN_HL_DURATION_SEC',
      ])
    }
    if (minScore !== null && minDuration !== null) {
      break
    }
  }

  return {
    minScore: minScore ?? DEFAULT_HIGHLIGHT_SCORE_THRESHOLD,
    minDuration: minDuration ?? DEFAULT_HIGHLIGHT_MIN_DURATION_SEC,
  }
}

export const __test__ = {
  resolveConfigValue,
  toNumber,
  metaEnv,
  runtimeEnv,
  globalEnv,
  toFiniteNumber,
  parseThresholdFromSource,
  resolveHighlightThresholds,
}