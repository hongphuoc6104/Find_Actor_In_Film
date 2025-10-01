import { toAbsoluteAssetUrl } from './assetUrls.js'
import {
  DEFAULT_HIGHLIGHT_MIN_DURATION_SEC,
  DEFAULT_HIGHLIGHT_SCORE_THRESHOLD,
  resolveHighlightThresholds,
} from './config.js'

const toFiniteNumber = (value) => {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

const normaliseDuration = (highlight, start, end) => {
  const parsed = toFiniteNumber(highlight?.duration)
  if (parsed !== null) {
    return parsed
  }
  if (start !== null && end !== null && end >= start) {
    return Number((end - start).toFixed(3))
  }
  return null
}

const resolveScoreCandidates = (highlight) => {
  const candidates = []
  const pushIfNumber = (value) => {
    const parsed = toFiniteNumber(value)
    if (parsed !== null) {
      candidates.push(parsed)
    }
  }

  pushIfNumber(highlight?.score)
  pushIfNumber(highlight?.max_score)
  pushIfNumber(highlight?.avg_similarity)
  pushIfNumber(highlight?.max_similarity)
  pushIfNumber(highlight?.min_similarity)
  pushIfNumber(highlight?.actor_similarity)

  return candidates
}

const resolveReasonThreshold = (sources, keys, fallback) => {
  for (const source of sources) {
    if (!source || typeof source !== 'object') {
      continue
    }
    for (const key of keys) {
      if (key in source) {
        const parsed = toFiniteNumber(source[key])
        if (parsed !== null) {
          return parsed
        }
      }
    }
  }
  return fallback
}


const convertAssetFields = (highlight) => {
  const assetKeys = [
    'clip',
    'clip_url',
    'clip_path',
    'video',
    'video_url',
    'video_path',
    'frame',
    'frame_url',
    'frame_path',
    'preview',
    'preview_url',
    'preview_path',
    'thumbnail',
    'image',
  ]

  assetKeys.forEach((key) => {
    const value = highlight[key]
    if (typeof value === 'string' && value) {
      highlight[key] = toAbsoluteAssetUrl(value)
    }
  })

  if (Array.isArray(highlight.supporting_detections)) {
    highlight.supporting_detections = highlight.supporting_detections.map((entry) => {
      if (!entry || typeof entry !== 'object') {
        return entry
      }
      const copy = { ...entry }
      if (typeof copy.frame === 'string' && copy.frame) {
        copy.frame = toAbsoluteAssetUrl(copy.frame)
      }
      if (typeof copy.frame_url === 'string' && copy.frame_url) {
        copy.frame_url = toAbsoluteAssetUrl(copy.frame_url)
      }
      return copy
    })
  }
}

export const normaliseHighlight = (rawHighlight, fallbackIndex = 0, thresholds = {}) => {
  if (!rawHighlight || typeof rawHighlight !== 'object') {
    return null
  }

  const start = toFiniteNumber(rawHighlight.start)
  const end = toFiniteNumber(rawHighlight.end)
  if (start === null || end === null) {
    return null
  }

  const duration = normaliseDuration(rawHighlight, start, end)
  const minDuration = Number.isFinite(thresholds.minDuration)
    ? thresholds.minDuration
    : DEFAULT_HIGHLIGHT_MIN_DURATION_SEC
  if (duration === null || duration < minDuration) {
    return null
  }

  const highlight = { ...rawHighlight }
  highlight.start = start
  highlight.end = end
  highlight.duration = duration

  const actorSimilarity = toFiniteNumber(rawHighlight.actor_similarity)
  if (actorSimilarity !== null) {
    highlight.actor_similarity = actorSimilarity
  } else if (highlight.actor_similarity !== undefined) {
    highlight.actor_similarity = null
  }

  const candidates = resolveScoreCandidates(rawHighlight)
  const bestScore = candidates.length ? Math.max(...candidates) : null
  const minScore = Number.isFinite(thresholds.minScore)
    ? thresholds.minScore
    : DEFAULT_HIGHLIGHT_SCORE_THRESHOLD
  const meetsScore = bestScore !== null && bestScore >= minScore
  const meetsActor = actorSimilarity !== null && actorSimilarity >= minScore

  if (!meetsScore && !meetsActor) {
    return null
  }

  if (bestScore !== null) {
    highlight.score = bestScore
  } else if (highlight.score === undefined && actorSimilarity !== null) {
    highlight.score = actorSimilarity
  }

  highlight.effective_score = toFiniteNumber(highlight.score)
  if (highlight.effective_score === null && actorSimilarity !== null) {
    highlight.effective_score = actorSimilarity
  }
  if (highlight.effective_score === null) {
    highlight.effective_score = 0
  }

  convertAssetFields(highlight)

  const rawId = rawHighlight.id ?? rawHighlight.key ?? rawHighlight.order
  if (rawId !== undefined && rawId !== null && rawId !== '') {
    highlight.id = String(rawId)
  } else {
    highlight.id = `highlight-${fallbackIndex}`
  }

  return highlight
}

export const filterHighlights = (highlights, options = {}) => {

  const cachedMeta =
    highlights && typeof highlights === 'object' ? highlights.__highlightFilterMeta : null

  const thresholdSources = [options, options?.support, options?.settings]
  if (cachedMeta) {
    thresholdSources.push(cachedMeta)
  }

  const thresholds = resolveHighlightThresholds(thresholdSources)

  const stats = {
    inCount: 0,
    outCount: 0,
    reasons: {
      det_score: resolveReasonThreshold(
        thresholdSources,
        ['det_score_threshold', 'detScoreThreshold', 'minScore', 'min_score'],
        thresholds.minScore,
      ),
      score: resolveReasonThreshold(
        thresholdSources,
        ['score_threshold', 'scoreThreshold', 'minScore', 'min_score'],
        thresholds.minScore,
      ),
      duration: resolveReasonThreshold(
        thresholdSources,
        ['minDuration', 'min_duration', 'MIN_HL_DURATION_SEC'],
        thresholds.minDuration,
      ),
    },
  }

  if (!Array.isArray(highlights)) {
    return { items: [], stats }
  }



  const normalised = []
  highlights.forEach((item, index) => {
    const highlight = normaliseHighlight(item, index, thresholds)
    if (highlight) {
      normalised.push(highlight)
    } else {
      stats.outCount += 1
    }
  })

  normalised.sort((a, b) => {
    const scoreDiff = (b.effective_score ?? 0) - (a.effective_score ?? 0)
    if (Math.abs(scoreDiff) > 1e-9) {
      return scoreDiff
    }
    const durationDiff = (b.duration ?? 0) - (a.duration ?? 0)
    if (Math.abs(durationDiff) > 1e-9) {
      return durationDiff
    }
    return (a.start ?? 0) - (b.start ?? 0)
  })

  const sorted = normalised.map((highlight, index) => ({
    ...highlight,
    order: index + 1,
  }))

  const meta = {
    minScore: thresholds.minScore,
    minDuration: thresholds.minDuration,
  }

  try {
    Object.defineProperty(sorted, '__highlightFilterMeta', {
      value: meta,
      enumerable: false,
      configurable: true,
    })
  } catch {}

  stats.inCount = sorted.length

  return { items: sorted, stats }
}

export const filterHighlightsArray = (highlights, options = {}) => {
  const result = filterHighlights(highlights, options)
  if (!result || typeof result !== 'object') {
    return []
  }
  return Array.isArray(result.items) ? result.items : []
}

export const __test__ = {
  toFiniteNumber,
  normaliseDuration,
  resolveScoreCandidates,
}