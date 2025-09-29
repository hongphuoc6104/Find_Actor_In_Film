import { toAbsoluteAssetUrl } from './assetUrls.js'
import {
  HIGHLIGHT_MIN_DURATION_SEC,
  HIGHLIGHT_SCORE_THRESHOLD,
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

export const normaliseHighlight = (rawHighlight, fallbackIndex = 0) => {
  if (!rawHighlight || typeof rawHighlight !== 'object') {
    return null
  }

  const start = toFiniteNumber(rawHighlight.start)
  const end = toFiniteNumber(rawHighlight.end)
  if (start === null || end === null) {
    return null
  }

  const duration = normaliseDuration(rawHighlight, start, end)
  if (duration === null || duration < HIGHLIGHT_MIN_DURATION_SEC) {
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
  const meetsScore = bestScore !== null && bestScore >= HIGHLIGHT_SCORE_THRESHOLD
  const meetsActor = actorSimilarity !== null && actorSimilarity >= HIGHLIGHT_SCORE_THRESHOLD

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

export const filterHighlights = (highlights) => {
  if (!Array.isArray(highlights)) {
    return []
  }

  const normalised = []
  highlights.forEach((item, index) => {
    const highlight = normaliseHighlight(item, index)
    if (highlight) {
      normalised.push(highlight)
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

  return normalised.map((highlight, index) => ({
    ...highlight,
    order: index + 1,
  }))
}

export const __test__ = {
  toFiniteNumber,
  normaliseDuration,
  resolveScoreCandidates,
}