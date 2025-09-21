const numberOrNull = (value) => {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

export const DEFAULT_TIME_EPSILON = 0.05

export const toBox = (entry) => {
  if (!entry) {
    return null
  }
  if (Array.isArray(entry) && entry.length >= 4) {
    const [x1, y1, x2, y2] = entry.map((value) => Number(value) || 0)
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
  }
  if (typeof entry === 'object') {
    const hasXYWH = ['x', 'y', 'width', 'height'].every((key) =>
      Number.isFinite(Number(entry[key]))
    )
    if (hasXYWH) {
      return {
        x: Number(entry.x) || 0,
        y: Number(entry.y) || 0,
        width: Number(entry.width) || 0,
        height: Number(entry.height) || 0,
      }
    }
    const hasCorners = ['x1', 'y1', 'x2', 'y2'].every((key) =>
      Number.isFinite(Number(entry[key]))
    )
    if (hasCorners) {
      const x1 = Number(entry.x1) || 0
      const y1 = Number(entry.y1) || 0
      const x2 = Number(entry.x2) || 0
      const y2 = Number(entry.y2) || 0
      return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
    }
  }
  return null
}

export const collectBoxesFromScene = (scene) => {
  if (!scene || typeof scene !== 'object') {
    return []
  }
  const boxes = []
  if (Array.isArray(scene.boxes)) {
    boxes.push(...scene.boxes)
  }
  if (scene.bbox) {
    boxes.push(scene.bbox)
  }
  return boxes.map(toBox).filter(Boolean)
}

export const collectBoxesFromTimelineEntry = (entry) => {
  if (!entry || typeof entry !== 'object') {
    return []
  }
  const boxes = []
  if (Array.isArray(entry.boxes)) {
    boxes.push(...entry.boxes)
  }
  if (entry.bbox) {
    boxes.push(entry.bbox)
  }
  return boxes.map(toBox).filter(Boolean)
}

const computeStartTimes = (timeline, fps, epsilon) =>
  timeline.map((entry, index) => {
    if (!entry || typeof entry !== 'object') {
      return index * epsilon
    }
    const candidates = [
      entry.clip_offset,
      entry.clipOffset,
      entry.offset,
      entry.relative_time,
      entry.relativeTime,
      entry.timestamp,
    ]
    for (const candidate of candidates) {
      const value = numberOrNull(candidate)
      if (value !== null) {
        return value
      }
    }
    return fps ? index / fps : index * epsilon
  })

const computeEndTime = (entry, index, starts, fps, epsilon) => {
  if (!entry || typeof entry !== 'object') {
    return starts[index] + (fps ? 1 / fps : epsilon)
  }
  const endCandidates = [
    entry.clip_end,
    entry.clipEnd,
    entry.end_offset,
    entry.endOffset,
    entry.end_timestamp,
    entry.endTimestamp,
  ]
  for (const candidate of endCandidates) {
    const value = numberOrNull(candidate)
    if (value !== null) {
      return value
    }
  }
  const duration = numberOrNull(entry.duration)
  if (duration !== null && duration > 0) {
    return starts[index] + duration
  }
  if (index + 1 < starts.length) {
    const nextStart = starts[index + 1]
    if (Number.isFinite(nextStart) && nextStart > starts[index]) {
      return nextStart
    }
  }
  return starts[index] + (fps ? 1 / fps : epsilon)
}

export const pickActiveTimelineEntry = (timeline, currentTime, clipFps) => {
  if (!Array.isArray(timeline) || !timeline.length) {
    return null
  }
  const fps = Number(clipFps)
  const hasFps = Number.isFinite(fps) && fps > 0
  const epsilon = hasFps ? 1 / fps : DEFAULT_TIME_EPSILON
  const starts = computeStartTimes(timeline, hasFps ? fps : null, epsilon)
  const safeTime = numberOrNull(currentTime) ?? 0

  let fallback = timeline[0]
  for (let index = 0; index < timeline.length; index += 1) {
    const entry = timeline[index]
    const start = starts[index]
    const end = computeEndTime(entry, index, starts, hasFps ? fps : null, epsilon)

    if (safeTime + epsilon >= start) {
      fallback = entry
    }

    if (safeTime >= start - epsilon && safeTime <= end + epsilon) {
      return entry
    }
  }

  return fallback
}

export const scaleBoxes = (boxes, width, height) => {
  if (!Array.isArray(boxes) || !boxes.length) {
    return []
  }
  const w = Number(width)
  const h = Number(height)
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
    return []
  }
  return boxes.map((box) => ({
    left: `${Math.max((box.x / w) * 100, 0)}%`,
    top: `${Math.max((box.y / h) * 100, 0)}%`,
    width: `${Math.min((box.width / w) * 100, 100)}%`,
    height: `${Math.min((box.height / h) * 100, 100)}%`,
  }))
}

export const computeOverlayBoxes = (
  scene,
  baseWidth,
  baseHeight,
  currentTime,
) => {
  const timeline = Array.isArray(scene?.timeline) ? scene.timeline : []
  const fps = Number(scene?.clip_fps)
  const activeEntry = pickActiveTimelineEntry(timeline, currentTime, fps)
  const boxes = activeEntry
    ? collectBoxesFromTimelineEntry(activeEntry)
    : collectBoxesFromScene(scene)
  return scaleBoxes(boxes, baseWidth, baseHeight)
}