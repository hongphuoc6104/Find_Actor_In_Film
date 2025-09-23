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

const computeStartTimes = (timeline, options) =>
  timeline.map((entry, index) => {
    if (!entry || typeof entry !== 'object') {
      return index * options.epsilon
    }
    const absoluteCandidates = [

      entry.time,
      entry.start,
      entry.start_time,
    ]
    for (const candidate of absoluteCandidates) {
      const value = numberOrNull(candidate)
      if (value !== null) {
        const adjusted =
          options.baseTimestamp !== null ? value - options.baseTimestamp : value
        return Math.max(adjusted, 0)
      }
    }
    const relativeCandidates = [
      entry.clip_offset,
      entry.clipOffset,
      entry.offset,
      entry.relative_time,
      entry.relativeTime,
      entry.timestamp,
      entry.relative_offset,
      entry.relativeOffset,
    ]
    for (const candidate of relativeCandidates) {
      const value = numberOrNull(candidate)
      if (value !== null) {
        return Math.max(value, 0)
      }
    }
    return options.fps ? index / options.fps : index * options.epsilon
  })

const computeEndTime = (entry, index, starts, options) => {
  if (!entry || typeof entry !== 'object') {
    return starts[index] + (options.fps ? 1 / options.fps : options.epsilon)
  }
  const absoluteEndCandidates = [
    entry.end,
    entry.until,
    entry.end_timestamp,
    entry.endTimestamp,
    entry.stop,
  ]
  for (const candidate of absoluteEndCandidates) {
    const value = numberOrNull(candidate)
    if (value !== null) {
      const adjusted =
        options.baseTimestamp !== null ? value - options.baseTimestamp : value
      return Math.max(adjusted, 0)
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
  return starts[index] + (options.fps ? 1 / options.fps : options.epsilon)
}

export const pickActiveTimelineEntry = (timeline, currentTime, clipFps) => {
  if (!Array.isArray(timeline) || !timeline.length) {
    return null
  }
  const options = {
    fps: null,
    epsilon: DEFAULT_TIME_EPSILON,
    baseTimestamp: null,
  }

  if (clipFps && typeof clipFps === 'object') {
    const fpsValue = numberOrNull(
      clipFps.fps ?? clipFps.clipFps ?? clipFps.frameRate ?? clipFps.videoFps,
    )
    if (fpsValue !== null && fpsValue > 0) {
      options.fps = fpsValue
      options.epsilon = 1 / fpsValue
    }
    const baseValue = numberOrNull(
      clipFps.sceneStart ??
        clipFps.baseTimestamp ??
        clipFps.start ??
        clipFps.startTime ??
        clipFps.offset ??
        clipFps.origin,
    )
    if (baseValue !== null) {
      options.baseTimestamp = baseValue
    }
  } else {
    const fpsValue = numberOrNull(clipFps)
    if (fpsValue !== null && fpsValue > 0) {
      options.fps = fpsValue
      options.epsilon = 1 / fpsValue
    }
  }

  const starts = computeStartTimes(timeline, options)
  const epsilon = options.epsilon
  const safeTime = numberOrNull(currentTime) ?? 0
  const relativeTime =
    options.baseTimestamp !== null ? safeTime - options.baseTimestamp : safeTime
  const safeRelative = Number.isFinite(relativeTime) ? relativeTime : 0

  let fallback = timeline[0]
  for (let index = 0; index < timeline.length; index += 1) {
    const entry = timeline[index]
    const start = starts[index]
    const end = computeEndTime(entry, index, starts, options)

    if (safeRelative + epsilon >= start) {
      fallback = entry
    }

    if (safeRelative >= start - epsilon && safeRelative <= end + epsilon) {
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
  const sceneStart = numberOrNull(
    scene?.start_time ??
      scene?.video_start_timestamp ??
      scene?.clip_start_timestamp ??
      scene?.timestamp,
  )
  const activeEntry = pickActiveTimelineEntry(timeline, currentTime, {
    fps,
    sceneStart,
  })
  const boxes = activeEntry
    ? collectBoxesFromTimelineEntry(activeEntry)
    : collectBoxesFromScene(scene)
  return scaleBoxes(boxes, baseWidth, baseHeight)
}