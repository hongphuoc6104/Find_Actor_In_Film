const numberOrNull = (value) => {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

export const DEFAULT_TIME_EPSILON = 0.05


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