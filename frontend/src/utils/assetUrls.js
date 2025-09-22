import { API_BASE_URL } from '../config.js'

const ABSOLUTE_URL_PATTERN = /^[a-zA-Z][a-zA-Z\d+\-.]*:/

const stripTrailingSlashes = (value) => value.replace(/\/+$/, '')

export const computeApiAssetBase = (base = API_BASE_URL) => {
  if (typeof base !== 'string') {
    return ''
  }

  const trimmed = base.trim()
  if (!trimmed) {
    return ''
  }

  let normalised = stripTrailingSlashes(trimmed)
  if (normalised.toLowerCase().endsWith('/api')) {
    normalised = normalised.slice(0, -4)
  }

  normalised = stripTrailingSlashes(normalised)
  return normalised || ''
}

const DEFAULT_ASSET_BASE = computeApiAssetBase()

export const toAbsoluteAssetUrl = (value, base = DEFAULT_ASSET_BASE) => {
  if (typeof value !== 'string') {
    return value ?? ''
  }

  const trimmed = value.trim()
  if (!trimmed) {
    return ''
  }

  if (ABSOLUTE_URL_PATTERN.test(trimmed) || trimmed.startsWith('//')) {
    return trimmed
  }

  const resolvedBase = computeApiAssetBase(base || DEFAULT_ASSET_BASE)
  if (!resolvedBase) {
    return trimmed
  }

  const baseForResolution = resolvedBase.endsWith('/')
    ? resolvedBase
    : `${resolvedBase}/`

  try {
    return new URL(trimmed, baseForResolution).toString()
  } catch (error) {
    const path = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
    return `${resolvedBase}${path}`
  }
}

export const __test__ = {
  computeApiAssetBase,
}