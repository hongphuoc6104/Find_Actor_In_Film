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

  let resolved = trimmed

  if (!ABSOLUTE_URL_PATTERN.test(trimmed) && !trimmed.startsWith('//')) {
    const origin =
      typeof window !== 'undefined' && window?.location?.origin
        ? stripTrailingSlashes(window.location.origin)
        : ''

    if (origin) {
      const relativePath = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
      resolved = `${origin}${relativePath}`
    }
  }

  let normalised = stripTrailingSlashes(resolved)
  if (normalised.toLowerCase().endsWith('/api')) {
    normalised = normalised.slice(0, -4)
  }

  normalised = stripTrailingSlashes(normalised)
  return normalised || ''
}

export const toAbsoluteAssetUrl = (value, base) => {

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

  const resolvedBase =
    typeof base === 'string' && base.trim()
      ? computeApiAssetBase(base)
      : computeApiAssetBase()
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