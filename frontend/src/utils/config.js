const metaEnv = typeof import.meta !== 'undefined' && import.meta?.env ? import.meta.env : {}
const runtimeEnv = typeof process !== 'undefined' && process?.env ? process.env : {}
const globalEnv = typeof window !== 'undefined' && window?.__APP_CONFIG__ ? window.__APP_CONFIG__ : {}

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

export const SEEK_PAD_SEC = toNumber(resolveConfigValue('SEEK_PAD_SEC'), 0)
export const PAUSE_TOLERANCE_SEC = toNumber(resolveConfigValue('PAUSE_TOLERANCE_SEC'), 0.2)
export const MIN_VIEWABLE_SEC = toNumber(resolveConfigValue('MIN_VIEWABLE_SEC'), 0.35)

export const __test__ = {
  resolveConfigValue,
  toNumber,
  metaEnv,
  runtimeEnv,
  globalEnv,
}