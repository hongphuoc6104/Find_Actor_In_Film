const metaEnv = typeof import.meta !== 'undefined' && import.meta?.env ? import.meta.env : {}
const runtimeEnv = typeof process !== 'undefined' && process?.env ? process.env : {}

export const API_BASE_URL =
  metaEnv.VITE_API_BASE_URL ?? runtimeEnv.VITE_API_BASE_URL ?? 'http://localhost:8000'