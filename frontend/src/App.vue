<template>
  <div class="app-shell">
    <header class="app-shell__header">
      <div class="app-shell__brand">Find Actor in Film</div>
      <nav class="app-shell__nav">
        <a
          href="/search"
          :class="{ active: currentPath === '/search' }"
          @click.prevent="navigate('/search')"
        >
          Tìm kiếm
        </a>
        <a
          href="/movies"
          :class="{ active: currentPath === '/movies' }"
          @click.prevent="navigate('/movies')"
        >
          Quản lý phim
        </a>
      </nav>
    </header>

    <main class="app-shell__main">
      <component :is="activeComponent" />
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import MovieManagementPage from './views/MovieManagementPage.vue'
import SearchPage from './views/SearchPage.vue'

const ROUTES = {
  '/search': SearchPage,
  '/movies': MovieManagementPage,
}

const DEFAULT_ROUTE = '/search'

const currentPath = ref(typeof window !== 'undefined' ? window.location.pathname : DEFAULT_ROUTE)
let removeListener = null

const normalizePath = (path) => {
  if (!path) {
    return DEFAULT_ROUTE
  }
  if (ROUTES[path]) {
    return path
  }
  return DEFAULT_ROUTE
}

const setPath = (path, replace = false) => {
  const normalized = normalizePath(path)
  if (typeof window !== 'undefined') {
    if (replace) {
      window.history.replaceState({}, '', normalized)
    } else {
      window.history.pushState({}, '', normalized)
    }
  }
  currentPath.value = normalized
}

const navigate = (path) => {
  if (path === currentPath.value) {
    return
  }
  setPath(path)
}

onMounted(() => {
  const normalized = normalizePath(currentPath.value)
  if (normalized !== currentPath.value) {
    setPath(normalized, true)
  } else {
    currentPath.value = normalized
  }

  const handlePopState = () => {
    currentPath.value = normalizePath(window.location.pathname)
  }
  window.addEventListener('popstate', handlePopState)
  removeListener = () => window.removeEventListener('popstate', handlePopState)
})

onBeforeUnmount(() => {
  if (typeof removeListener === 'function') {
    removeListener()
  }
})

const activeComponent = computed(() => ROUTES[currentPath.value] ?? ROUTES[DEFAULT_ROUTE])
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
}

.app-shell__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem 1.25rem;
  background: #ffffff;
  border-bottom: 1px solid rgba(148, 163, 184, 0.3);
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
  position: sticky;
  top: 0;
  z-index: 10;
}

.app-shell__brand {
  font-weight: 700;
  font-size: 1.35rem;
  color: #0f172a;
}

.app-shell__nav {
  display: flex;
  gap: 1.25rem;
  align-items: center;
}

.app-shell__nav a {
  text-decoration: none;
  color: #1e293b;
  font-weight: 600;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  transition: background 120ms ease, color 120ms ease;
}

.app-shell__nav a:hover {
  background: rgba(37, 99, 235, 0.12);
  color: #1d4ed8;
}

.app-shell__nav a.active {
  background: #2563eb;
  color: #f8fafc;
}

.app-shell__main {
  flex: 1;
  padding: 2.5rem 1.5rem 3rem;
  max-width: min(1400px, 100%);
  margin: 0 auto;
  width: 100%;
  display: block;
}

@media (max-width: 640px) {
  .app-shell__header {
    flex-direction: column;
    gap: 0.75rem;
    padding: 1.25rem 1.5rem;
    align-items: flex-start;
  }

  .app-shell__main {
    padding: 2rem 1rem 2.5rem;
  }
}
</style>
