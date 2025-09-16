<template>
  <main class="app">
    <header class="hero">
      <h1>Find Actor in Film</h1>
      <p>Upload a face image to search for matching characters across films.</p>
    </header>

    <FaceSearch />
  </main>
</template>

<script setup>
import FaceSearch from './components/FaceSearch.vue'
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.hero {
  text-align: center;
  display: grid;
  gap: 0.75rem;
}

.hero h1 {
  margin: 0;
  font-size: clamp(2rem, 3vw, 2.75rem);
  font-weight: 700;
}

.hero p {
  margin: 0 auto;
  max-width: 36rem;
  color: #4b5563;
  font-size: 1.05rem;
}
</style>