<template>
  <section class="search-page">
    <header class="search-page__hero">
      <h1>Find Actor in Film</h1>
      <p>Tải ảnh khuôn mặt để tìm các nhân vật tương ứng trong phim và xác nhận từng cảnh.</p>
    </header>

    <FaceSearch />
  </section>
</template>

<script setup>
import FaceSearch from '../components/FaceSearch.vue'
</script>

<style scoped>
.search-page {
  display: grid;
  gap: 2.5rem;
}

.search-page__hero {
  text-align: center;
  display: grid;
  gap: 0.75rem;
}

.search-page__hero h1 {
  margin: 0;
  font-size: clamp(2rem, 3vw, 2.75rem);
  font-weight: 700;
}

.search-page__hero p {
  margin: 0 auto;
  max-width: 38rem;
  color: #475569;
  font-size: 1.05rem;
}
</style>
