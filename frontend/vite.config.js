import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      // Cho phép import kiểu "@/..." trỏ tới src/
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
    open: false,
    hmr: {
      overlay: true,
    },
    // ✅ Bật proxy để FE gọi /movies, /recognize, /videos sang BE (cổng 8000)
    proxy: {
      '/movies': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/recognize': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/videos': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
