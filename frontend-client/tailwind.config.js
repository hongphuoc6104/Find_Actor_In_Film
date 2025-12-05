/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Thêm bộ màu của FaceLook Cinema luôn
        bgMain: '#050505',
        bgCard: '#121212',
        accent: '#E50914',
      }
    },
  },
  plugins: [],
}