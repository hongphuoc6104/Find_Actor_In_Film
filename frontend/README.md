# Frontend

This directory contains the Vite + Vue 3 single-page application for performing face-based character searches against the backend recognition service.

## Getting started

```bash
npm install
npm run dev
```

The development server runs on [http://localhost:5173](http://localhost:5173) by default.

Set the backend API endpoint in `.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

The `FaceSearch` component posts form data containing the selected image to `${VITE_API_BASE_URL}/recognize` and displays the candidates returned by the API.