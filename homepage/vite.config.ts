import react from '@vitejs/plugin-react'
import {defineConfig} from 'vite'

// Hashed assets land under /_home/ so they never collide with a page Zensical builds.
export default defineConfig({
  base: '/',
  plugins: [react()],
  build: {
    assetsDir: '_home',
    sourcemap: true,
    target: 'es2022',
  },
})
