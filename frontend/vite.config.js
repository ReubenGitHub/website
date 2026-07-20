import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '::',
    port: 3000,
    watch: {
      usePolling: true,
      interval: 100,
    },
    proxy: {
      '/api/simulation': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/physicsHub': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        ws: true,
      },
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'build',
  },
})
