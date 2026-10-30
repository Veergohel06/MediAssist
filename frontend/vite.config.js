import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/chat': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/reset': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/login': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/register': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/logout': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
});
