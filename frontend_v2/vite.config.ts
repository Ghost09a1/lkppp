import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        host: '127.0.0.1',
        port: 5173,
        strictPort: true,
        proxy: {
            '/api': 'http://localhost:8000',
            '/chat': 'http://localhost:8000',
            '/stt': 'http://localhost:8000',
            '/tts': 'http://localhost:8000',
            '/characters': 'http://localhost:8000',
            '/posts': 'http://localhost:8000',
            '/status': 'http://localhost:8000',
        }
    },
    build: {
        outDir: '../ui_v2',
        emptyOutDir: true,
    },
})
