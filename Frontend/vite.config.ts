// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // ✨ 이 줄이 있어야 합니다.

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // ✨ 이 플러그인이 활성화되어 있어야 합니다.
  ],
  server: {
    watch: {
      // public/longText.txt 파일의 변경은 무시하도록 설정
      // AI가 이 파일을 수정해도 더 이상 페이지가 새로고침되지 않습니다.
      ignored: ['**/public/longText.txt']
    },
    proxy: {
      // '/public'으로 시작하는 모든 요청을 백엔드 서버로 전달합니다.
      '/public': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // (만약 다른 프록시 설정이 있다면 여기에 추가합니다)
    }
  }
})