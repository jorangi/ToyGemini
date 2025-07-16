// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // ✨ 이 줄이 있어야 합니다.

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // ✨ 이 플러그인이 활성화되어 있어야 합니다.
  ],
  // css.postcss 설정 블록은 제거된 상태여야 합니다.
})