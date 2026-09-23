import { defineConfig } from 'vite';
export default defineConfig({
  server: { proxy: { '/api': process.env.RLCD_API ?? 'http://127.0.0.1:8765' } },
  build: { outDir: 'dist', emptyOutDir: true },
});
