import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { cpSync, existsSync } from 'fs'
import { resolve } from 'path'

// PDF.js (v6+) decodifica JBIG2/OpenJPEG/QCMS tramite moduli WASM che non sono
// importati da nessun file JS, quindi Vite non li rileva come asset da soli:
// vanno copiati esplicitamente nella build affinché wasmUrl (vedi ImageSearchWidget.tsx)
// possa raggiungerli in produzione.
function copyPdfjsWasm(): Plugin {
  return {
    name: 'copy-pdfjs-wasm',
    closeBundle() {
      const src = resolve(__dirname, 'node_modules/pdfjs-dist/wasm')
      const dest = resolve(__dirname, 'dist/assets/wasm')
      if (existsSync(src)) {
        cpSync(src, dest, { recursive: true })
      }
    }
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), copyPdfjsWasm()],
  base: '/assets/',  // Base path per gli asset
  build: {
    outDir: 'dist',
    assetsDir: 'assets',  // Metti gli asset in una sottocartella
    rollupOptions: {
      output: {
        entryFileNames: 'assets/index-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  }
})
