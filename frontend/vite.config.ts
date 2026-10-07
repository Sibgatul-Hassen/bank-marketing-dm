import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig, type Plugin } from 'vite'
import fs from 'node:fs'
import path from 'node:path'

const RESULTS = path.resolve(import.meta.dirname, '../results')

/** Serves ../results at /results in dev and copies it into dist/results on build.
 *  The Python pipeline is the only writer of results/; the frontend only reads it. */
function results(): Plugin {
  return {
    name: 'bank-results',
    configureServer(server) {
      server.watcher.add(RESULTS)
      server.middlewares.use('/results', (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? '/').split('?')[0])
        const file = path.join(RESULTS, rel)
        if (!file.startsWith(RESULTS) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
          res.statusCode = 404
          res.end('not built yet')
          return
        }
        res.setHeader('Content-Type', 'application/json; charset=utf-8')
        res.setHeader('Cache-Control', 'no-cache')
        fs.createReadStream(file).pipe(res)
        void next
      })
    },
    closeBundle() {
      if (fs.existsSync(RESULTS)) fs.cpSync(RESULTS, path.resolve(import.meta.dirname, 'dist/results'), { recursive: true })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), results()],
  base: process.env.GITHUB_ACTIONS ? '/bank-marketing-dm/' : '/',
  build: { chunkSizeWarningLimit: 5000 },
})
