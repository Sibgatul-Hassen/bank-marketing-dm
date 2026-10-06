// Renders every formula and glossary symbol in results/models/*/meta.json with KaTeX (throwOnError).
// Definition of Done: "every formula renders in KaTeX". Run: npm run check:katex
import fs from 'node:fs'
import path from 'node:path'
import katex from 'katex'

const dir = path.resolve(import.meta.dirname, '../../results/models')
let bad = 0, n = 0
for (const id of fs.readdirSync(dir)) {
  const p = path.join(dir, id, 'meta.json')
  if (!fs.existsSync(p)) continue
  const meta = JSON.parse(fs.readFileSync(p, 'utf8'))
  for (const f of meta.formulas) {
    for (const [what, tex] of [[f.id, f.latex], ...(f.glossary ?? []).map((g) => [`${f.id} glossary ${g.sym}`, g.sym])]) {
      n++
      try { katex.renderToString(tex, { throwOnError: true, displayMode: true }) }
      catch (e) { bad++; console.error(`FAIL ${id} ${what}: ${e.message}`) }
    }
  }
}
console.log(`${n - bad}/${n} expressions render`)
process.exit(bad ? 1 : 0)
