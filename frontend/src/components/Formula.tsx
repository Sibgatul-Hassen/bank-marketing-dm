import katex from 'katex'
import { useMemo, useState } from 'react'
import type { Formula } from '../data/types'

export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = useMemo(() => {
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore' }) }
    catch { return tex }
  }, [tex, display])
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function FormulaCard({ f, highlight }: { f: Formula; highlight?: boolean }) {
  const [copied, setCopied] = useState(false)
  return (
    <div id={`formula-${f.id}`} className="card" style={{ padding: '14px 16px', borderColor: highlight ? 'var(--accent)' : undefined, scrollMarginTop: 90 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="chip mono">{f.id}</span>
        <strong style={{ fontSize: 14 }}>{f.name}</strong>
        <div style={{ flex: 1 }} />
        <button className="btn" style={{ padding: '2px 8px', fontSize: 12 }} aria-label={`Copy LaTeX for ${f.name ?? f.id}`}
          onClick={() => { navigator.clipboard?.writeText(f.latex); setCopied(true); setTimeout(() => setCopied(false), 1200) }}>
          {copied ? 'Copied' : 'Copy LaTeX'}
        </button>
      </div>
      <div style={{ marginTop: 6 }}><Tex tex={f.latex} display /></div>
      <p className="secondary" style={{ fontSize: 14 }}>{f.caption}</p>
      {f.glossary?.length > 0 && (
        <dl style={{ margin: '10px 0 0', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 14px', fontSize: 13 }}>
          {f.glossary.map((g) => (
            <div key={g.sym} style={{ display: 'contents' }}>
              <dt><Tex tex={g.sym} /></dt>
              <dd className="mono secondary" style={{ margin: 0, fontSize: 12 }}>{g.means}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}
