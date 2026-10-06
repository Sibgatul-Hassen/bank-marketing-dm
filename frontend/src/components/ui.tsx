import { animate, motion, useInView, useReducedMotion } from 'framer-motion'
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import type { Assumption, Finding } from '../data/types'
import { EASE } from '../theme/palette'

export function Section({ title, eyebrow, children, id, intro }: { title?: string; eyebrow?: string; children: ReactNode; id?: string; intro?: ReactNode }) {
  return (
    <motion.section id={id} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }}
      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.06 } } }} style={{ marginTop: 48 }}>
      {(title || eyebrow) && (
        <Reveal>
          {eyebrow && <div className="eyebrow">{eyebrow}</div>}
          {title && <h2 style={{ marginTop: 4 }}>{title}</h2>}
          {intro && <div className="secondary" style={{ marginTop: 8, maxWidth: 720 }}>{intro}</div>}
        </Reveal>
      )}
      <div style={{ marginTop: title ? 18 : 0, display: 'flex', flexDirection: 'column', gap: 16 }}>{children}</div>
    </motion.section>
  )
}

export function Reveal({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <motion.div variants={{ hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: EASE } } }} style={style}>
      {children}
    </motion.div>
  )
}

export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true })
  const reduce = useReducedMotion()
  const [shown, setShown] = useState(reduce ? value : 0)
  useEffect(() => {
    if (!inView) return
    if (reduce) { setShown(value); return }
    const c = animate(0, value, { duration: 0.9, ease: EASE, onUpdate: setShown })
    return () => c.stop()
  }, [inView, value, reduce])
  return <span ref={ref}>{format(shown)}</span>
}

export function MetricTile({ label, value, format, hint, accent }: { label: string; value: number | null | undefined; format: (v: number) => string; hint?: string; accent?: boolean }) {
  return (
    <div className="card" style={{ padding: '12px 14px', minWidth: 0 }} title={hint}>
      <div className="eyebrow">{label}</div>
      <div className="num" style={{ fontSize: 22, fontWeight: 500, marginTop: 2, color: accent ? 'var(--accent-text)' : 'var(--text-primary)' }}>
        {value == null ? '—' : <CountUp value={value} format={format} />}
      </div>
      {hint && <div className="muted" style={{ fontSize: 11, lineHeight: 1.4 }}>{hint}</div>}
    </div>
  )
}

export const fmt = {
  auc: (v: number) => v.toFixed(4),
  pct: (v: number) => `${(v * 100).toFixed(1)}%`,
  pct2: (v: number) => `${(v * 100).toFixed(2)}%`,
  secs: (v: number) => (v < 1 ? `${(v * 1000).toFixed(0)} ms` : v < 120 ? `${v.toFixed(1)} s` : `${(v / 60).toFixed(1)} min`),
  int: (v: number) => Math.round(v).toLocaleString('en-US'),
  delta: (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(4)}`,
  num: (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toFixed(4)),
}

export function Placeholder({ what, detail }: { what: string; detail?: string }) {
  return (
    <div className="card" style={{ padding: 24, borderStyle: 'dashed', textAlign: 'center' }} role="status">
      <div style={{ fontWeight: 600 }}>{what} — not built yet</div>
      <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>{detail ?? 'Run the Python pipeline to generate this artifact; the page fills in automatically.'}</div>
    </div>
  )
}

export function Loading() {
  return <div className="muted" role="status" style={{ padding: 24 }}>Loading…</div>
}

const STATUS: Record<string, { cls: string; label: string }> = {
  ok: { cls: 'chip', label: 'ok' }, warn: { cls: 'chip chip-warn', label: 'warn' }, fail: { cls: 'chip chip-fail', label: 'fails' },
  required: { cls: 'chip chip-warn', label: 'required' }, unknown: { cls: 'chip chip-muted', label: 'unknown' },
}

export function AssumptionList({ items }: { items: Assumption[] }) {
  return (
    <div className="card scroll-x" style={{ padding: 4 }}>
      <table className="data">
        <thead><tr><th>Assumption</th><th>Status</th><th>Evidence on this data</th></tr></thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.text}><td>{a.text}</td><td><span className={STATUS[a.status].cls}>{STATUS[a.status].label}</span></td><td className="secondary">{a.evidence}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const FIND: Record<string, { color: string; label: string; icon: string }> = {
  result: { color: 'var(--accent)', label: 'Result', icon: '●' },
  limitation: { color: 'var(--warning)', label: 'Limitation', icon: '▲' },
  failure: { color: 'var(--negative)', label: 'Failure', icon: '■' },
}

export function FindingsList({ findings }: { findings: Finding[] }) {
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
      {findings.map((f, i) => (
        <motion.li key={i} initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.45, delay: i * 0.06, ease: EASE }}
          className="card" style={{ padding: '12px 14px', borderLeft: `3px solid ${FIND[f.type].color}`, display: 'flex', gap: 12 }}>
          <span aria-hidden="true" style={{ color: FIND[f.type].color, fontSize: 11, marginTop: 4 }}>{FIND[f.type].icon}</span>
          <div>
            <div className="eyebrow" style={{ color: FIND[f.type].color }}>{FIND[f.type].label}</div>
            <div>{f.text}</div>
          </div>
        </motion.li>
      ))}
    </ul>
  )
}

export function Callout({ children, tone = 'accent', title }: { children: ReactNode; tone?: 'accent' | 'warn' | 'neg'; title?: string }) {
  const c = tone === 'accent' ? 'var(--accent)' : tone === 'warn' ? 'var(--warning)' : 'var(--negative)'
  return (
    <div className="card" style={{ padding: '14px 16px', borderLeft: `3px solid ${c}` }}>
      {title && <div style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>}
      <div className="secondary">{children}</div>
    </div>
  )
}

export function Grid({ children, min = 160 }: { children: ReactNode; min?: number }) {
  return <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fill, minmax(min(${min}px, 100%), 1fr))`, gap: 12 }}>{children}</div>
}
