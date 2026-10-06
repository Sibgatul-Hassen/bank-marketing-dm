import { motion, useInView, useReducedMotion } from 'framer-motion'
import { useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { PlotlyFigure } from '../components/PlotlyFigure'
import { Callout, fmt, Loading, Placeholder, Reveal, Section } from '../components/ui'
import { useArtifact, useArtifacts } from '../data/loader'
import type { Curves, Explain, ShapExplain } from '../data/types'
import { DATA_COLORS, EASE, MODEL_COLORS, MODEL_NAMES, MODEL_ORDER } from '../theme/palette'

/* eslint-disable @typescript-eslint/no-explicit-any */
type Run = 'without_duration' | 'with_duration'
const CLASSIFIERS = ['decision_tree', 'catboost', 'tabnet', 'tabpfn']

// ------------------------------------------------------------------ comparator

const ROWS: { key: string; label: string; f: (v: number) => string; higher: boolean; max?: number }[] = [
  { key: 'pr_auc', label: 'PR-AUC', f: fmt.auc, higher: true, max: 1 },
  { key: 'roc_auc', label: 'ROC-AUC', f: fmt.auc, higher: true, max: 1 },
  { key: 'f1', label: 'F1', f: fmt.auc, higher: true, max: 1 },
  { key: 'recall', label: 'Recall', f: fmt.auc, higher: true, max: 1 },
  { key: 'precision', label: 'Precision', f: fmt.auc, higher: true, max: 1 },
  { key: 'capture_at_20', label: 'Captured @ top 20%', f: fmt.pct, higher: true, max: 1 },
  { key: 'total_seconds', label: 'Train + predict', f: fmt.secs, higher: false },
  { key: 'leak', label: 'Leak Δ AUC', f: fmt.delta, higher: false },
]

function value(e: any, key: string): number | null {
  if (!e?.runs) return key === 'roc_auc' ? e?.secondary?.roc_auc ?? null : null
  if (key === 'leak') return e.leak_delta.roc_auc
  if (key === 'total_seconds') return e.runs.without_duration.train_seconds + e.runs.without_duration.predict_seconds
  return e.runs.without_duration[key] ?? null
}

function Bar({ v, max, best, f }: { v: number | null; max: number; best: boolean | 'tie'; f: (v: number) => string }) {
  const reduce = useReducedMotion()
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span className="num" style={{ width: 70, textAlign: 'right', fontSize: 13, color: best === true ? 'var(--accent-text)' : 'var(--text-primary)', fontWeight: best === true ? 600 : 400 }}>{v == null ? '—' : f(v)}</span>
      <div style={{ flex: 1, height: 8, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
        <motion.div animate={{ width: `${v == null ? 0 : Math.max(2, Math.min(100, (Math.abs(v) / (max || 1)) * 100))}%` }} transition={{ duration: reduce ? 0 : 0.42, ease: EASE }}
          style={{ height: '100%', background: best === true ? 'var(--accent)' : 'var(--text-muted)', opacity: best === true ? 1 : 0.55 }} />
      </div>
      {best === 'tie' && <span className="chip chip-muted" style={{ fontSize: 10 }}>tie</span>}
    </div>
  )
}

function Comparator({ lb, pairs, curves, explains }: { lb: any; pairs: any; curves: Record<string, Curves | undefined>; explains: Record<string, Explain | undefined> }) {
  const [params, setParams] = useSearchParams()
  const a = params.get('a') ?? 'catboost', b = params.get('b') ?? 'tabpfn'
  const set = (k: 'a' | 'b', v: string) => { const p = new URLSearchParams(params); p.set(k, v); setParams(p, { replace: true }) }
  const ea = lb.models.find((m: any) => m.model_id === a), eb = lb.models.find((m: any) => m.model_id === b)
  const same = a === b
  const both = ea?.runs && eb?.runs

  const pr = pairs.pairs[`${a}|${b}`] ?? (pairs.pairs[`${b}|${a}`] ? { ...pairs.pairs[`${b}|${a}`], flipped: true } : null)
  let verdict = ''
  if (both && !same) {
    const ra = ea.runs.without_duration, rb = eb.runs.without_duration
    const d = ra.pr_auc - rb.pr_auc
    const lead = Math.abs(d) < 0.0005 ? null : d > 0 ? ea : eb
    const tA = ra.train_seconds + ra.predict_seconds, tB = rb.train_seconds + rb.predict_seconds
    const faster = tA < tB ? ea : eb, ratio = Math.max(tA, tB) / Math.max(Math.min(tA, tB), 1e-3)
    verdict = lead ? `${lead.display_name} leads on PR-AUC by ${Math.abs(d).toFixed(3)}` : 'The two tie on PR-AUC'
    verdict += ` (ROC-AUC ${fmt.auc(ra.roc_auc)} vs ${fmt.auc(rb.roc_auc)}). ${faster.display_name} is ${ratio.toFixed(ratio < 10 ? 1 : 0)}× faster end to end.`
    if (pr) verdict += ` McNemar ${pr.p_value < 0.0001 ? 'p < 0.0001' : `p = ${pr.p_value.toFixed(4)}`} — the difference in their errors ${pr.p_value < 0.05 ? 'is' : 'is not'} statistically significant at 0.05.`
    if ([a, b].includes('tabpfn')) verdict += ' TabPFN required no tuning and saw only 3,000 training rows.'
  } else if (!same) {
    verdict = `${(ea?.runs ? eb : ea)?.display_name} is not a classifier; its ROC-AUC is from a derived score (${(ea?.runs ? eb : ea)?.secondary?.metric}). Compare it as a description of the data, not as a competitor.`
  }

  const sel = (k: 'a' | 'b', v: string) => (
    <select className="select" aria-label={`Model ${k.toUpperCase()}`} value={v} onChange={(e) => set(k, e.target.value)} style={{ width: '100%' }}>
      {MODEL_ORDER.map((m) => <option key={m} value={m}>{MODEL_NAMES[m]}</option>)}
    </select>
  )
  const overlay = (kind: 'roc' | 'pr') => ({
    data: [a, b].filter((_, i) => !(i === 1 && same)).flatMap((m) => {
      const c = curves[m]
      if (!c) return []
      return [{ type: 'scatter', mode: 'lines', name: MODEL_NAMES[m], line: { color: MODEL_COLORS[m], width: 2.2 },
        x: kind === 'roc' ? c.roc.fpr : c.pr.recall, y: kind === 'roc' ? c.roc.tpr : c.pr.precision }]
    }).concat(kind === 'roc' ? [{ type: 'scatter', mode: 'lines', name: 'chance', line: { color: '#6B7E8F', dash: 'dot', width: 1 }, x: [0, 1], y: [0, 1] } as any]
      : [{ type: 'scatter', mode: 'lines', name: 'base rate', line: { color: '#6B7E8F', dash: 'dot', width: 1 }, x: [0, 1], y: [0.1127, 0.1127] } as any]),
    layout: { height: 300, title: kind === 'roc' ? 'ROC' : 'Precision–recall', xaxis: { title: kind === 'roc' ? 'false positive rate' : 'recall' },
      yaxis: { title: kind === 'roc' ? 'true positive rate' : 'precision' }, legend: { orientation: 'h', y: -0.25 }, margin: { t: 36 } },
  })
  const shapA = explains[a]?.kind === 'shap' ? (explains[a] as ShapExplain) : null
  const shapB = explains[b]?.kind === 'shap' ? (explains[b] as ShapExplain) : null

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 12, alignItems: 'center' }}>
        {sel('a', a)}<span className="muted">vs</span>{sel('b', b)}
      </div>
      {same ? (
        <p className="secondary" style={{ textAlign: 'center', padding: 24 }}>That is the same model on both sides — it ties with itself on every metric. Pick a different model for B.</p>
      ) : (<>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 18, marginTop: 16 }}>
          {[ea, eb].map((e, side) => (
            <div key={side}>
              <div style={{ fontWeight: 600, color: MODEL_COLORS[e.model_id], marginBottom: 6 }}>{e.display_name}</div>
              {ROWS.map((r) => {
                const va = value(ea, r.key), vb = value(eb, r.key)
                const v = side === 0 ? va : vb
                const max = r.max ?? Math.max(Math.abs(va ?? 0), Math.abs(vb ?? 0))
                let best: boolean | 'tie' = false
                if (va != null && vb != null) {
                  const tie = Math.abs(va - vb) < (r.key.includes('seconds') ? 0.01 : 0.0005)
                  best = tie ? 'tie' : (r.higher ? (side === 0 ? va > vb : vb > va) : (side === 0 ? va < vb : vb < va))
                }
                return (
                  <div key={r.key} style={{ display: 'grid', gridTemplateColumns: '120px 1fr', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span className="secondary" style={{ fontSize: 12.5 }}>{r.label}</span>
                    <Bar v={v} max={max} best={best} f={r.f} />
                  </div>
                )
              })}
            </div>
          ))}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 12, marginTop: 10 }}>
          <PlotlyFigure fig={overlay('roc')} height={300} label={`ROC overlay for ${MODEL_NAMES[a]} and ${MODEL_NAMES[b]}`} />
          <PlotlyFigure fig={overlay('pr')} height={300} label={`Precision-recall overlay for ${MODEL_NAMES[a]} and ${MODEL_NAMES[b]}`} />
        </div>
        {shapA && shapB && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginTop: 6, fontSize: 13 }}>
            {[shapA, shapB].map((s) => (
              <div key={s.model_id}>
                <div className="eyebrow">SHAP top 6 · {MODEL_NAMES[s.model_id]}</div>
                {s.global.features.slice(0, 6).map((f, i) => (
                  <div key={f} style={{ display: 'grid', gridTemplateColumns: '1fr 60px', gap: 6 }}>
                    <span className="mono" style={{ fontSize: 12 }}>{i + 1}. {f}</span><span className="num muted" style={{ textAlign: 'right' }}>{fmt.pct(s.global.share[i])}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
        <div className="card" style={{ marginTop: 14, padding: '12px 14px', background: 'var(--bg-elevated)' }}>
          <div className="eyebrow">Verdict — generated from the artifacts</div>
          <p style={{ marginTop: 4 }}>{verdict}</p>
          <p className="muted" style={{ fontSize: 12, marginTop: 4 }}>Winner decided by PR-AUC, not accuracy: with 11.27% positives, PR-AUC measures how well a model finds subscribers. Link: <span className="mono">/compare?a={a}&b={b}</span></p>
        </div>
      </>)}
    </div>
  )
}

// ------------------------------------------------------------------ leaderboard

const COLS: { key: string; label: string; f: (v: number) => string }[] = [
  { key: 'accuracy', label: 'Acc', f: fmt.auc }, { key: 'precision', label: 'Prec', f: fmt.auc }, { key: 'recall', label: 'Rec', f: fmt.auc },
  { key: 'f1', label: 'F1', f: fmt.auc }, { key: 'roc_auc', label: 'ROC-AUC', f: fmt.auc }, { key: 'pr_auc', label: 'PR-AUC', f: fmt.auc },
  { key: 'mcc', label: 'MCC', f: fmt.auc }, { key: 'train_seconds', label: 'Train', f: fmt.secs },
]

function Leaderboard({ lb }: { lb: any }) {
  const [mode, setMode] = useState<'without_duration' | 'with_duration' | 'both'>('without_duration')
  const [sort, setSort] = useState('pr_auc')
  const [withRef, setWithRef] = useState(false)
  const run: Run = mode === 'with_duration' ? 'with_duration' : 'without_duration'
  const rows = useMemo(() => {
    const src = [...lb.models.filter((m: any) => m.runs), ...(withRef ? lb.reference : [])]
    const k = sort
    return src.sort((x: any, y: any) => {
      const vx = k === 'leak' ? x.leak_delta.roc_auc : x.runs[run][k], vy = k === 'leak' ? y.leak_delta.roc_auc : y.runs[run][k]
      return k === 'train_seconds' ? vx - vy : vy - vx
    })
  }, [lb, sort, run, withRef])
  const unsup = lb.models.filter((m: any) => !m.runs)
  const th = (key: string, label: string) => (
    <th className="num" key={key}><button onClick={() => setSort(key)} style={{ all: 'unset', cursor: 'pointer', color: sort === key ? 'var(--accent-text)' : undefined }}
      aria-sort={sort === key ? 'descending' : 'none'}>{label}{sort === key ? ' ↓' : ''}</button></th>
  )
  const B = lb.baseline.runs.without_duration
  return (
    <div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 10 }} role="group" aria-label="Which run">
        {(['without_duration', 'with_duration', 'both'] as const).map((m) => (
          <button key={m} className={`btn ${mode === m ? 'active' : ''}`} aria-pressed={mode === m} onClick={() => setMode(m)}>
            {m === 'both' ? 'Both' : m === 'with_duration' ? 'With duration (leaky)' : 'Without duration'}</button>
        ))}
        <label className="secondary" style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
          <input type="checkbox" checked={withRef} onChange={(e) => setWithRef(e.target.checked)} /> include the six reference families
        </label>
      </div>
      <div className="card scroll-x" style={{ padding: 4 }}>
        <table className="data">
          <thead><tr><th>#</th><th>Model</th>{COLS.map((c) => th(c.key, c.label))}{th('leak', 'Leak Δ')}</tr></thead>
          <tbody>
            <tr style={{ background: 'var(--bg-elevated)' }}>
              <td className="muted">—</td><td><strong>Majority baseline</strong><div className="muted" style={{ fontSize: 11 }}>always "no"</div></td>
              {COLS.map((c) => <td key={c.key} className="num muted">{B[c.key] != null ? c.f(B[c.key]) : '—'}</td>)}<td className="num muted">—</td>
            </tr>
            {rows.map((e: any, i: number) => (
              <motion.tr key={e.model_id} layout transition={{ duration: 0.5, ease: EASE }} style={{ opacity: e.kind === 'reference' ? 0.8 : 1 }}>
                <td className="num">{i + 1}</td>
                <td>
                  {e.kind === 'project' ? <Link to={`/models/${e.model_id}`} style={{ color: MODEL_COLORS[e.model_id] }}>{e.display_name}</Link> : <span>{e.display_name}</span>}
                  <div className="muted" style={{ fontSize: 11 }}>{e.kind === 'reference' ? 'reference family' : e.family?.replace(/_/g, ' ')}</div>
                </td>
                {COLS.map((c) => (
                  <td key={c.key} className="num">
                    {mode === 'both'
                      ? <span>{c.f(e.runs.without_duration[c.key])}<br /><span className="muted" style={{ fontSize: 11 }}>{c.f(e.runs.with_duration[c.key])}</span></span>
                      : c.f(e.runs[run][c.key])}
                  </td>
                ))}
                <td className="num" style={{ color: 'var(--warning)' }}>{fmt.delta(e.leak_delta.roc_auc)}</td>
              </motion.tr>
            ))}
            {unsup.map((e: any) => (
              <tr key={e.model_id}>
                <td className="muted">·</td>
                <td><Link to={`/models/${e.model_id}`} style={{ color: MODEL_COLORS[e.model_id] }}>{e.display_name}</Link><div className="muted" style={{ fontSize: 11 }}>not a classifier</div></td>
                <td colSpan={COLS.length + 1} className="secondary" style={{ fontSize: 12.5 }}>{e.secondary.metric}: <span className="num">{fmt.auc(e.secondary.roc_auc)}</span> — {e.secondary.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>{mode === 'both' ? 'Top value: without duration; below: with duration. ' : ''}Sorted by PR-AUC by default. Each model is scored at its own F1-optimal threshold (chosen on training data); reference families at 0.5. Click a column to re-sort — rows slide to their new rank.</p>
    </div>
  )
}

// ------------------------------------------------------------------ slope chart (centrepiece)

function SlopeChart({ leak }: { leak: any }) {
  const [set, setSet] = useState<'reference' | 'project' | 'all'>('reference')
  const [hover, setHover] = useState<string | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const inView = useInView(ref, { once: true })
  const reduce = useReducedMotion()
  const rows: any[] = leak[set] ?? []
  const n = rows.length
  const W = 640, rowH = 34, H = n * rowH + 50, x1 = 210, x2 = W - 210
  const y = (r: number) => 30 + (r - 1) * rowH
  const color = (r: any, i: number) => MODEL_COLORS[r.model_id] ?? DATA_COLORS[i % DATA_COLORS.length]
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }} role="group" aria-label="Model set">
        {(['reference', 'project', 'all'] as const).map((s) => (
          <button key={s} className={`btn ${set === s ? 'active' : ''}`} aria-pressed={set === s} onClick={() => setSet(s)}>
            {s === 'reference' ? 'Six reference families (§3.6)' : s === 'project' ? 'Our four classifiers' : 'All ten'}</button>
        ))}
      </div>
      <div className="scroll-x">
        <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 520, maxWidth: W }} role="img"
          aria-label={`Rank by ROC-AUC with duration versus without: ${rows.map((r) => `${r.display_name} ${r.rank_with} to ${r.rank_without}`).join('; ')}`}>
          <text x={x1} y={14} textAnchor="middle" fontSize="12" fill="var(--text-secondary)">with duration (leaky)</text>
          <text x={x2} y={14} textAnchor="middle" fontSize="12" fill="var(--accent-text)">without duration</text>
          {rows.map((r, i) => {
            const moved = r.rank_with !== r.rank_without
            const dim = hover && hover !== r.model_id
            const c = color(r, i)
            return (
              <g key={r.model_id} onMouseEnter={() => setHover(r.model_id)} onMouseLeave={() => setHover(null)} style={{ cursor: 'default' }} opacity={dim ? 0.25 : 1}>
                <motion.line x1={x1} x2={x2} y1={y(r.rank_with)} y2={y(r.rank_without)} stroke={c} strokeWidth={moved ? 2.6 : 1.4}
                  strokeDasharray={moved ? undefined : '4 4'} initial={{ pathLength: reduce ? 1 : 0 }} animate={inView ? { pathLength: 1 } : {}}
                  transition={{ duration: 0.9, delay: reduce ? 0 : 0.1 * i, ease: EASE }} />
                <circle cx={x1} cy={y(r.rank_with)} r={5} fill={c} />
                <motion.circle cx={x2} cy={y(r.rank_without)} r={5} fill={c} initial={{ opacity: reduce ? 1 : 0 }} animate={inView ? { opacity: 1 } : {}} transition={{ delay: reduce ? 0 : 0.9 + 0.1 * i }} />
                <text x={x1 - 12} y={y(r.rank_with) + 4} textAnchor="end" fontSize="12" fill="var(--text-primary)">{r.rank_with}. {r.display_name} <tspan className="num" fill="var(--text-muted)">{r.auc_with.toFixed(3)}</tspan></text>
                <text x={x2 + 12} y={y(r.rank_without) + 4} fontSize="12" fill="var(--text-primary)">{r.rank_without}. {r.display_name} <tspan fill="var(--text-muted)">{r.auc_without.toFixed(3)}</tspan></text>
              </g>
            )
          })}
        </svg>
      </div>
      <p className="secondary" style={{ fontSize: 13.5 }}>Solid lines change rank; dashed lines keep it. Ranked by ROC-AUC, as in the plan's §3.6 table. Hover a line to isolate it.</p>
    </div>
  )
}

// ------------------------------------------------------------------ page

export function ComparisonPage() {
  const lb = useArtifact<any>('comparison/leaderboard.json')
  const leak = useArtifact<any>('comparison/leakage.json')
  const pairs = useArtifact<any>('comparison/pairwise.json')
  const radar = useArtifact<any>('comparison/radar.json')
  const analysis = useArtifact<any>('comparison/analysis.json')
  const ref = useArtifact<any>('comparison/reference_models.json')
  const curvesArr = useArtifacts<{ without_duration: Curves; with_duration: Curves }>(CLASSIFIERS.map((m) => `models/${m}/curves.json`))
  const explainArr = useArtifacts<Explain>(MODEL_ORDER.map((m) => `models/${m}/explain.json`))
  if (lb.loading || leak.loading) return <Loading />
  if (!lb.data || !leak.data || !pairs.data) return (<><h1>Comparison</h1><div style={{ marginTop: 20 }}><Placeholder what="Comparison artifacts" detail="Run python -m src.comparison after the model pages are built." /></div></>)

  const curves: Record<string, Curves | undefined> = Object.fromEntries(CLASSIFIERS.map((m, i) => [m, curvesArr.data[i]?.without_duration]))
  const explains = Object.fromEntries(MODEL_ORDER.map((m, i) => [m, explainArr.data[i]]))
  const L = lb.data, K = leak.data
  const sup = L.models.filter((m: any) => m.runs)
  const winner = [...sup].sort((a: any, b: any) => b.runs.without_duration.pr_auc - a.runs.without_duration.pr_auc)[0]

  const allCurves = (kind: 'roc' | 'pr' | 'lift') => {
    const traces: any[] = []
    sup.forEach((e: any) => {
      const c = curves[e.model_id]
      if (!c) return
      const [x, y] = kind === 'roc' ? [c.roc.fpr, c.roc.tpr] : kind === 'pr' ? [c.pr.recall, c.pr.precision] : [c.lift.fraction_called, c.lift.fraction_captured]
      traces.push({ type: 'scatter', mode: 'lines', name: e.display_name, x, y, line: { color: MODEL_COLORS[e.model_id], width: 2 } })
    })
    if (kind === 'roc' || kind === 'lift') traces.push({ type: 'scatter', mode: 'lines', name: kind === 'lift' ? 'random calling' : 'chance', x: [0, 1], y: [0, 1], line: { color: '#6B7E8F', dash: 'dot', width: 1 } })
    if (kind === 'pr') traces.push({ type: 'scatter', mode: 'lines', name: 'base rate 0.1127', x: [0, 1], y: [0.1127, 0.1127], line: { color: '#6B7E8F', dash: 'dot', width: 1 } })
    const t = { roc: ['ROC — all classifiers (without duration)', 'false positive rate', 'true positive rate'], pr: ['Precision–recall — all classifiers', 'recall', 'precision'], lift: ['Cumulative gains (lift curve)', 'fraction of clients called, best-scored first', 'fraction of all subscribers reached'] }[kind]
    return { data: traces, layout: { title: t[0], xaxis: { title: t[1], tickformat: kind === 'lift' ? '.0%' : undefined }, yaxis: { title: t[2], tickformat: kind === 'lift' ? '.0%' : undefined }, legend: { orientation: 'h', y: -0.22 }, height: 400 } }
  }

  const leakRows = [...K.all].sort((a: any, b: any) => b.delta_auc - a.delta_auc)
  const leakFig = {
    data: [
      { type: 'bar', orientation: 'h', name: 'with duration', y: leakRows.map((r: any) => r.display_name), x: leakRows.map((r: any) => r.auc_with), marker: { color: '#D4A373' }, hovertemplate: '%{y}: %{x:.4f}<extra>with</extra>' },
      { type: 'bar', orientation: 'h', name: 'without duration', y: leakRows.map((r: any) => r.display_name), x: leakRows.map((r: any) => r.auc_without), marker: { color: '#4FB3A0' }, hovertemplate: '%{y}: %{x:.4f}<extra>without</extra>' },
    ],
    layout: { barmode: 'group', title: 'ROC-AUC with vs without the leak (sorted by inflation)', xaxis: { range: [0.7, 1], title: 'ROC-AUC' }, margin: { l: 160 }, height: 460, legend: { orientation: 'h', y: -0.15 },
      annotations: leakRows.map((r: any) => ({ x: r.auc_with, y: r.display_name, text: `+${r.delta_auc.toFixed(3)}`, xanchor: 'left', showarrow: false, xshift: 6, yshift: 7, font: { size: 10 } })) },
  }

  const R = radar.data
  const radarFig = R && {
    data: Object.keys(R.scaled).map((m) => ({ type: 'scatterpolar', fill: 'toself', opacity: 0.55, name: MODEL_NAMES[m], line: { color: MODEL_COLORS[m] },
      r: [...R.axes.map((a: string) => R.scaled[m][a]), R.scaled[m][R.axes[0]]], theta: [...R.axes, R.axes[0]].map((a: string) => a.replace('_', '-')) })),
    layout: { polar: { radialaxis: { visible: true, range: [0, 1], showticklabels: false }, angularaxis: {} }, height: 420, legend: { orientation: 'h', y: -0.1 }, title: 'Trade-offs (min–max scaled)' },
  }

  const cap = (m: any) => m.runs.without_duration.capture_at_20
  const bestCap = [...sup].sort((a: any, b: any) => cap(b) - cap(a))[0]

  return (
    <article>
      <Reveal>
        <div className="eyebrow">Synthesis</div>
        <h1 style={{ fontSize: 40, marginTop: 4 }}>Comparison</h1>
        <p className="secondary" style={{ fontSize: 18, marginTop: 6 }}>
          Best ranking without the leak: <strong style={{ color: MODEL_COLORS[winner.model_id] }}>{winner.display_name}</strong>, PR-AUC {fmt.auc(winner.runs.without_duration.pr_auc)}.
          The more important result is what the leak does to the ranking.
        </p>
      </Reveal>

      <Section eyebrow="⭐ The headline" title="The leak does not inflate models equally — it reorders them">
        {K.finding?.inflation && <Callout title="Unequal inflation">{K.finding.inflation}</Callout>}
        {K.finding?.reorder && <Callout title="Reordering">{K.finding.reorder} A comparison run on leaky data measures each algorithm's capacity to exploit the leak, not its predictive ability.</Callout>}
        <SlopeChart leak={K} />
        <Callout tone="warn" title="How we position this">{K.finding?.claim}</Callout>
      </Section>

      <Section eyebrow="Head to head" title="Compare any two models">
        <Comparator lb={L} pairs={pairs.data} curves={curves} explains={explains} />
      </Section>

      <Section eyebrow="Leaderboard" title="All models, majority baseline pinned first">
        <Leaderboard lb={L} />
      </Section>

      <Section eyebrow="Visuals" title="Curves, leak impact, trade-offs">
        <Reveal><div className="card" style={{ padding: 12 }}><PlotlyFigure fig={allCurves('roc')} label="ROC overlay of all classifiers" height={400} /></div></Reveal>
        <Reveal><div className="card" style={{ padding: 12 }}><PlotlyFigure fig={allCurves('pr')} label="Precision-recall overlay of all classifiers" height={400} /></div></Reveal>
        <Reveal><div className="card" style={{ padding: 12 }}><PlotlyFigure fig={leakFig} label="Paired bars of ROC-AUC with and without duration" height={460} /></div></Reveal>
        {radarFig && <Reveal><div className="card" style={{ padding: 12 }}><PlotlyFigure fig={radarFig} label="Radar of accuracy, F1, AUC, speed and interpretability" height={420} /><p className="muted" style={{ fontSize: 12 }}>{R.note}</p></div></Reveal>}
        <Reveal>
          <div className="card" style={{ padding: 12 }}>
            <PlotlyFigure fig={allCurves('lift')} label="Lift curve" height={400} />
            <Callout title="The business reading">
              Calling the top 20% of clients as ranked by {bestCap.display_name} reaches <strong>{fmt.pct(cap(bestCap))}</strong> of all subscribers, against 20% for random selection —
              {' '}{(cap(bestCap) / 0.2).toFixed(1)}× as many subscribers per call. {sup.filter((m: any) => m !== bestCap).map((m: any) => `${m.display_name}: ${fmt.pct(cap(m))}`).join(' · ')}.
              {' '}This is an offline estimate on the test split; there was no live campaign to confirm it.
            </Callout>
          </div>
        </Reveal>
      </Section>

      <Section eyebrow="Analysis" title="Why each model ranked where it did">
        {analysis.data ? analysis.data.paragraphs.map((p: any) => (
          <Reveal key={p.model_id}>
            <div className="card" style={{ padding: '14px 16px', borderLeft: `3px solid ${MODEL_COLORS[p.model_id] ?? 'var(--border-strong)'}` }}>
              <h3 style={{ marginBottom: 6 }}>{p.title}</h3>
              <p className="secondary">{p.text}</p>
            </div>
          </Reveal>
        )) : <Placeholder what="Written analysis" />}
        {ref.data && <p className="muted" style={{ fontSize: 12 }}>{ref.data.note}</p>}
      </Section>
    </article>
  )
}
