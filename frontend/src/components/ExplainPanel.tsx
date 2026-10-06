import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useArtifact } from '../data/loader'
import type { ClusterExplain, Explain, RuleExplain, ShapExplain } from '../data/types'
import { useUI, type ExplainTab } from '../store'
import { EASE, MODEL_NAMES, MODEL_ORDER } from '../theme/palette'
import { PlotlyFigure } from './PlotlyFigure'
import { fmt, Loading, Placeholder } from './ui'

function useWidth() {
  const [w, setW] = useState(typeof window === 'undefined' ? 1400 : window.innerWidth)
  useEffect(() => {
    const on = () => setW(window.innerWidth)
    window.addEventListener('resize', on)
    return () => window.removeEventListener('resize', on)
  }, [])
  return w
}

const Sentence = ({ children }: { children: ReactNode }) => <p className="secondary" style={{ fontSize: 13.5, margin: '8px 0' }}>{children}</p>

// ------------------------------------------------------------------ SHAP

function ShapGlobal({ e }: { e: ShapExplain }) {
  const n = Math.min(12, e.global.features.length)
  const feats = e.global.features.slice(0, n).reverse()
  const share = e.global.share.slice(0, n).reverse()
  const dir = e.global.direction.slice(0, n).reverse()
  const bars = {
    data: [{ type: 'bar', orientation: 'h', x: share, y: feats, marker: { color: dir.map((d) => d === 'positive' ? '#4FB3A0' : d === 'negative' ? '#C88B8B' : '#7B9FC7') },
      hovertemplate: '%{y}: %{x:.1%} of attribution<extra></extra>' }],
    layout: { margin: { l: 110, r: 10, t: 10, b: 30 }, xaxis: { tickformat: '.0%' }, height: 300 },
  }
  const b = e.beeswarm
  const traces = b.features.map((f, i) => ({
    type: 'scattergl', mode: 'markers', name: f, showlegend: false,
    x: b.values[i], y: b.values[i].map((_, k) => b.features.length - 1 - i + (((k * 7919) % 100) / 100 - 0.5) * 0.6),
    text: b.labels[i].map((l) => `${f} = ${l}`),
    hovertemplate: '%{text}<br>SHAP %{x:.3f}<extra></extra>',
    marker: { size: 3.5, opacity: 0.75, color: b.feature_values[i].map((v) => v ?? 0.5),
      colorscale: [[0, '#7B9FC7'], [0.5, '#9FB0C0'], [1, '#D4A373']], cmin: 0, cmax: 1, showscale: false },
  }))
  const bee = {
    data: traces,
    layout: { height: 340, margin: { l: 110, r: 10, t: 10, b: 36 }, xaxis: { title: `SHAP value (${e.units})`, zeroline: true },
      yaxis: { tickvals: b.features.map((_, i) => b.features.length - 1 - i), ticktext: b.features, showgrid: false } },
  }
  return (
    <div>
      <Sentence>{e.global.plain_text}</Sentence>
      <PlotlyFigure fig={bars} height={300} label={`Share of SHAP attribution by feature for ${MODEL_NAMES[e.model_id]}`} />
      <p className="muted" style={{ fontSize: 11.5 }}>Teal = higher value pushes up · rose = pushes down · blue = mixed (categorical).</p>
      <ul style={{ fontSize: 12.5, paddingLeft: 16, margin: '8px 0' }} className="secondary">
        {e.global.direction_text.slice(0, 5).map((t) => <li key={t}>{t}</li>)}
      </ul>
      <div className="eyebrow" style={{ marginTop: 12 }}>Beeswarm — every dot is one client</div>
      <PlotlyFigure fig={bee} height={340} label="SHAP beeswarm plot" />
      <Sentence>Each dot is a client; its position is how much that feature moved their prediction. Colour = feature value (blue low → amber high; grey = categorical).</Sentence>
    </div>
  )
}

function ShapLocal({ e }: { e: ShapExplain }) {
  const { localIndex, setLocalIndex } = useUI()
  const i = Math.min(localIndex[e.model_id] ?? 0, e.local_examples.length - 1)
  const ex = e.local_examples[i]
  if (!ex) return <Sentence>No local examples.</Sentence>
  const c = [...ex.contributions].reverse()
  const fig = {
    data: [{ type: 'waterfall', orientation: 'h', base: ex.base_value, y: c.map((x) => x.feature), x: c.map((x) => x.value),
      measure: c.map(() => 'relative'), connector: { line: { color: 'rgba(159,176,192,0.3)' } },
      increasing: { marker: { color: '#4FB3A0' } }, decreasing: { marker: { color: '#C88B8B' } },
      hovertemplate: '%{y}: %{x:+.3f}<extra></extra>' }],
    layout: { height: 340, margin: { l: 150, r: 10, t: 10, b: 36 }, xaxis: { title: `model output (${e.units})` } },
  }
  return (
    <div>
      <label className="eyebrow" htmlFor="local-pick">Choose a client</label>
      <select id="local-pick" className="select" style={{ width: '100%', marginTop: 4 }} value={i} onChange={(ev) => setLocalIndex(e.model_id, Number(ev.target.value))}>
        {e.local_examples.map((x, k) => <option key={k} value={k}>#{x.client_id} — {x.label}</option>)}
      </select>
      <div className="card" style={{ padding: '10px 12px', marginTop: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span className="eyebrow">Client #{ex.client_id}</span>
          <span className={ex.y_true ? 'chip' : 'chip chip-muted'}>{ex.y_true ? 'subscribed' : 'did not subscribe'}</span>
        </div>
        <Sentence>{ex.plain_text}</Sentence>
      </div>
      <PlotlyFigure fig={fig} height={340} label={`SHAP waterfall for client ${ex.client_id}`} />
      <Sentence>Start at the average prediction ({fmt.num(ex.base_value)}); each bar adds one feature's push. The bars sum exactly to this client's output — SHAP's additivity property.</Sentence>
    </div>
  )
}

function ShapCompare({ e }: { e: ShapExplain }) {
  const prev = useUI((s) => s.previousModel)
  const options = MODEL_ORDER.filter((m) => m !== e.model_id && ['decision_tree', 'catboost', 'tabnet', 'tabpfn'].includes(m))
  const [other, setOther] = useState<string>(prev && options.includes(prev) ? prev : options[0])
  const o = useArtifact<Explain>(`models/${other}/explain.json`)
  const oe = o.data?.kind === 'shap' ? o.data : undefined
  const feats = useMemo(() => {
    if (!oe) return []
    const top = new Set([...e.global.features.slice(0, 8), ...oe.global.features.slice(0, 8)])
    return [...top]
  }, [e, oe])
  const share = (x: ShapExplain, f: string) => x.global.share[x.global.features.indexOf(f)] ?? 0
  return (
    <div>
      <label className="eyebrow" htmlFor="cmp-pick">Compare with</label>
      <select id="cmp-pick" className="select" style={{ width: '100%', marginTop: 4 }} value={other} onChange={(ev) => setOther(ev.target.value)}>
        {options.map((m) => <option key={m} value={m}>{MODEL_NAMES[m]}{m === prev ? ' (last viewed)' : ''}</option>)}
      </select>
      {o.loading ? <Loading /> : !oe ? <Placeholder what={`${MODEL_NAMES[other]} SHAP`} /> : (<>
        <PlotlyFigure height={420} label="SHAP share comparison" fig={{
          data: [
            { type: 'bar', orientation: 'h', name: MODEL_NAMES[e.model_id], y: [...feats].reverse(), x: [...feats].reverse().map((f) => share(e, f)), marker: { color: '#4FB3A0' } },
            { type: 'bar', orientation: 'h', name: MODEL_NAMES[other], y: [...feats].reverse(), x: [...feats].reverse().map((f) => share(oe, f)), marker: { color: '#D4A373' } },
          ],
          layout: { barmode: 'group', height: 420, margin: { l: 110, r: 10, t: 10, b: 30 }, xaxis: { tickformat: '.0%' }, legend: { orientation: 'h', y: -0.1 } },
        }} />
        <Sentence>
          {MODEL_NAMES[e.model_id]} leans most on <strong>{e.global.features[0]}</strong>; {MODEL_NAMES[other]} on <strong>{oe.global.features[0]}</strong>.
          {' '}Macro features take {fmt.pct(e.global.macro_share)} vs {fmt.pct(oe.global.macro_share)} of attribution.
          {' '}Shares are compared because the raw SHAP units differ between models ({e.units} vs {oe.units}).
        </Sentence>
      </>)}
    </div>
  )
}

// ------------------------------------------------------------------ clusters / rules

function ClusterPanel({ e, tab }: { e: ClusterExplain; tab: ExplainTab }) {
  const [sel, setSel] = useState(e.clusters.find((c) => c.id >= 0)?.id ?? 0)
  const sorted = [...e.clusters].sort((a, b) => b.subscribe_rate - a.subscribe_rate)
  if (tab === 'global') return (
    <div>
      <Sentence>{e.global.plain_text}</Sentence>
      <PlotlyFigure height={Math.max(260, sorted.length * 16)} label="Subscription rate per cluster" fig={{
        data: [{ type: 'bar', orientation: 'h', y: sorted.map((c) => c.name).reverse(), x: sorted.map((c) => c.subscribe_rate).reverse(),
          text: sorted.map((c) => `${c.size.toLocaleString()} clients`).reverse(), hovertemplate: '%{y}: %{x:.1%} · %{text}<extra></extra>', marker: { color: '#4FB3A0' } }],
        layout: { height: Math.max(260, sorted.length * 16), margin: { l: 110, r: 10, t: 10, b: 30 }, xaxis: { tickformat: '.0%' }, shapes: [{ type: 'line', x0: 0.1127, x1: 0.1127, yref: 'paper', y0: 0, y1: 1, line: { dash: 'dot', color: '#9FB0C0' } }] },
      }} />
      <Sentence>Dotted line = the 11.27% overall rate.</Sentence>
    </div>
  )
  const c = e.clusters.find((x) => x.id === sel) ?? e.clusters[0]
  if (tab === 'local') return (
    <div>
      <label className="eyebrow" htmlFor="cl-pick">Choose a cluster</label>
      <select id="cl-pick" className="select" style={{ width: '100%', marginTop: 4 }} value={sel} onChange={(ev) => setSel(Number(ev.target.value))}>
        {sorted.map((x) => <option key={x.id} value={x.id}>{x.name} — {fmt.pct(x.subscribe_rate)} ({x.size.toLocaleString()})</option>)}
      </select>
      <Sentence>{c.profile}</Sentence>
      <PlotlyFigure height={280} label={`Defining features of ${c.name}`} fig={{
        data: [{ type: 'bar', orientation: 'h', y: c.deviations.map((d) => d.feature).reverse(), x: c.deviations.map((d) => d.deviation).reverse(),
          text: c.deviations.map((d) => d.text).reverse(), hovertemplate: '%{text}<extra></extra>',
          marker: { color: c.deviations.map((d) => (d.deviation > 0 ? '#4FB3A0' : '#C88B8B')).reverse() } }],
        layout: { height: 280, margin: { l: 150, r: 10, t: 10, b: 30 }, xaxis: { title: 'deviation from population (z / scaled share)' } },
      }} />
      <ul className="secondary" style={{ fontSize: 12.5, paddingLeft: 16 }}>{c.deviations.map((d) => <li key={d.feature}>{d.text}</li>)}</ul>
    </div>
  )
  return <Sentence>Clusters are not a classifier, so there is no SHAP to compare. Compare the macro-feature share in the supervised models' panels with this page's finding: the clusters line up with the month of the call.</Sentence>
}

function RulePanel({ e, tab }: { e: RuleExplain; tab: ExplainTab }) {
  const [sel, setSel] = useState(0)
  if (tab === 'global') return (
    <div>
      <Sentence>{e.global.plain_text}</Sentence>
      <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {e.items.slice(0, 10).map((it, k) => (
          <li key={it.item} className="card" style={{ padding: '8px 10px', fontSize: 13 }}>
            <button className="mono" style={{ all: 'unset', cursor: 'pointer', color: 'var(--accent-text)' }} onClick={() => { setSel(k); useUI.getState().setExplainTab('local') }}>{it.item}</button>
            <div className="secondary" style={{ fontSize: 12 }}>{it.plain_text}</div>
          </li>
        ))}
      </ul>
    </div>
  )
  const it = e.items[Math.min(sel, e.items.length - 1)]
  if (tab === 'local') return (
    <div>
      <label className="eyebrow" htmlFor="it-pick">Choose an item</label>
      <select id="it-pick" className="select" style={{ width: '100%', marginTop: 4 }} value={sel} onChange={(ev) => setSel(Number(ev.target.value))}>
        {e.items.map((x, k) => <option key={x.item} value={k}>{x.item}</option>)}
      </select>
      <Sentence>{it.plain_text}</Sentence>
      <div className="scroll-x"><table className="data">
        <thead><tr><th>Rule</th><th className="num">lift</th><th className="num">conf</th><th className="num">coverage</th></tr></thead>
        <tbody>{it.rules.map((r) => <tr key={r.rule}><td className="mono" style={{ fontSize: 11.5 }}>{r.rule}</td><td className="num">{r.lift.toFixed(2)}</td><td className="num">{fmt.pct(r.confidence)}</td><td className="num">{fmt.pct2(r.coverage)}</td></tr>)}</tbody>
      </table></div>
    </div>
  )
  return <Sentence>Rules are not a per-client model, so there is no SHAP to compare. The items with the highest lift here (poutcome=success, cellular contact, low-rate months) are the same signals SHAP ranks highly in the supervised models.</Sentence>
}

// ------------------------------------------------------------------ shell

function PanelBody({ modelId }: { modelId: string }) {
  const { data, loading, error } = useArtifact<Explain>(`models/${modelId}/explain.json`)
  const tab = useUI((s) => s.explainTab)
  const setTab = useUI((s) => s.setExplainTab)
  if (loading) return <Loading />
  if (error || !data) return <Placeholder what="Explain panel" />
  const tabs: ExplainTab[] = ['global', 'local', 'compare']
  const kindLabel = data.kind === 'shap' ? `SHAP · ${data.explainer} · ${data.sample_size.toLocaleString()} clients` : data.kind === 'cluster_profile' ? 'Cluster profiles' : 'Rule contributions'
  return (
    <div>
      <div className="eyebrow">Explain</div>
      <div className="muted mono" style={{ fontSize: 11 }}>{kindLabel}</div>
      <div className="card" style={{ padding: '10px 12px', marginTop: 10, background: 'var(--bg-elevated)' }}>
        <div style={{ fontWeight: 600, fontSize: 14 }}>{data.narrative.headline}</div>
        <p className="secondary" style={{ fontSize: 13, marginTop: 4 }}>{data.narrative.body}</p>
        <p style={{ fontSize: 12, marginTop: 6, color: 'var(--warning)' }}>⚠ {data.narrative.caveat}</p>
      </div>
      <div role="tablist" aria-label="Explain views" style={{ display: 'flex', gap: 4, marginTop: 12, borderBottom: '1px solid var(--border)' }}>
        {tabs.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            style={{ all: 'unset', cursor: 'pointer', padding: '6px 10px', fontSize: 13, textTransform: 'capitalize',
              color: tab === t ? 'var(--accent-text)' : 'var(--text-secondary)', borderBottom: `2px solid ${tab === t ? 'var(--accent)' : 'transparent'}` }}>{t}</button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={tab} role="tabpanel" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.16 }} style={{ paddingTop: 10 }}>
          {data.kind === 'shap' ? (tab === 'global' ? <ShapGlobal e={data} /> : tab === 'local' ? <ShapLocal e={data} /> : <ShapCompare e={data} />)
            : data.kind === 'cluster_profile' ? <ClusterPanel e={data} tab={tab} /> : <RulePanel e={data as RuleExplain} tab={tab} />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export function ExplainPanel({ modelId }: { modelId: string }) {
  const open = useUI((s) => s.explainOpen)
  const setOpen = useUI((s) => s.setExplainOpen)
  const w = useWidth()
  const [sheetUp, setSheetUp] = useState(false)

  if (w >= 1280) {
    return (
      <AnimatePresence initial={false}>
        {open && (
          <motion.aside key="panel" aria-label="Explain panel" initial={{ x: 40, opacity: 0, width: 0 }} animate={{ x: 0, opacity: 1, width: 380 }} exit={{ x: 40, opacity: 0, width: 0 }}
            transition={{ duration: 0.28, ease: EASE }} className="themed"
            style={{ flexShrink: 0, borderLeft: '1px solid var(--border)', position: 'sticky', top: 'var(--header-h)', height: 'calc(100vh - var(--header-h))', overflowY: 'auto', background: 'var(--bg-surface)' }}>
            <div style={{ padding: '20px 16px 40px', width: 380 }}><PanelBody modelId={modelId} /></div>
          </motion.aside>
        )}
      </AnimatePresence>
    )
  }
  if (w >= 768) {
    return (
      <AnimatePresence>
        {open && (<>
          <motion.div key="scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 35 }} />
          <motion.aside key="drawer" aria-label="Explain panel" initial={{ x: 400 }} animate={{ x: 0 }} exit={{ x: 400 }} transition={{ duration: 0.28, ease: EASE }}
            style={{ position: 'fixed', right: 0, top: 'var(--header-h)', bottom: 0, width: 380, zIndex: 36, background: 'var(--bg-surface)', borderLeft: '1px solid var(--border)', overflowY: 'auto', padding: '20px 16px 40px' }}>
            <PanelBody modelId={modelId} />
          </motion.aside>
        </>)}
      </AnimatePresence>
    )
  }
  return (
    <motion.aside aria-label="Explain panel" drag="y" dragConstraints={{ top: 0, bottom: 0 }} dragElastic={0.2}
      onDragEnd={(_, info) => { if (info.offset.y < -40) setSheetUp(true); if (info.offset.y > 40) setSheetUp(false) }}
      animate={{ y: sheetUp ? 0 : 'calc(100% - 52px)' }} transition={{ duration: 0.28, ease: EASE }}
      style={{ position: 'fixed', left: 0, right: 0, bottom: 0, height: '78vh', zIndex: 36, background: 'var(--bg-surface)', borderTop: '1px solid var(--border-strong)', borderRadius: '16px 16px 0 0', boxShadow: '0 -8px 24px rgba(0,0,0,0.25)' }}>
      <button onClick={() => setSheetUp(!sheetUp)} aria-expanded={sheetUp} style={{ all: 'unset', cursor: 'pointer', display: 'block', width: '100%', textAlign: 'center', padding: '8px 0 6px' }}>
        <span style={{ display: 'block', width: 40, height: 4, borderRadius: 4, background: 'var(--border-strong)', margin: '0 auto 6px' }} />
        <span className="eyebrow">{sheetUp ? 'Swipe down to close' : 'Swipe up · Explain'}</span>
      </button>
      <div style={{ overflowY: 'auto', height: 'calc(100% - 52px)', padding: '4px 16px 40px' }}><PanelBody modelId={modelId} /></div>
    </motion.aside>
  )
}
