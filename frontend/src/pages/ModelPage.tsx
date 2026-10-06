import { useEffect } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ExtraView } from '../components/Extras'
import { FigureView } from '../components/FigureView'
import { FormulaCard } from '../components/Formula'
import { AssumptionList, Callout, FindingsList, fmt, Grid, Loading, MetricTile, Placeholder, Reveal, Section } from '../components/ui'
import { WorkflowDiagram } from '../components/WorkflowDiagram'
import { useArtifact } from '../data/loader'
import type { Findings, Meta, Metrics, ModelConfig } from '../data/types'
import { useUI } from '../store'
import { MODEL_NAMES, MODEL_ORDER } from '../theme/palette'

const FAMILY: Record<string, string> = {
  interpretable_baseline: 'Interpretable baseline', gradient_boosting: 'Gradient boosting', tabular_deep_learning: 'Tabular deep learning',
  foundation_model: 'Tabular foundation model', unsupervised: 'Unsupervised', pattern_mining: 'Pattern mining',
}

function HeaderMetrics({ m }: { m: Metrics }) {
  if (m.runs) {
    const r = m.runs.without_duration
    return (
      <Grid min={150}>
        <MetricTile label="ROC-AUC" value={r.roc_auc} format={fmt.auc} accent hint="without duration" />
        <MetricTile label="PR-AUC" value={r.pr_auc} format={fmt.auc} hint={`baseline ${fmt.auc(m.baselines?.majority_class.pr_auc ?? 0.1127)}`} />
        <MetricTile label="F1" value={r.f1} format={fmt.auc} hint={`threshold ${r.threshold?.toFixed(3)}`} />
        <MetricTile label="Train time" value={r.train_seconds ?? null} format={fmt.secs} hint="4 CPU cores" />
        <MetricTile label="Leak Δ AUC" value={m.leak_delta?.roc_auc ?? null} format={fmt.delta} hint="with − without duration" />
      </Grid>
    )
  }
  const tiles: [string, number | undefined, (v: number) => string, string?][] = m.task === 'unsupervised'
    ? [['Clusters', m.n_clusters as number, fmt.int], ['Noise', m.noise_fraction as number, fmt.pct], ['AMI with month', (m.ami as Record<string, number>)?.month, fmt.auc],
       ['AMI with subscription', (m.ami as Record<string, number>)?.['y (subscription)'], fmt.auc], ['Cluster-rate AUC', m.cluster_rate_auc_test as number, fmt.auc, 'held-out clients']]
    : [['Frequent itemsets', m.frequent_itemsets as number, fmt.int], ['Rules', m.rules_total as number, fmt.int], ['Actionable rules', m.rules_actionable as number, fmt.int, '→ y=yes, lift ≥ 1.2, non-redundant'],
       ['Rule-score AUC', m.rule_score_auc_test as number, fmt.auc, 'held-out clients']]
  return <Grid min={150}>{tiles.map(([l, v, f, h]) => <MetricTile key={l} label={l} value={v ?? null} format={f} hint={h} />)}</Grid>
}

function RunTable({ m }: { m: Metrics }) {
  if (!m.runs) return null
  const rows: [string, keyof typeof m.runs.with_duration, (v: number) => string][] = [
    ['Accuracy', 'accuracy', fmt.auc], ['Precision', 'precision', fmt.auc], ['Recall', 'recall', fmt.auc], ['F1', 'f1', fmt.auc],
    ['ROC-AUC', 'roc_auc', fmt.auc], ['PR-AUC', 'pr_auc', fmt.auc], ['MCC', 'mcc', fmt.auc], ['Brier', 'brier', fmt.auc],
    ['CV ROC-AUC', 'cv_roc_auc_mean', fmt.auc], ['Train', 'train_seconds', fmt.secs], ['Predict', 'predict_seconds', fmt.secs],
  ]
  const b = m.baselines?.majority_class
  const bv: Record<string, number> = b ? { accuracy: b.accuracy, roc_auc: b.roc_auc, pr_auc: b.pr_auc, f1: 0, recall: 0, precision: 0, mcc: 0 } : {}
  return (
    <div className="card scroll-x" style={{ padding: 4 }}>
      <table className="data">
        <thead><tr><th>Metric</th><th className="num">Majority baseline</th><th className="num">Without duration (primary)</th><th className="num">With duration (leaky)</th><th className="num">Δ</th></tr></thead>
        <tbody>
          {rows.map(([label, k, f]) => {
            const a = m.runs!.without_duration[k] as number | null | undefined, w = m.runs!.with_duration[k] as number | null | undefined
            return (
              <tr key={label}>
                <td>{label}</td>
                <td className="num muted">{bv[k as string] != null ? f(bv[k as string]) : ''}</td>
                <td className="num" style={{ color: 'var(--accent-text)' }}>{a == null ? '—' : f(a)}</td>
                <td className="num secondary">{w == null ? '—' : f(w)}</td>
                <td className="num muted">{a != null && w != null && !String(k).includes('seconds') ? fmt.delta(w - a) : ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12, padding: '8px 10px' }}>Protocol: {String(m.protocol?.split ?? '')}; threshold — {String(m.protocol?.threshold ?? '')}.</p>
    </div>
  )
}


interface CallRanking {
  score: string; how: string; steps: string[]; test_clients: number; subscribers: number; base_rate: number; roc_auc: number
  distinct_scores: number; top: { share: number; calls: number; hit_rate: number; lift: number; captured: number }[]
  within_month_auc: number; within_regime_auc: number; threshold_call_share?: number; summary: string; caveat: string
}

function CallRankingView({ r }: { r: CallRanking }) {
  return (
    <>
      <Reveal>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div className="eyebrow">Ranks clients by</div>
          <div style={{ fontSize: 17, fontWeight: 600, marginTop: 2, color: 'var(--accent-text)' }}>{r.score}</div>
          <p className="secondary" style={{ marginTop: 8 }}>{r.how}</p>
          <ol style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(180px, 100%), 1fr))', gap: 8, listStyle: 'none', padding: 0, margin: '12px 0 0' }}>
            {r.steps.map((st, i) => (
              <li key={i} className="card" style={{ padding: '8px 10px', background: 'var(--bg-elevated)', fontSize: 13 }}>
                <span className="mono muted" style={{ fontSize: 11 }}>{String(i + 1).padStart(2, '0')}</span><div>{st}</div>
              </li>
            ))}
          </ol>
          <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
            {r.distinct_scores.toLocaleString()} distinct scores across {r.test_clients.toLocaleString()} test clients
            {r.distinct_scores < 200 ? ' — many clients tie, so the list is called group by group.' : ' — almost every client gets their own position in the list.'}
            {r.threshold_call_share != null && ` At its F1-optimal threshold the model would call ${fmt.pct(r.threshold_call_share)} of clients.`}
          </p>
        </div>
      </Reveal>
      <Reveal>
        <div className="card scroll-x" style={{ padding: 4 }}>
          <table className="data">
            <thead><tr><th>Call the top…</th><th className="num">Calls</th><th className="num">Subscribe</th><th className="num">vs random</th><th className="num">Subscribers reached</th></tr></thead>
            <tbody>
              {r.top.map((t) => (
                <tr key={t.share}>
                  <td>{fmt.pct(t.share).replace('.0%', '%')} of clients</td><td className="num">{fmt.int(t.calls)}</td>
                  <td className="num" style={{ color: 'var(--accent-text)' }}>{fmt.pct(t.hit_rate)}</td><td className="num">{t.lift.toFixed(1)}×</td>
                  <td className="num">{fmt.pct(t.captured)}</td>
                </tr>
              ))}
              <tr><td className="muted">Random calling</td><td className="num muted">—</td><td className="num muted">{fmt.pct(r.base_rate)}</td><td className="num muted">1.0×</td><td className="num muted">= share called</td></tr>
            </tbody>
          </table>
          <p className="muted" style={{ fontSize: 12, padding: '8px 10px' }}>Held-out test set: {r.test_clients.toLocaleString()} clients, {r.subscribers.toLocaleString()} subscribers, scored without duration.</p>
        </div>
      </Reveal>
      <Reveal><Callout title="In one sentence">{r.summary}</Callout></Reveal>
      <Reveal><Callout tone="warn" title="Read with care">{r.caveat}</Callout></Reveal>
    </>
  )
}

export function ModelPage() {
  const { id = '' } = useParams()
  const visit = useUI((s) => s.visitModel)
  useEffect(() => { if (id) visit(id) }, [id, visit])
  const base = `models/${id}`
  const meta = useArtifact<Meta>(`${base}/meta.json`)
  const metrics = useArtifact<Metrics>(`${base}/metrics.json`)
  const config = useArtifact<ModelConfig>(`${base}/config.json`)
  const findings = useArtifact<Findings>(`${base}/findings.json`)
  const ranking = useArtifact<CallRanking>(`${base}/call_ranking.json`)

  if (meta.loading) return <Loading />
  if (!meta.data) return (<>
    <h1>{MODEL_NAMES[id] ?? id}</h1>
    <div style={{ marginTop: 20 }}><Placeholder what={`The ${MODEL_NAMES[id] ?? id} page`} /></div>
  </>)
  const m = meta.data
  const i = MODEL_ORDER.indexOf(id)

  return (
    <article>
      {/* ① HEADER */}
      <Reveal>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <span className="chip">{FAMILY[m.family] ?? m.family}</span>
          <span className="chip chip-muted">{m.year}</span>
          <span className="muted" style={{ fontSize: 12 }}>{m.reference}</span>
        </div>
        <h1 style={{ marginTop: 10, fontSize: 40 }}>{m.display_name}</h1>
        <p style={{ fontSize: 18, marginTop: 6 }} className="secondary">{m.one_liner}</p>
        <div style={{ marginTop: 18 }}>{metrics.data ? <HeaderMetrics m={metrics.data} /> : <Placeholder what="Metrics" />}</div>
        {findings.data && <div style={{ marginTop: 12 }}><Callout title="Headline">{findings.data.headline}</Callout></div>}
      </Reveal>

      <Section eyebrow="In plain words" title="What it does">
        <Reveal><p style={{ fontSize: 16, maxWidth: 760 }}>{m.plain_explanation}</p></Reveal>
      </Section>

      <Section eyebrow="The business question" title="How it decides who to call first">
        {ranking.data ? <CallRankingView r={ranking.data} /> : <Placeholder what="Call ranking" detail="Run python -m src.call_ranking." />}
      </Section>

      {/* ② WORKFLOW */}
      <Section eyebrow="Step 2" title="Workflow">
        <Reveal><WorkflowDiagram nodes={m.workflow} edges={m.edges} formulas={m.formulas} /></Reveal>
      </Section>

      {/* ③ FORMULAS */}
      <Section eyebrow="Step 3" title="Formulas">
        {m.formulas.map((f) => <Reveal key={f.id}><FormulaCard f={f} /></Reveal>)}
      </Section>

      {/* ④ CONFIG */}
      <Section eyebrow="Step 4" title="Configuration and assumptions" intro={config.data ? <span className="mono" style={{ fontSize: 13 }}>{config.data.estimator}</span> : undefined}>
        {config.data ? (
          <Reveal>
            <div className="card scroll-x" style={{ padding: 4 }}>
              <table className="data">
                <thead><tr><th>Hyperparameter</th><th>Value</th><th>Why</th></tr></thead>
                <tbody>{config.data.params.map((p) => (
                  <tr key={p.name}><td className="mono" style={{ fontSize: 12.5 }}>{p.name}</td>
                    <td className="mono" style={{ fontSize: 12.5, maxWidth: 220, wordBreak: 'break-word' }}>{Array.isArray(p.value) ? p.value.join(', ') : String(p.value)}</td>
                    <td className="secondary">{p.why}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </Reveal>
        ) : <Placeholder what="Configuration" />}
        <Reveal><AssumptionList items={m.assumptions} /></Reveal>
      </Section>

      {/* ⑤ RESULTS */}
      <Section eyebrow="Step 5" title="Results">
        {metrics.data?.runs && <Reveal><RunTable m={metrics.data} /></Reveal>}
        {m.figures.map((f) => <Reveal key={f.file}><FigureView base={base} fig={f} /></Reveal>)}
        {m.extras?.map((x) => (
          <Reveal key={x.file}>
            <div className="card" style={{ padding: '16px 18px', borderColor: 'var(--accent)' }}>
              <div className="eyebrow" style={{ color: 'var(--accent-text)' }}>★ Key exhibit</div>
              <h3 style={{ margin: '4px 0 12px' }}>{x.title}</h3>
              <ExtraView base={base} extra={x} />
            </div>
          </Reveal>
        ))}
      </Section>

      {/* ⑥ FINDINGS */}
      <Section eyebrow="Step 6" title="Findings — including what did not work">
        {findings.data ? (<>
          <FindingsList findings={findings.data.findings} />
          <Reveal><Callout title="How this differs from the other models">{findings.data.comparison_note}</Callout></Reveal>
          <Reveal><Callout tone="warn" title="Viva question">{findings.data.viva_answer}</Callout></Reveal>
        </>) : <Placeholder what="Findings" />}
      </Section>

      <nav aria-label="Model navigation" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 48, gap: 12 }}>
        {i > 0 ? <Link className="btn" to={`/models/${MODEL_ORDER[i - 1]}`}>← {MODEL_NAMES[MODEL_ORDER[i - 1]]}</Link> : <Link className="btn" to="/eda">← EDA</Link>}
        {i < MODEL_ORDER.length - 1 ? <Link className="btn" to={`/models/${MODEL_ORDER[i + 1]}`}>{MODEL_NAMES[MODEL_ORDER[i + 1]]} →</Link> : <Link className="btn" to="/compare">Comparison →</Link>}
      </nav>
    </article>
  )
}
