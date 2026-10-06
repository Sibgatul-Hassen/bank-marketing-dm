import { Link } from 'react-router-dom'
import { FigureView } from '../components/FigureView'
import { Callout, fmt, Grid, MetricTile, Reveal, Section } from '../components/ui'
import { useArtifact } from '../data/loader'
import type { Manifest } from '../data/types'
import { MODEL_COLORS, MODEL_NAMES } from '../theme/palette'

/* eslint-disable @typescript-eslint/no-explicit-any */

const LIMITS = [
  ['Scope', 'One Portuguese bank, May 2008 – November 2010, spanning the financial crisis. Five features are macroeconomic indicators tracking exactly that period; this would not transfer to another bank or decade.'],
  ['No novelty claim', 'This dataset has been modelled many times. The contribution is the leakage measurement and the explanations.'],
  ['Simulated deployment', 'No live campaign to test against; the lift curve is an offline estimate.'],
  ['Judgement calls', "'unknown' handling is one of three defensible choices. We report the affected counts so a reader can judge."],
]

export function Home() {
  const manifest = useArtifact<Manifest>('manifest.json').data
  const leak = useArtifact<any>('comparison/leakage.json').data
  const summary = useArtifact<any>('dataset/summary.json').data
  const big = leak?.reference?.length ? [...leak.reference].sort((a: any, b: any) => b.delta_auc - a.delta_auc) : []
  return (
    <article>
      <Reveal>
        <div className="eyebrow">Data Mining · Track 1 · UCI Bank Marketing</div>
        <h1 style={{ fontSize: 'clamp(30px, 5vw, 48px)', marginTop: 8, lineHeight: 1.1 }}>A leakage-aware, visually explained comparison of modern tabular learning methods</h1>
        <p className="secondary" style={{ fontSize: 18, marginTop: 14, maxWidth: 760 }}>
          A Portuguese bank sells term deposits by telephone. Calling everyone is expensive and only {summary ? fmt.pct2(summary.target.yes_rate) : '11.27%'} subscribe.
          <strong style={{ color: 'var(--text-primary)' }}> Which clients should be called?</strong>
        </p>
      </Reveal>

      <Section>
        <FigureView base="models/umap_hdbscan" fig={{ file: 'points3d.json', type: 'points3d', title: '8,000 clients in 3D',
          caption: 'Each point is a client, placed by UMAP so that similar clients sit together; teal = subscribed. Subscribers are spread across the whole cloud — the first hint that this problem is hard.' }} />
      </Section>

      <Section eyebrow="What we claim" title="The accuracy usually reported here depends on a feature you cannot have">
        <Reveal><p style={{ fontSize: 16 }}>
          Most published work on this dataset keeps <span className="mono">duration</span> — the length of the call, known only <em>after</em> it ends. We report every model twice
          and measure the inflation. It is not uniform, and it reorders the ranking.
        </p></Reveal>
        {big.length > 0 && (
          <Grid min={180}>
            <MetricTile label={`Leak gain · ${big[0].display_name}`} value={big[0].delta_auc} format={fmt.delta} accent hint="ROC-AUC, with − without" />
            <MetricTile label={`Leak gain · ${big[big.length - 1].display_name}`} value={big[big.length - 1].delta_auc} format={fmt.delta} hint="the least-inflated family" />
            <MetricTile label="Random Forest rank" value={big.find((r: any) => r.model_id === 'random_forest')?.rank_without ?? null} format={(v) => `${big.find((r: any) => r.model_id === 'random_forest')?.rank_with} → ${Math.round(v)}`} hint="with → without duration" />
          </Grid>
        )}
        <Grid min={240}>
          {[
            ['A · The leakage quantification', 'Every model trained with and without the leak, on identical splits — and the ranking compared.', '/compare'],
            ['B · An explained comparison', 'For each technique, the structural reason it ranked where it did — not just the score.', '/compare'],
            ['C · An inspectable tool', 'Workflows, formulas, configurations, figures, failures and SHAP beside every model.', '/models/decision_tree'],
          ].map(([t, d, to]) => (
            <Link key={t} to={to} className="card card-hover" style={{ padding: '14px 16px', color: 'var(--text-primary)' }}>
              <div style={{ fontWeight: 600 }}>{t}</div><p className="secondary" style={{ fontSize: 14, marginTop: 4 }}>{d}</p>
            </Link>
          ))}
        </Grid>
      </Section>

      <Section eyebrow="Pages" title="Nine pages, every one rendered from artifacts">
        <Grid min={250}>
          <Link to="/data" className="card card-hover" style={{ padding: '14px 16px', color: 'var(--text-primary)' }}><div style={{ fontWeight: 600 }}>Data & Preprocessing</div><p className="secondary" style={{ fontSize: 13.5 }}>Four data problems, the metadata discrepancy, and the split protocol.</p></Link>
          <Link to="/eda" className="card card-hover" style={{ padding: '14px 16px', color: 'var(--text-primary)' }}><div style={{ fontWeight: 600 }}>Exploratory Analysis</div><p className="secondary" style={{ fontSize: 13.5 }}>The month inversion, collinear macro features, the small-sample trap.</p></Link>
          {manifest?.models.map((m) => (
            <Link key={m.model_id} to={`/models/${m.model_id}`} className="card card-hover" style={{ padding: '14px 16px', color: 'var(--text-primary)', borderTop: `3px solid ${MODEL_COLORS[m.model_id]}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ fontWeight: 600 }}>{m.display_name ?? MODEL_NAMES[m.model_id]}</span>
                <span className={m.status === 'ok' ? 'chip' : 'chip chip-muted'}>{m.status === 'ok' ? 'built' : 'pending'}</span></div>
              <p className="secondary" style={{ fontSize: 13.5 }}>{m.one_liner ?? 'Not built yet.'}</p>
            </Link>
          ))}
          <Link to="/compare" className="card card-hover" style={{ padding: '14px 16px', color: 'var(--text-primary)' }}><div style={{ fontWeight: 600 }}>Comparison</div><p className="secondary" style={{ fontSize: 13.5 }}>Leaderboard, McNemar tests, the rank-reorder chart and the lift curve.</p></Link>
        </Grid>
      </Section>

      <Section eyebrow="Stated openly" title="Limitations">
        {LIMITS.map(([t, d]) => <Reveal key={t}><Callout tone="warn" title={t}>{d}</Callout></Reveal>)}
      </Section>
    </article>
  )
}
