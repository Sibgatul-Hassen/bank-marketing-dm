import { FigureView } from '../components/FigureView'
import { Callout, fmt, Grid, Loading, MetricTile, Placeholder, Reveal, Section } from '../components/ui'
import { WorkflowDiagram } from '../components/WorkflowDiagram'
import { useArtifact } from '../data/loader'
import type { FigureRef } from '../data/types'

/* eslint-disable @typescript-eslint/no-explicit-any */

const PIPELINE = [
  ['Raw CSV', 'bank-additional-full.csv, 41,188 × 21, separator ";".'],
  ['Verify hash', 'SHA256 of the file (line endings normalised) must match the team value, or loading stops.'],
  ['Drop duration', 'Call length is only known after the call — a target leak. Excluded from every model.'],
  ['Split pdays sentinel', '999 means "never contacted": becomes a was_contacted flag plus real day counts.'],
  ['Handle unknown', "Kept as its own level — it carries signal. Counts reported so you can judge."],
  ['Collapse default', 'Three "yes" rows in 41,188: keep only default_unknown.'],
  ['Encode', 'Per model: one-hot, native categoricals (CatBoost), or embeddings (TabNet).'],
  ['Scale', 'Only where distances or gradients need it — fitted inside each fold.'],
  ['Stratified split', '80/20 holdout, random_state 42; 5-fold CV inside the 80%.'],
  ['Cached artifacts', 'Every result written as JSON under results/ — what this site reads.'],
].map(([label, detail], i) => ({ id: `p${i}`, label, detail, formula_ref: null }))
const PIPE_EDGES = PIPELINE.slice(1).map((n, i) => [PIPELINE[i].id, n.id] as [string, string])

function Figures({ page, figs }: { page: string; figs: FigureRef[] }) {
  return <>{figs.filter((f) => f.page === page).map((f) => <Reveal key={f.file}><FigureView base="dataset" fig={f} /></Reveal>)}</>
}

export function DataPage() {
  const s = useArtifact<any>('dataset/summary.json')
  const q = useArtifact<any>('dataset/quality.json')
  const f = useArtifact<{ figures: FigureRef[] }>('dataset/figures.json')
  if (s.loading || q.loading) return <Loading />
  if (!s.data || !q.data) return <Placeholder what="Dataset artifacts" />
  const S = s.data, Q = q.data
  const [leak, sentinel, unknown, def] = Q.problems
  return (
    <article>
      <Reveal>
        <div className="eyebrow">Foundations</div>
        <h1 style={{ fontSize: 40, marginTop: 4 }}>Data & Preprocessing</h1>
        <p className="secondary" style={{ fontSize: 18, marginTop: 6 }}>Decisions no library can make for you.</p>
      </Reveal>
      <Section>
        <Callout title="Why this page exists even though three models preprocess internally">
          CatBoost, TabNet and TabPFN all encode and normalise on their own. None of them can decide that <span className="mono">duration</span> is unusable,
          that <span className="mono">999</span> is a sentinel, or that <span className="mono">unknown</span> means missing. Those are judgements about what the
          data <em>means</em>, and they change every downstream number.
        </Callout>
      </Section>

      <Section eyebrow="Integrity" title="Is everyone on the same data?">
        <Grid min={170}>
          <MetricTile label="Rows" value={S.rows} format={fmt.int} />
          <MetricTile label="Columns" value={S.columns} format={fmt.int} />
          <MetricTile label="Subscribed" value={S.target.yes_rate} format={fmt.pct2} accent />
          <MetricTile label="Majority baseline" value={S.target.majority_baseline_accuracy} format={fmt.pct2} hint="accuracy of always 'no'" />
        </Grid>
        <div className="card" style={{ padding: '12px 14px' }}>
          <div className="eyebrow">SHA256 (verified on load)</div>
          <div className="mono" style={{ fontSize: 12.5, wordBreak: 'break-all', color: 'var(--accent-text)' }}>{S.sha256}</div>
          <p className="muted" style={{ fontSize: 12, marginTop: 4 }}>{S.sha256_note} {S.source}. {S.citation}</p>
        </div>
      </Section>

      <Section eyebrow="Pipeline" title="From raw CSV to model input">
        <WorkflowDiagram nodes={PIPELINE} edges={PIPE_EDGES} formulas={[]} vertical />
      </Section>

      <Section eyebrow="The four data problems" title="What we found, and what we did">
        {[leak, sentinel, unknown, def].map((p: any, i: number) => (
          <Reveal key={p.id}>
            <div className="card" style={{ padding: '14px 16px' }}>
              <div className="eyebrow">Problem {i + 1}</div>
              <h3 style={{ margin: '2px 0 8px' }}>{p.title}</h3>
              {p.id === 'leak' && <p className="secondary">Correlation with the target <span className="num">{p.evidence.corr_with_target.toFixed(3)}</span> — the highest of any feature. Mean call
                <span className="num"> {p.evidence.mean_no.toFixed(0)}s</span> for "no" vs <span className="num">{p.evidence.mean_yes.toFixed(0)}s</span> for "yes". The minimum for "no" is
                <span className="num"> {p.evidence.min_no}s</span>; for "yes" it is <span className="num">{p.evidence.min_yes}s</span>: a zero-second call is guaranteed to be a "no", and the length is known only after hanging up.
                <br /><span className="muted" style={{ fontSize: 12.5 }}>Correction to our own plan: the next-strongest correlate is not <span className="mono">previous</span> (0.230) but <span className="mono">{Object.keys(p.evidence.next_highest)[0]}</span> ({(Object.values(p.evidence.next_highest)[0] as number).toFixed(3)}) by absolute value; <span className="mono">previous</span> is the next-highest <em>positive</em> one.</span></p>}
              {p.id === 'sentinel' && <p className="secondary"><span className="num">{fmt.int(p.evidence.rows_999)}</span> rows (<span className="num">{fmt.pct2(p.evidence.pct)}</span>) have pdays = 999. Real values run {p.evidence.real_range[0]}–{p.evidence.real_range[1]} days. Treated as a number, 999 tells a model these clients were called 2.7 years ago.</p>}
              {p.id === 'unknown' && (<>
                <div className="scroll-x"><table className="data"><thead><tr><th>Column</th><th className="num">unknown</th><th className="num">%</th></tr></thead>
                  <tbody>{p.evidence.columns.map((c: any) => <tr key={c.column}><td className="mono">{c.column}</td><td className="num">{fmt.int(c.unknown)}</td><td className="num">{fmt.pct2(c.pct)}</td></tr>)}</tbody></table></div>
                <p className="secondary" style={{ marginTop: 8 }}>housing and loan are unknown on the same <span className="num">{p.evidence.housing_and_loan_unknown_together}</span> rows. education=unknown subscribes at <span className="num">{fmt.pct2(p.evidence.education_unknown_rate)}</span> — above several known levels.</p>
              </>)}
              {p.id === 'default' && <p className="secondary">Counts: {Object.entries(p.evidence.counts).map(([k, v]) => `${k} ${fmt.int(v as number)}`).join(' · ')}. A column that is "yes" three times cannot teach anything; whether it is <em>unknown</em> (20.87%) might.</p>}
              <div className="chip" style={{ marginTop: 8 }}>Action</div> <span style={{ fontSize: 14 }}>{p.action}</span>
            </div>
          </Reveal>
        ))}
        <Callout tone="warn" title="The metadata discrepancy">{Q.metadata_discrepancy}</Callout>
        <Callout title={`${Q.duplicates.rows} duplicate rows`}>{Q.duplicates.note}</Callout>
      </Section>

      <Section eyebrow="Encoding" title="Why the encoding differs per model">
        <div className="card scroll-x" style={{ padding: 4 }}>
          <table className="data"><thead><tr><th>Model</th><th>Encoding</th></tr></thead>
            <tbody>{Object.entries(Q.encodings).map(([k, v]) => <tr key={k}><td className="mono">{k}</td><td className="secondary">{v as string}</td></tr>)}</tbody></table>
        </div>
      </Section>

      <Section eyebrow="Protocol" title="The split, stated explicitly">
        <Callout title="No information from the test set, ever">{Q.split_protocol}</Callout>
        {f.data && <Figures page="data" figs={f.data.figures} />}
      </Section>
    </article>
  )
}

const EDA_FLOW = ['Load', 'Profile types', 'Univariate distributions', 'Bivariate vs target', 'Correlation', 'Imbalance']
  .map((label, i) => ({ id: `e${i}`, label, detail: label, formula_ref: null }))

export function EDAPage() {
  const s = useArtifact<any>('dataset/summary.json')
  const f = useArtifact<{ figures: FigureRef[] }>('dataset/figures.json')
  if (s.loading) return <Loading />
  if (!s.data) return <Placeholder what="Dataset artifacts" />
  const S = s.data
  const trap = S.small_sample_trap
  const figs = f.data?.figures ?? []
  const fig = (file: string) => figs.find((x) => x.file === file)
  const show = (file: string) => { const x = fig(file); return x ? <Reveal key={file}><FigureView base="dataset" fig={x} /></Reveal> : null }
  const po = S.rates.poutcome, ct = S.rates.contact
  return (
    <article>
      <Reveal>
        <div className="eyebrow">Foundations</div>
        <h1 style={{ fontSize: 40, marginTop: 4 }}>Exploratory Analysis</h1>
        <p className="secondary" style={{ fontSize: 18, marginTop: 6 }}>Every number below is recomputed from the raw file — none is copied from the plan.</p>
      </Reveal>
      <Section eyebrow="Workflow" title="How we explored">
        <WorkflowDiagram nodes={EDA_FLOW} edges={EDA_FLOW.slice(1).map((n, i) => [EDA_FLOW[i].id, n.id] as [string, string])} formulas={[]} />
      </Section>

      <Section eyebrow="1" title="The strongest legitimate predictor: previous outcome"
        intro={<>{po.map((r: any) => `${r.level} ${fmt.pct2(r.rate)} (n=${fmt.int(r.n)})`).join(' · ')}</>}>
        {show('rate_poutcome.json')}
      </Section>
      <Section eyebrow="2" title="Contact channel" intro={<>{ct.map((r: any) => `${r.level} ${fmt.pct2(r.rate)} (n=${fmt.int(r.n)})`).join(' vs ')} — nearly 3×.</>}>
        {show('rate_contact.json')}
      </Section>
      <Section eyebrow="3" title="The month volume–rate inversion">
        {show('month_dual.json')}
        <Callout tone="warn">Do not read this causally. Months with few calls and high rates may reflect who the bank chose to call, not when clients are receptive.</Callout>
      </Section>
      <Section eyebrow="4" title="Three columns, one signal">
        {show('correlation.json')}
        <p className="secondary">emp.var.rate ↔ euribor3m r = <span className="num">{S.macro_correlation['emp.var.rate ~ euribor3m'].toFixed(3)}</span>, euribor3m ↔ nr.employed r =
          <span className="num"> {S.macro_correlation['euribor3m ~ nr.employed'].toFixed(3)}</span>. Consequences: UMAP collapses them into one "economy" axis, an independence-assuming model counts the
          same evidence three times, and any covariance inversion is ill-conditioned.</p>
      </Section>
      <Section eyebrow="5 ★" title="The small-sample trap">
        <div className="card" style={{ padding: '16px 18px', borderColor: 'var(--warning)' }}>
          <p><span className="mono">{trap.level}</span> shows a <strong className="num">{fmt.pct2(trap.rate)}</strong> subscription rate — double the average. But it comes from
            <strong className="num"> {trap.n}</strong> rows ({trap.yes} subscribers).</p>
          <p className="secondary" style={{ marginTop: 8 }}>One extra subscriber would move it to <span className="num">{fmt.pct2(trap.rate_with_one_more_yes)}</span> —
            each row is worth <span className="num">{trap.one_row_moves_points.toFixed(1)}</span> percentage points. The 95% Wilson interval is
            <span className="num"> {fmt.pct(trap.wilson_95[0])} – {fmt.pct(trap.wilson_95[1])}</span>: compatible with "below average" and "four times average".</p>
          <p className="secondary" style={{ marginTop: 8 }}><strong>Rule:</strong> never report a rate without its support. That is why every rate chart on this site prints <span className="mono">n</span>.</p>
        </div>
        {show('rate_education.json')}
      </Section>
      <Section eyebrow="6" title="A feature that does nothing" intro="day_of_week spans only 9.95% – 12.12%. Whether models agree is checked in each SHAP panel — and the answer is not uniform: see the decision tree's findings.">
        {show('rate_day_of_week.json')}
      </Section>
      <Section eyebrow="More" title="Occupation and numeric features">
        {show('rate_job.json')}
        {show('violins.json')}
      </Section>
    </article>
  )
}
