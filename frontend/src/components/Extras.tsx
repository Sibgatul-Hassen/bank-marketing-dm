import { useArtifact } from '../data/loader'
import type { Extra } from '../data/types'
import { DataTable } from './FigureView'
import { Tex } from './Formula'
import { fmt, Loading, Placeholder } from './ui'

/* eslint-disable @typescript-eslint/no-explicit-any */

function HandVerification({ d }: { d: any }) {
  const ok = d.checks.every((c: any) => Math.abs(c.hand - c.code) < 1e-9)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <p className="secondary">
        The course teaches these calculations by hand. Here they are, step by step, for <strong>{d.feature}</strong> at the root ({d.data}) —
        then checked against library output. {ok ? <span style={{ color: 'var(--accent-text)' }}>Every check matches to 1e-9.</span> : <span style={{ color: 'var(--negative)' }}>A check failed.</span>}
      </p>
      <div className="scroll-x">
        <table className="data">
          <thead><tr><th>Branch</th><th className="num">rows</th><th className="num">yes</th><th className="num">no</th><th className="num">rate</th></tr></thead>
          <tbody>
            <tr><td><strong>root S</strong></td><td className="num">{fmt.int(d.root.n)}</td><td className="num">{fmt.int(d.root.yes)}</td><td className="num">{fmt.int(d.root.no)}</td><td className="num">{fmt.pct(d.root.yes / d.root.n)}</td></tr>
            {d.levels.map((l: any) => (
              <tr key={l.level}><td>{d.feature} = {l.level}</td><td className="num">{fmt.int(l.n)}</td><td className="num">{fmt.int(l.yes)}</td><td className="num">{fmt.int(l.no)}</td><td className="num">{fmt.pct(l.yes / l.n)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(380px, 100%), 1fr))', gap: 14 }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>By hand</div>
          <ol style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
            {d.steps.map((s: any, i: number) => (
              <li key={i}>
                <div><strong>{s.label}</strong> <span className="chip mono" style={{ fontSize: 10 }}>{s.formula_ref}</span></div>
                <div className="mono secondary" style={{ fontSize: 11.5, wordBreak: 'break-word' }}>{s.expr}</div>
                <div className="mono" style={{ color: 'var(--accent-text)' }}>= {s.value.toFixed(6)}</div>
              </li>
            ))}
          </ol>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>Checked against code</div>
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Quantity</th><th className="num">hand</th><th className="num">code</th><th>source</th></tr></thead>
              <tbody>{d.checks.map((c: any) => (
                <tr key={c.what}><td>{c.what}</td><td className="num">{c.hand.toFixed(6)}</td><td className="num">{c.code.toFixed(6)}</td>
                  <td className="muted" style={{ fontSize: 12 }}>{Math.abs(c.hand - c.code) < 1e-9 ? '✓ ' : '✗ '}{c.source}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>{d.note}</p>
        </div>
      </div>
      <div>
        <div className="eyebrow" style={{ marginBottom: 6 }}>Every categorical feature at the root — why gain ratio exists</div>
        <DataTable spec={{ columns: ['feature', 'levels', 'info_gain', 'split_info', 'gain_ratio', 'gini_gain'], rows: d.all_features }} />
        <p className="secondary" style={{ fontSize: 13, marginTop: 8 }}>
          ID3 (information gain) picks <strong>{d.c45.id3_pick}</strong>; CART (Gini gain) picks <strong>{d.c45.gini_pick}</strong>. C4.5 first keeps features with
          at least the average gain ({d.c45.average_info_gain.toFixed(4)} → {d.c45.survivors.join(', ')}), then takes the best ratio: <strong>{d.c45.pick}</strong>.
          Note how high-cardinality <span className="mono">job</span> and <span className="mono">month</span> score well on gain but are penalised by split information.
        </p>
      </div>
    </div>
  )
}

function RuleCard({ r, tone, title }: { r: any; tone: string; title: string }) {
  return (
    <div className="card" style={{ padding: '14px 16px', borderTop: `3px solid ${tone}`, flex: 1, minWidth: 240 }}>
      <div className="eyebrow" style={{ color: tone }}>{title}</div>
      <div className="mono" style={{ fontSize: 13, margin: '6px 0 10px' }}>{r.antecedent.join(' + ')} ⇒ {r.consequent.join(' + ')}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6, fontSize: 13 }}>
        <span className="secondary">confidence</span><span className="num" style={{ fontSize: 18 }}>{fmt.pct(r.confidence)}</span>
        <span className="secondary">lift</span><span className="num" style={{ fontSize: 18, color: tone }}>{r.lift.toFixed(3)}</span>
        <span className="secondary">support</span><span className="num">{fmt.pct2(r.support)}</span>
        <span className="secondary">coverage</span><span className="num">{fmt.pct2(r.coverage)}</span>
      </div>
    </div>
  )
}

function ConfidenceTrap({ d }: { d: any }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <RuleCard r={d.trap} tone="var(--negative)" title="High confidence, useless" />
        <RuleCard r={d.useful} tone="var(--accent)" title="Lower confidence, useful" />
      </div>
      <p className="secondary">{d.explanation}</p>
      <div className="card" style={{ padding: '10px 14px' }}>
        <Tex display tex={String.raw`\text{lift} = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)} \quad\Rightarrow\quad \frac{${(d.trap.confidence).toFixed(3)}}{${d.base_rate_no.toFixed(3)}} = ${d.trap.lift.toFixed(3)} \qquad \frac{${d.useful.confidence.toFixed(3)}}{${d.base_rate_yes.toFixed(3)}} = ${d.useful.lift.toFixed(2)}`} />
      </div>
    </div>
  )
}

export function ExtraView({ base, extra }: { base: string; extra: Extra }) {
  const { data, loading, error } = useArtifact<any>(`${base}/${extra.file}`)
  if (loading) return <Loading />
  if (error || !data) return <Placeholder what={extra.title} />
  if (extra.kind === 'hand_verification') return <HandVerification d={data} />
  if (extra.kind === 'confidence_trap') return <ConfidenceTrap d={data} />
  return <Placeholder what={extra.title} detail={`No renderer for ${extra.kind}`} />
}
