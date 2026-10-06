import { lazy, Suspense, useMemo, useState } from 'react'
import { useArtifact } from '../data/loader'
import type { FigureRef } from '../data/types'
import { PlotlyFigure } from './PlotlyFigure'
import type { Point3, TreeNode } from './ThreeViews'
import { fmt, Loading, Placeholder } from './ui'

const PointCloud3D = lazy(() => import('./ThreeViews').then((m) => ({ default: m.PointCloud3D })))
const Tree3D = lazy(() => import('./ThreeViews').then((m) => ({ default: m.Tree3D })))
const MaskBars3D = lazy(() => import('./ThreeViews').then((m) => ({ default: m.MaskBars3D })))

/* eslint-disable @typescript-eslint/no-explicit-any */

export function DataTable({ spec }: { spec: { columns: string[]; rows: Record<string, any>[]; format?: Record<string, string> } }) {
  const f = (col: string, v: any) => {
    if (v == null || (typeof v === 'number' && !isFinite(v))) return '—'
    const kind = spec.format?.[col]
    if (typeof v !== 'number') return String(v)
    if (kind === 'pct') return fmt.pct(v)
    if (kind === 'int') return fmt.int(v)
    return fmt.num(v)
  }
  return (
    <div className="scroll-x">
      <table className="data">
        <thead><tr>{spec.columns.map((c) => <th key={c} className={typeof spec.rows[0]?.[c] === 'number' ? 'num' : ''}>{c.replace(/_/g, ' ')}</th>)}</tr></thead>
        <tbody>
          {spec.rows.map((r, i) => (
            <tr key={i}>{spec.columns.map((c) => <td key={c} className={typeof r[c] === 'number' ? 'num' : ''}>{f(c, r[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Tree2D({ nodes }: { nodes: TreeNode[] }) {
  const { pos, w, h } = useMemo(() => {
    const kids: Record<number, TreeNode[]> = {}
    nodes.forEach((n) => { if (n.parent != null) (kids[n.parent] ??= []).push(n) })
    const p: Record<number, { x: number; y: number }> = {}
    let leaf = 0
    const walk = (n: TreeNode): number => {
      const c = kids[n.id] ?? []
      const x = c.length ? c.map(walk).reduce((a, b) => a + b, 0) / c.length : leaf++
      p[n.id] = { x, y: n.depth }
      return x
    }
    const root = nodes.find((n) => n.parent == null)
    if (root) walk(root)
    return { pos: p, w: Math.max(leaf, 1), h: Math.max(...nodes.map((n) => n.depth)) + 1 }
  }, [nodes])
  const CW = 118, RH = 112, NW = 108, NH = 64
  const W = w * CW, H = h * RH
  return (
    <div className="scroll-x">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fitted decision tree, top three levels" style={{ display: 'block', margin: '0 auto' }}>
        {nodes.map((n) => {
          if (n.parent == null) return null
          const a = pos[n.parent], b = pos[n.id]
          const x1 = a.x * CW + CW / 2, y1 = a.y * RH + NH + 8, x2 = b.x * CW + CW / 2, y2 = b.y * RH + 8
          return (
            <g key={`e${n.id}`}>
              <path d={`M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} />
              <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 2} textAnchor="middle" fontSize="9.5" fill="var(--text-muted)">{(n.edge ?? '').slice(0, 24)}</text>
            </g>
          )
        })}
        {nodes.map((n) => {
          const p = pos[n.id]
          const x = p.x * CW + (CW - NW) / 2, y = p.y * RH + 8
          const end = n.leaf || n.truncated
          return (
            <g key={n.id}>
              <rect x={x} y={y} width={NW} height={NH} rx={9} fill="var(--bg-elevated)" stroke={end ? 'var(--border-strong)' : 'var(--accent)'} strokeWidth={1} />
              <rect x={x} y={y + NH - 5} width={NW * Math.min(1, n.rate)} height={5} rx={2} fill="var(--class-yes)" />
              <text x={x + NW / 2} y={y + 17} textAnchor="middle" fontSize="10.5" fontWeight={600} fill="var(--text-primary)">
                {end ? (n.truncated ? '… continues' : 'leaf') : (n.split ?? '').slice(0, 20)}
              </text>
              <text x={x + NW / 2} y={y + 33} textAnchor="middle" fontSize="10" fill="var(--text-secondary)" className="num">{n.samples.toLocaleString()} clients</text>
              <text x={x + NW / 2} y={y + 47} textAnchor="middle" fontSize="10" fill="var(--accent-text)" className="num">{(n.rate * 100).toFixed(1)}% yes</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function ThreeFallback({ height }: { height: number }) {
  return <div style={{ height }}><Loading /></div>
}

function SpecView({ spec, fig, base }: { spec: any; fig: FigureRef; base: string }) {
  const [three, setThree] = useState(false)
  const [slow, setSlow] = useState(false)
  const type = spec.type ?? fig.type
  if (type === 'table') return <DataTable spec={spec} />
  if (type === 'tree') {
    return (
      <div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          <button className={`btn ${!three ? 'active' : ''}`} onClick={() => setThree(false)}>2D</button>
          <button className={`btn ${three && !slow ? 'active' : ''}`} onClick={() => { setThree(true); setSlow(false) }}>3D</button>
          <span className="muted" style={{ fontSize: 12, alignSelf: 'center' }}>{three && !slow ? 'Click a node to focus it; drag to orbit.' : 'Bar under each node = subscription rate.'}</span>
        </div>
        {three && !slow
          ? <Suspense fallback={<ThreeFallback height={420} />}><Tree3D nodes={spec.nodes} onFallback={() => setSlow(true)} /></Suspense>
          : <Tree2D nodes={spec.nodes} />}
        {slow && <p className="muted" style={{ fontSize: 12 }}>3D ran below 30 fps on this device; showing the 2D version.</p>}
      </div>
    )
  }
  if (type === 'points3d') {
    if (slow) {
      const pts: Point3[] = spec.points
      const mk = (yv: number, name: string, color: string) => {
        const s = pts.filter((p) => p.y_true === yv)
        return { type: 'scattergl', mode: 'markers', name, x: s.map((p) => p.x), y: s.map((p) => p.y), marker: { size: 3, color, opacity: 0.7 } }
      }
      return (<>
        <PlotlyFigure fig={{ data: [mk(0, 'not subscribed', '#6B7E8F'), mk(1, 'subscribed', '#4FB3A0')], layout: { title: '3D projection, first two axes (2D fallback)' } }} label="2D fallback of the 3D UMAP point cloud" />
        <p className="muted" style={{ fontSize: 12 }}>3D ran below 30 fps on this device; showing a 2D view instead.</p>
      </>)
    }
    return <Suspense fallback={<ThreeFallback height={460} />}><PointCloud3D points={spec.points} clusters={spec.clusters} onFallback={() => setSlow(true)} /></Suspense>
  }
  if (type === 'mask3d') {
    if (slow) return <p className="muted">3D ran below 30 fps; see the mask heatmap above for the same data.</p>
    return <Suspense fallback={<ThreeFallback height={420} />}><MaskBars3D steps={spec.steps} features={spec.features} values={spec.values} onFallback={() => setSlow(true)} /></Suspense>
  }
  if (spec.data) return <PlotlyFigure fig={spec} label={`${fig.title}. ${fig.caption}`} />
  return <Placeholder what={fig.title} detail={`Unknown figure type in ${base}/${fig.file}`} />
}

export function FigureView({ base, fig }: { base: string; fig: FigureRef }) {
  const { data, error, loading } = useArtifact<any>(`${base}/figures/${fig.file}`)
  return (
    <figure className="card" style={{ margin: 0, padding: '14px 16px' }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{fig.title}</div>
      {loading ? <Loading /> : error || !data ? <Placeholder what={fig.title} /> : <SpecView spec={data} fig={fig} base={base} />}
      <figcaption className="secondary" style={{ fontSize: 13.5, marginTop: 10 }}>{fig.caption}</figcaption>
    </figure>
  )
}
