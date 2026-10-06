import { AnimatePresence, motion, useInView, useReducedMotion } from 'framer-motion'
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Formula, WorkflowNode } from '../data/types'
import { EASE } from '../theme/palette'
import { Tex } from './Formula'

interface Props { nodes: WorkflowNode[]; edges: [string, string][]; formulas: Formula[]; vertical?: boolean }

const NODE_H = 76
const GAP = 22

/** Generic workflow diagram driven entirely by meta.json. Forward edges are straight; an edge to an
 *  earlier node (a loop: boosting rounds, TabNet steps, tree recursion) is drawn as a curved return path. */
export function WorkflowDiagram({ nodes, edges, formulas, vertical: forceVertical }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)
  const [active, setActive] = useState<string | null>(null)
  const inView = useInView(wrap, { once: true, margin: '-40px' })
  const reduce = useReducedMotion()

  useLayoutEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const vertical = forceVertical || width < 640
  const idx = useMemo(() => Object.fromEntries(nodes.map((n, i) => [n.id, i])), [nodes])
  const n = nodes.length
  const nodeW = vertical ? Math.max(160, Math.min(width - 60, 420)) : Math.max(92, (width - GAP * (n - 1)) / n)
  const loops = edges.filter(([a, b]) => idx[b] <= idx[a])
  const loopPad = loops.length ? 46 : 8
  const pos = (i: number) => (vertical ? { x: 0, y: i * (NODE_H + GAP) } : { x: i * (nodeW + GAP), y: 0 })
  const H = vertical ? n * (NODE_H + GAP) - GAP : NODE_H + loopPad
  const W = Math.max(0, vertical ? nodeW + loopPad : width)

  const paths = edges.map(([a, b], k) => {
    const i = idx[a], j = idx[b]
    const pa = pos(i), pb = pos(j)
    let d: string
    let loop = false
    if (j <= i) {
      loop = true
      if (vertical) {
        const x = nodeW, ya = pa.y + NODE_H / 2, yb = pb.y + NODE_H / 2, bulge = 30 + 8 * (k % 2)
        d = `M ${x} ${ya} C ${x + bulge} ${ya}, ${x + bulge} ${yb}, ${x + 6} ${yb}`
      } else {
        const y = NODE_H, xa = pa.x + nodeW / 2, xb = pb.x + nodeW / 2, bulge = loopPad - 6
        d = `M ${xa} ${y} C ${xa} ${y + bulge}, ${xb} ${y + bulge}, ${xb} ${y + 6}`
      }
    } else if (vertical) {
      d = `M ${nodeW / 2} ${pa.y + NODE_H} L ${nodeW / 2} ${pb.y - 4}`
    } else {
      const y = NODE_H / 2
      d = j === i + 1 ? `M ${pa.x + nodeW} ${y} L ${pb.x - 4} ${y}` : `M ${pa.x + nodeW / 2} 0 C ${pa.x + nodeW / 2} -26, ${pb.x + nodeW / 2} -26, ${pb.x + nodeW / 2} -4`
    }
    return { d, loop, key: `${a}-${b}`, order: Math.max(i, j) }
  })

  const activeNode = nodes.find((x) => x.id === active)
  const refs = (activeNode?.formula_ref ?? '').split(',').map((s) => s.trim()).filter(Boolean)

  return (
    <div>
      <div ref={wrap} style={{ position: 'relative', width: '100%', height: H + 8, marginTop: vertical ? 0 : 8 }}>
        <svg width={W} height={H + 8} style={{ position: 'absolute', inset: 0, overflow: 'visible' }} aria-hidden="true">
          <defs>
            <marker id="wf-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
            </marker>
            <marker id="wf-arrow-loop" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--amber)" />
            </marker>
          </defs>
          {paths.map((p) => (
            <motion.path key={p.key} d={p.d} fill="none" stroke={p.loop ? 'var(--amber)' : 'var(--text-muted)'} strokeWidth={1.5}
              strokeDasharray={p.loop ? '5 4' : undefined} markerEnd={`url(#${p.loop ? 'wf-arrow-loop' : 'wf-arrow'})`}
              initial={{ pathLength: reduce ? 1 : 0, opacity: reduce ? 1 : 0 }}
              animate={inView ? { pathLength: 1, opacity: 1 } : {}}
              transition={{ duration: 0.35, delay: reduce ? 0 : n * 0.18 + p.order * 0.05, ease: EASE }} />
          ))}
          {loops.length > 0 && !vertical && (
            <text x={W / 2} y={H + 4} textAnchor="middle" fontSize="11" fill="var(--amber)">↺ repeat</text>
          )}
        </svg>
        {nodes.map((node, i) => {
          const p = pos(i)
          const on = active === node.id
          return (
            <motion.button key={node.id} type="button" onClick={() => setActive(on ? null : node.id)} aria-expanded={on}
              aria-label={`Step ${i + 1}: ${node.label}`}
              initial={{ opacity: reduce ? 1 : 0, y: reduce ? 0 : 8 }} animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.3, delay: reduce ? 0 : i * 0.18, ease: EASE }}
              style={{ position: 'absolute', left: p.x, top: p.y, width: nodeW, height: NODE_H, borderRadius: 12, cursor: 'pointer', textAlign: 'left',
                padding: '8px 10px', font: 'inherit', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 2,
                background: on ? 'var(--accent-subtle)' : 'var(--bg-elevated)', color: 'var(--text-primary)',
                border: `1px solid ${on ? 'var(--accent)' : 'var(--border-strong)'}`, transition: 'border-color 160ms, background-color 160ms' }}>
              <span className="mono" style={{ fontSize: 10, color: on ? 'var(--accent-text)' : 'var(--text-muted)' }}>{String(i + 1).padStart(2, '0')}{node.formula_ref ? ` · ${node.formula_ref}` : ''}</span>
              <span style={{ fontSize: nodeW < 120 ? 11.5 : 12.5, lineHeight: 1.25, fontWeight: 500 }}>{node.label}</span>
            </motion.button>
          )
        })}
      </div>
      <AnimatePresence initial={false}>
        {activeNode && (
          <motion.div key={activeNode.id} initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.28, ease: EASE }} style={{ overflow: 'hidden' }}>
            <div className="card" style={{ padding: '14px 16px', marginTop: 14, borderColor: 'var(--accent)' }}>
              <div style={{ fontWeight: 600 }}>{activeNode.label}</div>
              <p className="secondary" style={{ marginTop: 4 }}>{activeNode.detail}</p>
              {refs.map((r) => {
                const f = formulas.find((x) => x.id === r)
                return f ? (
                  <div key={r} style={{ marginTop: 8 }}>
                    <span className="chip mono">{f.id}</span> <span className="muted" style={{ fontSize: 13 }}>{f.name}</span>
                    <Tex tex={f.latex} display />
                  </div>
                ) : null
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>Click a step to see what it does and its formula. Dashed amber arrows are loops.</p>
    </div>
  )
}
