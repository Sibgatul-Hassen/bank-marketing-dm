import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { useUI } from '../store'
import { cssVar, DATA_COLORS, EASE } from '../theme/palette'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Fig = { data: any[]; layout?: any }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let plotlyPromise: Promise<any> | null = null
const getPlotly = () => (plotlyPromise ??= import('plotly.js-dist-min').then((m) => m.default ?? m))

const DARK_LOW = '#1E2832'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function themeData(data: any[], light: boolean) {
  if (!light) return data
  // Sequential scales in the artifacts start at the dark surface colour; in light mode start at the light one.
  return data.map((t) => (Array.isArray(t.colorscale)
    ? { ...t, colorscale: t.colorscale.map(([s, c]: [number, string]) => [s, c === DARK_LOW ? '#E5EAEF' : c]) }
    : t))
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function themeLayout(layout: any = {}, height?: number) {
  const text = cssVar('--text-primary'), sec = cssVar('--text-secondary'), grid = cssVar('--grid'), line = cssVar('--border-strong')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const axis = (a: any = {}) => ({ gridcolor: grid, linecolor: line, zerolinecolor: line, tickfont: { color: sec, size: 11 }, ...a,
    title: typeof a.title === 'string' ? { text: a.title, font: { color: sec, size: 12 } } : { ...(a.title ?? {}), font: { color: sec, size: 12, ...(a.title?.font ?? {}) } } })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const out: any = {
    ...layout,
    font: { family: 'Inter, system-ui, sans-serif', color: text, size: 12 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    colorway: DATA_COLORS,
    autosize: true,
    height: height ?? layout.height ?? 380,
    margin: { l: 56, r: 24, t: 48, b: 48, ...(layout.margin ?? {}) },
    hoverlabel: { bgcolor: cssVar('--bg-overlay'), bordercolor: line, font: { color: text, family: 'Inter, system-ui, sans-serif', size: 12 } },
    legend: { font: { color: sec, size: 11 }, bgcolor: 'rgba(0,0,0,0)', ...(layout.legend ?? {}) },
    title: layout.title ? { ...(typeof layout.title === 'string' ? { text: layout.title } : layout.title), font: { size: 13, color: sec }, x: 0, xanchor: 'left' } : undefined,
  }
  for (const k of Object.keys(layout)) if (/^[xy]axis\d*$/.test(k)) out[k] = axis(layout[k])
  if (!out.xaxis) out.xaxis = axis()
  if (!out.yaxis) out.yaxis = axis()
  if (layout.polar) out.polar = { ...layout.polar, bgcolor: 'rgba(0,0,0,0)', radialaxis: axis(layout.polar.radialaxis), angularaxis: axis(layout.polar.angularaxis) }
  if (Array.isArray(layout.annotations))
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    out.annotations = layout.annotations.map((a: any) => ({ ...a, font: { color: sec, size: 11, ...(a.font ?? {}), ...(a.font?.color ? {} : { color: sec }) } }))
  return out
}

export function PlotlyFigure({ fig, label, height }: { fig: Fig; label: string; height?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const theme = useUI((s) => s.theme)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); io.disconnect() } }, { rootMargin: '200px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!visible || !ref.current) return
    let alive = true
    const el = ref.current
    getPlotly().then((Plotly) => {
      if (!alive) return
      Plotly.react(el, themeData(fig.data, theme === 'light'), themeLayout(fig.layout, height), { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'] })
    })
    return () => { alive = false }
  }, [visible, fig, theme, height])

  useEffect(() => {
    const el = ref.current
    return () => { if (el) getPlotly().then((P) => P.purge(el)) }
  }, [])

  return (
    <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={visible ? { opacity: 1, scale: 1 } : {}} transition={{ duration: 0.4, ease: EASE }}
      role="img" aria-label={label} style={{ minHeight: height ?? fig.layout?.height ?? 380, width: '100%' }}>
      <div ref={ref} style={{ width: '100%' }} />
    </motion.div>
  )
}
