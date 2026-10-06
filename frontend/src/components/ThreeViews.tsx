import { Line, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { useReducedMotion } from 'framer-motion'
import { useEffect, useMemo, useRef, useState, type CSSProperties, type MutableRefObject } from 'react'
import * as THREE from 'three'
import { useUI } from '../store'
import { cssVar } from '../theme/palette'

// ---------------------------------------------------------------- shared

function FpsGuard({ onSlow }: { onSlow: () => void }) {
  const frames = useRef<number[]>([])
  const fired = useRef(false)
  useFrame(() => {
    const now = performance.now()
    frames.current.push(now)
    while (frames.current.length && now - frames.current[0] > 2000) frames.current.shift()
    const elapsed = now - (frames.current[0] ?? now)
    if (!fired.current && elapsed > 1900 && frames.current.length / (elapsed / 1000) < 30) {
      fired.current = true
      setTimeout(onSlow, 0) // leave the render loop before React unmounts the canvas
    }
  })
  return null
}

/** Labels as plain DOM in an overlay, positioned each frame by projecting 3D anchors.
 *  Avoids drei <Html>, which mounts one React root per label and races on unmount under React 19. */
type LabelEls = MutableRefObject<(HTMLDivElement | null)[]>

function ProjectedLabels({ anchors, els }: { anchors: [number, number, number][]; els: LabelEls }) {
  const v = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera, size }) => {
    anchors.forEach((a, i) => {
      const el = els.current[i]
      if (!el) return
      v.set(a[0], a[1], a[2]).project(camera)
      const x = ((v.x + 1) / 2) * size.width, y = ((1 - v.y) / 2) * size.height
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`
      el.style.opacity = v.z < 1 && v.z > -1 ? '1' : '0'
    })
  })
  return null
}

function LabelLayer({ texts, els, style }: { texts: string[]; els: LabelEls; style?: CSSProperties }) {
  return (
    <div aria-hidden="true" style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
      {texts.map((t, i) => (
        <div key={i} ref={(el) => { els.current[i] = el }} style={{ position: 'absolute', left: 0, top: 0, whiteSpace: 'nowrap', fontSize: 11, fontFamily: 'Inter, sans-serif', opacity: 0, ...style }}>{t}</div>
      ))}
    </div>
  )
}

function useBg() {
  const theme = useUI((s) => s.theme)
  return useMemo(() => ({ theme, muted: cssVar('--text-muted') || '#7F92A3', text: cssVar('--text-primary') || '#E6EDF3' }), [theme])
}

// ---------------------------------------------------------------- UMAP point cloud

export interface Point3 { x: number; y: number; z: number; cluster: number; y_true: number; membership: number; age: number; job: string; month: string; poutcome: string }

function Cloud({ points, colorBy, clusterColors, onHover }: { points: Point3[]; colorBy: 'class' | 'cluster'; clusterColors: Record<number, string>; onHover: (i: number | null) => void }) {
  const ref = useRef<THREE.Points>(null)
  const matRef = useRef<THREE.PointsMaterial>(null)
  const reduce = useReducedMotion()
  const theme = useUI((s) => s.theme)
  const { positions, center, scale } = useMemo(() => {
    const p = new Float32Array(points.length * 3)
    const c = new THREE.Vector3()
    points.forEach((q, i) => { p.set([q.x, q.y, q.z], i * 3); c.add(new THREE.Vector3(q.x, q.y, q.z)) })
    c.divideScalar(points.length || 1)
    let r = 0
    points.forEach((q) => { r = Math.max(r, Math.hypot(q.x - c.x, q.y - c.y, q.z - c.z)) })
    return { positions: p, center: c, scale: 4 / (r || 1) }
  }, [points])
  const colors = useMemo(() => {
    const col = new Float32Array(points.length * 3)
    const yes = new THREE.Color('#4FB3A0'), no = new THREE.Color(theme === 'light' ? '#9AA8B5' : '#56687A')
    points.forEach((q, i) => {
      const c = colorBy === 'class' ? (q.y_true ? yes : no) : new THREE.Color(clusterColors[q.cluster] ?? '#3A4652')
      col.set([c.r, c.g, c.b], i * 3)
    })
    return col
  }, [points, colorBy, clusterColors, theme])
  const start = useRef(performance.now())
  useFrame(() => {
    if (matRef.current) matRef.current.opacity = reduce ? 0.9 : Math.min(0.9, ((performance.now() - start.current) / 1200) * 0.9)
  })
  return (
    <group scale={scale} position={[-center.x * scale, -center.y * scale, -center.z * scale]}>
      <points ref={ref} onPointerMove={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover(e.index ?? null) }} onPointerOut={() => onHover(null)}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial ref={matRef} size={0.07} vertexColors transparent opacity={0} sizeAttenuation depthWrite={false} />
      </points>
    </group>
  )
}

function RaycastThreshold() {
  const { raycaster } = useThree()
  useEffect(() => { raycaster.params.Points = { threshold: 0.03 } }, [raycaster])
  return null
}

export function PointCloud3D({ points, clusters, height = 460, onFallback }: {
  points: Point3[]; clusters: { id: number; color: string; subscribe_rate: number; profile: string }[]; height?: number; onFallback: () => void
}) {
  const [colorBy, setColorBy] = useState<'class' | 'cluster'>('class')
  const [hover, setHover] = useState<number | null>(null)
  const reduce = useReducedMotion()
  const clusterColors = useMemo(() => Object.fromEntries(clusters.map((c) => [c.id, c.color])), [clusters])
  const p = hover != null ? points[hover] : null
  const cl = p ? clusters.find((c) => c.id === p.cluster) : null
  return (
    <div style={{ position: 'relative' }}>
      <div style={{ position: 'absolute', zIndex: 2, top: 10, left: 10, display: 'flex', gap: 6 }} role="group" aria-label="Colour points by">
        <button className={`btn ${colorBy === 'class' ? 'active' : ''}`} onClick={() => setColorBy('class')} aria-pressed={colorBy === 'class'}>By subscription</button>
        <button className={`btn ${colorBy === 'cluster' ? 'active' : ''}`} onClick={() => setColorBy('cluster')} aria-pressed={colorBy === 'cluster'}>By cluster</button>
      </div>
      <div style={{ height, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', background: 'radial-gradient(ellipse at center, var(--bg-elevated), var(--bg-base))' }}
        role="img" aria-label={`3D UMAP projection of ${points.length.toLocaleString()} clients, coloured by ${colorBy === 'class' ? 'subscription' : 'cluster'}`}>
        <Canvas camera={{ position: [0, 0, 9], fov: 45 }} dpr={[1, 2]} frameloop="always">
          <RaycastThreshold />
          <Cloud points={points} colorBy={colorBy} clusterColors={clusterColors} onHover={setHover} />
          <OrbitControls enablePan={false} autoRotate={!reduce && hover == null} autoRotateSpeed={0.5} minDistance={3} maxDistance={18} />
          <FpsGuard onSlow={onFallback} />
        </Canvas>
      </div>
      {p && (
        <div className="card" style={{ position: 'absolute', right: 10, top: 10, padding: '10px 12px', fontSize: 12, maxWidth: 260, pointerEvents: 'none', background: 'var(--bg-overlay)' }}>
          <div style={{ fontWeight: 600, color: p.y_true ? 'var(--accent-text)' : 'var(--text-secondary)' }}>{p.y_true ? 'Subscribed' : 'Did not subscribe'}</div>
          <div className="secondary">Age {p.age} · {p.job} · called in {p.month}</div>
          <div className="secondary">Previous campaign: {p.poutcome}</div>
          <div className="muted" style={{ marginTop: 4 }}>{p.cluster < 0 ? 'Noise — no cluster' : `Cluster ${p.cluster} (${((cl?.subscribe_rate ?? 0) * 100).toFixed(1)}% subscribe) · membership ${p.membership.toFixed(2)}`}</div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 14, fontSize: 12, marginTop: 8 }} className="secondary">
        {colorBy === 'class' ? (<>
          <span><span style={{ color: '#4FB3A0' }}>●</span> subscribed</span><span><span style={{ color: '#6B7E8F' }}>●</span> not subscribed</span>
        </>) : <span>Colours = HDBSCAN clusters; dark grey = noise.</span>}
        <span className="muted">Drag to orbit · scroll to zoom · hover for a client card</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- decision tree in 3D

export interface TreeNode { id: number; parent: number | null; edge: string | null; depth: number; samples: number; yes: number; rate: number; leaf: boolean; truncated: boolean; split?: string }

function layoutTree(nodes: TreeNode[]) {
  const kids: Record<number, TreeNode[]> = {}
  nodes.forEach((n) => { if (n.parent != null) (kids[n.parent] ??= []).push(n) })
  const pos: Record<number, [number, number, number]> = {}
  let leaf = 0
  const walk = (n: TreeNode): number => {
    const c = kids[n.id] ?? []
    const x = c.length ? c.map(walk).reduce((a, b) => a + b, 0) / c.length : leaf++
    pos[n.id] = [x, -n.depth * 1.6, -n.depth * 1.2]
    return x
  }
  const root = nodes.find((n) => n.parent == null)
  if (root) walk(root)
  const mid = (leaf - 1) / 2
  Object.values(pos).forEach((p) => { p[0] = (p[0] - mid) * 1.25 })
  return { pos, kids }
}

function FocusCamera({ target }: { target: THREE.Vector3 }) {
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null
  useFrame(() => {
    if (!controls) return
    controls.target.lerp(target, 0.08)
    controls.update()
  })
  return null
}

export function Tree3D({ nodes, height = 420, onFallback }: { nodes: TreeNode[]; height?: number; onFallback: () => void }) {
  const { pos } = useMemo(() => layoutTree(nodes), [nodes])
  const [sel, setSel] = useState<number | null>(null)
  const bg = useBg()
  const els = useRef<(HTMLDivElement | null)[]>([])
  const target = useMemo(() => new THREE.Vector3(...(sel != null ? pos[sel] : [0, -2.4, -1.8])), [sel, pos])
  const maxN = Math.max(...nodes.map((n) => n.samples))
  const radius = (n: TreeNode) => 0.12 + 0.3 * Math.sqrt(n.samples / maxN)
  const anchors = nodes.map((n) => [pos[n.id][0], pos[n.id][1] - radius(n) - 0.3, pos[n.id][2]] as [number, number, number])
  const texts = nodes.map((n) => (n.depth > 1 && n.id !== sel ? '' : n.leaf || n.truncated ? `${(n.rate * 100).toFixed(0)}% yes · ${n.samples.toLocaleString()}` : n.split ?? ''))
  const s = sel != null ? nodes.find((n) => n.id === sel) : null
  return (
    <div style={{ position: 'relative', height, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }} role="img" aria-label="Decision tree drawn in 3D, one plane per depth">
      <Canvas camera={{ position: [0, 1, 11], fov: 45 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} />
        {nodes.map((n) => n.parent != null && (
          <Line key={`e${n.id}`} points={[pos[n.parent], pos[n.id]]} color={bg.muted} lineWidth={1} transparent opacity={0.7} />
        ))}
        {nodes.map((n) => {
          const c = new THREE.Color('#6B7E8F').lerp(new THREE.Color('#4FB3A0'), Math.min(1, n.rate / 0.6))
          return (
            <mesh key={n.id} position={pos[n.id]} onClick={(e) => { e.stopPropagation(); setSel(sel === n.id ? null : n.id) }}>
              <sphereGeometry args={[radius(n), 24, 24]} />
              <meshStandardMaterial color={c} emissive={sel === n.id ? '#4FB3A0' : '#000000'} emissiveIntensity={0.4} />
            </mesh>
          )
        })}
        <ProjectedLabels anchors={anchors} els={els} />
        <OrbitControls makeDefault enablePan={false} minDistance={4} maxDistance={20} />
        <FocusCamera target={target} />
        <FpsGuard onSlow={onFallback} />
      </Canvas>
      <LabelLayer texts={texts} els={els} style={{ color: bg.text }} />
      {s && (
        <div className="card" style={{ position: 'absolute', right: 10, top: 10, padding: '8px 12px', fontSize: 12, background: 'var(--bg-overlay)', pointerEvents: 'none' }}>
          <div style={{ fontWeight: 600 }}>{s.leaf ? 'Leaf' : s.truncated ? 'Subtree (continues)' : s.split}</div>
          <div className="secondary">{s.samples.toLocaleString()} clients · {s.yes.toLocaleString()} subscribed ({(s.rate * 100).toFixed(1)}%)</div>
          {s.edge && <div className="muted">reached via: {s.edge}</div>}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- TabNet masks in 3D

export function MaskBars3D({ steps, features, values, height = 420, onFallback }: { steps: number; features: string[]; values: number[][]; height?: number; onFallback: () => void }) {
  const bg = useBg()
  const els = useRef<(HTMLDivElement | null)[]>([])
  const max = Math.max(...values.flat(), 1e-6)
  const nF = features.length
  const fx = (f: number) => (f - (nF - 1) / 2) * 0.55
  const sz = (st: number) => (st - (steps - 1) / 2) * 1.3
  const anchors: [number, number, number][] = [
    ...features.map((_, i) => [fx(i), -0.3, sz(steps - 1) + 1.1] as [number, number, number]),
    ...Array.from({ length: steps }, (_, st) => [fx(0) - 1.3, 0, sz(st)] as [number, number, number]),
  ]
  const texts = [...features, ...Array.from({ length: steps }, (_, st) => `step ${st + 1}`)]
  return (
    <div style={{ position: 'relative', height, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
      role="img" aria-label={`TabNet feature masks: ${steps} decision steps by ${nF} features, bar height = mask weight`}>
      <Canvas camera={{ position: [0, 7, 13], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[6, 10, 6]} intensity={0.8} />
        {values.map((row, st) => row.map((v, f) => {
          const h = 0.05 + 4 * (v / max)
          const c = new THREE.Color('#2A4A48').lerp(new THREE.Color('#9FCBB4'), v / max)
          return (
            <mesh key={`${st}-${f}`} position={[fx(f), h / 2, sz(st)]}>
              <boxGeometry args={[0.4, h, 0.9]} />
              <meshStandardMaterial color={c} />
            </mesh>
          )
        }))}
        <ProjectedLabels anchors={anchors} els={els} />
        <OrbitControls enablePan={false} minDistance={6} maxDistance={24} />
        <FpsGuard onSlow={onFallback} />
      </Canvas>
      <LabelLayer texts={texts} els={els} style={{ color: bg.muted, fontSize: 10 }} />
    </div>
  )
}
