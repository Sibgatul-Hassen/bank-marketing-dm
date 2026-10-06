import { AnimatePresence, motion, useScroll, useSpring } from 'framer-motion'
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { useArtifact } from '../data/loader'
import type { Manifest } from '../data/types'
import { useUI } from '../store'
import { EASE, MODEL_NAMES, MODEL_ORDER } from '../theme/palette'

export function LogoMark({ size = 28 }: { size?: number }) {
  // An abstract branching node: a decision-tree split that is also a graph neighbourhood.
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <g stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" fill="none">
        <path d="M16 7v7M16 14l-7 6M16 14l7 6M9 20l-2.5 5.5M9 20l3 5.5M23 20l2.5 5.5" />
        <path d="M23 20l-11 5.5" strokeOpacity="0.35" strokeDasharray="1.5 2.5" />
      </g>
      <circle cx="16" cy="7" r="2.8" fill="var(--accent)" />
      <circle cx="9" cy="20" r="2.3" fill="var(--accent)" />
      <circle cx="23" cy="20" r="2.3" fill="var(--amber)" />
      <circle cx="6.5" cy="25.5" r="1.6" fill="var(--text-muted)" />
      <circle cx="12" cy="25.5" r="1.6" fill="var(--text-muted)" />
      <circle cx="25.5" cy="25.5" r="1.6" fill="var(--text-muted)" />
    </svg>
  )
}

const NAV = [
  { group: 'Foundations', items: [{ to: '/', label: 'Overview' }, { to: '/data', label: 'Data & Preprocessing' }, { to: '/eda', label: 'Exploratory Analysis' }] },
  { group: 'Models', items: MODEL_ORDER.map((id) => ({ to: `/models/${id}`, label: '', id })) },
  { group: 'Synthesis', items: [{ to: '/compare', label: 'Comparison' }] },
]

function SunMoon({ theme }: { theme: string }) {
  return theme === 'dark' ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
  )
}

function Header({ onMenu, showExplainToggle }: { onMenu: () => void; showExplainToggle: boolean }) {
  const { theme, toggleTheme, explainOpen, setExplainOpen } = useUI()
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 200, damping: 40 })
  return (
    <header style={{ position: 'fixed', inset: '0 0 auto 0', height: 'var(--header-h)', zIndex: 40, backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)', background: 'color-mix(in srgb, var(--bg-base) 78%, transparent)', borderBottom: '1px solid var(--border)' }}>
      <div style={{ height: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '0 16px' }}>
        <button className="btn icon-btn lg:hidden" onClick={onMenu} aria-label="Open navigation">
          <svg width="18" height="18" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8" fill="none" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
        </button>
        <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-primary)' }} aria-label="Home">
          <LogoMark />
          <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
            <span style={{ fontWeight: 600, fontSize: 15 }}>Bank Marketing</span>
            <span className="muted hidden sm:block" style={{ fontSize: 11 }}>Leakage-aware comparison of tabular learning methods</span>
          </span>
        </NavLink>
        <div style={{ flex: 1 }} />
        {showExplainToggle && (
          <button className="btn" onClick={() => setExplainOpen(!explainOpen)} aria-pressed={explainOpen} aria-label="Toggle explain panel">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></svg>
            <span className="hidden sm:inline">Explain</span>
          </button>
        )}
        <button className="btn icon-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
          <SunMoon theme={theme} />
        </button>
        <a className="btn icon-btn" href="https://github.com/Sibgatul-Hassen/bank-marketing-dm" target="_blank" rel="noreferrer" aria-label="GitHub repository">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.6 22.4c.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.5-3.9-1.5-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0C17 4.7 18 5 18 5c.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.2c0 .3.2.7.8.6A11.5 11.5 0 0 0 12 .5z" /></svg>
        </a>
      </div>
      <motion.div style={{ scaleX, transformOrigin: '0 0', position: 'absolute', left: 0, right: 0, bottom: -1, height: 2, background: 'var(--accent)' }} />
    </header>
  )
}

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const manifest = useArtifact<Manifest>('manifest.json').data
  const status = (id: string) => manifest?.models.find((m) => m.model_id === id)
  return (
    <nav aria-label="Pages" style={{ padding: '20px 12px', display: 'flex', flexDirection: 'column', gap: 18 }}>
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="eyebrow" style={{ padding: '0 10px 6px' }}>{g.group}</div>
          {g.items.map((it) => {
            const id = (it as { id?: string }).id
            const st = id ? status(id) : undefined
            const label = id ? (st?.display_name ?? MODEL_NAMES[id] ?? id) : it.label
            const dot = id ? (st?.status === 'ok' ? 'var(--accent)' : st?.status === 'invalid' ? 'var(--warning)' : 'var(--border-strong)') : null
            const done = it.to === '/compare' ? manifest?.comparison_ok : it.to === '/data' || it.to === '/eda' ? manifest?.dataset_ok : undefined
            return (
              <NavLink key={it.to} to={it.to} end onClick={onNavigate} className="navlink"
                style={({ isActive }) => ({ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px', borderRadius: 8, fontSize: 14,
                  color: isActive ? 'var(--accent-text)' : 'var(--text-secondary)', background: isActive ? 'var(--accent-subtle)' : 'transparent' })}>
                <span aria-hidden="true" style={{ width: 7, height: 7, borderRadius: 99, flexShrink: 0,
                  background: dot ?? (done === undefined ? 'transparent' : done ? 'var(--accent)' : 'var(--border-strong)'),
                  border: dot || done !== undefined ? 'none' : '1px solid var(--border-strong)' }} />
                <span>{label}</span>
                {id && st?.status !== 'ok' && <span className="sr-only">(not built yet)</span>}
              </NavLink>
            )
          })}
        </div>
      ))}
      <p className="muted" style={{ fontSize: 11, padding: '0 10px' }}>● built and validated · ○ not built yet</p>
    </nav>
  )
}

function Footer() {
  return (
    <footer className="footer themed" style={{ borderTop: '1px solid var(--border)', minHeight: 56, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '6px 24px', padding: '12px 16px', fontSize: 13, color: 'var(--text-muted)' }}>
      <div className="footer-left" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span className="copyright-symbol" style={{ color: 'var(--accent)', display: 'inline-block', transition: 'transform 160ms var(--ease)' }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.2)')} onMouseLeave={(e) => (e.currentTarget.style.transform = 'none')}>©</span>
        <span className="copyright-name" style={{ color: 'var(--text-secondary)' }}>Sibgatul Hassen</span>
      </div>
      <div className="footer-center" style={{ flex: 1, textAlign: 'center', minWidth: 200 }}>Data Mining · United International University</div>
      <div className="footer-right">2026 · UCI Bank Marketing (Moro et al., 2014)</div>
    </footer>
  )
}

export function Layout({ children, explain }: { children: ReactNode; explain?: ReactNode }) {
  const [menu, setMenu] = useState(false)
  const loc = useLocation()
  const theme = useUI((s) => s.theme)
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])
  useEffect(() => { window.scrollTo({ top: 0 }); setMenu(false) }, [loc.pathname])
  const panel = !!explain

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <a href="#main" className="sr-only focus:not-sr-only" style={{ position: 'absolute', zIndex: 60, padding: 8 }}>Skip to content</a>
      <Header onMenu={() => setMenu(true)} showExplainToggle={panel} />
      <div style={{ display: 'flex', flex: 1, paddingTop: 'var(--header-h)' }}>
        <aside className="hidden lg:block themed" style={{ width: 'var(--nav-w)', flexShrink: 0, borderRight: '1px solid var(--border)', position: 'sticky', top: 'var(--header-h)', height: 'calc(100vh - var(--header-h))', overflowY: 'auto' }}>
          <Nav />
        </aside>
        <AnimatePresence>
          {menu && (
            <>
              <motion.div key="scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setMenu(false)}
                style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 45 }} />
              <motion.aside key="drawer" initial={{ x: -260 }} animate={{ x: 0 }} exit={{ x: -260 }} transition={{ duration: 0.28, ease: EASE }}
                style={{ position: 'fixed', top: 0, bottom: 0, left: 0, width: 260, background: 'var(--bg-surface)', zIndex: 50, overflowY: 'auto', borderRight: '1px solid var(--border)' }}>
                <div style={{ padding: '16px 16px 0', display: 'flex', alignItems: 'center', gap: 10 }}><LogoMark /><strong>Pages</strong></div>
                <Nav onNavigate={() => setMenu(false)} />
              </motion.aside>
            </>
          )}
        </AnimatePresence>
        <div style={{ flex: 1, minWidth: 0, display: 'flex' }}>
          <main id="main" style={{ flex: 1, minWidth: 0, padding: '32px 16px 64px' }}>
            <AnimatePresence mode="wait">
              <motion.div key={loc.pathname} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.38, ease: EASE }} style={{ maxWidth: 900, margin: '0 auto' }}>
                {children}
              </motion.div>
            </AnimatePresence>
          </main>
          {panel && explain}
        </div>
      </div>
      <Footer />
    </div>
  )
}
