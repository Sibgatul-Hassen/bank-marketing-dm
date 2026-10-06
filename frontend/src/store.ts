import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'dark' | 'light'
export type ExplainTab = 'global' | 'local' | 'compare'

interface UIState {
  theme: Theme
  explainOpen: boolean
  explainTab: ExplainTab
  localIndex: Record<string, number>
  currentModel: string | null
  previousModel: string | null
  setTheme: (t: Theme) => void
  toggleTheme: () => void
  setExplainOpen: (v: boolean) => void
  setExplainTab: (t: ExplainTab) => void
  setLocalIndex: (model: string, i: number) => void
  visitModel: (id: string) => void
}

export const useUI = create<UIState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      explainOpen: true,
      explainTab: 'global',
      localIndex: {},
      currentModel: null,
      previousModel: null,
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set({ theme: get().theme === 'dark' ? 'light' : 'dark' }),
      setExplainOpen: (explainOpen) => set({ explainOpen }),
      setExplainTab: (explainTab) => set({ explainTab }),
      setLocalIndex: (model, i) => set({ localIndex: { ...get().localIndex, [model]: i } }),
      visitModel: (id) => {
        const cur = get().currentModel
        if (cur === id) return
        set({ previousModel: cur, currentModel: id })
      },
    }),
    { name: 'bm-theme', partialize: (s) => ({ theme: s.theme, explainOpen: s.explainOpen, explainTab: s.explainTab }) },
  ),
)
