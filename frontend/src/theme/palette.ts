// §10.2 data palette — deliberately restrained, colourblind-safe, max six series.
export const DATA_COLORS = ['#4FB3A0', '#D4A373', '#7B9FC7', '#C88B8B', '#A89BC4', '#8FA880']
export const SEQUENTIAL = ['#1E2832', '#2A4A48', '#36695E', '#4A8A77', '#6FAB93', '#9FCBB4']
export const DIVERGING = ['#C88B8B', '#D9B3B3', '#E8E3E0', '#A9CFC4', '#4FB3A0']
export const CLASS = { yes: '#4FB3A0', no: '#6B7E8F' }
export const EASE = [0.22, 1, 0.36, 1] as const

export const MODEL_ORDER = ['decision_tree', 'catboost', 'tabnet', 'tabpfn', 'umap_hdbscan', 'fp_growth']
export const MODEL_COLORS: Record<string, string> = {
  decision_tree: '#7B9FC7', catboost: '#4FB3A0', tabnet: '#A89BC4', tabpfn: '#D4A373',
  umap_hdbscan: '#8FA880', fp_growth: '#C88B8B',
}
export const MODEL_NAMES: Record<string, string> = {
  decision_tree: 'Decision Tree', catboost: 'CatBoost', tabnet: 'TabNet', tabpfn: 'TabPFN',
  umap_hdbscan: 'UMAP + HDBSCAN', fp_growth: 'FP-Growth',
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
